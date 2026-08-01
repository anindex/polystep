"""Candidate scoring for the NN paths.

``NNCostEvaluator`` dispatches each call to the cheapest path the model and call shape
allow: a bmm plan for pure MLPs, ``vmap`` over the parameter dict, or an in-place
weight swap that holds one weight set regardless of candidate count.

The site-aware evaluators below it (``SiteVmapEvaluator``, ``SparseDeltaEvaluator``,
``SubspaceDeltaEvaluator``, ``FactoredEvaluator``) batch only the one parameter a
candidate perturbs, so the layers ahead of it run once per chunk.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, TYPE_CHECKING, Union

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from .projection import SparseRandomProjection
from .solvers._shared import loss_buffer_dtype

if TYPE_CHECKING:
    from .hybrid_subspace import HybridSubspace


# Smallest candidate batch worth keeping the batched paths for. Below it the weight
# stack no longer amortizes, so the in-place path's single weight set is preferable.
_MIN_BATCHED_CANDIDATES = 8


def _is_vmap_error(e: BaseException) -> bool:
    """Whether an exception came from the vmap transform rather than the model.

    A bare "batched" would also match an error in the user's forward and demote it to the
    ~N-times-slower loop, hiding it. Match functorch's own markers: "batched tensor" is
    the BatchedTensor repr, "vmap"/"functorch"/"torch.func" name the transform, and
    "randomness" is its op guard for a forward that draws its own noise.
    """
    msg = str(e).lower()
    return any(k in msg for k in ("vmap", "functorch", "torch.func", "batched tensor", "randomness"))


def auto_detect_chunk_size(
    model: nn.Module,
    safety_factor: float = 2.0,
    compile_overhead: bool = False,
) -> Optional[int]:
    """Estimate safe vmap chunk_size from model size and GPU memory.

    Returns None when the model is on CPU (no memory limit needed),
    even if the machine has a GPU available. On GPU, estimates
    per-evaluation memory as 4x parameter memory (conservative
    heuristic for activations + intermediates) and divides available
    GPU memory by this estimate.

    Args:
        model: The model to estimate for.
        safety_factor: Divisor for extra safety margin (default 2.0).
        compile_overhead: When True, multiply the safety factor by 1.5
            to account for the 10-20% extra peak memory ``torch.compile``
            pulls in for CUDA graph capture.

    Returns:
        Recommended chunk_size, or None if model is on CPU.
    """
    # Check actual device of model parameters, not global CUDA availability
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        return None  # No parameters

    if param_device.type != "cuda":
        return None

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    # Heuristic: 4x (params + buffers) for activations + intermediates
    per_eval_bytes = (param_bytes + buffer_bytes) * 4

    if per_eval_bytes <= 0:
        return None

    # torch.compile reserves CUDA-graph workspaces and intermediate
    # buffers that push peak memory ~10-20% above the eager path; bake
    # the headroom into the safety factor so the first call does not OOM.
    effective_safety = safety_factor * 1.5 if compile_overhead else safety_factor

    free_mem, _ = torch.cuda.mem_get_info(param_device)
    chunk = max(1, int(free_mem / (per_eval_bytes * effective_safety)))
    return chunk


_UNSET = object()  # sentinel distinguishing "not yet computed" from None


def _autocast_ctx(device: torch.device, dtype: Optional[torch.dtype]):
    """Autocast frame for a candidate forward, or a no-op when ``dtype`` is None.

    Only the arithmetic is affected; the parameters stay at their own dtype, so a
    candidate perturbation is not rounded away before it reaches the loss the way
    ``mixed_precision`` casts it away.
    """
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype or torch.bfloat16,
        enabled=dtype is not None,
    )


def _reusable(buffer: torch.Tensor, target: torch.Tensor) -> bool:
    """Whether ``buffer`` can back up ``target`` in place.

    Device and dtype matter as much as shape: after ``model.to(...)`` or a cast, a
    stale buffer restores through a silent conversion or from the wrong device.
    """
    return buffer.shape == target.shape and buffer.dtype == target.dtype and buffer.device == target.device


def _uniform_float_dtype(*param_sources) -> Optional[torch.dtype]:
    """The one float dtype every parameter shares, or None if they differ.

    None means the model computes in more than one dtype and casts internally, so a
    caller must leave its inputs and its hand-written arithmetic alone.
    """
    seen = set()
    for src in param_sources:
        values = src.values() if isinstance(src, dict) else [src]
        for v in values:
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                seen.add(v.dtype)
    if len(seen) != 1:
        return None
    return seen.pop()


def _batched_loss_kind(loss_fn) -> Optional[str]:
    """Name the per-sample reduction the batched-linear path can reproduce.

    The bmm forward does not depend on the loss; only the final reduction does. A
    configured loss (class weights, label smoothing, a custom ignore_index, a non-mean
    reduction) has no such reproduction and falls back to vmap, which calls the real
    ``loss_fn``. Returns None in that case.

    A subclass that overrides ``forward`` also returns None: the reductions below are
    the functional form of the base class, so reproducing a subclass that adds a term
    would score every candidate on a different objective than the closure.
    """
    if getattr(loss_fn, "reduction", None) != "mean":
        return None
    for base, kind in ((nn.CrossEntropyLoss, "cross_entropy"), (nn.MSELoss, "mse"), (nn.L1Loss, "l1")):
        if not isinstance(loss_fn, base):
            continue
        if type(loss_fn).forward is not base.forward:
            return None
        if kind == "cross_entropy" and not (
            loss_fn.weight is None and loss_fn.ignore_index == -100 and float(loss_fn.label_smoothing) == 0.0
        ):
            return None
        return kind
    return None


def _reduce_per_candidate(outputs, targets, loss_fn, loss_kind):
    """Reduce ``(N, B, ...)`` outputs to one loss per candidate.

    Shared by the bmm and factored paths, which build the same ``outputs`` by different
    routes. ``targets=None`` defers to ``loss_fn``; otherwise the reduction is the
    per-sample form named by ``loss_kind``.
    """
    if targets is None:
        return loss_fn(outputs).mean(dim=1) if outputs.dim() > 2 else loss_fn(outputs)

    n = outputs.shape[0]
    if loss_kind == "cross_entropy":
        tgt = targets.unsqueeze(0).expand(n, -1)  # (N, B)
        return (
            torch.nn.functional.cross_entropy(
                outputs.reshape(n * targets.shape[0], -1), tgt.reshape(-1), reduction="none"
            )
            .reshape(n, -1)
            .mean(dim=1)
        )  # (N,)

    # Cast before expand, or the cast materializes the whole (N, *target) tensor rather
    # than the one target. Promote, so FP64 targets keep their precision against FP32.
    dtype = torch.promote_types(outputs.dtype, targets.dtype)
    tgt = targets.to(dtype).unsqueeze(0).expand(n, *targets.shape)
    fn = torch.nn.functional.mse_loss if loss_kind == "mse" else torch.nn.functional.l1_loss
    return fn(outputs.to(dtype), tgt, reduction="none").flatten(1).mean(dim=1)  # (N,)


class NNCostEvaluator:
    """Vectorized NN cost evaluation via vmap + functional_call.

    Evaluates a model at N batched parameter configurations. Uses
    ``torch.vmap`` for vectorized inference; falls back to a sequential
    Python loop if vmap fails (one-time warning emitted).

    Args:
        model: The ``nn.Module`` to evaluate. Will be put in eval mode.
        loss_fn: Loss function with signature:

            - ``loss_fn(output, targets) -> scalar`` (supervised), or
            - ``loss_fn(output) -> scalar`` (unsupervised, targets=None).
        chunk_size: vmap ``chunk_size`` for memory control.
            ``None`` = evaluate all at once (no chunking).
            ``"auto"`` = auto-detect from model size and GPU memory.
            Positive int = evaluate in chunks of this size.
        compile_vmap: If True, wrap the vmap evaluation in
            ``torch.compile(mode="default")`` for Inductor kernel fusion
            (fusion only, NOT CUDA graphs, so launch overhead is not
            eliminated; expect a modest win). Falls back to
            eager on failure. Best for CUDA models. Default False. For the
            launch-bound CUDA-graph win on large recurrent nets, use the
            in-place ``compile_forward`` path instead.
        compile_forward: On the in-place path, compile the
            forward+loss closure with ``torch.compile(mode="reduce-overhead")``
            (CUDA graphs). Unlike ``compile_vmap`` this does eliminate per-kernel
            launch overhead, because the in-place path has no vmap chunk-concat.
            The Python swap loop stays; only the captured ``model(inputs)`` is
            replayed per candidate. Requires CUDA and a static input/param shape;
            falls back to eager on failure. ``None`` (default) enables it wherever the
            in-place path runs, which is auto for >500K-param GPU models or forced with
            ``use_inplace=True``; ``False`` opts out.
        per_sample: Diagnostic mode. If True, ``evaluate`` returns the unreduced
            ``(N, B)`` per-sample losses instead of ``(N,)``, at no extra forward
            cost: the values are computed either way and the batch mean discards
            them. Requires ``loss_fn.reduction == "none"``. Supported on the vmap
            and sequential paths; the batched-linear and in-place paths raise.
            Default False.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        chunk_size: Union[None, int, str] = None,
        compile_vmap: bool = False,
        use_inplace: Optional[bool] = None,
        compile_forward: Optional[bool] = None,
        per_sample: bool = False,
        autocast_dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.autocast_dtype = autocast_dtype
        # A reducing loss_fn returns (N,) here, silently, which is the shape per_sample
        # exists to avoid. Reject it rather than hand back the wrong thing.
        if per_sample and getattr(loss_fn, "reduction", "none") != "none":
            raise ValueError(
                f"per_sample=True needs loss_fn.reduction='none', got "
                f"{loss_fn.reduction!r}. A reducing loss collapses the batch before "
                f"evaluate() can return the (N, B) tensor."
            )
        self.per_sample = per_sample
        self._chunk_size_raw = chunk_size
        self._chunk_size_cached = _UNSET  # lazily computed for "auto" mode
        self._vmap_failed = False
        self._warned = False
        self._compile_vmap = compile_vmap
        self._compiled_vmap_fn = None
        self._compile_failed = False

        # In-place forward+loss compile (CUDA graphs, launch-bound win).
        # Resolved after in-place auto-detection below.
        self._compiled_fwd_loss = None
        self._compile_forward_failed = False
        self._compile_forward_verified = False
        # Restore buffers for the two in-place paths, allocated on first use.
        self._inplace_backup = None
        self._subspace_backup = None
        self._cast_cache = None  # (inputs, dtype, converted) for the per-chunk cast
        self._input_dtype_cache = _UNSET

        # Force eval mode for consistent behavior (frozen BN stats, no dropout)
        model.eval()

        # Fast batched-linear evaluator (MLP-only models), when the loss has a
        # per-sample form the bmm path can reproduce.
        loss_kind = _batched_loss_kind(loss_fn)
        self._batched_linear = (
            BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind) if loss_kind is not None else None
        )

        # The batched paths hold N copies of the weights and activations at once; the
        # in-place path holds one, at the cost of N sequential forwards. Pick in-place
        # only when the batched footprint would not fit, rather than on a parameter
        # count, which ignores how many candidates and how much free memory there are.
        # Pass use_inplace=True/False to override.
        if use_inplace is not None:
            self._use_inplace = use_inplace
        else:
            self._use_inplace = self._batched_footprint_exceeds_free_memory(model)
        # An explicit True also means "run the real forward". A forward that reads state
        # functional_call cannot substitute (a weight repacked at build time, a numpy
        # reconstruction, anything held by object reference) scores every candidate the
        # same on the stateless paths. Auto-detection only judges memory, so it does not
        # disqualify them.
        self._inplace_forced = use_inplace is True

        # The in-place path is a Python loop of N sequential forwards, so it is
        # launch-bound, which is what CUDA graphs fix, and it is the only path the flag
        # affects. An explicit value still wins; a compile failure falls back to eager.
        if compile_forward is None:
            self._compile_forward = self._use_inplace
        else:
            self._compile_forward = compile_forward

        # Cache frozen buffers from the real model (shared across all particles)
        self._buffers = dict(model.named_buffers())

        # Cache param dict for in-place evaluation (avoids O(L) module traversal per call).
        # remove_duplicate=False so tied weights resolve under every module path they
        # appear at: batch_unflatten emits an alias key per shared entry, and the
        # deduplicated dict would report those aliases as unknown parameters.
        self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))

    def _autocast(self, device: torch.device):
        """Run the candidate forward in ``autocast_dtype``, or a no-op when unset."""
        return _autocast_ctx(device, self.autocast_dtype)

    def reset_vmap(self) -> None:
        """Rebuild every cache that encodes the model's current shape.

        Call after swapping a layer, rebinding a buffer, replacing a Parameter, or
        moving the model. The bmm plan, the buffer dict and the parameter dict all
        pin the model as it was at construction: without this a swapped activation
        keeps scoring through the old plan and a rebound buffer keeps its old value,
        both silently.
        """
        self._vmap_failed = False
        self._warned = False
        # Symmetric recovery: a prior compile failure should not stay latched
        # after the model/device changed under us.
        self._compile_failed = False
        self._compiled_vmap_fn = None
        self._compile_forward_failed = False
        self._compiled_fwd_loss = None
        self._compile_forward_verified = False
        self._buffers = dict(self.model.named_buffers())
        self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))
        self._inplace_backup = None
        self._subspace_backup = None
        self._cast_cache = None
        self._input_dtype_cache = _UNSET
        loss_kind = _batched_loss_kind(self.loss_fn)
        self._batched_linear = (
            BatchedLinearEvaluator.try_build(self.model, self.loss_fn, loss_kind) if loss_kind is not None else None
        )

    @property
    def chunk_size(self) -> Optional[int]:
        """Resolved chunk_size for vmap.

        Returns None (no chunking), or a positive int. When
        ``chunk_size="auto"`` was passed, queries GPU memory once to compute
        a safe value (returns None on CPU), then caches the result.
        """
        if self._chunk_size_raw == "auto":
            if self._chunk_size_cached is _UNSET:
                self._chunk_size_cached = auto_detect_chunk_size(
                    self.model,
                    compile_overhead=self._compile_vmap,
                )
            return self._chunk_size_cached
        return self._chunk_size_raw

    @torch.inference_mode()
    def evaluate(
        self,
        stacked_params: dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate model at batched parameters.

        Args:
            stacked_params: ``{key: (N, *param_shape)}`` stacked param dicts
                (from ``ParamLayout.batch_unflatten()``).
            inputs: Input data batch (broadcast to all N evaluations).
            targets: Optional targets (broadcast to all N evaluations).

        Returns:
            Losses tensor of shape ``(N,)``, or ``(N, B)`` when ``per_sample``.
        """
        # Already eval (set in __init__), so this is a bool check rather than an
        # O(L) module walk. Restore on exit for external callers.
        was_training = self.model.training
        if was_training:
            self.model.eval()

        # Match float inputs to the (maybe bf16) param dtype so bmm/vmap don't hit a
        # dtype mismatch under mixed precision. Integer inputs and targets untouched.
        # From the consuming layer, not from whichever entry is first: an FP64 scalar
        # ahead of an FP32 Linear would cast the batch to FP64 and crash it.
        inputs = self.cast_inputs(inputs, self._input_dtype(stacked_params))

        try:
            # One autocast frame over every candidate-scoring path below.
            with self._autocast(inputs.device):
                return self._dispatch(stacked_params, inputs, targets)
        finally:
            if was_training:
                self.model.train()

    def cast_inputs(self, inputs: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
        """Cast float inputs to ``dtype``, reusing the last result.

        The step calls the evaluators once per chunk with the same batch, so without
        this a 52-chunk sweep runs 52 identical conversions.
        """
        if dtype is None or not inputs.is_floating_point() or inputs.dtype == dtype:
            return inputs
        cached = self._cast_cache
        if cached is not None and cached[0] is inputs and cached[1] is dtype:
            return cached[2]
        out = inputs.to(dtype)
        self._cast_cache = (inputs, dtype, out)
        return out

    def _input_dtype(self, stacked_params) -> Optional[torch.dtype]:
        """Float dtype the first input-consuming layer expects, or None to leave it be.

        Cached: ``evaluate`` runs once per chunk, and walking every module there showed
        up in the step profile. ``reset_vmap`` clears it.
        """
        if not stacked_params:
            return None
        if self._input_dtype_cache is _UNSET:
            self._input_dtype_cache = next(
                (
                    m.weight.dtype
                    for m in self.model.modules()
                    if getattr(m, "weight", None) is not None and m.weight.is_floating_point()
                ),
                None,
            )
        if self._input_dtype_cache is not None:
            return self._input_dtype_cache
        first = next(iter(stacked_params.values()))
        return first.dtype if first.is_floating_point() else None

    def _dispatch(self, stacked_params, inputs, targets):
        """Pick the cheapest scoring path this model and call shape allow."""
        # In-place weight swap: O(1 x activation) memory regardless of N. Checked
        # before the bmm path, whose (N, d_out, d_in) stack is the O(N x activation)
        # cost this exists to avoid.
        if self._use_inplace:
            if self.per_sample:
                raise NotImplementedError(
                    "per_sample is not supported on the in-place path. Construct the "
                    "evaluator with use_inplace=False for diagnostic runs; the (N, B) "
                    "tensor defeats the point of the in-place path's O(1) memory anyway."
                )
            return self._evaluate_inplace(stacked_params, inputs, targets)

        # Batched bmm for Linear-only models. Supervised only; the pre-flattened
        # input reproduces the real forward only when a Flatten precedes every Linear.
        # Soft-label targets are (B, C) probabilities; the cross-entropy reduction below
        # assumes (B,) class indices and would fail in expand. vmap calls the real loss.
        if (
            self._batched_linear is not None
            and targets is not None
            and (inputs.dim() == 2 or self._batched_linear.leading_flatten)
            and not self.per_sample
            and not (self._batched_linear.loss_kind == "cross_entropy" and targets.dim() != 1)
        ):
            return self._batched_linear.evaluate(stacked_params, inputs, targets)

        if self._vmap_failed:
            result = self._evaluate_loop(stacked_params, inputs, targets)
        else:
            try:
                result = self._evaluate_vmap(stacked_params, inputs, targets)
            except Exception as e:
                # Only catch vmap/functorch errors; re-raise real bugs.
                if not _is_vmap_error(e):
                    raise
                if not self._warned:
                    warnings.warn(
                        f"vmap failed for {type(self.model).__name__}: {e}. Falling back to sequential "
                        f"evaluation (~N x slower). If this masks a real bug in the model's forward, "
                        f"the error text is above.",
                        stacklevel=2,
                    )
                    self._warned = True
                self._vmap_failed = True
                result = self._evaluate_loop(stacked_params, inputs, targets)
        return result

    def _evaluate_vmap(self, stacked_params, inputs, targets):
        """Vectorized evaluation via vmap + functional_call.

        When ``compile_vmap=True`` (set at init), wraps the batched evaluation
        in ``torch.compile(mode="default")`` for **Inductor kernel fusion only**.
        This does NOT capture CUDA graphs and so does NOT eliminate per-kernel
        launch overhead, ``mode="reduce-overhead"`` (CUDA graphs) is
        not used here because vmap's chunked-output concatenation
        conflicts with CUDA-graph tensor ownership. Expect a modest fusion win, not the
        launch-bound speedup.
        The CUDA-graph / launch-elimination lever lives on the in-place path
        (``compile_forward``), which has no chunk-concat. Falls back to eager
        vmap permanently on compilation failure (with a one-time warning).
        """
        buffers = self._buffers
        loss_fn = self.loss_fn
        model = self.model
        resolved_chunk = self.chunk_size
        per_sample = self.per_sample

        # inputs/targets are explicit args (in_dims=None), not closed-over, so a
        # cached torch.compile graph does not bake in the first call's batch.
        def single_eval(params, inputs, targets):
            # Buffers win on a key collision: they are frozen model state and are not
            # part of the OT optimization.
            full_dict = {**params, **buffers}
            output = functional_call(model, full_dict, (inputs,))
            if targets is not None:
                loss = loss_fn(output, targets)
            else:
                loss = loss_fn(output)
            if not per_sample:
                if loss.dim() > 0:
                    loss = loss.mean()
            elif loss.dim() == 0:
                # A callable carrying no ``reduction`` attribute passes the constructor
                # check, so a reducing one is only caught here.
                raise ValueError(
                    "per_sample=True needs an unreduced loss_fn, but it returned a scalar. Return one value per sample."
                )
            elif loss.dim() > 1:
                # An unreduced loss over non-scalar targets keeps the target's trailing
                # dims. per_sample promises one value per sample, so fold them.
                loss = loss.flatten(1).mean(dim=1)
            return loss

        batched = vmap(single_eval, in_dims=(0, None, None), chunk_size=resolved_chunk)

        # Compiled path: torch.compile on the vmapped forward for Inductor kernel
        # FUSION ONLY (mode="default", no CUDA graphs; see the method docstring).
        # Only attempted with compile_vmap=True. Lazy-compiled on first call.
        if self._compile_vmap and not self._compile_failed:
            if self._compiled_vmap_fn is None:
                try:
                    # "default" = fusion, no CUDA graphs. "reduce-overhead" is slower
                    # here: vmap already amortizes the launches graphs would remove.
                    self._compiled_vmap_fn = torch.compile(
                        batched,
                        mode="default",
                        fullgraph=False,
                    )
                except Exception as e:  # noqa: BLE001
                    self._compile_failed = True
                    self._compiled_vmap_fn = None
                    warnings.warn(
                        f"compile_vmap: torch.compile failed for "
                        f"{type(self.model).__name__} ({e}); using eager vmap for the "
                        f"rest of the run.",
                        stacklevel=2,
                    )

            if self._compiled_vmap_fn is not None:
                try:
                    return self._compiled_vmap_fn(stacked_params, inputs, targets)
                except Exception as e:  # noqa: BLE001
                    # Execution of the compiled graph failed - fall back permanently.
                    self._compile_failed = True
                    self._compiled_vmap_fn = None
                    warnings.warn(
                        f"compile_vmap: compiled forward raised at run time for "
                        f"{type(self.model).__name__} ({e}); using eager vmap for the "
                        f"rest of the run.",
                        stacklevel=2,
                    )

        return batched(stacked_params, inputs, targets)

    @staticmethod
    def _batched_footprint_exceeds_free_memory(model: nn.Module) -> bool:
        """Whether a batched candidate stack is too large for the device to be worth it.

        The batched paths are chunked, so N alone never decides this: what does is
        whether even a small batch of weight copies fits. Below the threshold the
        chunked batch wins on wall clock; above it the in-place path's one weight set
        is the only thing that fits.

        CPU always answers False: there is no hard ceiling to fall off, and N sequential
        forwards are slower there than a chunked batch.
        """
        try:
            first = next(model.parameters())
        except StopIteration:
            return False
        if first.device.type != "cuda":
            return False
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        free, _total = torch.cuda.mem_get_info(first.device)
        # Half the free memory, so activations and allocator fragmentation still fit.
        return _MIN_BATCHED_CANDIDATES * param_bytes > free * 0.5

    def _evaluate_loop(self, stacked_params, inputs, targets):
        """Sequential fallback when vmap is incompatible."""
        if not stacked_params:
            return torch.zeros(0, device=inputs.device)
        N = next(iter(stacked_params.values())).shape[0]
        if N == 0:
            return torch.zeros(0, device=inputs.device)
        losses = []
        for i in range(N):
            full_dict = {**{k: v[i] for k, v in stacked_params.items()}, **self._buffers}
            output = functional_call(self.model, full_dict, (inputs,))
            loss = self.loss_fn(output, targets) if targets is not None else self.loss_fn(output)
            if not self.per_sample:
                if loss.dim() > 0:
                    loss = loss.mean()
            elif loss.dim() == 0:
                raise ValueError(
                    "per_sample=True needs an unreduced loss_fn, but it returned a scalar. Return one value per sample."
                )
            elif loss.dim() > 1:
                loss = loss.flatten(1).mean(dim=1)
            losses.append(loss)
        return torch.stack(losses)

    def _forward_loss(self, inputs, targets):
        """Eager forward + reduced scalar loss on the model's CURRENT params."""
        output = self.model(inputs)
        loss = self.loss_fn(output, targets) if targets is not None else self.loss_fn(output)
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def _forward_loss_fn(self):
        """Return the per-candidate forward+loss callable for the in-place path.

        With ``compile_forward=True`` on CUDA, lazily compile a forward+loss
        closure with ``mode="reduce-overhead"`` (CUDA graphs), so replaying it
        per candidate eliminates kernel-launch overhead: the launch-bound win.
        The closure reads the model's current parameters, which the swap loop
        mutates via ``.data.copy_`` / ``_foreach_copy_`` between calls; ``copy_``
        preserves storage addresses, so graph replay reads the fresh weights.
        Safe here (unlike the vmap path) because there is no chunk-concat.
        Falls back to the eager closure permanently on failure.
        """
        if not (self._compile_forward and not self._compile_forward_failed):
            return self._forward_loss
        first = next(self.model.parameters(), None)
        if first is None or not first.is_cuda:
            return self._forward_loss  # CUDA-graph capture needs CUDA
        if self._compiled_fwd_loss is None:
            model, loss_fn = self.model, self.loss_fn

            def fwd_loss(inputs, targets):
                output = model(inputs)
                loss = loss_fn(output, targets) if targets is not None else loss_fn(output)
                if loss.dim() > 0:
                    loss = loss.mean()
                return loss

            try:
                self._compiled_fwd_loss = torch.compile(fwd_loss, mode="reduce-overhead", fullgraph=False)
            except Exception as e:  # noqa: BLE001
                self._compile_forward_failed = True
                self._compiled_fwd_loss = None
                warnings.warn(
                    f"compile_forward: torch.compile failed for "
                    f"{type(self.model).__name__} ({e}); using eager forward.",
                    stacklevel=2,
                )
                return self._forward_loss
        return self._compiled_fwd_loss

    def _verified_forward_loss_fn(self, inputs, targets):
        """``_forward_loss_fn`` with its first invocation guarded.

        ``compile_forward`` compiles with ``mode="reduce-overhead"``, so CUDA-graph
        capture happens on the first *call*, not at ``torch.compile`` time. Without
        this the capture failure escapes as an unguarded error mid-loop, while the
        ``compile_vmap`` path falls back cleanly. Costs one extra forward, once per
        evaluator, against the N forwards the caller is about to run.
        """
        fwd_loss = self._forward_loss_fn()
        if fwd_loss is self._forward_loss or self._compile_forward_verified:
            return fwd_loss
        try:
            fwd_loss(inputs, targets)
        except Exception as e:
            self._compile_forward_failed = True
            self._compiled_fwd_loss = None
            warnings.warn(
                "compile_forward: compiled forward raised at run time for "
                f"{type(self.model).__name__} ({e}); using eager forward for the rest of the run.",
                stacklevel=2,
            )
            return self._forward_loss
        self._compile_forward_verified = True
        return fwd_loss

    def _evaluate_inplace(self, stacked_params, inputs, targets):
        """Memory-minimal evaluation via in-place weight swapping.

        Evaluates N parameter configurations sequentially by directly
        modifying model weights in-place. Uses only O(1 x activation)
        memory regardless of N, making it feasible for large models
        where vmap would OOM.

        Inspired by MeZO's in-place perturbation strategy and ZO2's
        sequential evaluation. Uses ``torch.inference_mode()`` to
        eliminate autograd overhead and view tracking.

        Wall-clock: roughly N times slower than vmap for small models, but for large
        models where each forward already saturates the GPU the gap narrows.
        """
        if not stacked_params:
            return torch.zeros(0, device=inputs.device)
        N = next(iter(stacked_params.values())).shape[0]
        device = inputs.device
        losses = torch.empty(N, device=device, dtype=loss_buffer_dtype(next(iter(stacked_params.values())).dtype))

        param_dict = self._param_dict_cache
        if not stacked_params.keys() <= param_dict.keys():
            # Model gained or renamed params since construction; refresh the cache
            # so new keys aren't silently evaluated with stale weights.
            param_dict = self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))
        unknown = stacked_params.keys() - param_dict.keys()
        if unknown:
            # Silently skipping these would evaluate every candidate at the base
            # weights and report a flat cost matrix as a real result.
            raise ValueError(
                f"{type(self.model).__name__} has no parameter(s) {sorted(unknown)}"
                f"; the candidate configurations cannot be applied. "
                f"Known parameters: {sorted(param_dict)}."
            )
        # Reused across calls: this runs once per chunk, so cloning the whole weight set
        # every time allocates the model again on every chunk of every step.
        backup = self._inplace_backup
        if (
            backup is None
            or backup.keys() != stacked_params.keys()
            or any(not _reusable(backup[k], param_dict[k].data) for k in stacked_params)
        ):
            backup = self._inplace_backup = {k: param_dict[k].data.clone() for k in stacked_params}
        else:
            for key in stacked_params:
                backup[key].copy_(param_dict[key].data)
        original_params = backup

        fwd_loss = self._verified_forward_loss_fn(inputs, targets)
        # Resolve the destination param views once so each swap is a single fused
        # foreach op instead of one kernel launch per parameter tensor.
        dst_keys = list(stacked_params)
        dsts = [param_dict[k].data for k in dst_keys]
        try:
            for i in range(N):
                # Swap all candidate weights in-place with one fused copy.
                torch._foreach_copy_(dsts, [stacked_params[k][i] for k in dst_keys])
                # Already under inference_mode from evaluate(). Store the detached
                # scalar and read it before the next replay overwrites the graph buffer.
                losses[i] = fwd_loss(inputs, targets).detach()
        finally:
            # Always restore, even on error. One fused copy, like the swap above.
            torch._foreach_copy_(dsts, [original_params[k] for k in dst_keys])

        return losses

    def evaluate_subspace_inplace(
        self,
        subspace: "HybridSubspace",
        projections: dict,
        base_sd: dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fused subspace-reconstruct + forward via in-place weight swap.

        Instead of materialising all ``N`` configurations' full weights
        up front via ``reconstruct_batch`` and then evaluating, this
        method reconstructs one configuration at a time directly into the
        model parameters using ``apply_perturbation_inplace`` and runs a
        single forward pass per configuration.

        Peak memory is ``O(model_params + batch * activation)``
        independent of ``N``; no stacked parameter dict is ever allocated.

        Args:
            subspace: HybridSubspace with ``apply_perturbation_inplace``.
            projections: Per-layer projection matrices.
            base_sd: Base (unperturbed) state_dict.
            flat_subspace_batch: (N, subspace_dim) subspace coordinates.
            inputs: Input data batch.
            targets: Optional targets.

        Returns:
            Losses tensor of shape (N,).
        """
        N = flat_subspace_batch.shape[0]
        device = inputs.device

        # Cast float inputs to the param dtype: under mixed_precision the model is BF16
        # while inputs arrive FP32. Integer inputs (token ids, targets) are untouched.
        first = next(self.model.parameters(), None)
        if first is not None:
            inputs = self.cast_inputs(inputs, first.dtype)

        losses = torch.empty(N, device=device, dtype=loss_buffer_dtype(flat_subspace_batch.dtype))

        was_training = self.model.training
        if was_training:
            self.model.eval()

        fwd_loss = self._verified_forward_loss_fn(inputs, targets)
        # Reuse the cached parameter dict; refresh only if it no longer covers the
        # entries this subspace perturbs.
        param_dict = self._param_dict_cache
        if not param_dict.keys() >= {s.entry_key for s in subspace.specs} & set(base_sd):
            param_dict = self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))
        if hasattr(subspace, "prepare_inplace"):
            subspace.prepare_inplace(base_sd)
        # Snapshot only the entries that will be written, not the whole state_dict, into
        # a buffer reused across chunks rather than reallocated per call.
        _keys = [k for k in base_sd if k in param_dict]
        entry_params = self._subspace_backup
        if (
            entry_params is None
            or list(entry_params) != _keys
            or any(not _reusable(entry_params[k], param_dict[k].data) for k in _keys)
        ):
            entry_params = self._subspace_backup = {k: param_dict[k].data.clone() for k in _keys}
        else:
            for k in _keys:
                entry_params[k].copy_(param_dict[k].data)
        try:
            for i in range(N):
                # Reconstruct weights for config i directly into model params
                subspace.apply_perturbation_inplace(
                    projections,
                    self.model,
                    base_sd,
                    flat_subspace_batch[i],
                    param_dict=param_dict,
                )
                # Forward pass - already under inference_mode from caller
                # (compiled + CUDA-graph-replayed when compile_forward=True).
                with self._autocast(device):
                    losses[i] = fwd_loss(inputs, targets).detach()
        finally:
            if hasattr(subspace, "release_inplace"):
                subspace.release_inplace()
            restore = list(entry_params)
            torch._foreach_copy_([param_dict[k].data for k in restore], [entry_params[k] for k in restore])
            if was_training:
                self.model.train()

        return losses


_SEQ_EQUIVALENT_WARNED = set()


def _warn_if_sequential_equivalent(model: nn.Module, supported: tuple) -> None:
    """Warn once when a hand-written forward is the only thing blocking the fast paths.

    Every batched evaluator requires ``type(model).forward is nn.Sequential.forward``,
    because the plan is rebuilt from ``named_children()`` and a custom forward can apply
    ops those children do not name. A model that is otherwise a plain MLP therefore opts
    itself out of the bmm and delta paths by defining a ``forward`` identical to the one
    it would have inherited, and the only symptom is that the step is several times
    slower.
    """
    key = id(type(model))
    if key in _SEQ_EQUIVALENT_WARNED:
        return
    children = list(model.named_children())
    if not children:
        return
    flat = []
    for name, mod in children:
        flat.extend(mod.named_children() if isinstance(mod, nn.Sequential) else [(name, mod)])
    if not any(isinstance(m, nn.Linear) for _, m in flat):
        return
    if not all(isinstance(m, supported) for _, m in flat):
        return
    # Require an activation (or flatten) among the children. Without one the forward
    # applies its nonlinearity inline, the children are not the whole computation, and
    # subclassing nn.Sequential would change the model rather than speed it up.
    if not any(not isinstance(m, nn.Linear) for _, m in flat):
        return
    # Compare parameter mass, not names: a wrapper module prefixes them ("net.0.weight"),
    # so a name-set test would miss exactly the wrap-a-Sequential case.
    linear_numel = sum(p.numel() for _, m in flat if isinstance(m, nn.Linear) for p in m.parameters())
    if linear_numel != sum(p.numel() for p in model.parameters()):
        return
    _SEQ_EQUIVALENT_WARNED.add(key)
    warnings.warn(
        f"{type(model).__name__} is built from Linear/activation layers only, but defines its own "
        f"forward, so the batched and delta evaluators decline it and every candidate goes through "
        f"vmap. Subclass nn.Sequential instead (pass an OrderedDict to keep the same state_dict "
        f"keys) to take the fast path.",
        stacklevel=3,
    )


# Layer types the plan understands, paired with the tag ``evaluate`` dispatches on.
# The tag comes from the matched base class, never from the concrete class name: a
# subclass of nn.Flatten named otherwise would otherwise be *called* where the plan
# means to skip it. Linear is first so it wins over any later overlap.
_TAGGED_LAYERS = (
    (nn.Linear, "linear"),
    (nn.Flatten, "flatten"),
    (nn.Dropout, "dropout"),
    (nn.ReLU, "activation"),
    (nn.LeakyReLU, "activation"),
    (nn.Sigmoid, "activation"),
    (nn.Tanh, "activation"),
    (nn.GELU, "activation"),
    (nn.SiLU, "activation"),
)
_SUPPORTED_LAYERS = tuple(base for base, _ in _TAGGED_LAYERS)


def _probe_elementwise(module: nn.Module) -> bool:
    """Does the declared contract hold on a sample? One forward at build time.

    A softmax, layer norm or per-tensor quantizer answers differently for a slice than
    for the same positions of the whole. The outlier sits outside every probed slice so
    any reduction moves, and the slices span all three axes because a row-wise reduction
    survives a leading-dim one.
    """
    x = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4).clone()
    x[-1, -1, -1] = 37.0
    slices = ((slice(None, 1), Ellipsis), (0, Ellipsis), (Ellipsis, slice(None, 2)))
    try:
        with torch.no_grad():
            full = module(x)
            if full.shape != x.shape or full.dtype != x.dtype:
                return False
            return all(torch.equal(full[s], module(x[s])) for s in slices)
    except Exception:
        return False


def _elementwise_entry(submod: nn.Module) -> bool:
    """Whether ``submod`` declared ``module(x)[i] == module(x[i])`` and can keep it.

    A parameter would have to be batched per candidate and a buffer is state the plan
    calls one activation several times over, so both are refused outright.
    """
    if not getattr(submod, "polystep_elementwise", False):
        return False
    if any(True for _ in submod.parameters()) or any(True for _ in submod.buffers()):
        return False
    return _probe_elementwise(submod)


def _weight_transforms(module: Optional[nn.Module]):
    """``(weight_fn, bias_fn)`` for a layer whose forward is ``x @ Q(w).t() + Qb(b)``.

    ``(None, None)`` for a plain Linear. Both must be elementwise: the batched path
    transforms the stacked weight, and the delta path needs ``Q(w + d) - Q(w)`` where a
    plain Linear takes ``d``.
    """
    if module is None:
        return None, None
    return getattr(module, "polystep_weight_transform", None), getattr(module, "polystep_bias_transform", None)


def _declares_weight_transform(model: nn.Module) -> bool:
    """Whether any submodule transforms its own weight before the matmul."""
    return any(getattr(m, "polystep_weight_transform", None) is not None for m in model.modules())


def _linear_like(submod: nn.Module) -> bool:
    """Whether a weight-transforming layer has the shape the plan needs of a Linear."""
    if getattr(submod, "polystep_weight_transform", None) is None:
        return False
    params = dict(submod.named_parameters())
    if set(params) - {"weight", "bias"} or "weight" not in params:
        return False
    if params["weight"].dim() != 2 or ("bias" in params and params["bias"].dim() != 1):
        return False
    return not any(True for _ in submod.buffers())


def _has_hooks(module: nn.Module) -> bool:
    """Whether ``module`` carries a forward hook. The hand-built matmuls never run it."""
    return bool(
        getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
        or getattr(module, "_forward_hooks_with_kwargs", None)
    )


def _plan_children(model: nn.Module):
    """``(name, submodule)`` in forward order, one entry per position.

    ``named_children`` dedupes by module identity, so ``Sequential(fc, ReLU(), fc)``
    yields ``fc`` once and the plan drops the second application.
    """
    for name, mod in model._modules.items():
        if mod is None:
            continue
        if isinstance(mod, nn.Sequential):
            for sub, m in mod._modules.items():
                if m is not None:
                    yield f"{name}.{sub}", m
        else:
            yield name, mod


def _plan_params_distinct(layer_keys) -> bool:
    """True when no two plan entries share a parameter tensor.

    Tied weights put one tensor under two plan names; the stacked dict only carries the
    first, so ``evaluate`` raises KeyError on the second.
    """
    seen: set[int] = set()
    for _name, tag, module in layer_keys:
        if tag != "linear":
            continue
        for p in module.parameters(recurse=False):
            if id(p) in seen:
                return False
            seen.add(id(p))
    return True


def _plan_keys_all_trainable(layer_keys, available: set) -> bool:
    """True when every Linear weight and bias in the plan is in ``available``.

    The delta paths read each Linear's weight straight out of ``base_sd``, which
    holds only trainable parameters. A frozen weight raises KeyError there and a
    frozen bias comes back None, silently evaluating a bias-free network.
    """
    for name, tag, module in layer_keys:
        if tag != "linear":
            continue
        if f"{name}.weight" not in available:
            return False
        if getattr(module, "bias", None) is not None and f"{name}.bias" not in available:
            return False
    return True


class BatchedLinearEvaluator:
    """Fast batched evaluation for pure-MLP models.

    Replaces vmap + functional_call with explicit ``torch.bmm`` per Linear
    layer, eliminating vmap dispatch overhead. For N parameter configs of a
    k-layer MLP, performs k batched matmuls instead of N sequential forward
    passes or a single vmap call.

    Supported layers: ``nn.Linear``, ``nn.ReLU``, ``nn.LeakyReLU``,
    ``nn.Sigmoid``, ``nn.Tanh``, ``nn.GELU``, ``nn.SiLU``, ``nn.Flatten``,
    and ``nn.Dropout`` (eval mode), plus any module setting
    ``polystep_elementwise = True``. Models containing any other layer
    cause ``try_build`` to return ``None``.

    The elementwise contract a declaring module must satisfy: ``module(x)[i] ==
    module(x[i])``, no parameters, no buffers, no Python state, no in-place writes,
    shape and dtype preserving, deterministic, callable at 2-D and 3-D alike.
    Continuity and differentiability are not required, because the delta algebra is an
    exact finite difference: a step function is reproduced exactly and a smooth
    ``LayerNorm`` would be silently wrong.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, loss_kind: str = "cross_entropy"):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys  # ordered list of (name, tag, module)
        # evaluate() flattens the input once up front and then treats every
        # 'flatten' entry as a no-op, which is only equivalent to the real forward
        # when the Flatten runs before any Linear. try_build rejects the rest.
        self.leading_flatten = any(tag == "flatten" for _, tag, _ in layer_keys)

    @classmethod
    def try_build(
        cls, model: nn.Module, loss_fn: Callable, loss_kind: str = "cross_entropy"
    ) -> "BatchedLinearEvaluator | None":
        """Build if model is compatible, else return None."""
        # The bmm plan is rebuilt from named_children(), which misses activations
        # applied inline in a custom forward (e.g. torch.relu) and would compute a
        # wrong, activation-free loss. Only trust real nn.Sequential forwards.
        if type(model).forward is not nn.Sequential.forward:
            _warn_if_sequential_equivalent(model, _SUPPORTED_LAYERS)
            return None
        if _has_hooks(model):
            return None

        # Build an ordered (name, tag, module) plan. The module is kept so evaluate()
        # applies an activation's EXACT semantics (LeakyReLU.negative_slope,
        # GELU.approximate, ...) instead of hardcoded functional defaults, and so a
        # Linear's declared weight transform is reachable. Unsupported -> vmap fallback.
        def _entry(full, submod):
            # The delta evaluators start from a view of the caller's input batch and
            # share activations across candidates, so an in-place activation would
            # write through to tensors it does not own. vmap copies, so defer to it.
            if getattr(submod, "inplace", False):
                return None
            # A hook rewrites the output the plan is about to compute by hand.
            if _has_hooks(submod):
                return None
            for base, tag in _TAGGED_LAYERS:
                if not isinstance(submod, base):
                    continue
                # A subclass that overrides forward does something its base's tag does
                # not describe, so it belongs on the vmap path even though it matches.
                if type(submod).forward is not base.forward:
                    return None
                # A non-default Flatten reshapes differently from the bmm path's
                # leading flatten; defer those models to the (correct) vmap path.
                if tag == "flatten" and (submod.start_dim, submod.end_dim) != (1, -1):
                    return None
                return (full, tag, submod)
            if _linear_like(submod):
                return (full, "linear", submod)
            if _elementwise_entry(submod):
                return (full, "activation", submod)
            return None

        layer_keys = []
        seen_linear = False
        for full, submod in _plan_children(model):
            entry = _entry(full, submod)
            if entry is None:
                return None  # unsupported layer (Conv2d, non-default Flatten, ...)
            # evaluate() flattens once before the first layer and then skips
            # 'flatten' entries, so a Flatten placed *after* a Linear would be
            # applied at the wrong point: Linear(3,3) -> ReLU -> Flatten ->
            # Linear(6,2) on a (B, 2, 3) input pre-flattens to (B, 6) and then
            # fails in the first Linear. Defer those models to vmap.
            if entry[1] == "flatten" and seen_linear:
                return None
            seen_linear = seen_linear or entry[1] == "linear"
            layer_keys.append(entry)
        if not layer_keys:
            return None
        if not _plan_params_distinct(layer_keys):
            return None
        # Verify all named parameters are covered by detected Linear layers.
        # Models with extra parameters (e.g., learned scales) need vmap.
        # remove_duplicate=False, or a tied parameter hides under its first name.
        linear_param_keys = set()
        for name, tag, _ in layer_keys:
            if tag == "linear":
                linear_param_keys.add(f"{name}.weight")
                linear_param_keys.add(f"{name}.bias")
        model_param_keys = {n for n, _ in model.named_parameters(remove_duplicate=False)}
        if not model_param_keys.issubset(linear_param_keys):
            return None
        # Every path built on this plan pins one param dtype, so a mixed-dtype model
        # raises. Hybrid and Factored allow those; vmap handles them, so decline.
        if len({p.dtype for p in model.parameters() if p.is_floating_point()}) > 1:
            return None
        return cls(model, loss_fn, layer_keys, loss_kind)

    @torch.inference_mode()
    def evaluate(
        self,
        stacked_params: dict[str, torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched forward via bmm for Linear layers."""
        N = next(iter(stacked_params.values())).shape[0]
        # x: (N, B, features) - expand input across N param configs
        if inputs.dim() == 2:
            x = inputs.unsqueeze(0).expand(N, -1, -1)  # (N, B, in_feat)
        else:
            # Flatten spatial dims for non-2D inputs
            x = inputs.reshape(inputs.shape[0], -1).unsqueeze(0).expand(N, -1, -1)

        for name, tag, module in self._layer_keys:
            if tag == "linear":
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                wq, bq = _weight_transforms(module)
                # Frozen parameters are not in the candidate stack; they are constant
                # across candidates, so the module's own tensor stands in. bmm does not
                # broadcast its batch dim, hence the stride-0 expand.
                W = stacked_params.get(w_key)
                if W is None:
                    W = module.weight.unsqueeze(0).expand(N, -1, -1)
                if wq is not None:
                    # Elementwise on the weight, so transforming the whole stack is the
                    # same as transforming each candidate's weight on its own.
                    W = wq(W)
                bias = stacked_params.get(b_key)
                if bias is None and getattr(module, "bias", None) is not None:
                    bias = module.bias.unsqueeze(0)
                if bias is not None and bq is not None:
                    bias = bq(bias)
                # (N, B, in) @ (N, in, out) -> (N, B, out). baddbmm folds the bias into
                # the same kernel; adding it afterwards costs a second full-size tensor.
                if bias is not None:
                    x = torch.baddbmm(bias.unsqueeze(1), x, W.transpose(1, 2))
                else:
                    x = torch.bmm(x, W.transpose(1, 2))
            elif tag in ("flatten", "dropout"):
                pass  # input is pre-flattened; dropout is identity in eval mode
            else:
                # Activation: apply the real module so its configuration
                # (LeakyReLU negative_slope, GELU approximate, ...) is exact.
                x = module(x)

        return _reduce_per_candidate(x, targets, self.loss_fn, self.loss_kind)


def _resolve_layout_entry(self, offsets: torch.Tensor, pdim: int, span=None):
    """The one layout entry every candidate in this chunk perturbs, or None.

    A candidate perturbs ``pdim`` contiguous flat positions starting at its offset.
    The site-aware paths need all of them inside a single parameter; a chunk that
    straddles two entries, or runs into the layout's padding tail, falls back.

    ``span`` is ``(lo, hi)`` as Python ints when the caller already knows the chunk
    bounds. Passing it avoids reducing ``offsets`` on device and syncing the result.
    """
    if span is None:
        lo, hi = torch.stack([offsets.min(), offsets.max()]).tolist()
        hi += pdim
    else:
        lo, hi = span

    idx = int(torch.searchsorted(self._starts, torch.tensor(lo), right=True)) - 1
    if idx < 0 or idx >= len(self._keys):
        return None
    entry = self._entry_by_key[self._keys[idx]]
    if lo < entry.offset or hi > entry.offset + entry.numel:
        return None
    return entry


class SiteVmapEvaluator:
    """Evaluate candidates that all perturb one parameter tensor, batching only it.

    ``vmap`` over the whole parameter dict gives every entry a candidate axis, so the
    model is replicated N times even though a candidate differs from the base in one
    contiguous run. Batching the perturbed tensor alone and leaving the rest shared
    means the graph ahead of that tensor runs once, and the candidate axis appears only
    where it is first consumed.

    This makes no assumption about the module set, unlike
    :class:`SparseDeltaEvaluator`, which is faster where it applies but only handles a
    Sequential of Linear layers. Anything ``torch.func`` can trace works here:
    convolutions, normalisation, attention, a custom ``forward``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        layout,
        buffers: dict,
        per_sample: bool = False,
        owner: "NNCostEvaluator | None" = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.per_sample = per_sample
        self._buffers = buffers
        self._owner = owner
        self._entry_by_key = {e.key: e for e in layout.entries}
        self._starts = torch.tensor([e.offset for e in layout.entries], dtype=torch.long)
        self._keys = [e.key for e in layout.entries]

    @classmethod
    def try_build(cls, evaluator: "NNCostEvaluator", layout) -> "SiteVmapEvaluator | None":
        """Decline only where the caller's own vmap path is already unusable.

        A tied weight puts one flat position under two module paths, so replacing a
        single entry would apply half the perturbation. An explicit ``use_inplace=True``
        also declines: this path scores through ``functional_call``, not the real forward.
        """
        if evaluator._vmap_failed or evaluator._inplace_forced or layout.shared_groups:
            return None
        return cls(
            evaluator.model, evaluator.loss_fn, layout, evaluator._buffers, evaluator.per_sample, owner=evaluator
        )

    def _retired(self) -> bool:
        """Whether a vmap failure has permanently sent this path back to the dense one."""
        return self._owner is not None and self._owner._vmap_failed

    def resolve_site(self, offsets: torch.Tensor, pdim: int, span=None):
        """The one layout entry this chunk perturbs, or None to use the dense path."""
        if self._retired():
            return None
        return _resolve_layout_entry(self, offsets, pdim, span)

    def resolve_spec(self, subspace, lo: int, hi: int, sub_dim: int):
        """The per-layer spec owning coordinates ``[lo, hi)``, or None if they straddle.

        A per-layer subspace gives each parameter its own coordinate block, so a run
        inside one block moves exactly one parameter and the rest stay shared.
        Coordinates past ``sub_dim`` are particle padding with nothing behind them.
        """
        if hi > sub_dim or self._retired():
            return None
        for spec in subspace.specs:
            if spec.flat_start <= lo and hi <= spec.flat_end:
                return spec
        return None

    @torch.inference_mode()
    def evaluate_subspace(self, projections, bary_sd, spec, col_start, dcoords, inputs, targets=None):
        """Losses for candidates that offset one spec's coordinates from the barycentre.

        Every candidate in group ``g`` moves the same ``pdim`` coordinates, so only the
        columns ``[col_start + g*pdim, col_start + (g+1)*pdim)`` of the projection reach
        the output. Batching that column run per group turns the correction into a bmm
        costing ``O(n * pdim * num_params)`` instead of the ``O(n * num_coords *
        num_params)`` a dense coordinate row would pay.

        Args:
            projections: Per-layer projections, keyed by entry.
            bary_sd: Parameters at the current coordinates, shared across candidates.
            spec: The one layer whose coordinates every candidate perturbs.
            col_start: First coordinate of the chunk's run, relative to the spec.
            dcoords: ``(n_groups, n_cand, pdim)`` offset from the barycentre. An offset,
                not an absolute position: ``bary_sd`` already carries the barycentre.
            inputs: Input batch, shared across candidates.
            targets: Optional targets, shared across candidates.
        """
        base = bary_sd[spec.entry_key]
        n_groups, n_cand, pdim = dcoords.shape
        n = n_groups * n_cand
        n_cols = n_groups * pdim
        P = projections[spec.entry_key] if spec.is_projected else None

        if isinstance(P, torch.Tensor):
            # P is (num_params, num_coords). .t() before the split so groups lead.
            blk = P.narrow(1, col_start, n_cols).t().reshape(n_groups, pdim, -1)
            delta = torch.bmm(dcoords.to(blk.dtype), blk).reshape(n, -1)
        else:
            # Sparse projections and unprojected specs take a full-width coordinate row.
            wide = dcoords.new_zeros(n, spec.flat_end - spec.flat_start)
            cols = torch.arange(col_start, col_start + n_cols, device=dcoords.device)
            wide.scatter_(
                1,
                cols.reshape(n_groups, 1, pdim).expand(n_groups, n_cand, pdim).reshape(n, pdim),
                dcoords.reshape(n, pdim),
            )
            delta = P.project(wide) if P is not None else wide

        site = (base.reshape(1, -1) + delta.to(base.dtype)).reshape(-1, *base.shape)
        return self._vmap_over_site(spec.entry_key, bary_sd, site, inputs, targets)

    @torch.inference_mode()
    def evaluate(self, base_sd, entry, local_idx, values, inputs, targets=None):
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates.

        Args:
            base_sd: Unperturbed parameters, keyed like ``state_dict``.
            entry: The layout entry every candidate perturbs.
            local_idx: ``(G, pdim)`` indices into that flattened parameter.
            values: ``(G, C, pdim)`` replacement values at those indices.
            inputs: Input batch, shared across candidates.
            targets: Optional targets, shared across candidates.
        """
        groups, cand, pdim = values.shape
        n = groups * cand
        base = base_sd[entry.key]

        # Only this entry gets a candidate axis; every other parameter stays shared.
        site = base.reshape(-1).unsqueeze(0).expand(n, -1).clone()
        rows = local_idx.unsqueeze(1).expand(groups, cand, pdim).reshape(n, pdim)
        site.scatter_(1, rows, values.reshape(n, pdim))
        return self._vmap_over_site(entry.key, base_sd, site.reshape(n, *base.shape), inputs, targets)

    def _vmap_over_site(self, key, base_sd, site, inputs, targets):
        """Map the model over ``site``, an ``(n, *shape)`` stack for parameter ``key``.

        ``in_dims`` batches that argument alone, so every other parameter is passed
        once and the work ahead of it is not repeated per candidate.
        """
        shared = {k: v for k, v in base_sd.items() if k != key}
        shared.update(self._buffers)
        model, loss_fn, per_sample = self.model, self.loss_fn, self.per_sample

        # The step calls this directly, so the cast and autocast frame that
        # NNCostEvaluator.evaluate applies have to be repeated here. Cast to the model's
        # dtype, not the site's: on a mixed-dtype model the perturbed layer need not be
        # the one the input reaches first, and casting to it feeds the wrong dtype in.
        # A mixed model casts internally in its own forward, so leave the input alone.
        cast_to = _uniform_float_dtype(shared, site.dtype)
        if cast_to is not None and inputs.is_floating_point() and inputs.dtype != cast_to:
            inputs = self._owner.cast_inputs(inputs, cast_to) if self._owner is not None else inputs.to(cast_to)

        # The step calls this directly, so NNCostEvaluator.evaluate's eval-mode frame
        # never runs and a train-mode model would score some chunks with dropout live.
        was_training = model.training
        if was_training:
            model.eval()

        def single_eval(site_param, inputs, targets):
            output = functional_call(model, {**shared, key: site_param}, (inputs,))
            loss = loss_fn(output, targets) if targets is not None else loss_fn(output)
            if not per_sample:
                if loss.dim() > 0:
                    loss = loss.mean()
            elif loss.dim() == 0:
                raise ValueError(
                    "per_sample=True needs an unreduced loss_fn, but it returned a scalar. Return one value per sample."
                )
            elif loss.dim() > 1:
                loss = loss.flatten(1).mean(dim=1)
            return loss

        # No autocast frame: under vmap it casts activations but not a Conv2d bias, so
        # a conv model raises "Input type (BFloat16) and bias type (float)".
        # candidate_autocast covers the paths NNCostEvaluator.evaluate dispatches to.
        try:
            return vmap(single_eval, in_dims=(0, None, None))(site, inputs, targets)
        except Exception as e:
            # The step calls this directly, so NNCostEvaluator.evaluate's fallback never
            # sees the failure and the run dies instead of degrading. Usual cause: a
            # forward that draws its own randomness, which vmap's default
            # randomness="error" rejects. "different" is not a substitute, it changes the
            # numbers. Score this chunk in a loop; resolve_site and resolve_spec then
            # decline every later chunk.
            if not _is_vmap_error(e):
                raise
            if self._owner is not None:
                self._owner._vmap_failed = True
            warnings.warn(
                f"vmap failed on the site-aware path for {type(model).__name__}: {e}. "
                "Falling back to a sequential loop; later chunks use the dense path.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.stack([single_eval(site[i], inputs, targets) for i in range(site.shape[0])])
        finally:
            if was_training:
                model.train()


class SparseDeltaEvaluator:
    """Evaluate full-space candidates that each perturb one contiguous parameter run.

    In full-space mode a candidate is the base parameter vector with a single particle
    row replaced, so it differs from the base in ``particle_dim`` contiguous scalars,
    all inside one weight or bias. Every layer other than that one therefore holds the
    *shared* base weight, and the dense path's per-candidate ``bmm`` against an
    ``(N, d_out, d_in)`` stack is doing N times the arithmetic and N times the weight
    traffic for nothing.

    So::

        layers before the perturbed one   one forward at batch 1
        the perturbed layer               one shared GEMM plus a <=particle_dim scatter
        layers after it                   one (N*B, d_in) @ (d_in, d_out) GEMM

    The win is largest for a shallow net whose parameters concentrate in an early
    layer, which is the shape the full-space path is for. Deep nets should use a
    subspace instead.

    Handles the same module set as :class:`BatchedLinearEvaluator`; ``try_build``
    returns ``None`` for anything else.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, layout, loss_kind: str):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys
        # entry_key -> (index into _layer_keys, is_weight)
        self._site = {}
        for idx, (name, tag, _) in enumerate(layer_keys):
            if tag == "linear":
                self._site[f"{name}.weight"] = (idx, True)
                self._site[f"{name}.bias"] = (idx, False)
        self._entry_by_key = {e.key: e for e in layout.entries}
        # Flat spans of the entries this path can perturb, for the containment test.
        self._starts = torch.tensor([e.offset for e in layout.entries], dtype=torch.long)
        self._keys = [e.key for e in layout.entries]

    @classmethod
    def try_build(cls, model: nn.Module, loss_fn: Callable, layout) -> "SparseDeltaEvaluator | None":
        """Reuse BatchedLinearEvaluator's compatibility check and layer plan."""
        loss_kind = _batched_loss_kind(loss_fn)
        if loss_kind is None:
            return None
        # A tied weight means one flat position moves two module paths, so the
        # single-site correction below would apply only half the perturbation.
        if layout.shared_groups:
            return None
        plan = BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind)
        if plan is None:
            return None
        if not _plan_keys_all_trainable(plan._layer_keys, {e.key for e in layout.entries}):
            return None
        return cls(model, loss_fn, plan._layer_keys, layout, loss_kind)

    def resolve_site(self, offsets: torch.Tensor, pdim: int, span=None):
        """Return the layout entry every candidate in this chunk perturbs, or None.

        Narrows :func:`_resolve_layout_entry` to the entries this path can correct,
        which are the weights and biases of the detected Linear layers.
        """
        entry = _resolve_layout_entry(self, offsets, pdim, span)
        return entry if entry is not None and entry.key in self._site else None

    @torch.inference_mode()
    def evaluate(self, base_sd, entry_key, local_idx, values, inputs, targets=None):
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates.

        The perturbed positions are a function of the particle, not of the vertex, so
        the ``C`` candidates of a group share one ``local_idx`` row. Every gather here
        runs at ``G`` rows and broadcasts over ``C``; passing ``C = 1`` degrades to one
        group per candidate, which is what a screened (non particle-major) chunk needs.

        Args:
            base_sd: Unperturbed parameters, keyed like ``state_dict``.
            entry_key: The one parameter every candidate perturbs.
            local_idx: ``(G, pdim)`` indices into that flattened parameter.
            values: ``(G, C, pdim)`` replacement values at those indices.
            inputs: Input batch, shared across candidates.
            targets: Optional targets, shared across candidates.
        """
        groups, cand, pdim = values.shape
        n = groups * cand
        site_idx, is_weight = self._site[entry_key]
        param_dtype = next(iter(base_sd.values())).dtype
        if inputs.is_floating_point() and inputs.dtype != param_dtype:
            inputs = inputs.to(param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        batch = x.shape[0]
        # Deltas, not absolute values: the correction adds onto the base output. When the
        # site layer transforms its own tensors, the output moves by the difference of
        # the transformed values, which for a piecewise-constant transform is nothing
        # like the raw perturbation.
        site_q = _weight_transforms(self._layer_keys[site_idx][2])[0 if is_weight else 1]
        base_param = base_sd[entry_key].reshape(-1)
        base_at_site = base_param[local_idx].unsqueeze(1)
        if site_q is not None:
            delta = (site_q(values) - site_q(base_at_site)).to(param_dtype)  # (G, C, pdim)
        else:
            delta = (values - base_at_site).to(param_dtype)

        def at(t, cols):
            """``(G, 1, B, pdim)`` holding ``t[:, cols]``, broadcastable over C.

            Gathers the transposed view: the copy then runs over contiguous rows, which
            is several times faster than gathering dim 1 and permuting afterwards.
            """
            return t.t().index_select(0, cols.reshape(-1)).reshape(groups, pdim, batch).permute(0, 2, 1).unsqueeze(1)

        # Three states: before the site `x` is shared at batch 1; from the site to the
        # next Linear the candidates differ only at columns `dcols` by `dvals`, and
        # carrying that pair instead of an (N, B, d_out) tensor is the whole win; after
        # it `xn` holds dense per-candidate activations.
        dcols = dvals = xn = None

        for idx, (name, tag, module) in enumerate(self._layer_keys):
            if tag in ("flatten", "dropout"):
                continue

            if tag != "linear":
                if xn is not None:
                    xn = module(xn)
                elif dvals is None:
                    x = module(x)
                else:
                    # Elementwise, so the delta stays inside dcols and the activation
                    # of the gathered entries is the gather of the activation. Recompute
                    # the perturbed entries only, then re-express them as a difference.
                    at_cols = at(x, dcols)
                    x = module(x)
                    dvals = module(at_cols + dvals) - module(at_cols)
                continue

            wq, bq = _weight_transforms(module)
            weight = base_sd[f"{name}.weight"]
            bias = base_sd.get(f"{name}.bias")
            if wq is not None:
                weight = wq(weight)
            if bias is not None and bq is not None:
                bias = bq(bias)

            if xn is not None:
                xn = xn @ weight.t()
                if bias is not None:
                    xn = xn + bias
                continue

            out = x @ weight.t()  # (B, d_out), the weight is shared
            if bias is not None:
                out = out + bias

            if dvals is not None:
                # Mix the confined delta into every output unit: it goes dense here.
                w_cols = weight.index_select(1, dcols.reshape(-1)).reshape(-1, groups, pdim)
                corr = torch.einsum("ogp,gcbp->gcbo", w_cols, dvals)
                xn = out + corr.reshape(n, batch, -1)
                dcols = dvals = None
                continue

            if idx == site_idx:
                if is_weight:
                    # dW[r, c] adds x[:, c] * dW[r, c] to output unit r.
                    d_in = weight.shape[1]
                    dcols = local_idx // d_in
                    dvals = at(x, local_idx % d_in) * delta.unsqueeze(2)
                else:
                    dcols = local_idx
                    dvals = delta.unsqueeze(2).expand(groups, cand, batch, pdim)
                if pdim > 1:
                    # A contiguous run usually lies within one weight row, so several
                    # entries of dcols name the same output unit. The activation is
                    # nonlinear, so it has to see their sum: give every member of a
                    # duplicate group the group total, then keep only the first.
                    same = dcols.unsqueeze(2) == dcols.unsqueeze(1)  # (G, pdim, pdim)
                    dvals = torch.einsum("gpq,gcbq->gcbp", same.to(dvals.dtype), dvals)
                    earlier = torch.tril(torch.ones_like(same[0]), -1).bool()
                    dvals = dvals * (~(same & earlier).any(dim=1))[:, None, None, :]
            x = out

        if xn is None:
            xn = x.expand(n, batch, x.shape[-1])
            if dvals is not None:
                # The site was the last Linear, so nothing mixed the delta back in.
                index = dcols[:, None, None, :].expand(groups, cand, batch, pdim).reshape(n, batch, pdim)
                xn = xn.clone().scatter_add_(2, index, dvals.reshape(n, batch, pdim))
        return _reduce_per_candidate(xn, targets, self.loss_fn, self.loss_kind)


class SubspaceDeltaEvaluator:
    """Evaluate subspace candidates that each perturb one particle's coordinate run.

    A candidate is the barycenter with one particle row replaced, so its coordinate
    vector differs from the barycenter in ``pdim`` consecutive entries. When that run
    lies inside a single layer's coordinate block, only that layer's weight moves and
    the perturbation is linear in ``pdim`` coordinates::

        W_n = W_bary + reshape(sum_j d_nj * P[:, j])
        x @ W_n.T  = x @ W_bary.T + sum_j d_nj * (x @ M_j.T),   M_j = reshape(P[:, j])

    So the ``M_j`` products are computed once per column and reused by every candidate
    that shares the particle. Layers other than the site hold the shared barycenter
    weight, so they run once at batch 1 before the site and once at ``(N*B, d_in)``
    after it. Nothing of shape ``(N, d_out, d_in)`` is built.

    ``resolve_site`` returns ``None`` whenever the run is not contained in one block:
    a coordinate block whose width is not a multiple of ``pdim`` shifts every later
    particle off the block grid, and unprojected full-width entries and biases sit in
    the same coordinate space. Containment is checked, never assumed. The caller falls
    back to ``reconstruct_batch`` in all those cases.

    Handles the same module set as :class:`BatchedLinearEvaluator`.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, loss_kind: str):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys
        # entry_key -> (index into _layer_keys, is_weight)
        self._site = {}
        for idx, (name, tag, _) in enumerate(layer_keys):
            if tag == "linear":
                self._site[f"{name}.weight"] = (idx, True)
                self._site[f"{name}.bias"] = (idx, False)

    @classmethod
    def try_build(cls, model: nn.Module, loss_fn: Callable, subspace) -> "SubspaceDeltaEvaluator | None":
        """Reuse BatchedLinearEvaluator's compatibility check and layer plan."""
        if not hasattr(subspace, "specs"):
            return None
        # This path never forms the perturbed weight: the correction is a linear map of
        # the coordinate delta through the projection basis. A weight transform breaks
        # that linearity, and reconstructing per candidate is what the path exists to
        # avoid, so the model goes to vmap instead.
        if _declares_weight_transform(model):
            return None
        loss_kind = _batched_loss_kind(loss_fn)
        if loss_kind is None:
            return None
        plan = BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind)
        if plan is None:
            return None
        if not _plan_keys_all_trainable(plan._layer_keys, {spec.entry_key for spec in subspace.specs}):
            return None
        return cls(model, loss_fn, plan._layer_keys, loss_kind)

    def resolve_site(self, subspace, lo: int, hi: int, sub_dim: int):
        """The spec owning coordinates ``[lo, hi)``, or ``None`` if they straddle.

        ``lo``/``hi`` bound every candidate's run in the chunk, so one containing spec
        covers them all. Coordinates past ``sub_dim`` are particle padding with no
        parameter behind them.

        Two corrections exist: a projected weight, whose coordinates reach the output
        through the basis, and an unprojected bias, whose coordinates are the delta
        itself. An unprojected weight matches neither (its coordinates index the
        flattened weight, not output units), so it falls back.
        """
        if hi > sub_dim:
            return None
        for spec in subspace.specs:
            if spec.flat_start <= lo and hi <= spec.flat_end:
                site = self._site.get(spec.entry_key)
                if site is None:
                    return None
                _, is_weight = site
                return spec if is_weight == spec.is_projected else None
        return None

    def _basis_products(self, spec, P, col_start, n_cols, x, acc_dtype):
        """``(n_cols, d_out, B)`` holding ``x @ M_j.T`` for the column run at ``col_start``."""
        d_out, d_in = spec.original_shape
        if isinstance(P, SparseRandomProjection):
            # A column of P has nnz_per_col nonzeros, so M_j is that sparse: gather the
            # entries and scatter-add their contributions instead of forming M_j.
            cols = torch.arange(col_start, col_start + n_cols, device=x.device)
            rows, vals = P.columns(cols, x.device, x.dtype)  # (n_cols, nnz)
            out_idx = torch.div(rows, d_in, rounding_mode="floor")  # (n_cols, nnz)
            in_idx = rows.remainder(d_in)
            contrib = x.index_select(1, in_idx.reshape(-1)).reshape(x.shape[0], n_cols, -1)
            contrib = contrib.permute(1, 2, 0) * vals.unsqueeze(-1).to(acc_dtype)  # (n_cols, nnz, B)
            g = torch.zeros(n_cols, d_out, x.shape[0], device=x.device, dtype=acc_dtype)
            return g.scatter_add_(1, out_idx.unsqueeze(-1).expand_as(contrib), contrib)
        # P is (d_out*d_in, num_coords), so P.view(d_out, d_in, -1)[o, i, j] is M_j[o, i]
        # with no copy; gathering and transposing copies n_cols * d_out * d_in per chunk.
        # Batching over columns rather than d_out is faster at the same FLOPs.
        blk = P.view(d_out, d_in, -1).narrow(2, col_start, n_cols).permute(2, 1, 0).to(acc_dtype)
        return torch.matmul(x.to(acc_dtype), blk).permute(0, 2, 1)  # (n_cols, B, d_out) -> (n_cols, d_out, B)

    @torch.inference_mode()
    def evaluate(
        self,
        subspace,
        projections: dict,
        bary_sd: dict[str, torch.Tensor],
        spec,
        col_start: int,
        dcoords: torch.Tensor,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates.

        Args:
            subspace: The subspace whose ``specs`` ``spec`` came from.
            projections: Per-layer projections, keyed like ``state_dict``.
            bary_sd: Barycenter weights, already reconstructed at the current ``X``.
            spec: The layer spec ``resolve_site`` returned.
            col_start: First column into ``spec``'s projection. Groups are
                particle-major and contiguous, so the run is
                ``[col_start, col_start + G * pdim)``.
            dcoords: ``(G, C, pdim)`` coordinate deltas from the barycenter.
            inputs: Input batch, shared across candidates.
            targets: Optional targets, shared across candidates.
        """
        n_groups, n_cand, pdim = dcoords.shape
        site_idx, is_weight = self._site[spec.entry_key]
        param_dtype = next(iter(bary_sd.values())).dtype
        # Accumulate the correction in fp32: it is a small delta on top of a full-scale
        # output, so a half-precision add drops it entirely.
        acc_dtype = torch.float32 if param_dtype in (torch.float16, torch.bfloat16) else param_dtype
        if inputs.is_floating_point() and inputs.dtype != param_dtype:
            inputs = inputs.to(param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        n = n_groups * n_cand
        xn = None

        for idx, (name, tag, module) in enumerate(self._layer_keys):
            if tag in ("flatten", "dropout"):
                continue
            if tag != "linear":
                if xn is None:
                    x = module(x)
                else:
                    xn = module(xn)
                continue

            weight = bary_sd[f"{name}.weight"]
            bias = bary_sd.get(f"{name}.bias")

            if xn is not None:
                xn = xn @ weight.t()
                if bias is not None:
                    xn = xn + bias
                continue

            out = x @ weight.t()  # (B, d_out), the barycenter weight is shared
            if bias is not None:
                out = out + bias

            if idx == site_idx:
                if is_weight:
                    g = self._basis_products(
                        spec, projections[spec.entry_key], col_start, n_groups * pdim, x, acc_dtype
                    ).reshape(n_groups, pdim, weight.shape[0], x.shape[0])
                    corr = torch.einsum("gcp,gpob->gcbo", dcoords.to(acc_dtype), g)
                else:
                    # A bias site is unprojected: the coordinates are the bias delta,
                    # constant across the batch.
                    cols = torch.arange(col_start, col_start + n_groups * pdim, device=x.device).view(n_groups, pdim)
                    corr = torch.zeros(n_groups, n_cand, weight.shape[0], device=x.device, dtype=acc_dtype)
                    corr.scatter_(2, cols.unsqueeze(1).expand(n_groups, n_cand, pdim), dcoords.to(acc_dtype))
                    corr = corr.unsqueeze(2)
                xn = (out.to(acc_dtype) + corr.reshape(n, -1, weight.shape[0])).to(param_dtype)
                continue
            x = out

        if xn is None:
            xn = x.unsqueeze(0).expand(n, *x.shape)
        return _reduce_per_candidate(xn, targets, self.loss_fn, self.loss_kind)


class FactoredEvaluator:
    """Evaluate N low-rank candidates of an MLP without materializing any weight.

    For a :class:`~polystep.factored_subspace.FactoredSubspace`, candidate ``n``
    perturbs a Linear layer by ``dW_n = A_n @ B`` with ``B`` shared. So::

        x (W + A_n B)^T = x W^T + (x B^T) A_n^T

    The base term is one GEMM against the shared weight and the correction is
    rank-``r``. Nothing of shape ``(N, d_out, d_in)`` is ever built, which is what the
    materializing path spends most of a step on.

    Handles the same module set as :class:`BatchedLinearEvaluator`; ``try_build``
    returns ``None`` for anything else so the caller falls back.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, loss_kind: str = "cross_entropy"):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys

    @classmethod
    def try_build(cls, model: nn.Module, loss_fn: Callable) -> "FactoredEvaluator | None":
        """Reuse BatchedLinearEvaluator's compatibility check and layer plan."""
        loss_kind = _batched_loss_kind(loss_fn)
        if loss_kind is None:
            return None
        # The correction here is `(x B^T) A_n^T`, linear in the factor. A weight
        # transform breaks that, so those models fall back to vmap.
        if _declares_weight_transform(model):
            return None
        plan = BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind)
        if plan is None:
            return None
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        if not _plan_keys_all_trainable(plan._layer_keys, trainable):
            return None
        return cls(model, loss_fn, plan._layer_keys, loss_kind)

    @torch.inference_mode()
    def evaluate(
        self,
        subspace,
        projections: dict,
        base_sd: dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Losses of shape ``(N,)`` for ``N`` coordinate vectors."""
        N = flat_subspace_batch.shape[0]
        specs = {s.entry_key: s for s in subspace.specs}

        param_dtype = next(iter(base_sd.values())).dtype
        if inputs.is_floating_point() and inputs.dtype != param_dtype:
            inputs = inputs.to(param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        # (B, d_in), shared across candidates until the first perturbed layer.
        shared = True

        for name, tag, module in self._layer_keys:
            if tag == "linear":
                w_key, b_key = f"{name}.weight", f"{name}.bias"
                W = base_sd[w_key]
                base = x @ W.t()  # (B, d_out) while shared, else (N, B, d_out)

                spec = specs.get(w_key)
                if spec is not None and spec.is_projected:
                    coords = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
                    d_out = spec.original_shape[0]
                    A = coords.reshape(N, d_out, subspace.ranks[w_key]).to(param_dtype)
                    B = projections[w_key].to(param_dtype)
                    if shared:
                        # x is (B, d_in) -> one (B, r) product feeds every candidate.
                        corr = torch.einsum("nor,br->nbo", A, x @ B.t())
                        base = base.unsqueeze(0) + corr
                        shared = False
                    else:
                        base = base + torch.bmm(x @ B.t(), A.transpose(1, 2))
                elif not shared:
                    pass  # unperturbed layer, base already has the candidate dim
                x = base

                if b_key in base_sd:
                    bias = base_sd[b_key]
                    spec_b = specs.get(b_key)
                    if spec_b is not None and not spec_b.is_projected:
                        db = flat_subspace_batch[:, spec_b.flat_start : spec_b.flat_end].to(param_dtype)
                        if shared:
                            x = x.unsqueeze(0) + (bias + db).unsqueeze(1)
                            shared = False
                        else:
                            x = x + (bias + db).unsqueeze(1)
                    else:
                        x = x + bias
            elif tag in ("flatten", "dropout"):
                continue  # input is pre-flattened; dropout is identity in eval mode
            else:
                x = module(x)

        if not shared:
            out = x  # (N, B, out_features)
        else:
            out = x.unsqueeze(0).expand(N, -1, -1)

        return _reduce_per_candidate(out, targets, self.loss_fn, self.loss_kind)
