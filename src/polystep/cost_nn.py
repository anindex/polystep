"""Vectorized NN cost evaluation via vmap + functional_call.

Evaluates a neural network at batched parameter configurations using
``torch.vmap`` for vectorized evaluation -- each candidate parameter set
is evaluated in parallel by swapping model parameters via
``torch.func.functional_call``. Falls back to a sequential Python loop
if vmap is incompatible with the model (one-time warning emitted).

When ``chunk_size`` is set, evaluations are batched to bound peak GPU
memory. ``auto_detect_chunk_size`` estimates a safe value based on model
size and available GPU memory.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, TYPE_CHECKING, Union

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

if TYPE_CHECKING:
    from .hybrid_subspace import HybridSubspace
    from .transform import ParamLayout


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
    # the headroom into the safety factor so we don't OOM at first call.
    effective_safety = safety_factor * 1.5 if compile_overhead else safety_factor

    free_mem, _ = torch.cuda.mem_get_info(param_device)
    chunk = max(1, int(free_mem / (per_eval_bytes * effective_safety)))
    return chunk


_UNSET = object()  # sentinel distinguishing "not yet computed" from None


def _batched_loss_kind(loss_fn) -> Optional[str]:
    """Name the per-sample reduction the batched-linear path can reproduce.

    The bmm forward does not depend on the loss; only the final reduction does. A
    configured loss (class weights, label smoothing, a custom ignore_index, a non-mean
    reduction) has no such reproduction and falls back to vmap, which calls the real
    ``loss_fn``. Returns None in that case.
    """
    if getattr(loss_fn, "reduction", None) != "mean":
        return None
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        if loss_fn.weight is None and loss_fn.ignore_index == -100 and float(loss_fn.label_smoothing) == 0.0:
            return "cross_entropy"
        return None
    if isinstance(loss_fn, nn.MSELoss):
        return "mse"
    if isinstance(loss_fn, nn.L1Loss):
        return "l1"
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

    tgt = targets.unsqueeze(0).expand(n, *targets.shape).to(outputs.dtype)
    fn = torch.nn.functional.mse_loss if loss_kind == "mse" else torch.nn.functional.l1_loss
    return fn(outputs, tgt, reduction="none").flatten(1).mean(dim=1)  # (N,)


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
            eliminated; expect ~2-4x, architecture-dependent). Falls back to
            eager on failure. Best for CUDA models. Default False. For the
            launch-bound CUDA-graph win on large recurrent nets, use the
            in-place ``compile_forward`` path instead.
        compile_forward: If True, on the in-place path compile the
            forward+loss closure with ``torch.compile(mode="reduce-overhead")``
            (CUDA graphs). Unlike ``compile_vmap`` this DOES eliminate per-kernel
            launch overhead: the launch-bound win: because the in-place path
            has no vmap chunk-concat. The Python swap loop stays; only the
            captured ``model(inputs)`` is replayed per candidate. Requires CUDA
            and a static input/param shape; falls back to eager on failure.
            Default False. (No effect unless the in-place path is used --
            auto for >500K-param GPU models, or force with ``use_inplace=True``.)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        chunk_size: Union[None, int, str] = None,
        compile_vmap: bool = False,
        use_inplace: Optional[bool] = None,
        compile_forward: bool = False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self._chunk_size_raw = chunk_size
        self._chunk_size_cached = _UNSET  # lazily computed for "auto" mode
        self._vmap_failed = False
        self._warned = False
        self._compile_vmap = compile_vmap
        self._compiled_vmap_fn = None
        self._compile_failed = False

        # In-place forward+loss compile (CUDA graphs, launch-bound win).
        self._compile_forward = compile_forward
        self._compiled_fwd_loss = None
        self._compile_forward_failed = False
        self._compile_forward_verified = False

        # Force eval mode for consistent behavior (frozen BN stats, no dropout)
        model.eval()

        # Fast batched-linear evaluator (MLP-only models), when the loss has a
        # per-sample form the bmm path can reproduce.
        loss_kind = _batched_loss_kind(loss_fn)
        self._batched_linear = (
            BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind) if loss_kind is not None else None
        )

        # Auto-detect whether to use memory-efficient in-place evaluation.
        # For very large models (>500K params) on GPU, vmap materializes N
        # copies of all intermediate activations simultaneously, causing
        # O(N x activation) memory. In-place evaluation uses O(1 x activation)
        # regardless of N. For models ≤500K params (e.g. MNISTNet ~102K,
        # CIFAR10MLP ~199K), chunked vmap is fast and fits in GPU memory.
        # Pass use_inplace=True/False to override auto-detection.
        if use_inplace is not None:
            self._use_inplace = use_inplace
        else:
            n_params = sum(p.numel() for p in model.parameters())
            try:
                on_gpu = next(model.parameters()).device.type == "cuda"
            except StopIteration:
                on_gpu = False
            self._use_inplace = on_gpu and n_params > 500_000

        # Cache frozen buffers from the real model (shared across all particles)
        self._buffers = dict(model.named_buffers())

        # Cache param dict for in-place evaluation (avoids O(L) module traversal per call).
        # remove_duplicate=False so tied weights resolve under every module path they
        # appear at: batch_unflatten emits an alias key per shared entry, and the
        # deduplicated dict would report those aliases as unknown parameters.
        self._param_dict_cache = dict(self.model.named_parameters())

    def reset_vmap(self) -> None:
        """Reset the vmap/compile failure flags so both are attempted again.

        Useful after changing the model architecture (e.g., swapping layers),
        moving the model to a different device, or upgrading PyTorch (vmap
        op coverage and torch.compile support both expand across releases).
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
            Losses tensor of shape ``(N,)``.
        """
        # Model should already be in eval mode (set in __init__).
        # Fast path: was_training is False 99.9% of the time (single bool check,
        # no O(L) module traversal). Restore on exit for external callers.
        was_training = self.model.training
        if was_training:
            self.model.eval()

        # Match float inputs to the (maybe bf16) param dtype so bmm/vmap don't hit a
        # dtype mismatch under mixed precision. Integer inputs and targets untouched.
        if stacked_params:
            param_dtype = next(iter(stacked_params.values())).dtype
            if inputs.is_floating_point() and inputs.dtype != param_dtype:
                inputs = inputs.to(param_dtype)

        try:
            # Fast path: batched bmm for Linear-only models (MLP)
            # Only used for supervised (targets != None) with cross-entropy.
            # The bmm path pre-flattens the input, which only reproduces the real
            # forward when a Flatten precedes every Linear.
            if (
                self._batched_linear is not None
                and targets is not None
                and (inputs.dim() == 2 or self._batched_linear.leading_flatten)
            ):
                return self._batched_linear.evaluate(stacked_params, inputs, targets)

            # Memory-efficient path: in-place weight swap for large GPU models.
            # Uses O(1 x activation) memory instead of O(N x activation).
            if self._use_inplace:
                return self._evaluate_inplace(stacked_params, inputs, targets)

            if self._vmap_failed:
                result = self._evaluate_loop(stacked_params, inputs, targets)
            else:
                try:
                    result = self._evaluate_vmap(stacked_params, inputs, targets)
                except Exception as e:
                    # Only catch vmap/functorch-specific errors; re-raise real bugs.
                    # Keywords are deliberately narrow: a bare "batched" would also
                    # match a real bug in a user forward ("batched input not
                    # supported by op X") and silently demote it to the ~N-times-slower
                    # sequential loop, hiding the real error. Match functorch's own
                    # markers instead ("batched tensor" is the BatchedTensor repr;
                    # "vmap"/"functorch"/"torch.func" name the transform; "randomness"
                    # is vmap's stochastic-op guard).
                    msg = str(e).lower()
                    is_vmap_issue = any(
                        k in msg for k in ("vmap", "functorch", "torch.func", "batched tensor", "randomness")
                    )
                    if not is_vmap_issue:
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
        finally:
            if was_training:
                self.model.train()

    def _evaluate_vmap(self, stacked_params, inputs, targets):
        """Vectorized evaluation via vmap + functional_call.

        When ``compile_vmap=True`` (set at init), wraps the batched evaluation
        in ``torch.compile(mode="default")`` for **Inductor kernel fusion only**.
        This does NOT capture CUDA graphs and so does NOT eliminate per-kernel
        launch overhead, ``mode="reduce-overhead"`` (CUDA graphs) is
        deliberately not used here because vmap's chunked-output concatenation
        conflicts with CUDA-graph tensor ownership. Expect a modest fusion win
        (roughly 2-4x, architecture-dependent), not the launch-bound speedup.
        The CUDA-graph / launch-elimination lever lives on the in-place path
        (``compile_forward``), which has no chunk-concat. Falls back to eager
        vmap permanently on compilation failure (with a one-time warning).
        """
        buffers = self._buffers
        loss_fn = self.loss_fn
        model = self.model
        resolved_chunk = self.chunk_size

        # inputs/targets are explicit args (in_dims=None), not closed-over, so a
        # cached torch.compile graph does not bake in the first call's batch.
        def single_eval(params, inputs, targets):
            # Buffers override params intentionally: stacked_params contains only
            # trainable entries from ParamLayout, while self._buffers holds frozen
            # model state (e.g., BatchNorm running_mean/var, num_batches_tracked).
            # If a key appears in both, the buffer value is authoritative because
            # buffers are not part of the OT optimization and must stay frozen.
            full_dict = {**params, **buffers}
            output = functional_call(model, full_dict, (inputs,))
            if targets is not None:
                loss = loss_fn(output, targets)
            else:
                loss = loss_fn(output)
            if loss.dim() > 0:
                loss = loss.mean()
            return loss

        batched = vmap(single_eval, in_dims=(0, None, None), chunk_size=resolved_chunk)

        # Compiled path: torch.compile on the vmapped forward for Inductor kernel
        # FUSION ONLY (mode="default", no CUDA graphs -- see the method docstring).
        # Only attempted with compile_vmap=True. Lazy-compiled on first call.
        if self._compile_vmap and not self._compile_failed:
            if self._compiled_vmap_fn is None:
                try:
                    # "default" = kernel fusion without CUDA graphs. "reduce-overhead"
                    # (CUDA graphs) has tensor-ownership conflicts with vmap's chunked
                    # output concatenation, so the launch-elimination win is not
                    # reachable here; it lives on the in-place compile_forward path.
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

    def _evaluate_loop(self, stacked_params, inputs, targets):
        """Sequential fallback when vmap is incompatible."""
        if not stacked_params:
            return torch.zeros(0, device=inputs.device)
        N = next(iter(stacked_params.values())).shape[0]
        if N == 0:
            return torch.zeros(0, device=inputs.device)
        losses = []
        for i in range(N):
            params_i = {k: v[i] for k, v in stacked_params.items()}
            full_dict = {**params_i, **self._buffers}
            output = functional_call(self.model, full_dict, (inputs,))
            if targets is not None:
                loss = self.loss_fn(output, targets)
            else:
                loss = self.loss_fn(output)
            if loss.dim() > 0:
                loss = loss.mean()
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

        Wall-clock: ~Nx slower than vmap for small models, but for
        large models where each forward pass already saturates the GPU,
        the overhead is minimal (~2x vs vmap).
        """
        if not stacked_params:
            return torch.zeros(0, device=inputs.device)
        N = next(iter(stacked_params.values())).shape[0]
        device = inputs.device
        losses = torch.empty(N, device=device)

        # Save original weights (one copy, regardless of N)
        original_params = {}
        param_dict = self._param_dict_cache
        if not stacked_params.keys() <= param_dict.keys():
            # Model gained or renamed params since construction; refresh the cache
            # so new keys aren't silently evaluated with stale weights.
            param_dict = self._param_dict_cache = dict(self.model.named_parameters())
        unknown = stacked_params.keys() - param_dict.keys()
        if unknown:
            # Silently skipping these would evaluate every candidate at the base
            # weights and report a flat cost matrix as a real result.
            raise ValueError(
                f"{type(self.model).__name__} has no parameter(s) {sorted(unknown)}"
                f"; the candidate configurations cannot be applied. "
                f"Known parameters: {sorted(param_dict)}."
            )
        for key in stacked_params:
            original_params[key] = param_dict[key].data.clone()

        fwd_loss = self._verified_forward_loss_fn(inputs, targets)
        # Resolve the destination param views once so each swap is a single fused
        # foreach op instead of one kernel launch per parameter tensor.
        dst_keys = list(stacked_params)
        dsts = [param_dict[k].data for k in dst_keys]
        try:
            for i in range(N):
                # Swap all candidate weights in-place with one fused copy.
                torch._foreach_copy_(dsts, [stacked_params[k][i] for k in dst_keys])
                # Forward pass - already under inference_mode from evaluate()
                # (compiled + CUDA-graph-replayed when compile_forward=True).
                # .item()-free: store the detached scalar, read out before the
                # next replay overwrites the graph's static output buffer.
                losses[i] = fwd_loss(inputs, targets).detach()
        finally:
            # Always restore original weights, even on error
            for key, orig in original_params.items():
                if key in param_dict:
                    param_dict[key].data.copy_(orig)

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

        # Cast float inputs to the model param dtype. This path is called
        # directly, so under mixed_precision the model is BF16 while inputs may
        # arrive FP32; without the cast the forward mismatches. Integer inputs
        # (token ids, class targets) are left untouched.
        try:
            param_dtype = next(self.model.parameters()).dtype
            if inputs.is_floating_point() and inputs.dtype != param_dtype:
                inputs = inputs.to(param_dtype)
        except StopIteration:
            pass

        losses = torch.empty(N, device=device)

        was_training = self.model.training
        if was_training:
            self.model.eval()

        fwd_loss = self._verified_forward_loss_fn(inputs, targets)
        # Reuse the cached parameter dict; refresh only if it no longer covers the
        # entries this subspace perturbs.
        param_dict = self._param_dict_cache
        if not param_dict.keys() >= {s.entry_key for s in subspace.specs} & set(base_sd):
            param_dict = self._param_dict_cache = dict(self.model.named_parameters())
        if hasattr(subspace, "prepare_inplace"):
            subspace.prepare_inplace(base_sd)
        # Snapshot only the entries that will be written, not the whole state_dict.
        entry_params = {k: param_dict[k].data.clone() for k in base_sd if k in param_dict}
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
                losses[i] = fwd_loss(inputs, targets).detach()
        finally:
            if hasattr(subspace, "release_inplace"):
                subspace.release_inplace()
            for key, entry in entry_params.items():
                param_dict[key].data.copy_(entry)
            if was_training:
                self.model.train()

        return losses


class BatchedLinearEvaluator:
    """Fast batched evaluation for pure-MLP models.

    Replaces vmap + functional_call with explicit ``torch.bmm`` per Linear
    layer, eliminating vmap dispatch overhead. For N parameter configs of a
    k-layer MLP, performs k batched matmuls instead of N sequential forward
    passes or a single vmap call.

    Supported layers: ``nn.Linear``, ``nn.ReLU``, ``nn.LeakyReLU``,
    ``nn.Sigmoid``, ``nn.Tanh``, ``nn.GELU``, ``nn.SiLU``, ``nn.Flatten``,
    and ``nn.Dropout`` (eval mode). Models containing any other layer
    cause ``try_build`` to return ``None``.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, loss_kind: str = "cross_entropy"):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys  # ordered list of (name, type_tag)
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
            return None
        supported = (nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Flatten, nn.Dropout)

        # Build an ordered (name, tag, module) plan. ``module`` is None for
        # Linear layers (handled by bmm) and the actual nn.Module for
        # activations so evaluate() applies their EXACT semantics
        # (LeakyReLU.negative_slope, GELU.approximate, ...) instead of
        # hardcoded functional defaults. Unsupported -> None (vmap fallback).
        def _entry(full, submod):
            if isinstance(submod, nn.Linear):
                return (full, "linear", None)
            # A non-default Flatten reshapes differently from the bmm path's
            # leading flatten; defer those models to the (correct) vmap path.
            if isinstance(submod, nn.Flatten) and (submod.start_dim, submod.end_dim) != (1, -1):
                return None
            if isinstance(submod, supported):
                return (full, type(submod).__name__.lower(), submod)
            return None

        layer_keys = []
        seen_linear = False
        for name, mod in model.named_children():
            if isinstance(mod, nn.Sequential):
                # Walk one level of Sequential
                children = [(f"{name}.{sub}", m) for sub, m in mod.named_children()]
            else:
                children = [(name, mod)]
            for full, submod in children:
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
        # Verify all named parameters are covered by detected Linear layers.
        # Models with extra parameters (e.g., learned scales) need vmap.
        linear_param_keys = set()
        for name, tag, _ in layer_keys:
            if tag == "linear":
                linear_param_keys.add(f"{name}.weight")
                linear_param_keys.add(f"{name}.bias")
        model_param_keys = {n for n, _ in model.named_parameters()}
        if not model_param_keys.issubset(linear_param_keys):
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
                W = stacked_params[w_key]  # (N, out, in)
                # bmm: (N, B, in) @ (N, in, out) -> (N, B, out)
                x = torch.bmm(x, W.transpose(1, 2))  # (N, B, out)
                if b_key in stacked_params:
                    x = x + stacked_params[b_key].unsqueeze(1)  # (N, 1, out) broadcast
            elif tag in ("flatten", "dropout"):
                pass  # input is pre-flattened; dropout is identity in eval mode
            else:
                # Activation: apply the real module so its configuration
                # (LeakyReLU negative_slope, GELU approximate, ...) is exact.
                x = module(x)

        return _reduce_per_candidate(x, targets, self.loss_fn, self.loss_kind)


def compute_nn_cost_matrix(
    evaluator: NNCostEvaluator,
    X_probe: torch.Tensor,
    layout: ParamLayout,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute NN cost matrix from probe points via vectorized evaluation.

    Flattens the (P, V, K) probe structure into a single batch dimension,
    evaluates all probe points via the evaluator, then reshapes and averages
    over the probe dimension K to produce the (P, V) cost matrix.

    Args:
        evaluator: NNCostEvaluator instance.
        X_probe: Probe points of shape ``(P, V, K, D)`` where D is the
            particle flat dimension (rows * particle_dim).
        layout: ParamLayout for converting particles to param dicts.
        inputs: Input data batch (shared across all evaluations).
        targets: Optional targets (shared across all evaluations).

    Returns:
        Cost matrix of shape ``(P, V)``.
    """
    P, V, K, D = X_probe.shape

    # Flatten (P, V, K) into single batch dimension N = P*V*K
    flat_probes = X_probe.reshape(P * V * K, D)

    # Convert to stacked param dicts: {key: (N, *param_shape)}
    stacked_params = layout.batch_unflatten(flat_probes)

    # Evaluate all probe points
    losses = evaluator.evaluate(stacked_params, inputs, targets)  # (N,)

    # Reshape and average over probe dimension K
    cost_matrix = losses.reshape(P, V, K).mean(dim=-1)  # (P, V)
    return cost_matrix


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
        self.layout = layout
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
        return cls(model, loss_fn, plan._layer_keys, layout, loss_kind)

    def resolve_site(self, offsets: torch.Tensor, pdim: int):
        """Return the layout entry every candidate in this chunk perturbs, or None.

        ``offsets`` are flat start indices of each candidate's perturbed run. The fast
        path needs one shared site, so a chunk that straddles an entry boundary, or
        whose particle rows do (the layout packs entries back to back with no
        ``particle_dim`` alignment), falls back to the dense path.
        """
        lo, hi_start = torch.stack([offsets.min(), offsets.max()]).tolist()  # one transfer
        hi = hi_start + pdim
        idx = int(torch.searchsorted(self._starts, torch.tensor(lo), right=True)) - 1
        if idx < 0:
            return None
        entry = self._entry_by_key[self._keys[idx]]
        if lo < entry.offset or hi > entry.offset + entry.numel:
            return None  # spans two entries, or runs into the layout padding
        return entry if entry.key in self._site else None

    @torch.inference_mode()
    def evaluate(self, base_sd, entry_key, local_idx, values, inputs, targets=None):
        """Losses for N candidates perturbing ``entry_key`` at ``local_idx``.

        Args:
            base_sd: Unperturbed parameters, keyed like ``state_dict``.
            entry_key: The one parameter every candidate perturbs.
            local_idx: ``(N, pdim)`` indices into that flattened parameter.
            values: ``(N, pdim)`` replacement values at those indices.
            inputs: Input batch, shared across candidates.
            targets: Optional targets, shared across candidates.
        """
        n, pdim = local_idx.shape
        site_idx, is_weight = self._site[entry_key]
        param_dtype = next(iter(base_sd.values())).dtype
        if inputs.is_floating_point() and inputs.dtype != param_dtype:
            inputs = inputs.to(param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        batch = x.shape[0]
        # Deltas, not absolute values: the correction adds onto the base output.
        base_param = base_sd[entry_key].reshape(-1)
        delta = (values - base_param[local_idx]).to(param_dtype)  # (N, pdim)

        # Three states. Before the site, `x` is shared at batch 1 and there is no
        # candidate dimension anywhere. From the site until the next Linear, the
        # candidates differ from `x` only at columns `dcols`, by `dvals`; carrying that
        # (N, B, pdim) pair instead of an (N, B, d_out) tensor is the whole point. The
        # next Linear mixes those columns into every output, so from there on `xn`
        # holds dense per-candidate activations.
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
                    # Elementwise, so the delta stays inside dcols. Recompute the
                    # perturbed entries only, then re-express them as a difference.
                    at_cols = x.index_select(1, dcols.reshape(-1)).reshape(batch, n, pdim).permute(1, 0, 2)
                    x = module(x)
                    dvals = module(at_cols + dvals) - x.index_select(1, dcols.reshape(-1)).reshape(
                        batch, n, pdim
                    ).permute(1, 0, 2)
                continue

            weight = base_sd[f"{name}.weight"]
            bias = base_sd.get(f"{name}.bias")

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
                w_cols = weight.index_select(1, dcols.reshape(-1)).reshape(-1, n, pdim)
                xn = out.unsqueeze(0) + torch.einsum("onp,nbp->nbo", w_cols, dvals)
                dcols = dvals = None
                continue

            if idx == site_idx:
                if is_weight:
                    # dW[r, c] adds x[:, c] * dW[r, c] to output unit r.
                    d_in = weight.shape[1]
                    dcols = local_idx // d_in
                    cols = (local_idx % d_in).reshape(-1)
                    dvals = x.index_select(1, cols).reshape(batch, n, pdim).permute(1, 0, 2) * delta.unsqueeze(1)
                else:
                    dcols = local_idx
                    dvals = delta.unsqueeze(1).expand(n, batch, pdim)
                if pdim > 1:
                    # A contiguous run usually lies within one weight row, so several
                    # entries of dcols name the same output unit. The activation is
                    # nonlinear, so it has to see their sum: give every member of a
                    # duplicate group the group total, then keep only the first.
                    same = dcols.unsqueeze(2) == dcols.unsqueeze(1)  # (N, pdim, pdim)
                    dvals = torch.einsum("npq,nbq->nbp", same.to(dvals.dtype), dvals)
                    earlier = torch.tril(torch.ones_like(same[0]), -1).bool()
                    dvals = dvals * (~(same & earlier).any(dim=1)).unsqueeze(1)
            x = out

        if xn is None:
            xn = x.unsqueeze(0).expand(n, batch, x.shape[-1])
            if dvals is not None:
                # The site was the last Linear, so nothing mixed the delta back in.
                xn = xn.clone().scatter_add_(2, dcols.unsqueeze(1).expand(n, batch, pdim), dvals)
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
        self.leading_flatten = any(tag == "flatten" for _, tag, _ in layer_keys)

    @classmethod
    def try_build(cls, model: nn.Module, loss_fn: Callable) -> "FactoredEvaluator | None":
        """Reuse BatchedLinearEvaluator's compatibility check and layer plan."""
        loss_kind = _batched_loss_kind(loss_fn)
        if loss_kind is None:
            return None
        plan = BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind)
        return None if plan is None else cls(model, loss_fn, plan._layer_keys, loss_kind)

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
