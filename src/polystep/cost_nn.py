"""Candidate scoring for the NN paths.

``NNCostEvaluator`` dispatches to the cheapest path the model allows: bmm for pure
MLPs, ``vmap``, or an in-place weight swap. The site-aware evaluators below batch only
the one parameter a candidate perturbs.
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


# Smallest candidate batch worth the batched paths; below it the in-place path wins.
_MIN_BATCHED_CANDIDATES = 8


def _is_vmap_error(e: BaseException) -> bool:
    """Whether an exception came from the vmap transform rather than the model.

    Matches functorch's own markers, so a user-forward error is not demoted to the
    slow loop.
    """
    msg = str(e).lower()
    return any(k in msg for k in ("vmap", "functorch", "torch.func", "batched tensor", "randomness"))


def auto_detect_chunk_size(
    model: nn.Module,
    safety_factor: float = 2.0,
    compile_overhead: bool = False,
) -> Optional[int]:
    """Estimate a safe vmap chunk_size from model size and GPU memory.

    Returns None on CPU. On GPU, estimates per-evaluation memory as 4x parameter
    memory (activations + intermediates) and divides free memory by it.
    """
    # Check the model's actual device, not global CUDA availability.
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        return None  # No parameters

    if param_device.type != "cuda":
        return None

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    # 4x (params + buffers) for activations + intermediates.
    per_eval_bytes = (param_bytes + buffer_bytes) * 4

    if per_eval_bytes <= 0:
        return None

    # Bake torch.compile's CUDA-graph headroom into the safety factor.
    effective_safety = safety_factor * 1.5 if compile_overhead else safety_factor

    free_mem, _ = torch.cuda.mem_get_info(param_device)
    chunk = max(1, int(free_mem / (per_eval_bytes * effective_safety)))
    return chunk


_UNSET = object()  # sentinel distinguishing "not yet computed" from None


def _autocast_ctx(device: torch.device, dtype: Optional[torch.dtype]):
    """Autocast frame for a candidate forward, or a no-op when ``dtype`` is None.

    Parameters keep their own dtype, so a perturbation is not rounded away first.
    """
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype or torch.bfloat16,
        enabled=dtype is not None,
    )


def _reusable(buffer: torch.Tensor, target: torch.Tensor) -> bool:
    """Whether ``buffer`` can back up ``target`` in place (shape, dtype, device)."""
    return buffer.shape == target.shape and buffer.dtype == target.dtype and buffer.device == target.device


def _uniform_float_dtype(*param_sources) -> Optional[torch.dtype]:
    """The one float dtype every parameter shares, or None if they differ.

    None means the model casts internally, so the caller must not touch dtypes.
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
    """Name the per-sample reduction the batched-linear path can reproduce, else None.

    A configured loss has no reproduction and falls back to vmap; a subclass that
    overrides ``forward`` would score candidates on a different objective.
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


def _fold_candidate_loss(loss, per_sample):
    """One value per candidate, or one per sample under ``per_sample``."""
    if not per_sample:
        return loss.mean() if loss.dim() > 0 else loss
    if loss.dim() == 0:
        # A reducing callable without ``reduction`` passes the constructor check and is
        # only caught here.
        raise ValueError(
            "per_sample=True needs an unreduced loss_fn, but it returned a scalar. Return one value per sample."
        )
    # An unreduced loss keeps the target's trailing dims; fold them to one per sample.
    return loss.flatten(1).mean(dim=1) if loss.dim() > 1 else loss


def _reduce_per_candidate(outputs, targets, loss_fn, loss_kind):
    """Reduce ``(N, B, ...)`` outputs to one loss per candidate."""
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

    # Cast before expand, or the cast materializes the whole (N, *target) tensor;
    # promote so FP64 targets keep precision.
    dtype = torch.promote_types(outputs.dtype, targets.dtype)
    tgt = targets.to(dtype).unsqueeze(0).expand(n, *targets.shape)
    fn = torch.nn.functional.mse_loss if loss_kind == "mse" else torch.nn.functional.l1_loss
    return fn(outputs.to(dtype), tgt, reduction="none").flatten(1).mean(dim=1)  # (N,)


def cast_inputs_memo(holder, inputs: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Cast float inputs to ``dtype``, memoized on ``holder`` so a chunk loop casts once."""
    if dtype is None or not inputs.is_floating_point() or inputs.dtype == dtype:
        return inputs
    cached = getattr(holder, "_cast_cache", None)
    if cached is not None and cached[0] is inputs and cached[1] is dtype:
        return cached[2]
    out = inputs.to(dtype)
    holder._cast_cache = (inputs, dtype, out)
    return out


class NNCostEvaluator:
    """Vectorized NN cost evaluation via vmap + functional_call.

    Evaluates a model at N batched parameter configurations, falling back to a
    sequential loop if vmap fails.

    Args:
        model: The ``nn.Module`` to evaluate, put in eval mode.
        loss_fn: ``loss_fn(output, targets) -> scalar`` or ``loss_fn(output) -> scalar``.
        chunk_size: vmap ``chunk_size`` (``None`` = no chunking, ``"auto"`` = detect).
        compile_vmap: Wrap vmap in ``torch.compile(mode="default")`` (fusion only).
        compile_forward: Compile the in-place path's forward with
            ``mode="reduce-overhead"`` (CUDA graphs); ``None`` enables it there.
        per_sample: Return unreduced ``(N, B)`` losses instead of ``(N,)``. Needs
            ``loss_fn.reduction == "none"``.
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
        # A reducing loss_fn returns (N,), the shape per_sample exists to avoid.
        if per_sample and getattr(loss_fn, "reduction", "none") != "none":
            raise ValueError(
                f"per_sample=True needs loss_fn.reduction='none', got "
                f"{loss_fn.reduction!r}. A reducing loss collapses the batch before "
                f"evaluate() can return the (N, B) tensor."
            )
        self.per_sample = per_sample
        self._chunk_size_raw = chunk_size
        self._chunk_size_cached = _UNSET  # lazily computed for "auto"
        self._vmap_failed = False
        self._warned = False
        self._compile_vmap = compile_vmap
        self._compiled_vmap_fn = None
        self._compile_failed = False

        # In-place forward+loss compile (CUDA graphs), resolved after auto-detection.
        self._compiled_fwd_loss = None
        self._compile_forward_failed = False
        self._compile_forward_verified = False
        # Restore buffers for the two in-place paths, allocated on first use.
        self._inplace_backup = None
        self._subspace_backup = None
        self._cast_cache = None  # (inputs, dtype, converted) for the per-chunk cast
        self._input_dtype_cache = _UNSET

        # Eval mode: frozen BN stats, no dropout.
        model.eval()

        # Fast batched-linear evaluator (MLP-only), when the loss has a bmm-friendly form.
        loss_kind = _batched_loss_kind(loss_fn)
        self._batched_linear = (
            BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind) if loss_kind is not None else None
        )

        # Pick in-place only when the batched footprint would not fit (not on a
        # parameter count); pass use_inplace=True/False to override.
        if use_inplace is not None:
            self._use_inplace = use_inplace
        else:
            self._use_inplace = self._batched_footprint_exceeds_free_memory(model)
        # An explicit True also means "run the real forward": a forward that reads
        # state functional_call cannot substitute scores every candidate the same on
        # the stateless paths, and auto-detection only judges memory.
        self._inplace_forced = use_inplace is True

        # Only the in-place loop (N sequential forwards) is launch-bound, so the flag
        # only affects it; an explicit value wins, failure falls back to eager.
        if compile_forward is None:
            self._compile_forward = self._use_inplace
        else:
            self._compile_forward = compile_forward

        # Frozen buffers shared across all particles.
        self._buffers = dict(model.named_buffers())

        # remove_duplicate=False so tied weights resolve under every module path: a
        # deduplicated dict would report their aliases as unknown parameters.
        self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))

    def _autocast(self, device: torch.device):
        """Run the candidate forward in ``autocast_dtype``, or a no-op when unset."""
        return _autocast_ctx(device, self.autocast_dtype)

    def reset_vmap(self) -> None:
        """Rebuild every cache that pins the model's shape at construction.

        Call after swapping a layer, rebinding a buffer, replacing a Parameter, or
        moving the model.
        """
        self._vmap_failed = False
        self._warned = False
        # A prior compile failure should not stay latched after the model changed.
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
        """Resolved vmap chunk_size: None, or a positive int (cached for "auto")."""
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

        Returns losses of shape ``(N,)``, or ``(N, B)`` under ``per_sample``.
        """
        # Already eval (set in __init__), so this is a bool check, not an O(L) walk.
        was_training = self.model.training
        if was_training:
            self.model.eval()

        # Match float inputs to the consuming layer's dtype (integers untouched); an
        # earlier FP64 scalar would otherwise cast the batch wrong.
        inputs = self.cast_inputs(inputs, self._input_dtype(stacked_params))

        try:
            # One autocast frame over every candidate-scoring path below.
            with self._autocast(inputs.device):
                return self._dispatch(stacked_params, inputs, targets)
        finally:
            if was_training:
                self.model.train()

    def cast_inputs(self, inputs: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
        """Cast float inputs to ``dtype``, reusing the last result across chunks."""
        return cast_inputs_memo(self, inputs, dtype)

    def _input_dtype(self, stacked_params) -> Optional[torch.dtype]:
        """Float dtype the first input-consuming layer expects, cached; None to skip."""
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
        # In-place swap is O(1 x activation), checked before the bmm path's O(N) stack.
        if self._use_inplace:
            if self.per_sample:
                raise NotImplementedError(
                    "per_sample is not supported on the in-place path. Construct the "
                    "evaluator with use_inplace=False for diagnostic runs; the (N, B) "
                    "tensor defeats the point of the in-place path's O(1) memory anyway."
                )
            return self._evaluate_inplace(stacked_params, inputs, targets)

        # Batched bmm for Linear-only models, supervised only. Soft-label targets are
        # (B, C) and would fail the cross-entropy expand, so vmap handles them.
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

        ``compile_vmap=True`` wraps it in ``torch.compile(mode="default")`` for
        kernel fusion only (no CUDA graphs); the launch-bound win lives on the
        in-place path. Falls back to eager vmap permanently on failure.
        """
        buffers = self._buffers
        loss_fn = self.loss_fn
        model = self.model
        resolved_chunk = self.chunk_size
        per_sample = self.per_sample

        # inputs/targets are explicit args, so a cached graph does not bake in the
        # first call's batch.
        def single_eval(params, inputs, targets):
            # Buffers win on a key collision: frozen model state, not part of the solve.
            full_dict = {**params, **buffers}
            output = functional_call(model, full_dict, (inputs,))
            if targets is not None:
                loss = loss_fn(output, targets)
            else:
                loss = loss_fn(output)
            return _fold_candidate_loss(loss, per_sample)

        batched = vmap(single_eval, in_dims=(0, None, None), chunk_size=resolved_chunk)

        # torch.compile on the vmapped forward for kernel fusion only; lazy on first call.
        if self._compile_vmap and not self._compile_failed:
            if self._compiled_vmap_fn is None:
                try:
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
                    # Compiled graph failed at run time; fall back permanently.
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
        """Whether even a small batch of weight copies fits, regardless of N.

        CPU always answers False: there is no ceiling to fall off.
        """
        try:
            first = next(model.parameters())
        except StopIteration:
            return False
        if first.device.type != "cuda":
            return False
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        free, _total = torch.cuda.mem_get_info(first.device)
        # Half the free memory, so activations and fragmentation still fit.
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
            losses.append(_fold_candidate_loss(loss, self.per_sample))
        return torch.stack(losses)

    def _forward_loss(self, inputs, targets):
        """Eager forward + reduced scalar loss on the model's current params."""
        output = self.model(inputs)
        loss = self.loss_fn(output, targets) if targets is not None else self.loss_fn(output)
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def _forward_loss_fn(self):
        """Per-candidate forward+loss callable, compiled with CUDA graphs when asked.

        The swap loop mutates params via ``copy_``, which preserves storage addresses,
        so graph replay reads the fresh weights. Falls back to eager on failure.
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

        CUDA-graph capture happens on the first call, so it is checked once here
        rather than failing mid-loop.
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
        """Memory-minimal evaluation via in-place weight swapping (O(1 x activation))."""
        if not stacked_params:
            return torch.zeros(0, device=inputs.device)
        N = next(iter(stacked_params.values())).shape[0]
        device = inputs.device
        losses = torch.empty(N, device=device, dtype=loss_buffer_dtype(next(iter(stacked_params.values())).dtype))

        param_dict = self._param_dict_cache
        if not stacked_params.keys() <= param_dict.keys():
            # Refresh the cache so new keys aren't silently evaluated with stale weights.
            param_dict = self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))
        unknown = stacked_params.keys() - param_dict.keys()
        if unknown:
            # Skipping these would report a flat cost matrix as a real result.
            raise ValueError(
                f"{type(self.model).__name__} has no parameter(s) {sorted(unknown)}"
                f"; the candidate configurations cannot be applied. "
                f"Known parameters: {sorted(param_dict)}."
            )
        # Reuse the backup across chunks instead of cloning the whole weight set each
        # call.
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
        # Resolve the destination views once so each swap is one fused foreach op.
        dst_keys = list(stacked_params)
        dsts = [param_dict[k].data for k in dst_keys]
        try:
            for i in range(N):
                # One fused copy per candidate.
                torch._foreach_copy_(dsts, [stacked_params[k][i] for k in dst_keys])
                # Store the detached scalar before the next replay overwrites it.
                losses[i] = fwd_loss(inputs, targets).detach()
        finally:
            # Always restore, even on error.
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

        Reconstructs one configuration at a time with ``apply_perturbation_inplace``;
        peak memory is independent of N and no stacked parameter dict is allocated.
        """
        N = flat_subspace_batch.shape[0]
        device = inputs.device

        # Cast float inputs to the param dtype (integers untouched).
        first = next(self.model.parameters(), None)
        if first is not None:
            inputs = self.cast_inputs(inputs, first.dtype)

        losses = torch.empty(N, device=device, dtype=loss_buffer_dtype(flat_subspace_batch.dtype))

        was_training = self.model.training
        if was_training:
            self.model.eval()

        fwd_loss = self._verified_forward_loss_fn(inputs, targets)
        # Refresh the cached parameter dict only if it no longer covers these entries.
        param_dict = self._param_dict_cache
        if not param_dict.keys() >= {s.entry_key for s in subspace.specs} & set(base_sd):
            param_dict = self._param_dict_cache = dict(self.model.named_parameters(remove_duplicate=False))
        if hasattr(subspace, "prepare_inplace"):
            subspace.prepare_inplace(base_sd)
        # Snapshot only the written entries into a buffer reused across chunks.
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
                # Reconstruct config i directly into the model params.
                subspace.apply_perturbation_inplace(
                    projections,
                    self.model,
                    base_sd,
                    flat_subspace_batch[i],
                    param_dict=param_dict,
                )
                # Under inference_mode from the caller; compiled when compile_forward.
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
    """Warn once when a hand-written forward is all that blocks the fast paths.

    The plan is rebuilt from ``named_children()``, so a custom forward can apply ops
    those children do not name.
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
    # Require an activation (or flatten) among the children, or the forward applies its
    # nonlinearity inline and the children are not the whole computation.
    if not any(not isinstance(m, nn.Linear) for _, m in flat):
        return
    # Compare parameter mass, not names: a wrapper prefixes them ("net.0.weight").
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
# The tag comes from the matched base class, not the class name. Linear is first so it
# wins over any later overlap.
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
    """Check the elementwise contract on a sample at build time.

    A softmax or layer norm answers differently for a slice than for the whole. The
    outlier sits outside every slice, and the slices span all three axes.
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

    Parameters and buffers are refused outright.
    """
    if not getattr(submod, "polystep_elementwise", False):
        return False
    if any(True for _ in submod.parameters()) or any(True for _ in submod.buffers()):
        return False
    return _probe_elementwise(submod)


def _weight_transforms(module: Optional[nn.Module]):
    """``(weight_fn, bias_fn)`` for ``x @ Q(w).t() + Qb(b)``, else ``(None, None)``.

    Both must be elementwise, so the delta path can use ``Q(w + d) - Q(w)``.
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
    """Whether ``module`` carries a forward hook the hand-built matmuls never run."""
    return bool(
        getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
        or getattr(module, "_forward_hooks_with_kwargs", None)
    )


def _plan_children(model: nn.Module):
    """``(name, submodule)`` in forward order, one entry per position.

    ``named_children`` dedupes by identity, so a reused module appears once.
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

    Tied weights put one tensor under two names; the stacked dict only carries the first.
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

    ``base_sd`` holds only trainable parameters; a frozen bias would silently score a
    bias-free network.
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
    """Fast batched evaluation for pure-MLP models via ``torch.bmm`` per Linear.

    Supported layers: ``nn.Linear``, the common activations, ``nn.Flatten``,
    ``nn.Dropout`` (eval), plus any module with ``polystep_elementwise = True``;
    anything else makes ``try_build`` return None.

    The elementwise contract: ``module(x)[i] == module(x[i])``, no params, buffers,
    Python state, or in-place writes; shape/dtype preserving and deterministic.
    Continuity is not required because the delta algebra is an exact finite difference.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list, loss_kind: str = "cross_entropy"):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_kind = loss_kind
        self._layer_keys = layer_keys  # ordered list of (name, tag, module)
        # Flatten is a no-op only when it runs before any Linear; try_build rejects
        # the rest.
        self.leading_flatten = any(tag == "flatten" for _, tag, _ in layer_keys)

    @classmethod
    def try_build(
        cls, model: nn.Module, loss_fn: Callable, loss_kind: str = "cross_entropy"
    ) -> "BatchedLinearEvaluator | None":
        """Build if model is compatible, else return None."""
        # The plan is rebuilt from named_children(), which misses inline activations in
        # a custom forward; only trust real nn.Sequential forwards.
        if type(model).forward is not nn.Sequential.forward:
            _warn_if_sequential_equivalent(model, _SUPPORTED_LAYERS)
            return None
        if _has_hooks(model):
            return None

        # Build an ordered (name, tag, module) plan; the module is kept so evaluate()
        # applies an activation's exact semantics and a weight transform is reachable.
        def _entry(full, submod):
            # In-place activations would write through tensors the delta path shares
            # across candidates, so defer them to vmap.
            if getattr(submod, "inplace", False):
                return None
            if _has_hooks(submod):
                return None
            for base, tag in _TAGGED_LAYERS:
                if not isinstance(submod, base):
                    continue
                # A subclass overriding forward does more than its base's tag describes.
                if type(submod).forward is not base.forward:
                    return None
                # A non-default Flatten reshapes differently from the leading flatten.
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
            # A Flatten after a Linear would be applied at the wrong point; defer.
            if entry[1] == "flatten" and seen_linear:
                return None
            seen_linear = seen_linear or entry[1] == "linear"
            layer_keys.append(entry)
        if not layer_keys:
            return None
        if not _plan_params_distinct(layer_keys):
            return None
        # All named parameters must be covered by detected Linear layers
        # (remove_duplicate=False, or a tied parameter hides under its first name).
        linear_param_keys = set()
        for name, tag, _ in layer_keys:
            if tag == "linear":
                linear_param_keys.add(f"{name}.weight")
                linear_param_keys.add(f"{name}.bias")
        model_param_keys = {n for n, _ in model.named_parameters(remove_duplicate=False)}
        if not model_param_keys.issubset(linear_param_keys):
            return None
        # Every path built on this plan pins one param dtype, so decline mixed-dtype
        # models (vmap handles them).
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
        # Expand the input across the N param configs.
        if inputs.dim() == 2:
            x = inputs.unsqueeze(0).expand(N, -1, -1)  # (N, B, in_feat)
        else:
            # Flatten spatial dims for non-2D inputs.
            x = inputs.reshape(inputs.shape[0], -1).unsqueeze(0).expand(N, -1, -1)

        for name, tag, module in self._layer_keys:
            if tag == "linear":
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                wq, bq = _weight_transforms(module)
                # Frozen params are constant across candidates, so the module's tensor
                # stands in; bmm does not broadcast its batch dim, hence the expand.
                W = stacked_params.get(w_key)
                if W is None:
                    W = module.weight.unsqueeze(0).expand(N, -1, -1)
                if wq is not None:
                    # Elementwise, so transforming the whole stack equals per-candidate.
                    W = wq(W)
                bias = stacked_params.get(b_key)
                if bias is None and getattr(module, "bias", None) is not None:
                    bias = module.bias.unsqueeze(0)
                if bias is not None and bq is not None:
                    bias = bq(bias)
                # baddbmm folds the bias into the same kernel.
                if bias is not None:
                    x = torch.baddbmm(bias.unsqueeze(1), x, W.transpose(1, 2))
                else:
                    x = torch.bmm(x, W.transpose(1, 2))
            elif tag in ("flatten", "dropout"):
                pass  # input is pre-flattened; dropout is identity in eval mode
            else:
                # Apply the real module so its configuration is exact.
                x = module(x)

        return _reduce_per_candidate(x, targets, self.loss_fn, self.loss_kind)


def _resolve_layout_entry(self, offsets: torch.Tensor, pdim: int, span=None):
    """The one layout entry every candidate in this chunk perturbs, or None.

    ``span`` is ``(lo, hi)`` when the caller knows the chunk bounds, avoiding a device
    reduce and sync.
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

    The graph ahead of the perturbed tensor runs once, and the candidate axis appears
    only where it is first consumed. Makes no assumption about the module set, unlike
    :class:`SparseDeltaEvaluator`.
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
        """Decline where the caller's vmap path is unusable, or on tied weights."""
        if evaluator._vmap_failed or evaluator._inplace_forced or layout.shared_groups:
            return None
        return cls(
            evaluator.model, evaluator.loss_fn, layout, evaluator._buffers, evaluator.per_sample, owner=evaluator
        )

    def _retired(self) -> bool:
        """Whether a vmap failure has sent this path back to the dense one."""
        return self._owner is not None and self._owner._vmap_failed

    def resolve_site(self, offsets: torch.Tensor, pdim: int, span=None):
        """The one layout entry this chunk perturbs, or None to use the dense path."""
        if self._retired():
            return None
        return _resolve_layout_entry(self, offsets, pdim, span)

    def resolve_spec(self, subspace, lo: int, hi: int, sub_dim: int):
        """The per-layer spec owning coordinates ``[lo, hi)``, or None if they straddle.

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

        Only the columns ``[col_start, col_start + n_groups*pdim)`` of the projection
        reach the output, so the correction is a bmm over ``pdim`` columns per group.
        """
        base = bary_sd[spec.entry_key]
        n_groups, n_cand, pdim = dcoords.shape
        n = n_groups * n_cand
        n_cols = n_groups * pdim
        P = projections[spec.entry_key] if spec.is_projected else None

        if isinstance(P, torch.Tensor):
            # P is (num_params, num_coords); .t() first so groups lead.
            blk = P.narrow(1, col_start, n_cols).t().reshape(n_groups, pdim, -1)
            delta = torch.bmm(dcoords.to(blk.dtype), blk).reshape(n, -1)
        else:
            # Sparse/unprojected specs take a full-width coordinate row.
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
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates."""
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

        ``in_dims`` batches that argument alone, so the work ahead of it runs once.
        """
        shared = {k: v for k, v in base_sd.items() if k != key}
        shared.update(self._buffers)
        model, loss_fn, per_sample = self.model, self.loss_fn, self.per_sample

        # Repeat evaluate()'s cast/autocast frame here. Cast to the model's dtype, not
        # the site's: the perturbed layer need not be the one the input reaches first.
        cast_to = _uniform_float_dtype(shared, site.dtype)
        if cast_to is not None and inputs.is_floating_point() and inputs.dtype != cast_to:
            inputs = self._owner.cast_inputs(inputs, cast_to) if self._owner is not None else inputs.to(cast_to)

        # Repeat evaluate()'s eval-mode frame, or train-mode dropout stays live.
        was_training = model.training
        if was_training:
            model.eval()

        def single_eval(site_param, inputs, targets):
            output = functional_call(model, {**shared, key: site_param}, (inputs,))
            loss = loss_fn(output, targets) if targets is not None else loss_fn(output)
            return _fold_candidate_loss(loss, per_sample)

        # No autocast frame: under vmap it casts activations but not a Conv2d bias, so
        # a conv model raises a dtype mismatch.
        try:
            return vmap(single_eval, in_dims=(0, None, None))(site, inputs, targets)
        except Exception as e:
            # The step calls this directly, so the failure never reaches evaluate()'s
            # fallback. Usual cause is a forward that draws its own randomness, which
            # vmap rejects. Score this chunk in a loop; later chunks use the dense path.
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

    Every layer but the perturbed one holds the shared base weight, so the dense path's
    per-candidate bmm against an ``(N, d_out, d_in)`` stack does N times the work for
    nothing. The win is largest for a shallow net whose parameters concentrate in an
    early layer; deep nets should use a subspace instead.

    Handles the same module set as :class:`BatchedLinearEvaluator`.
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
        # A tied weight moves two module paths, so a single-site correction applies only
        # half the perturbation.
        if layout.shared_groups:
            return None
        plan = BatchedLinearEvaluator.try_build(model, loss_fn, loss_kind)
        if plan is None:
            return None
        if not _plan_keys_all_trainable(plan._layer_keys, {e.key for e in layout.entries}):
            return None
        return cls(model, loss_fn, plan._layer_keys, layout, loss_kind)

    def resolve_site(self, offsets: torch.Tensor, pdim: int, span=None):
        """The entry every candidate in this chunk perturbs, narrowed to what this path
        corrects (the detected Linear weights and biases)."""
        entry = _resolve_layout_entry(self, offsets, pdim, span)
        return entry if entry is not None and entry.key in self._site else None

    @torch.inference_mode()
    def evaluate(self, base_sd, entry_key, local_idx, values, inputs, targets=None):
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates.

        The perturbed positions are a function of the particle, not the vertex, so a
        group's ``C`` candidates share one ``local_idx`` row.
        """
        groups, cand, pdim = values.shape
        n = groups * cand
        site_idx, is_weight = self._site[entry_key]
        param_dtype = next(iter(base_sd.values())).dtype
        inputs = cast_inputs_memo(self, inputs, param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        batch = x.shape[0]
        # Deltas, not absolute values: the correction adds onto the base output, so a
        # transformed site moves by the difference of the transformed values.
        site_q = _weight_transforms(self._layer_keys[site_idx][2])[0 if is_weight else 1]
        base_param = base_sd[entry_key].reshape(-1)
        base_at_site = base_param[local_idx].unsqueeze(1)
        if site_q is not None:
            delta = (site_q(values) - site_q(base_at_site)).to(param_dtype)  # (G, C, pdim)
        else:
            delta = (values - base_at_site).to(param_dtype)

        def at(t, cols):
            """``(G, 1, B, pdim)`` holding ``t[:, cols]``, broadcastable over C.

            Gathers the transposed view, whose copy runs over contiguous rows.
            """
            return t.t().index_select(0, cols.reshape(-1)).reshape(groups, pdim, batch).permute(0, 2, 1).unsqueeze(1)

        # Before the site `x` is shared at batch 1; through the site the candidates
        # differ only at `dcols` by `dvals`; after it `xn` is dense.
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
                    # Elementwise: recompute the perturbed entries, then re-express the
                    # delta as a difference.
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
                    # Several dcols entries can name the same output unit, and the
                    # nonlinear activation must see their sum.
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

    The correction is linear in ``pdim`` coordinates:
    ``x @ W_n.T = x @ W_bary.T + sum_j d_nj * (x @ M_j.T)`` with ``M_j = reshape(P[:, j])``,
    so the ``M_j`` products are computed once per column and reused. Nothing of shape
    ``(N, d_out, d_in)`` is built.

    ``resolve_site`` returns None when the run is not contained in one block, so the
    caller falls back to ``reconstruct_batch``. Handles the same module set as
    :class:`BatchedLinearEvaluator`.
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
        # The correction is a linear map of the coordinate delta, so a weight transform
        # breaks it and the model goes to vmap.
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

        Only a projected weight or an unprojected bias matches a correction; an
        unprojected weight indexes the flattened weight, not output units.
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
            # A column of P is sparse, so gather the entries and scatter-add instead of
            # forming M_j.
            cols = torch.arange(col_start, col_start + n_cols, device=x.device)
            rows, vals = P.columns(cols, x.device, x.dtype)  # (n_cols, nnz)
            out_idx = torch.div(rows, d_in, rounding_mode="floor")  # (n_cols, nnz)
            in_idx = rows.remainder(d_in)
            contrib = x.index_select(1, in_idx.reshape(-1)).reshape(x.shape[0], n_cols, -1)
            contrib = contrib.permute(1, 2, 0) * vals.unsqueeze(-1).to(acc_dtype)  # (n_cols, nnz, B)
            g = torch.zeros(n_cols, d_out, x.shape[0], device=x.device, dtype=acc_dtype)
            return g.scatter_add_(1, out_idx.unsqueeze(-1).expand_as(contrib), contrib)
        # P.view(d_out, d_in, -1)[o, i, j] is M_j[o, i] with no copy; batching over
        # columns rather than d_out is faster at the same FLOPs.
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
        """Losses of shape ``(G*C,)`` for ``G`` particle groups of ``C`` candidates."""
        n_groups, n_cand, pdim = dcoords.shape
        site_idx, is_weight = self._site[spec.entry_key]
        param_dtype = next(iter(bary_sd.values())).dtype
        # Accumulate the correction in fp32: a half-precision add would drop the small
        # delta on top of a full-scale output.
        acc_dtype = torch.float32 if param_dtype in (torch.float16, torch.bfloat16) else param_dtype
        inputs = cast_inputs_memo(self, inputs, param_dtype)
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
                    # An unprojected bias site: the coordinates are the bias delta.
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

    A candidate perturbs a Linear by ``dW_n = A_n @ B`` with ``B`` shared, so
    ``x (W + A_n B)^T = x W^T + (x B^T) A_n^T``: the base term is one GEMM and the
    correction is rank-``r``. Nothing of shape ``(N, d_out, d_in)`` is built.

    Handles the same module set as :class:`BatchedLinearEvaluator`.
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
        # The correction is linear in the factor, so a weight transform falls back to vmap.
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
        inputs = cast_inputs_memo(self, inputs, param_dtype)
        x = inputs if inputs.dim() == 2 else inputs.reshape(inputs.shape[0], -1)
        # Shared across candidates until the first perturbed layer.
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
                        # x is (B, d_in), so one (B, r) product feeds every candidate.
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
