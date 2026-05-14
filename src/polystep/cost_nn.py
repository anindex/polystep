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

    if param_device.type != 'cuda':
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
            ``torch.compile(mode="reduce-overhead")`` for kernel fusion
            and CUDA graph capture. Falls back to eager on failure.
            Best for CUDA models with static shapes. Default False.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        chunk_size: Union[None, int, str] = None,
        compile_vmap: bool = False,
        use_inplace: Optional[bool] = None,
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

        # Force eval mode for consistent behavior (frozen BN stats, no dropout)
        model.eval()

        # Try to build fast batched-linear evaluator (MLP-only models).
        # Only for CrossEntropyLoss since the bmm path hardcodes it.
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            self._batched_linear = BatchedLinearEvaluator.try_build(model, loss_fn)
        else:
            self._batched_linear = None

        # Auto-detect whether to use memory-efficient in-place evaluation.
        # For very large models (>500K params) on GPU, vmap materializes N
        # copies of all intermediate activations simultaneously, causing
        # O(N × activation) memory. In-place evaluation uses O(1 × activation)
        # regardless of N. For models ≤500K params (e.g. MNISTNet ~102K,
        # CIFAR10MLP ~199K), chunked vmap is fast and fits in GPU memory.
        # Pass use_inplace=True/False to override auto-detection.
        if use_inplace is not None:
            self._use_inplace = use_inplace
        else:
            n_params = sum(p.numel() for p in model.parameters())
            try:
                on_gpu = next(model.parameters()).device.type == 'cuda'
            except StopIteration:
                on_gpu = False
            self._use_inplace = on_gpu and n_params > 500_000

        # Cache frozen buffers from the real model (shared across all particles)
        self._buffers = dict(model.named_buffers())

        # Cache param dict for in-place evaluation (avoids O(L) module traversal per call)
        self._param_dict_cache = dict(self.model.named_parameters())

    def reset_vmap(self) -> None:
        """Reset the vmap failure flag so vmap is attempted again.

        Useful after changing the model architecture (e.g., swapping layers),
        moving the model to a different device, or upgrading PyTorch where
        a previously unsupported op may now work under vmap.
        """
        self._vmap_failed = False
        self._warned = False

    @property
    def chunk_size(self) -> Optional[int]:
        """Resolved chunk_size for vmap.

        Returns None (no chunking), or a positive int. When
        ``chunk_size="auto"`` was passed, queries GPU memory once to compute
        a safe value (returns None on CPU), then caches the result.
        """
        if self._chunk_size_raw == "auto":
            if self._chunk_size_cached is _UNSET:
                self._chunk_size_cached = auto_detect_chunk_size(self.model)
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

        try:
            # Fast path: batched bmm for Linear-only models (MLP)
            # Only used for supervised (targets != None) with cross-entropy.
            if self._batched_linear is not None and targets is not None:
                return self._batched_linear.evaluate(stacked_params, inputs, targets)

            # Memory-efficient path: in-place weight swap for large GPU models.
            # Uses O(1 × activation) memory instead of O(N × activation).
            if self._use_inplace:
                return self._evaluate_inplace(stacked_params, inputs, targets)

            if self._vmap_failed:
                result = self._evaluate_loop(stacked_params, inputs, targets)
            else:
                try:
                    result = self._evaluate_vmap(stacked_params, inputs, targets)
                except Exception as e:
                    # Only catch vmap/functorch-related errors; re-raise real bugs
                    msg = str(e).lower()
                    is_vmap_issue = any(
                        k in msg
                        for k in ("vmap", "functorch", "batched tensor", "batched", "randomness")
                    )
                    if not is_vmap_issue:
                        raise
                    if not self._warned:
                        warnings.warn(
                            f"vmap failed for {type(self.model).__name__}: {e}. "
                            f"Falling back to sequential evaluation.",
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
        in ``torch.compile(mode="reduce-overhead")`` for kernel fusion and
        CUDA graph capture. Falls back to eager vmap on compilation failure.
        """
        buffers = self._buffers
        loss_fn = self.loss_fn
        model = self.model
        resolved_chunk = self.chunk_size

        def single_eval(params):
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

        batched = vmap(single_eval, in_dims=(0,), chunk_size=resolved_chunk)

        # Compiled path: torch.compile on vmap for kernel fusion + CUDA graphs.
        # Only attempted on CUDA with compile_vmap=True. Lazy-compiled on first call.
        if self._compile_vmap and not self._compile_failed:
            if self._compiled_vmap_fn is None:
                try:
                    # Use "default" mode: kernel fusion without CUDA graphs.
                    # "reduce-overhead" (CUDA graphs) has tensor ownership conflicts
                    # with vmap's chunked output concatenation.
                    self._compiled_vmap_fn = torch.compile(
                        batched, mode="default", fullgraph=False,
                    )
                except Exception:
                    self._compile_failed = True
                    self._compiled_vmap_fn = None

            if self._compiled_vmap_fn is not None:
                try:
                    return self._compiled_vmap_fn(stacked_params)
                except Exception:
                    # Compilation or execution failed - fall back to eager permanently
                    self._compile_failed = True
                    self._compiled_vmap_fn = None

        return batched(stacked_params)

    def _evaluate_loop(self, stacked_params, inputs, targets):
        """Sequential fallback when vmap is incompatible."""
        N = next(iter(stacked_params.values())).shape[0]
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
        for key in stacked_params:
            if key in param_dict:
                original_params[key] = param_dict[key].data.clone()

        try:
            for i in range(N):
                # Swap weights in-place - no copies, just overwrite .data
                for key in stacked_params:
                    if key in param_dict:
                        param_dict[key].data.copy_(stacked_params[key][i])

                # Forward pass under inference_mode - minimal memory
                with torch.inference_mode():
                    output = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(output, targets)
                    else:
                        loss = self.loss_fn(output)
                    if loss.dim() > 0:
                        loss = loss.mean()
                # .item()-free: store tensor directly, detach from inference graph
                losses[i] = loss.detach()
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

        EGGROLL-inspired: instead of materializing ALL N configs' full
        weights via ``reconstruct_batch`` then evaluating, this reconstructs
        ONE config at a time directly into model weights using
        ``apply_perturbation_inplace``, then runs a single forward pass.

        Memory: O(model_params + 1 × batch × activation) regardless of N.
        No stacked_params dict is ever allocated.

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
        losses = torch.empty(N, device=device)

        was_training = self.model.training
        if was_training:
            self.model.eval()

        try:
            for i in range(N):
                # Reconstruct weights for config i directly into model params
                subspace.apply_perturbation_inplace(
                    projections, self.model, base_sd, flat_subspace_batch[i],
                )
                # Forward pass - inference memory only
                with torch.inference_mode():
                    output = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(output, targets)
                    else:
                        loss = self.loss_fn(output)
                    if loss.dim() > 0:
                        loss = loss.mean()
                losses[i] = loss.detach()
        finally:
            # Restore base weights
            param_dict = dict(self.model.named_parameters())
            for key, base in base_sd.items():
                if key in param_dict:
                    param_dict[key].data.copy_(base)
            if was_training:
                self.model.train()

        return losses


class BatchedLinearEvaluator:
    """Fast batched evaluation for models composed only of Linear + stateless layers.

    Replaces vmap + functional_call with explicit ``torch.bmm`` per Linear
    layer, eliminating vmap dispatch overhead. For N parameter configs of a
    k-layer MLP, performs k batched matmuls instead of N sequential forward
    passes or a single vmap call.

    Only supports: ``nn.Linear``, ``nn.ReLU``, ``nn.Flatten``, ``nn.Dropout``
    (eval mode), ``nn.BatchNorm1d`` (eval mode with frozen stats).
    Returns None from ``try_build`` if the model contains unsupported layers.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, layer_keys: list):
        self.model = model
        self.loss_fn = loss_fn
        self._layer_keys = layer_keys  # ordered list of (name, type_tag)

    @classmethod
    def try_build(cls, model: nn.Module, loss_fn: Callable) -> "BatchedLinearEvaluator | None":
        """Build if model is compatible, else return None."""
        supported = (nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh,
                     nn.Flatten, nn.Dropout)
        layer_keys = []
        for name, mod in model.named_children():
            if isinstance(mod, nn.Linear):
                layer_keys.append((name, 'linear'))
            elif isinstance(mod, nn.Sequential):
                # Walk one level of Sequential
                for subname, submod in mod.named_children():
                    full = f"{name}.{subname}"
                    if isinstance(submod, nn.Linear):
                        layer_keys.append((full, 'linear'))
                    elif isinstance(submod, supported):
                        layer_keys.append((full, type(submod).__name__.lower()))
                    else:
                        return None  # unsupported layer
            elif isinstance(mod, supported):
                layer_keys.append((name, type(mod).__name__.lower()))
            else:
                return None  # unsupported layer (Conv2d, etc.)
        if not layer_keys:
            return None
        # Verify all named parameters are covered by detected Linear layers.
        # Models with extra parameters (e.g., learned scales) need vmap.
        linear_param_keys = set()
        for name, tag in layer_keys:
            if tag == 'linear':
                linear_param_keys.add(f"{name}.weight")
                linear_param_keys.add(f"{name}.bias")
        model_param_keys = {n for n, _ in model.named_parameters()}
        if not model_param_keys.issubset(linear_param_keys):
            return None
        return cls(model, loss_fn, layer_keys)

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

        for name, tag in self._layer_keys:
            if tag == 'linear':
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                W = stacked_params[w_key]  # (N, out, in)
                # bmm: (N, out, in) @ (N, in, B)^T → (N, B, out)
                x = torch.bmm(x, W.transpose(1, 2))  # (N, B, out)
                if b_key in stacked_params:
                    x = x + stacked_params[b_key].unsqueeze(1)  # (N, 1, out) broadcast
            elif tag == 'relu':
                x = torch.relu(x)
            elif tag == 'leakyrelu':
                x = torch.nn.functional.leaky_relu(x)
            elif tag == 'sigmoid':
                x = torch.sigmoid(x)
            elif tag == 'tanh':
                x = torch.tanh(x)
            elif tag in ('flatten', 'dropout'):
                pass  # already flat / eval mode = identity

        # x: (N, B, out_features) - compute per-config loss
        if targets is not None:
            tgt = targets.unsqueeze(0).expand(N, -1)  # (N, B)
            # Reshape for cross-entropy: (N*B, C) vs (N*B,)
            losses = torch.nn.functional.cross_entropy(
                x.reshape(N * targets.shape[0], -1),
                tgt.reshape(-1),
                reduction='none',
            ).reshape(N, -1).mean(dim=1)  # (N,)
        else:
            losses = self.loss_fn(x).mean(dim=1) if x.dim() > 2 else self.loss_fn(x)
        return losses


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
