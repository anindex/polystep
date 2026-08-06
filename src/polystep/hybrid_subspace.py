"""Per-layer projections with coordinated rotation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from .solvers._shared import decomposition_dtype, thin_qr

from .projection.sparse import SparseRandomProjection
from .subspace import ProjectedAbsorbMixin, ProjectionSpec, SvdRatioMixin, _stable_entry_seed, absorb_due

if TYPE_CHECKING:
    import torch.nn as nn
    from .blockwise import BlockConfig
    from .transform import ParamLayout


LayerProjectionSpec = ProjectionSpec


def _mixed_entry_dtypes(layout: "ParamLayout") -> "Optional[Dict[str, torch.dtype]]":
    """Per-entry dtypes, or None when they already agree."""
    dtypes = {e.key: e.dtype for e in layout.entries}
    floats = {d for d in dtypes.values() if d.is_floating_point}
    return dtypes if len(floats) > 1 else None


def _require_positive_rank(rank: int, name: str) -> None:
    """Reject a zero rank, which yields a spec with no coordinates."""
    if rank < 1:
        raise ValueError(f"{name} must be >= 1, got {rank}.")


def _require_tall(spec: "LayerProjectionSpec") -> None:
    """Reject a wide spec, which has no orthonormal basis."""
    if spec.num_params < spec.num_coords:
        raise ValueError(
            f"{spec.entry_key}: num_coords ({spec.num_coords}) exceeds num_params "
            f"({spec.num_params}), which has no orthonormal projection."
        )


def _scale_specs_to_budget(
    specs: list,
    max_dim: "Optional[int]",
    current_dim: int,
) -> "Tuple[list, int]":
    """Scale projected specs so the total stays within ``max_dim``."""
    if max_dim is None or current_dim <= max_dim:
        return specs, current_dim

    unprojected_dim = sum(s.num_coords for s in specs if not s.is_projected)
    n_projected = sum(1 for s in specs if s.is_projected)
    target_projected = max_dim - unprojected_dim

    if target_projected < n_projected:
        # Unprojected width alone exceeds the budget, so the cap is unreachable.
        target_projected = n_projected
        warnings.warn(
            f"max_subspace_dim={max_dim} is unreachable: unprojected parameters take "
            f"{unprojected_dim} coordinates and {n_projected} projected layers need one "
            f"each, so the subspace is {unprojected_dim + n_projected}.",
            stacklevel=3,
        )

    projected_dim = current_dim - unprojected_dim
    if projected_dim <= target_projected:
        return specs, current_dim

    scale = target_projected / projected_dim
    remaining, left = target_projected, n_projected
    new_specs: list = []
    new_offset = 0
    for spec in specs:
        if spec.is_projected:
            left -= 1
            headroom = min(spec.num_params, remaining - left)
            num_coords = min(max(1, round(spec.num_coords * scale)), headroom)
            remaining -= num_coords
            # Full width means the projection is the identity; perturb directly.
            spec = replace(spec, num_coords=num_coords, is_projected=num_coords < spec.num_params)
        new_specs.append(replace(spec, flat_start=new_offset, flat_end=new_offset + spec.num_coords))
        new_offset += spec.num_coords
    return new_specs, new_offset


@dataclass
class HybridSubspace(ProjectedAbsorbMixin, SvdRatioMixin):
    """Hybrid subspace: per-layer projections with a shared rotation schedule."""

    specs: Tuple[LayerProjectionSpec, ...]
    subspace_dim: int
    compression_ratio: float
    seed: int = 0
    rotation_mode: str = "displacement"
    rotation_interval: int = 0
    svd_ratio_init: float = 0.0
    svd_ratio_final: float = 0.5
    displacement_history_size: int = 5
    absorb_mode: str = "stagnation"
    absorb_patience: int = 20
    absorb_interval: int = 0
    sparse_threshold_bytes: int = 1_000_000_000  # 1GB: layers exceeding this use sparse projection
    # Redraw via displacement-SVD at the absorb boundary; the duals are already reset there.
    absorb_aligned_active: bool = False

    _total_params: int = 0
    # Budget control settings (preserved across rank transitions)
    _max_subspace_dim: Optional[int] = None
    # Per-entry dtype; on mixed models each entry's projection is built at its own dtype.
    # Empty means uniform.
    _entry_dtypes: Optional[Dict[str, torch.dtype]] = None

    def __post_init__(self) -> None:
        if self.rotation_interval != 0:
            warnings.warn(
                "HybridSubspace works best with rotation_interval=0. "
                "Non-zero values cause dual potential resets that degrade accuracy.",
                stacklevel=2,
            )
        if self.compression_ratio == 0.0 and self._total_params > 0:
            object.__setattr__(self, "compression_ratio", self.subspace_dim / self._total_params)

    # Factory methods

    @classmethod
    def from_layout(
        cls,
        layout: "ParamLayout",
        rank: int,
        seed: int = 0,
        max_subspace_dim: Optional[int] = None,
        **kwargs,
    ) -> "HybridSubspace":
        """Create a HybridSubspace with fixed rank from a ParamLayout."""
        _require_positive_rank(rank, "rank")
        specs = []
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                effective_rank = min(rank, d_in, d_out)
                num_params = math.prod(shape)
                # Capped at num_params: a wider matrix has no orthonormal basis and the
                # fallback gain sqrt(num_params/num_coords) breaks the step radii.
                num_coords = min(d_out * effective_rank + effective_rank * d_in, num_params)
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                        # Full width means the projection is the identity; perturb directly.
                        is_projected=num_coords < num_params,
                    )
                )
                offset += num_coords
            else:
                # 1D params (bias, LayerNorm) are perturbed directly.
                num_elements = entry.numel
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_elements,
                        num_coords=num_elements,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                        is_projected=False,
                    )
                )
                offset += num_elements

        specs, offset = _scale_specs_to_budget(specs, max_subspace_dim, offset)

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        return cls(
            specs=tuple(specs),
            subspace_dim=offset,
            compression_ratio=compression,
            seed=seed,
            _total_params=total_params,
            _max_subspace_dim=max_subspace_dim,
            _entry_dtypes=_mixed_entry_dtypes(layout),
            **kwargs,
        )

    @classmethod
    def auto_from_layout(
        cls,
        layout: "ParamLayout",
        compression_ratio: int = 16,
        min_rank: int = 4,
        max_rank: int = 64,
        seed: int = 0,
        max_subspace_dim: Optional[int] = None,
        **kwargs,
    ) -> "HybridSubspace":
        """Create a HybridSubspace with auto per-layer rank selection."""
        _require_positive_rank(min_rank, "min_rank")
        _require_positive_rank(max_rank, "max_rank")
        specs = []
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                min_dim = min(d_in, d_out)
                auto_rank = max(min_rank, min(max_rank, min_dim // compression_ratio))
                effective_rank = min(auto_rank, d_in, d_out)
                num_params = math.prod(shape)
                num_coords = min(d_out * effective_rank + effective_rank * d_in, num_params)
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                        # Full width means the projection is the identity; perturb directly.
                        is_projected=num_coords < num_params,
                    )
                )
                offset += num_coords
            else:
                num_elements = entry.numel
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_elements,
                        num_coords=num_elements,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                        is_projected=False,
                    )
                )
                offset += num_elements

        specs, offset = _scale_specs_to_budget(specs, max_subspace_dim, offset)

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        return cls(
            specs=tuple(specs),
            subspace_dim=offset,
            compression_ratio=compression,
            seed=seed,
            _total_params=total_params,
            _max_subspace_dim=max_subspace_dim,
            _entry_dtypes=_mixed_entry_dtypes(layout),
            **kwargs,
        )

    # Projection management

    def init_projections(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Union[torch.Tensor, SparseRandomProjection]]:
        """Per-layer projection matrices, memoized by ``(device, dtype)``.

        Callers must not write into the result.
        """
        cached = getattr(self, "_init_projections_cache", None)
        if cached is not None and cached[0] == (device, dtype):
            return cached[1]
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}
        for spec in self.specs:
            if not spec.is_projected:
                continue
            if self._use_sparse(spec):
                entry_seed = _stable_entry_seed(self.seed, spec.entry_key, 0)
                projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                P = self._get_projection(spec, device, dtype, step=0)
                projections[spec.entry_key] = P
        # Not an __init__ field, so dataclasses.replace drops it, matching _fused_P.
        self._init_projections_cache = ((device, dtype), projections)
        return projections

    def entry_dtype(self, spec: LayerProjectionSpec, default: torch.dtype) -> torch.dtype:
        """The dtype this entry's projection is built at (its own parameter dtype)."""
        return (self._entry_dtypes or {}).get(spec.entry_key, default)

    def _use_sparse(self, spec: LayerProjectionSpec) -> bool:
        """Whether this layer's dense projection would exceed the byte threshold."""
        return spec.num_params * spec.num_coords * 4 > self.sparse_threshold_bytes

    def _get_projection(
        self,
        spec: LayerProjectionSpec,
        device: torch.device,
        dtype: torch.dtype,
        step: int = 0,
    ) -> torch.Tensor:
        """Build the ``(num_params, num_coords)`` projection for one entry, QR-orthogonalized."""
        dtype = self.entry_dtype(spec, dtype)
        # QR-orthogonal Gaussian columns give isotropic unit-norm perturbations (SubZero, ICCV 2025).
        entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(entry_seed)

        # Generate on CPU then move; a CUDA generator has no CPU path.
        P_raw = torch.randn(
            spec.num_params,
            spec.num_coords,
            generator=gen,
            dtype=dtype,
            device="cpu",
        )
        _require_tall(spec)
        # geqrf has no bf16/fp16 CPU kernel; orthogonalize in fp32, cast back.
        qr_in = P_raw.to(decomposition_dtype(P_raw.dtype))
        P, _ = thin_qr(qr_in)  # (num_params, num_coords)
        P = P.to(dtype=P_raw.dtype).to(device=device)

        return P

    # Rotation coordination

    def rotate_all(
        self,
        projections: Dict[str, torch.Tensor],
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Rotate all layer projections on the shared schedule, by mode.

        ``'random'`` redraws every basis; ``'displacement'`` keeps SVD-derived directions per layer.
        """
        # Step 0 is the freshly seeded basis; rotating it discards a basis nothing has scored yet.
        if self.rotation_interval <= 0 or step <= 0 or step % self.rotation_interval != 0:
            return projections

        # Sparse tensors lack .device/.dtype; take them from the first dense projection.
        device, dtype = torch.device("cpu"), torch.float32  # safe defaults
        for p in projections.values():
            if isinstance(p, torch.Tensor):
                device, dtype = p.device, p.dtype
                break

        use_random = (
            self.rotation_mode == "random" or displacement_history is None or displacement_history.shape[0] == 0
        )

        if not use_random:
            # Fall back to random when the history is non-finite or near zero.
            if not torch.isfinite(displacement_history).all():
                use_random = True
            elif (displacement_history * displacement_history).sum().item() < 1e-20:
                use_random = True

        if use_random:
            return self._rotate_all_random(device, dtype, step)
        else:
            svd_ratio = self.get_svd_ratio(step, total_steps)
            return self._rotate_all_displacement(
                projections,
                displacement_history,
                svd_ratio,
                device,
                dtype,
                step,
            )

    def _rotate_all_random(
        self,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> Dict[str, Union[torch.Tensor, SparseRandomProjection]]:
        """Regenerate all projections with new seeds (random mode)."""
        new_projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}
        for spec in self.specs:
            if not spec.is_projected:
                continue
            if self._use_sparse(spec):
                entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step)
                new_projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                P = self._get_projection(spec, device, dtype, step=step)
                new_projections[spec.entry_key] = P
        return new_projections

    def _rotate_all_displacement(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        displacement_history: torch.Tensor,
        svd_ratio: float,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> Dict[str, Union[torch.Tensor, SparseRandomProjection]]:
        """Rotate each layer toward its displacement-history SVD directions; sparse layers redraw randomly."""
        new_projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}

        for spec in self.specs:
            if not spec.is_projected:
                continue  # 1D params have no stored projection
            P = projections[spec.entry_key]
            if isinstance(P, SparseRandomProjection):
                # SVD rotation needs a dense matmul; fall back to a random redraw.
                entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step)
                new_projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                layer_disp = displacement_history[:, spec.flat_start : spec.flat_end]
                new_P = self._rotate_layer_displacement(
                    P,
                    spec,
                    layer_disp,
                    svd_ratio,
                    device,
                    dtype,
                    step,
                )
                new_projections[spec.entry_key] = new_P

        return new_projections

    def _rotate_layer_displacement(
        self,
        P_old: torch.Tensor,
        spec: LayerProjectionSpec,
        layer_displacement: torch.Tensor,
        svd_ratio: float,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> torch.Tensor:
        """Rotate one layer's projection: keep top SVD directions, fill the rest randomly."""
        _require_tall(spec)
        # The rotated basis must come back at this entry's dtype, not the coordinates'.
        dtype = self.entry_dtype(spec, dtype)
        if svd_ratio <= 0.0:
            return self._get_projection(spec, device, dtype, step=step)

        k_svd = max(1, int(svd_ratio * spec.num_coords))
        k_random = spec.num_coords - k_svd

        # Displacement history in full parameter space: (num_params, history_len).
        D_full = P_old @ layer_displacement.T.to(P_old.dtype)

        if not torch.isfinite(D_full).all():
            return self._get_projection(spec, device, dtype, step=step)

        svd_dtype = D_full.dtype
        D_full = D_full.to(decomposition_dtype(svd_dtype))

        # The history is short, so the full SVD costs about the same as a randomized one.
        U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
        k_svd = min(k_svd, U.shape[1])
        U_top = U[:, :k_svd]
        k_random = spec.num_coords - k_svd
        if U_top.dtype != svd_dtype:
            U_top = U_top.to(svd_dtype)

        # Drawn on CPU so the seed is portable across devices.
        entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step, "random")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(entry_seed)

        Z_random = torch.randn(
            spec.num_params,
            k_random,
            generator=gen,
            dtype=dtype,
            device="cpu",
        ).to(device)

        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        combined = torch.cat([U_top, Z_random], dim=1)
        orig_dtype = combined.dtype
        Q, _ = thin_qr(combined.to(decomposition_dtype(orig_dtype)))
        return Q.to(orig_dtype) if Q.dtype != orig_dtype else Q

    # Absorb coordination

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Whether to fold the perturbation into the base weights this step."""
        return absorb_due(
            self.absorb_mode,
            self.absorb_patience,
            self.absorb_interval,
            stagnation_count,
            iteration,
        )

    # Core reconstruction methods

    def apply_perturbation(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct a state_dict from the base weights and one subspace vector."""
        result = {}
        for spec in self.specs:
            chunk = flat_subspace[spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]

            if spec.is_projected:
                P = projections[spec.entry_key]
                if isinstance(P, SparseRandomProjection):
                    delta = P.project(chunk)  # (num_params,)
                    result[spec.entry_key] = (base.reshape(-1) + delta).reshape(spec.original_shape)
                else:
                    result_flat = torch.addmm(
                        base.reshape(1, -1),
                        chunk.unsqueeze(0).to(P.dtype),
                        P.t(),
                    )
                    result[spec.entry_key] = result_flat.reshape(spec.original_shape)
            else:
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape).to(base.dtype)

        return result

    def _gather_fused_coords(self, flat_subspace: torch.Tensor) -> torch.Tensor:
        """Slice out the coordinates the fused dense blocks consume, cast to the block's dtype."""
        first_start = self._fused_dense_specs[0][0].flat_start
        last_end = self._fused_dense_specs[-1][0].flat_end
        total = sum(s.num_coords for s, _ in self._fused_dense_specs)
        if last_end - first_start == total:
            coords = flat_subspace[..., first_start:last_end]
        else:
            coords = torch.cat(
                [flat_subspace[..., s.flat_start : s.flat_end] for s, _ in self._fused_dense_specs],
                dim=-1,
            )
        fused_P = getattr(self, "_fused_P", None)
        if fused_P is not None and coords.dtype != fused_P.dtype:
            coords = coords.to(fused_P.dtype)
        return coords

    def prepare_inplace(self, base_sd: Dict[str, torch.Tensor]) -> None:
        """Cache the concatenated dense base row for a run of in-place candidates."""
        if getattr(self, "_fused_P", None) is None or not self._fused_dense_specs:
            return
        self._fused_base_row_cache = torch.cat(
            [base_sd[spec.entry_key].reshape(-1) for spec, _ in self._fused_dense_specs]
        ).unsqueeze(0)

    def release_inplace(self) -> None:
        """Drop the cached base row and scratch buffer."""
        self._fused_base_row_cache = None
        self._inplace_delta_buffer = None

    def _fused_base_row(self, base_sd: Dict[str, torch.Tensor]) -> torch.Tensor:
        cached = getattr(self, "_fused_base_row_cache", None)
        if cached is not None:
            return cached
        return torch.cat([base_sd[spec.entry_key].reshape(-1) for spec, _ in self._fused_dense_specs]).unsqueeze(0)

    def build_fused_projection(
        self,
        projections: Dict[str, Union[torch.Tensor, "SparseRandomProjection"]],
    ) -> None:
        """Build a fused block-diagonal projection from the dense per-layer projections.

        Skips a single block, and declines past ``max_fused_bytes`` where the padding read outweighs the saved launches.
        """
        dense_blocks = []
        self._fused_dense_specs = []  # specs participating in fused matmul
        self._fused_sparse_specs = []  # specs needing per-layer sparse path
        self._fused_bias_specs = []  # 1D params (identity, no projection)
        self._fused_odd_dtype_specs = []  # dense, but not the fused block's dtype
        total_params = 0
        total_coords = 0

        # block_diag needs one dtype; on mixed models fuse the majority and leave the rest per-layer.
        dense_dtypes: Dict[torch.dtype, int] = {}
        for spec in self.specs:
            P = projections.get(spec.entry_key)
            if spec.is_projected and isinstance(P, torch.Tensor):
                dense_dtypes[P.dtype] = dense_dtypes.get(P.dtype, 0) + spec.num_params
        fuse_dtype = max(dense_dtypes, key=dense_dtypes.get) if dense_dtypes else None

        for spec in self.specs:
            P = projections.get(spec.entry_key)
            if not spec.is_projected:
                self._fused_bias_specs.append(spec)
            elif isinstance(P, SparseRandomProjection):
                self._fused_sparse_specs.append((spec, P))
            elif P.dtype is not fuse_dtype:
                self._fused_odd_dtype_specs.append((spec, P))
            else:
                self._fused_dense_specs.append((spec, total_params))
                dense_blocks.append(P)
                total_params += spec.num_params
                total_coords += spec.num_coords

        fused_bytes = total_params * total_coords * 4
        max_fused_bytes = 32 * 1024 * 1024

        if len(dense_blocks) > 1 and fused_bytes <= max_fused_bytes:
            self._fused_P = torch.block_diag(*dense_blocks)
        else:
            self._fused_P = None
        self.release_inplace()

    def apply_perturbation_inplace(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        model: "nn.Module",
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
        param_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Write the perturbation straight into the model's parameter tensors."""
        if param_dict is None:
            param_dict = dict(model.named_parameters())

        # Fused path: one addmm plus one grouped copy per candidate, whatever the layer count.
        fused_P = getattr(self, "_fused_P", None)
        if fused_P is not None and self._fused_dense_specs:
            coords = self._gather_fused_coords(flat_subspace)
            buf = self._inplace_delta_buffer
            if buf is None or buf.shape[1] != fused_P.shape[0] or buf.dtype != fused_P.dtype:
                buf = fused_P.new_empty((1, fused_P.shape[0]))
                self._inplace_delta_buffer = buf
            torch.addmm(self._fused_base_row(base_sd), coords.unsqueeze(0), fused_P.t(), out=buf)
            dsts, srcs = [], []
            for spec, offset in self._fused_dense_specs:
                param = param_dict.get(spec.entry_key)
                if param is None:
                    continue
                dsts.append(param.data)
                srcs.append(buf[0, offset : offset + spec.num_params].view(spec.original_shape))
            if dsts:
                torch._foreach_copy_(dsts, srcs)
            remaining = (
                [s for s, _ in self._fused_sparse_specs]
                + [s for s, _ in self._fused_odd_dtype_specs]
                + list(self._fused_bias_specs)
            )
        else:
            remaining = self.specs

        for spec in remaining:
            key = spec.entry_key
            if key not in param_dict:
                continue
            chunk = flat_subspace[spec.flat_start : spec.flat_end]
            base = base_sd[key]
            param = param_dict[key]

            if spec.is_projected:
                P = projections[key]
                if isinstance(P, SparseRandomProjection):
                    delta = P.project(chunk)
                    param.data.copy_((base.reshape(-1) + delta).reshape(spec.original_shape))
                elif param.data.is_contiguous():
                    torch.addmm(
                        base.reshape(1, -1),
                        chunk.unsqueeze(0).to(P.dtype),
                        P.t(),
                        out=param.data.reshape(1, -1),
                    )
                else:
                    res = torch.addmm(base.reshape(1, -1), chunk.unsqueeze(0).to(P.dtype), P.t())
                    param.data.copy_(res.reshape(spec.original_shape))
            else:
                param.data.copy_(base + chunk.reshape(spec.original_shape).to(base.dtype))

    def reconstruct_batch(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points.

        The dense entries alias one buffer, so callers must treat them as read-only.
        """
        N = flat_subspace_batch.shape[0]
        result = {}

        if getattr(self, "_fused_P", None) is not None and self._fused_dense_specs:
            fused_coords = self._gather_fused_coords(flat_subspace_batch)
            fused_delta = fused_coords @ self._fused_P.t()

            for spec, param_offset in self._fused_dense_specs:
                block = fused_delta[:, param_offset : param_offset + spec.num_params]
                block.add_(base_sd[spec.entry_key].reshape(1, -1))
                # unflatten keeps this a view; reshape would copy the whole (N, num_params).
                result[spec.entry_key] = block.unflatten(1, spec.original_shape)

            for spec, P in self._fused_sparse_specs:
                chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
                # project() already copies, so adding first fuses the two passes into one.
                delta = base_sd[spec.entry_key].reshape(1, -1) + P.project(chunk)
                result[spec.entry_key] = delta.reshape(N, *spec.original_shape)

            for spec, P in self._fused_odd_dtype_specs:
                chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
                delta = (chunk.to(P.dtype) @ P.t()).reshape(N, *spec.original_shape)
                result[spec.entry_key] = delta.add_(base_sd[spec.entry_key])

            for spec in self._fused_bias_specs:
                chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
                base = base_sd[spec.entry_key]
                delta = chunk.reshape(N, *spec.original_shape).to(base.dtype)
                result[spec.entry_key] = base.unsqueeze(0) + delta

            return result

        for spec in self.specs:
            chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]

            if spec.is_projected:
                P = projections[spec.entry_key]
                if isinstance(P, SparseRandomProjection):
                    result[spec.entry_key] = (base.reshape(1, -1) + P.project(chunk)).reshape(N, *spec.original_shape)
                else:
                    result[spec.entry_key] = (chunk.to(P.dtype) @ P.t()).reshape(N, *spec.original_shape).add_(base)
            else:
                delta = chunk.reshape(N, *spec.original_shape).to(base.dtype)
                result[spec.entry_key] = base.unsqueeze(0) + delta

        return result


def create_hybrid_blocks(
    hybrid: HybridSubspace,
    particle_dim: int = 8,
) -> "List[BlockConfig]":
    """One ``BlockConfig`` per spec, so each layer gets its own OT solve.

    Use the block's flat ranges, not the spec's, to split and reassemble.
    """
    from .blockwise import BlockConfig

    blocks = []
    offset = 0  # contiguous offset that accounts for padding

    for i, spec in enumerate(hybrid.specs):
        num_coords = spec.flat_end - spec.flat_start
        padded_coords = num_coords + (-num_coords % particle_dim)
        num_particles = padded_coords // particle_dim

        # Use a contiguous offset, not spec.flat_start, to avoid gaps or overlaps.
        blocks.append(
            BlockConfig(
                name=spec.entry_key,
                leaf_indices=(i,),
                flat_start=offset,
                flat_end=offset + padded_coords,
                num_particles=num_particles,
                particle_dim=particle_dim,
            )
        )
        offset += padded_coords

    # `offset` (sum of padded block dims) may exceed hybrid.subspace_dim; callers must
    # allocate vectors of size `offset`, not hybrid.subspace_dim.
    return blocks
