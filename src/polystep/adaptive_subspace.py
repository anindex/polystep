"""Rotating orthogonal projection subspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from .solvers._shared import decomposition_dtype, thin_qr
from .subspace import ProjectedAbsorbMixin, SvdRatioMixin, absorb_due
import torch.nn as nn

if TYPE_CHECKING:
    from .transform import ParamLayout


def _spawn_cpu_generator(generator: Optional[torch.Generator]) -> Optional[torch.Generator]:
    """Mirror a non-CPU generator onto CPU, seeded from a fresh draw."""
    if generator is None or generator.device.type == "cpu":
        return generator
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)
    return cpu_gen


def _draw_basis_gaussian(rows, cols, device, dtype, generator):
    """Draw a ``(rows, cols)`` Gaussian for basis construction."""
    target = torch.device(device)
    if generator is None:
        return torch.randn(rows, cols, device=target, dtype=dtype)
    Z = torch.randn(rows, cols, generator=_spawn_cpu_generator(generator), device="cpu", dtype=dtype)
    return Z.to(device=target) if target.type != "cpu" else Z


@dataclass(frozen=True)
class EntrySpec:
    """The mapping of one parameter entry into the flat vector."""

    entry_key: str
    original_shape: Tuple[int, ...]
    num_params: int
    flat_start: int
    flat_end: int


@dataclass
class AdaptiveSubspace(ProjectedAbsorbMixin, SvdRatioMixin):
    """Adaptive subspace with a single rotating orthogonal projection."""

    full_dim: int
    subspace_dim: int
    compression_ratio: float = 0.0
    rotation_mode: str = "displacement"
    svd_ratio_init: float = 0.0
    svd_ratio_final: float = 0.5
    displacement_history_size: int = 5
    absorb_mode: str = "stagnation"
    absorb_patience: int = 20
    absorb_interval: int = 0
    rotation_interval: int = 1
    _entry_specs: Tuple[EntrySpec, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.subspace_dim <= self.full_dim:
            raise ValueError(
                f"subspace_dim must be in (0, full_dim={self.full_dim}], got {self.subspace_dim}. "
                "A reduced QR cannot return more orthonormal columns than rows, so the "
                "projection would come back narrower than the coordinates allocated for it."
            )
        if self.compression_ratio == 0.0 and self.full_dim > 0:
            object.__setattr__(self, "compression_ratio", self.subspace_dim / self.full_dim)

    def init_projection(
        self,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Build the initial random orthogonal projection ``(full_dim, subspace_dim)``."""
        projection_dtype = dtype if dtype is not None else torch.float32
        projection_device = device if device is not None else "cpu"
        return self._make_orthogonal_basis(
            self.full_dim,
            self.subspace_dim,
            device=projection_device,
            dtype=projection_dtype,
            generator=generator,
        )

    @staticmethod
    def _make_orthogonal_basis(
        rows: int,
        cols: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Create an orthogonal matrix via QR with a fixed sign."""
        # Half precision has no QR kernel; decomposition_dtype upcasts it. Keep QR on GPU for CUDA.
        target_device = torch.device(device)
        qr_device = target_device if target_device.type == "cuda" else torch.device("cpu")
        compute_dtype = decomposition_dtype(dtype)
        Z = _draw_basis_gaussian(rows, cols, qr_device, compute_dtype, generator)
        P, R = thin_qr(Z)
        # Positive diagonal in R removes the QR sign ambiguity.
        d = torch.sign(torch.diagonal(R))
        d[d == 0] = 1.0
        P = (P * d)[:, :cols]  # QR may return a full square Q; slice back to cols
        if dtype != compute_dtype:
            P = P.to(dtype=dtype)
        if P.device != target_device:
            P = P.to(device=target_device)
        return P

    @torch.inference_mode()
    def rotate(
        self,
        projection: torch.Tensor,
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        history_is_full: bool = False,
    ) -> torch.Tensor:
        """Rotate the basis by mode: random redraw or displacement-SVD."""
        device = projection.device
        dtype = projection.dtype

        use_random = (
            self.rotation_mode == "random" or displacement_history is None or displacement_history.shape[0] == 0
        )

        if not use_random:
            if not torch.isfinite(displacement_history).all():
                use_random = True
            else:
                disp_norm_sq = (displacement_history * displacement_history).sum().item()
                if disp_norm_sq < 1e-20:
                    use_random = True

        if use_random:
            return self._rotate_random(device, dtype, generator)
        else:
            svd_ratio = self.get_svd_ratio(step, total_steps)
            return self._rotate_displacement(
                projection,
                displacement_history,
                svd_ratio,
                device,
                dtype,
                generator,
                history_is_full,
            )

    @torch.inference_mode()
    def _rotate_random(
        self,
        device: str | torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw an entirely new QR-orthogonalized basis."""
        return self._make_orthogonal_basis(
            self.full_dim,
            self.subspace_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    @torch.inference_mode()
    def _rotate_displacement(
        self,
        projection: torch.Tensor,
        displacement_history: torch.Tensor,
        svd_ratio: float,
        device: str | torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        history_is_full: bool = False,
    ) -> torch.Tensor:
        """Rotate the basis: keep top SVD directions, fill the rest randomly."""
        # svd_ratio 0 means a fresh random basis.
        if svd_ratio <= 0.0:
            return self._rotate_random(device, dtype, generator)

        k_svd = max(1, int(svd_ratio * self.subspace_dim))
        k_random = self.subspace_dim - k_svd

        # A full-space history keeps the basis each row was measured in; the caller states the frame.
        if history_is_full:
            D_full = displacement_history.T
        else:
            D_full = projection @ displacement_history.T

        # Fall back to random on non-finite history.
        if not torch.isfinite(D_full).all():
            return self._rotate_random(device, dtype, generator)

        compute_dtype = decomposition_dtype(dtype)
        D_full = D_full.to(compute_dtype)

        # The history is short, so the full SVD costs about the same.
        U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
        k_svd = min(k_svd, U.shape[1])
        U_top = U[:, :k_svd]
        k_random = self.subspace_dim - k_svd

        Z_random = _draw_basis_gaussian(self.full_dim, k_random, device, compute_dtype, generator)
        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        combined = torch.cat([U_top, Z_random], dim=1)
        P_new, R = thin_qr(combined)
        d = torch.sign(torch.diagonal(R))
        d[d == 0] = 1.0
        P_new = P_new * d
        P_new = P_new[:, : self.subspace_dim]
        if P_new.dtype != dtype:
            P_new = P_new.to(dtype)

        if P_new.device != device:
            P_new = P_new.to(device=device)

        return P_new

    def apply_perturbation(
        self,
        projection,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct a state_dict from the base weights and one subspace vector."""
        from .projection import SparseRandomProjection

        if isinstance(projection, SparseRandomProjection):
            delta_flat = projection.project(flat_subspace)  # (full_dim,)
        else:
            delta_flat = projection @ flat_subspace  # (full_dim,)
        result: Dict[str, torch.Tensor] = {}
        for spec in self._entry_specs:
            delta_chunk = delta_flat[spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            result[spec.entry_key] = base + delta_chunk.reshape(spec.original_shape)
        return result

    def reconstruct_batch(
        self,
        projection,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points; entries alias one buffer, treat as read-only."""
        from .projection import SparseRandomProjection

        if isinstance(projection, SparseRandomProjection):
            # project() returns a transposed result; make it contiguous.
            delta_batch = projection.project(flat_subspace_batch).contiguous()  # (N, full_dim)
        else:
            delta_batch = flat_subspace_batch @ projection.T  # (N, full_dim)
        result: Dict[str, torch.Tensor] = {}
        for spec in self._entry_specs:
            delta_chunk = delta_batch[:, spec.flat_start : spec.flat_end]
            delta_chunk.add_(base_sd[spec.entry_key].reshape(1, -1))
            # unflatten keeps this a view; reshape would copy the whole (N, num_params).
            result[spec.entry_key] = delta_chunk.unflatten(1, spec.original_shape)
        return result

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Whether to fold the perturbation into the base weights this step."""
        return absorb_due(
            self.absorb_mode,
            self.absorb_patience,
            self.absorb_interval,
            stagnation_count,
            iteration,
        )

    @classmethod
    def auto_from_params(
        cls,
        model: nn.Module,
        compression_target: float = 0.05,
        min_rank: int = 64,
        max_rank: int = 4096,
        **kwargs,
    ) -> AdaptiveSubspace:
        """Create an AdaptiveSubspace from an nn.Module with auto rank."""
        from .transform import ParamLayout

        layout = ParamLayout.from_module(model)
        full_dim = layout.total_params
        subspace_dim = max(min_rank, min(max_rank, int(full_dim * compression_target)))
        subspace_dim = min(subspace_dim, full_dim)

        entry_specs = cls._build_entry_specs(layout)

        return cls(
            full_dim=full_dim,
            subspace_dim=subspace_dim,
            _entry_specs=tuple(entry_specs),
            **kwargs,
        )

    @classmethod
    def from_layout(
        cls,
        layout: "ParamLayout",
        rank: int,
        **kwargs,
    ) -> AdaptiveSubspace:
        """Create an AdaptiveSubspace from a ParamLayout with explicit rank."""
        full_dim = layout.total_params
        subspace_dim = min(rank, full_dim)

        entry_specs = cls._build_entry_specs(layout)

        return cls(
            full_dim=full_dim,
            subspace_dim=subspace_dim,
            _entry_specs=tuple(entry_specs),
            **kwargs,
        )

    @staticmethod
    def _build_entry_specs(layout: "ParamLayout") -> List[EntrySpec]:
        """Build the EntrySpec list from a ParamLayout."""
        specs: List[EntrySpec] = []
        for entry in layout.entries:
            specs.append(
                EntrySpec(
                    entry_key=entry.key,
                    original_shape=entry.shape,
                    num_params=entry.numel,
                    flat_start=entry.offset,
                    flat_end=entry.offset + entry.numel,
                )
            )
        return specs
