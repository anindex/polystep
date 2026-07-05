"""Subspace compression for neural network parameters.

Reduces the OT problem from ``N`` parameters to a ``K``-dimensional
subspace (``K << N``) and projects back to full parameter space, dropping
the OT solver's memory from O(N^2) to O(K^2).

Two factorisations are provided:

- :class:`LowRankSubspace` - ``delta_W = B @ A``, bilinear in the subspace
  coords. Near zero this bilinearity flattens the cost landscape and
  often gives uniform OT transport; useful for scalability diagnostics.
- :class:`LinearSubspace` - ``delta_W = P @ coords`` with a fixed random
  projection ``P``. Linear in the subspace coords, so probing
  ``coords + alpha * direction`` produces a proportional cost change.
  Recommended for training.

All factors/coords are packed into a single flat subspace vector with
per-layer offsets. That vector *is* the particle array for the OT solver
(reshaped to ``(num_particles, particle_dim)``).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch


def _stable_entry_seed(*parts: object) -> int:
    """Hash a tuple of seed components to a 31-bit integer, deterministically
    across processes (unlike Python's salted ``hash``).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.adler32(key)) & 0x7FFFFFFF


if TYPE_CHECKING:
    from .transform import ParamLayout


@dataclass(frozen=True)
class FactorSpec:
    """Describes the low-rank decomposition of a single parameter entry.

    Attributes:
        entry_key: ParamLayout entry key (e.g., "fc1.weight").
        original_shape: Original parameter shape.
        is_lowrank: True for 2D+ params (B@A compression), False for 1D (full).
        b_shape: Shape of B factor (d_out, r), or () for 1D params.
        a_shape: Shape of A factor (r, d_in), or () for 1D params.
        flat_start: Start offset into the flat subspace vector.
        flat_end: End offset into the flat subspace vector.
    """

    entry_key: str
    original_shape: Tuple[int, ...]
    is_lowrank: bool
    b_shape: Tuple[int, ...]
    a_shape: Tuple[int, ...]
    flat_start: int
    flat_end: int


@dataclass
class LowRankSubspace:
    """Low-rank subspace compression for neural network parameters.

    Packs per-layer B and A factors into a single flat subspace vector.
    Supports both fixed-rank and auto-rank (per-layer) modes.

    Attributes:
        specs: Per-parameter decomposition specs.
        rank: Fixed rank used (0 for auto mode with per-layer ranks).
        subspace_dim: Total flat subspace vector size (sum of all factor elements).
        compression_ratio: subspace_dim / total_params for diagnostics.
    """

    specs: Tuple[FactorSpec, ...]
    rank: int
    subspace_dim: int
    compression_ratio: float

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_layout(cls, layout: ParamLayout, rank: int) -> LowRankSubspace:
        """Create a LowRankSubspace with fixed rank from a ParamLayout.

        For each 2D+ parameter entry, computes effective_rank = min(rank, d_in, d_out)
        and creates B=(d_out, effective_rank), A=(effective_rank, d_in) factors.
        1D parameters are stored as full perturbations (no compression).

        Args:
            layout: ParamLayout describing the model's parameter structure.
            rank: Fixed rank for all low-rank decompositions.

        Returns:
            LowRankSubspace with per-layer FactorSpecs and flat offsets.
        """
        specs = []
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                # 2D+ param: low-rank B@A decomposition
                # Conv: (C_out, C_in, H, W) -> d_out=C_out, d_in=C_in*H*W
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                effective_rank = min(rank, d_in, d_out)
                b_shape = (d_out, effective_rank)
                a_shape = (effective_rank, d_in)
                num_elements = d_out * effective_rank + effective_rank * d_in
                specs.append(
                    FactorSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_lowrank=True,
                        b_shape=b_shape,
                        a_shape=a_shape,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements
            else:
                # 1D param (bias, LayerNorm): full perturbation
                num_elements = entry.numel
                specs.append(
                    FactorSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_lowrank=False,
                        b_shape=(),
                        a_shape=(),
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        return cls(
            specs=tuple(specs),
            rank=rank,
            subspace_dim=offset,
            compression_ratio=compression,
        )

    @classmethod
    def auto_from_layout(
        cls,
        layout: ParamLayout,
        compression_ratio: int = 16,
        min_rank: int = 4,
        max_rank: int = 64,
    ) -> LowRankSubspace:
        """Create a LowRankSubspace with auto per-layer rank selection.

        Computes per-layer rank proportional to min(d_in, d_out) without
        user tuning: rank = max(min_rank, min(max_rank, min(d_in, d_out) // compression_ratio)).

        Args:
            layout: ParamLayout describing the model's parameter structure.
            compression_ratio: Divisor for rank computation (default 16).
            min_rank: Minimum rank per layer (default 4).
            max_rank: Maximum rank per layer (default 64).

        Returns:
            LowRankSubspace with per-layer auto-selected ranks (rank=0).
        """
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
                b_shape = (d_out, effective_rank)
                a_shape = (effective_rank, d_in)
                num_elements = d_out * effective_rank + effective_rank * d_in
                specs.append(
                    FactorSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_lowrank=True,
                        b_shape=b_shape,
                        a_shape=a_shape,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements
            else:
                num_elements = entry.numel
                specs.append(
                    FactorSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_lowrank=False,
                        b_shape=(),
                        a_shape=(),
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        return cls(
            specs=tuple(specs),
            rank=0,  # 0 signals auto mode
            subspace_dim=offset,
            compression_ratio=compression,
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def apply_perturbation(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct full state_dict from base params + flat subspace vector.

        For low-rank entries, slices out B and A factors, computes delta = B @ A,
        reshapes to original shape, and adds to base. For 1D entries, adds chunk
        directly.

        Args:
            base_sd: Base state_dict with original parameter values.
            flat_subspace: 1D tensor of shape (subspace_dim,).

        Returns:
            New state_dict with perturbed parameters.
        """
        result = {}
        for spec in self.specs:
            chunk = flat_subspace[spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_lowrank:
                b_size = spec.b_shape[0] * spec.b_shape[1]
                B = chunk[:b_size].reshape(spec.b_shape)
                A = chunk[b_size:].reshape(spec.a_shape)
                delta = (B @ A).reshape(spec.original_shape)
                result[spec.entry_key] = base + delta
            else:
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape)
        return result

    def reconstruct_batch(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points using torch.bmm.

        Args:
            base_sd: Base state_dict with original parameter values.
            flat_subspace_batch: 2D tensor of shape (N, subspace_dim).

        Returns:
            Dict ``{key: (N, *original_shape)}`` with batched perturbed params.
        """
        N = flat_subspace_batch.shape[0]
        result = {}
        for spec in self.specs:
            chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_lowrank:
                b_size = spec.b_shape[0] * spec.b_shape[1]
                B = chunk[:, :b_size].reshape(N, *spec.b_shape)
                A = chunk[:, b_size:].reshape(N, *spec.a_shape)
                delta_2d = torch.bmm(B, A)  # (N, d_out, d_in)
                delta = delta_2d.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base.unsqueeze(0) + delta
            else:
                delta = chunk.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base.unsqueeze(0) + delta
        return result

    def absorb(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fold subspace perturbation into base weights and zero the subspace.

        Args:
            base_sd: Base state_dict to absorb perturbation into.
            flat_subspace: Current subspace vector of shape (subspace_dim,).

        Returns:
            Tuple of (new_base_sd, zeroed_subspace_vector).
        """
        new_sd = self.apply_perturbation(base_sd, flat_subspace)
        return new_sd, torch.zeros_like(flat_subspace)


# ======================================================================
# LinearSubspace: random projection (linear mapping)
# ======================================================================


@dataclass(frozen=True)
class ProjectionSpec:
    """Describes the random projection mapping for a single parameter entry.

    Attributes:
        entry_key: ParamLayout entry key (e.g., "fc1.weight").
        original_shape: Original parameter shape.
        is_projected: True for 2D+ params (random projection), False for 1D (full).
        num_params: Total elements in this parameter (d_out * d_in).
        num_coords: Number of subspace coordinates allocated.
        flat_start: Start offset into the flat subspace vector.
        flat_end: End offset into the flat subspace vector.
    """

    entry_key: str
    original_shape: Tuple[int, ...]
    is_projected: bool
    num_params: int
    num_coords: int
    flat_start: int
    flat_end: int


class LinearSubspace:
    """Linear subspace compression via fixed random projection matrices.

    For each 2D+ parameter, weight perturbation is:
        delta_W = (P @ coords).reshape(original_shape)
    where P is a fixed random projection matrix of shape (num_params, num_coords).

    Key property: This is LINEAR in coords. When the OT solver probes at
    coords + alpha * direction, the weight change is P @ (coords + alpha * direction)
    = P @ coords + alpha * (P @ direction), which is proportional. The OT solver
    sees meaningful cost differences, enabling convergence.

    Unlike LowRankSubspace (B@A, bilinear), this avoids the near-zero gradient
    problem that causes uniform OT transport.

    Note on radius scaling: LinearSubspace requires larger ``step_radius``
    (typically 30x the default) because the random projection matrix has
    1/sqrt(N) scaling, which dilutes the perturbation magnitude in full
    parameter space.

    Example::

        from polystep import LinearSubspace, PolyStepOptimizer

        subspace = LinearSubspace.from_layout(optimizer.layout, rank=8)
        optimizer = PolyStepOptimizer(
            model, subspace=subspace, step_radius=4.5,
        )

    See Also:
        ``LowRankSubspace`` for bilinear B@A compression (scalability only).
        ``PolyStepOptimizer`` for subspace integration.

    Attributes:
        specs: Per-parameter projection specs.
        subspace_dim: Total flat subspace vector size.
        compression_ratio: subspace_dim / total_params for diagnostics.
        seed: Seed for deterministic projection matrix generation.
    """

    def __init__(
        self,
        specs: Tuple[ProjectionSpec, ...],
        subspace_dim: int,
        compression_ratio: float,
        seed: int = 0,
    ) -> None:
        self.specs = specs
        self.subspace_dim = subspace_dim
        self.compression_ratio = compression_ratio
        self.seed = seed
        self._projections: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_layout(
        cls,
        layout: ParamLayout,
        rank: int,
        seed: int = 0,
        max_subspace_dim: Optional[int] = None,
    ) -> LinearSubspace:
        """Create a LinearSubspace with fixed rank from a ParamLayout.

        Uses the SAME formula as LowRankSubspace for num_coords per layer:
        num_coords = d_out * effective_rank + effective_rank * d_in.
        This makes subspace_dim identical, so switching between the two
        is a drop-in replacement.

        Args:
            layout: ParamLayout describing the model's parameter structure.
            rank: Rank parameter (controls compression level).
            seed: Seed for deterministic projection matrices.
            max_subspace_dim: Optional cap on total subspace dimension.

        Returns:
            LinearSubspace with per-layer ProjectionSpecs and flat offsets.
        """
        specs = []
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                effective_rank = min(rank, d_in, d_out)
                # Same formula as LowRankSubspace for drop-in compatibility
                num_coords = d_out * effective_rank + effective_rank * d_in
                num_params = math.prod(shape)
                specs.append(
                    ProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_projected=True,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                    )
                )
                offset += num_coords
            else:
                num_elements = entry.numel
                specs.append(
                    ProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_projected=False,
                        num_params=num_elements,
                        num_coords=num_elements,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements

        # Apply budget cap if specified
        from .hybrid_subspace import _scale_specs_to_budget

        specs, offset = _scale_specs_to_budget(specs, max_subspace_dim, offset)

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        inst = cls(
            specs=tuple(specs),
            subspace_dim=offset,
            compression_ratio=compression,
            seed=seed,
        )
        inst._max_subspace_dim = max_subspace_dim
        return inst

    @classmethod
    def auto_from_layout(
        cls,
        layout: ParamLayout,
        compression_ratio: int = 16,
        min_rank: int = 4,
        max_rank: int = 64,
        seed: int = 0,
        max_subspace_dim: Optional[int] = None,
    ) -> LinearSubspace:
        """Create a LinearSubspace with auto per-layer rank selection.

        Same auto-rank logic as LowRankSubspace: per-layer rank proportional
        to min(d_in, d_out) without user tuning.

        Args:
            layout: ParamLayout describing the model's parameter structure.
            compression_ratio: Divisor for rank computation (default 16).
            min_rank: Minimum rank per layer (default 4).
            max_rank: Maximum rank per layer (default 64).
            seed: Seed for deterministic projection matrices.

        Returns:
            LinearSubspace with per-layer auto-selected coordinate counts.
        """
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
                num_coords = d_out * effective_rank + effective_rank * d_in
                num_params = math.prod(shape)
                specs.append(
                    ProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_projected=True,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                    )
                )
                offset += num_coords
            else:
                num_elements = entry.numel
                specs.append(
                    ProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        is_projected=False,
                        num_params=num_elements,
                        num_coords=num_elements,
                        flat_start=offset,
                        flat_end=offset + num_elements,
                    )
                )
                offset += num_elements

        # Apply budget cap if specified
        from .hybrid_subspace import _scale_specs_to_budget

        specs, offset = _scale_specs_to_budget(specs, max_subspace_dim, offset)

        total_params = layout.total_params
        compression = offset / total_params if total_params > 0 else 0.0

        inst = cls(
            specs=tuple(specs),
            subspace_dim=offset,
            compression_ratio=compression,
            seed=seed,
        )
        inst._max_subspace_dim = max_subspace_dim
        return inst

    # ------------------------------------------------------------------
    # Projection matrix generation
    # ------------------------------------------------------------------

    def _get_projection(
        self,
        spec: ProjectionSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or generate the projection matrix for a parameter entry.

        Generates P of shape (num_params, num_coords) deterministically from
        self.seed and spec.entry_key. Columns are scaled by 1/sqrt(num_coords)
        for unit-variance output (standard random projection scaling).

        Args:
            spec: ProjectionSpec for this parameter.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Projection matrix P of shape (num_params, num_coords).
        """
        cache_key = (spec.entry_key, str(device), str(dtype))
        if cache_key in self._projections:
            cached = self._projections[cache_key]
            if cached.device == device and cached.dtype == dtype:
                return cached

        # Deterministic seed from global seed + entry key
        entry_seed = _stable_entry_seed(self.seed, spec.entry_key)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(entry_seed)

        # Generate on CPU then move (Generator doesn't support CUDA)
        P = torch.randn(
            spec.num_params,
            spec.num_coords,
            generator=gen,
            dtype=dtype,
            device="cpu",
        )
        # Scale for unit-variance output
        P = P * (1.0 / math.sqrt(spec.num_coords))
        P = P.to(device=device)

        self._projections[cache_key] = P
        return P

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def apply_perturbation(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct full state_dict from base params + flat subspace vector.

        For projected entries, slices coords, matrix-multiplies with projection P,
        reshapes to original shape, and adds to base. For 1D entries, adds directly.

        Args:
            base_sd: Base state_dict with original parameter values.
            flat_subspace: 1D tensor of shape (subspace_dim,).

        Returns:
            New state_dict with perturbed parameters.
        """
        result = {}
        for spec in self.specs:
            chunk = flat_subspace[spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_projected:
                P = self._get_projection(spec, base.device, base.dtype)
                delta = (P @ chunk).reshape(spec.original_shape)
                result[spec.entry_key] = base + delta
            else:
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape)
        return result

    def reconstruct_batch(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points.

        For projected entries: coords @ P.T gives (N, num_params), then reshape.
        This avoids bmm -- simple matmul.

        Args:
            base_sd: Base state_dict with original parameter values.
            flat_subspace_batch: 2D tensor of shape (N, subspace_dim).

        Returns:
            Dict ``{key: (N, *original_shape)}`` with batched perturbed params.
        """
        N = flat_subspace_batch.shape[0]
        result = {}
        for spec in self.specs:
            chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_projected:
                P = self._get_projection(spec, base.device, base.dtype)
                # chunk: (N, num_coords), P: (num_params, num_coords)
                # delta_flat = chunk @ P.T -> (N, num_params)
                delta_flat = chunk @ P.t()
                delta = delta_flat.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base.unsqueeze(0) + delta
            else:
                delta = chunk.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base.unsqueeze(0) + delta
        return result

    def absorb(
        self,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fold subspace perturbation into base weights and zero the subspace.

        Args:
            base_sd: Base state_dict to absorb perturbation into.
            flat_subspace: Current subspace vector of shape (subspace_dim,).

        Returns:
            Tuple of (new_base_sd, zeroed_subspace_vector).
        """
        new_sd = self.apply_perturbation(base_sd, flat_subspace)
        return new_sd, torch.zeros_like(flat_subspace)
