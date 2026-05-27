"""HybridSubspace: per-layer projections with coordinated rotation.

Combines ``LinearSubspace``'s per-layer structure (better per-step coverage)
with ``AdaptiveSubspace``'s synchronized rotation coordination.

Key design principles:

1. **Per-layer projections**: Like LinearSubspace, each parameter entry has
   its own projection matrix P_layer of shape (num_params, num_coords). This
   gives better per-step coverage than AdaptiveSubspace's global projection.

2. **Synchronized rotation**: Like AdaptiveSubspace, all layer projections
   rotate together on the same schedule (every N steps). This maintains
   global coordination.

3. **1/sqrt(N) scaling**: Projections use LinearSubspace's scaling convention
   (P = randn * 1/sqrt(num_coords)) for unit-variance output, NOT QR-orthonormal
   columns. This matches proven LinearSubspace behavior.

4. **Displacement-biased rotation**: In 'displacement' mode, rotation is
   biased toward productive directions discovered during optimization, but
   applied per-layer using that layer's portion of the displacement history.

The key insight: AdaptiveSubspace's global projection covers ~0.25% of
parameters per step vs LinearSubspace's ~4.3%. HybridSubspace addresses
this by using per-layer projections while keeping AdaptiveSubspace's
rotation coordination.

Example::

    from polystep import HybridSubspace, ParamLayout
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(784, 128), nn.Linear(128, 10))
    layout = ParamLayout.from_module(model)

    # Create with auto rank selection
    hybrid = HybridSubspace.auto_from_layout(layout)

    # Initialize per-layer projections
    projections = hybrid.init_projections(torch.device('cpu'), torch.float32)

    # Each iteration: rotate all projections together
    projections = hybrid.rotate_all(projections, step=i, total_steps=100)

See Also:
    ``LinearSubspace`` for fixed per-layer projections (no rotation).
    ``AdaptiveSubspace`` for single global rotating projection.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from .projection.sparse import SparseRandomProjection

if TYPE_CHECKING:
    import torch.nn as nn
    from .blockwise import BlockConfig
    from .transform import ParamLayout


@dataclass(frozen=True)
class LayerProjectionSpec:
    """Describes the projection mapping for a single parameter entry.

    Each layer has its own projection matrix P_layer of shape
    (num_params, num_coords) that maps subspace coordinates to parameter
    perturbations.

    Attributes:
        entry_key: state_dict key (e.g., "fc1.weight").
        original_shape: Original parameter shape.
        num_params: Total elements in this parameter (d_out * d_in for 2D).
        num_coords: Number of subspace coordinates allocated to this layer.
        flat_start: Start offset into the global subspace vector.
        flat_end: End offset into the global subspace vector.
        is_projected: True for 2D+ params (uses projection), False for 1D
            (biases pass through directly without projection).
    """

    entry_key: str
    original_shape: Tuple[int, ...]
    num_params: int
    num_coords: int
    flat_start: int
    flat_end: int
    is_projected: bool = True


def _scale_specs_to_budget(
    specs: list,
    max_dim: "Optional[int]",
    current_dim: int,
) -> "Tuple[list, int]":
    """Proportionally scale projected specs to fit within max_dim budget.

    Only scales specs that are already projected (2D+ weight matrices).
    Unprojected 1D params (biases, LayerNorm) are kept at full size since
    they are tiny and projecting them adds overhead without benefit.

    Returns (new_specs, new_total_dim). No-op when max_dim is None or
    current_dim already fits.
    """
    if max_dim is None or current_dim <= max_dim:
        return specs, current_dim

    # Compute budget available for projected specs after reserving 1D params
    unprojected_dim = sum(s.num_coords for s in specs if not s.is_projected)
    projected_dim = current_dim - unprojected_dim
    target_projected = max(1, max_dim - unprojected_dim)

    if projected_dim <= target_projected:
        return specs, current_dim

    scale = target_projected / projected_dim
    new_specs: list = []
    new_offset = 0
    for spec in specs:
        if spec.is_projected:
            new_coords = max(1, round(spec.num_coords * scale))
            new_coords = min(new_coords, spec.num_params)
            new_specs.append(replace(spec,
                num_coords=new_coords,
                flat_start=new_offset,
                flat_end=new_offset + new_coords,
                is_projected=True,
            ))
        else:
            # Keep 1D params at full size, no projection
            new_specs.append(replace(spec,
                flat_start=new_offset,
                flat_end=new_offset + spec.num_coords,
            ))
        new_offset += new_specs[-1].num_coords
    return new_specs, new_offset


@dataclass
class HybridSubspace:
    """Hybrid subspace compression with per-layer projections and global rotation.

    Combines LinearSubspace's per-layer structure with AdaptiveSubspace's
    synchronized rotation coordination. Each layer has its own projection
    matrix, but all projections rotate together on the same schedule.

    Two rotation modes are supported:

    - ``'random'``: Regenerates all layer projections with new seeds.
      Simple but effective for exploration.

    - ``'displacement'``: Uses SVD of recent displacement history to retain
      productive directions per layer. The fraction of SVD-derived directions
      increases linearly from ``svd_ratio_init`` to ``svd_ratio_final``.

    Example::

        from polystep import HybridSubspace
        from polystep.transform import ParamLayout
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(784, 128), nn.Linear(128, 10))
        layout = ParamLayout.from_module(model)

        # Auto rank selection based on compression ratio
        hybrid = HybridSubspace.auto_from_layout(layout, min_rank=4, max_rank=64)

        # Initialize projections
        projections = hybrid.init_projections(torch.device('cpu'), torch.float32)

        # Rotate all projections together
        projections = hybrid.rotate_all(projections, step=1, total_steps=100)

    Attributes:
        specs: Per-layer projection specifications.
        subspace_dim: Total subspace dimension (sum of all layer num_coords).
        compression_ratio: subspace_dim / total_params.
        seed: Base seed for deterministic projection generation.
        rotation_mode: 'random' or 'displacement' (default 'displacement').
        rotation_interval: Rotate every N steps (default 1).
        svd_ratio_init: Starting SVD ratio for displacement mode (default 0.0).
        svd_ratio_final: Ending SVD ratio for displacement mode (default 0.5).
        displacement_history_size: Rolling window for displacement history (default 5).
        absorb_mode: 'stagnation' or 'periodic' (default 'stagnation').
        absorb_patience: Steps of stagnation before absorb (default 20).
        absorb_interval: Periodic absorb interval; 0 = disabled (default 0).
    """

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
    projection_mode: str = "random"  # 'random' (dense Gaussian) or 'structured' (block-diagonal)

    # Track total params for compression ratio calculation
    _total_params: int = 0
    # Budget control settings (preserved across rank transitions)
    _max_subspace_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.rotation_interval != 0:
            warnings.warn(
                "HybridSubspace works best with rotation_interval=0. "
                "Non-zero values cause dual potential resets that degrade accuracy.",
                stacklevel=2,
            )
        if self.compression_ratio == 0.0 and self._total_params > 0:
            object.__setattr__(
                self, "compression_ratio", self.subspace_dim / self._total_params
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_layout(
        cls,
        layout: "ParamLayout",
        rank: int,
        seed: int = 0,
        max_subspace_dim: Optional[int] = None,
        **kwargs,
    ) -> "HybridSubspace":
        """Create a HybridSubspace with fixed rank from a ParamLayout.

        Uses the SAME formula as LinearSubspace for num_coords per layer:
        num_coords = d_out * effective_rank + effective_rank * d_in.
        This makes subspace_dim identical, so switching from LinearSubspace
        is a drop-in replacement.

        Args:
            layout: ParamLayout describing the model's parameter structure.
            rank: Rank parameter (controls compression level).
            seed: Seed for deterministic projection matrices.
            max_subspace_dim: Optional cap on total subspace dimension.
                When set, all layers' num_coords are proportionally scaled
                down to fit within this budget. Default None (no cap).
            **kwargs: Additional arguments (rotation_mode, svd_ratio_*, etc.).

        Returns:
            HybridSubspace with per-layer LayerProjectionSpecs.

        Example::

            layout = ParamLayout.from_module(model)
            hybrid = HybridSubspace.from_layout(layout, rank=8)
        """
        specs = []
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                effective_rank = min(rank, d_in, d_out)
                # Same formula as LinearSubspace for drop-in compatibility
                num_coords = d_out * effective_rank + effective_rank * d_in
                num_params = math.prod(shape)
                specs.append(LayerProjectionSpec(
                    entry_key=entry.key,
                    original_shape=shape,
                    num_params=num_params,
                    num_coords=num_coords,
                    flat_start=offset,
                    flat_end=offset + num_coords,
                    is_projected=True,
                ))
                offset += num_coords
            else:
                # 1D param (bias, LayerNorm): full perturbation, no projection
                num_elements = entry.numel
                specs.append(LayerProjectionSpec(
                    entry_key=entry.key,
                    original_shape=shape,
                    num_params=num_elements,
                    num_coords=num_elements,
                    flat_start=offset,
                    flat_end=offset + num_elements,
                    is_projected=False,
                ))
                offset += num_elements

        # Apply budget cap if specified
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
        """Create a HybridSubspace with auto per-layer rank selection.

        Same auto-rank logic as LinearSubspace: per-layer rank proportional
        to min(d_in, d_out) without user tuning.

        Args:
            layout: ParamLayout describing the model's parameter structure.
            compression_ratio: Divisor for rank computation (default 16).
            min_rank: Minimum rank per layer (default 4).
            max_rank: Maximum rank per layer (default 64).
            seed: Seed for deterministic projection matrices.
            max_subspace_dim: Optional cap on total subspace dimension.
            **kwargs: Additional arguments (rotation_mode, svd_ratio_*, etc.).

        Returns:
            HybridSubspace with per-layer auto-selected ranks.

        Example::

            layout = ParamLayout.from_module(model)
            hybrid = HybridSubspace.auto_from_layout(layout, min_rank=4, max_rank=64)
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
                specs.append(LayerProjectionSpec(
                    entry_key=entry.key,
                    original_shape=shape,
                    num_params=num_params,
                    num_coords=num_coords,
                    flat_start=offset,
                    flat_end=offset + num_coords,
                    is_projected=True,
                ))
                offset += num_coords
            else:
                num_elements = entry.numel
                specs.append(LayerProjectionSpec(
                    entry_key=entry.key,
                    original_shape=shape,
                    num_params=num_elements,
                    num_coords=num_elements,
                    flat_start=offset,
                    flat_end=offset + num_elements,
                    is_projected=False,
                ))
                offset += num_elements

        # Apply budget cap if specified
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
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Projection management
    # ------------------------------------------------------------------

    def init_projections(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Union[torch.Tensor, SparseRandomProjection]]:
        """Generate per-layer projection matrices.

        Creates a projection matrix P_layer of shape (num_params, num_coords)
        for each layer using deterministic seeded random generation. Each
        projection is scaled by 1/sqrt(num_coords) for unit-variance output,
        matching LinearSubspace's scaling convention.

        Layers whose dense projection would exceed ``sparse_threshold_bytes``
        automatically use ``SparseRandomProjection`` instead, reducing memory
        by ~100-1000x for large layers.

        Args:
            device: Target device for projection matrices.
            dtype: Target dtype for projection matrices.

        Returns:
            Dict mapping entry_key to projection matrix (dense Tensor or
            SparseRandomProjection for large layers).

        Example::

            projections = hybrid.init_projections(torch.device('cpu'), torch.float32)
            for key, P in projections.items():
                print(f'{key}: {type(P).__name__}')
        """
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}
        for spec in self.specs:
            dense_bytes = spec.num_params * spec.num_coords * 4  # float32
            if spec.is_projected and dense_bytes > self.sparse_threshold_bytes:
                entry_seed = hash((self.seed, spec.entry_key, 0)) % (2**31)
                projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                P = self._get_projection(spec, device, dtype, step=0)
                projections[spec.entry_key] = P
        return projections

    def _get_projection(
        self,
        spec: LayerProjectionSpec,
        device: torch.device,
        dtype: torch.dtype,
        step: int = 0,
    ) -> torch.Tensor:
        """Generate the projection matrix for a parameter entry.

        Generates P of shape (num_params, num_coords) deterministically from
        self.seed, spec.entry_key, and step. Dispatches on ``projection_mode``:

        - ``'random'``: Dense Gaussian projection with 1/sqrt(num_coords) scaling.
        - ``'structured'``: Block-diagonal projection preserving weight matrix
          row structure (SubZero-inspired).

        For 1D parameters (biases), returns identity-like mapping since
        num_params == num_coords (regardless of projection_mode).

        Args:
            spec: LayerProjectionSpec for this parameter.
            device: Target device.
            dtype: Target dtype.
            step: Current step (used for rotation seed).

        Returns:
            Projection matrix P of shape (num_params, num_coords).
        """
        # 1D params: identity-like projection
        if spec.num_params == spec.num_coords:
            return torch.eye(
                spec.num_params, dtype=dtype, device=device,
            )

        if self.projection_mode == 'structured':
            return self._init_structured_projection(spec, device, dtype, step)

        # Default: dense random Gaussian projection with QR-orthogonal columns.
        # QR orthogonalization (inspired by SubZero, ICCV 2025) gives lower
        # variance perturbations than i.i.d. scaled Gaussian columns, improving
        # per-step signal quality at negligible extra cost.
        entry_seed = hash((self.seed, spec.entry_key, step)) % (2**31)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(entry_seed)

        # Generate on CPU then move (Generator doesn't support CUDA)
        P_raw = torch.randn(
            spec.num_params, spec.num_coords,
            generator=gen, dtype=dtype, device='cpu',
        )
        # QR-orthogonalize columns when num_params >= num_coords (tall matrix).
        # This produces orthonormal columns, giving isotropic perturbations.
        # For wide matrices (more coords than params), fall back to scaled Gaussian.
        if spec.num_params >= spec.num_coords:
            P, _ = torch.linalg.qr(P_raw, mode='reduced')  # (num_params, num_coords)
        else:
            P = P_raw * (1.0 / math.sqrt(spec.num_coords))
        P = P.to(device=device)

        return P

    def _init_structured_projection(
        self,
        spec: LayerProjectionSpec,
        device: torch.device,
        dtype: torch.dtype,
        step: int = 0,
    ) -> torch.Tensor:
        """Create block-diagonal structured projection for a layer.

        For a weight matrix of shape (d_out, d_in), treats each row as a group.
        The projection matrix is block-diagonal: each d_in-sized block gets its
        own small random projection. This preserves within-row correlations.

        SubZero insight (Malladi et al. 2024): instead of projecting all
        d_out*d_in params together (losing structure), project groups of d_in
        params independently (preserving row structure of the weight matrix).

        For layers where d_out*d_in = num_params:
        - Split into d_out blocks of d_in params each
        - Each block gets a (d_in, block_coords) random projection
        - Assemble into a block-diagonal (num_params, num_coords) matrix

        Args:
            spec: LayerProjectionSpec for this parameter.
            device: Target device.
            dtype: Target dtype.
            step: Current step (used for rotation seed).

        Returns:
            Block-diagonal projection matrix of shape (num_params, num_coords).
        """
        d_out = spec.original_shape[0]
        d_in = spec.num_params // d_out

        # Distribute coords across blocks
        coords_per_block = max(1, math.ceil(spec.num_coords / d_out))
        actual_total_coords = coords_per_block * d_out

        # Deterministic seed from global seed + entry key + step
        entry_seed = hash((self.seed, spec.entry_key, step)) % (2**31)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(entry_seed)

        # Build block-diagonal projection on CPU
        P = torch.zeros(spec.num_params, actual_total_coords, dtype=dtype, device='cpu')
        for i in range(d_out):
            row_start = i * d_in
            row_end = row_start + d_in
            col_start = i * coords_per_block
            col_end = col_start + coords_per_block
            block = torch.randn(
                d_in, coords_per_block,
                generator=gen, dtype=dtype, device='cpu',
            )
            block.mul_(1.0 / math.sqrt(coords_per_block))
            P[row_start:row_end, col_start:col_end] = block

        # Handle rounding: pad or truncate to match num_coords
        if actual_total_coords < spec.num_coords:
            pad = torch.zeros(
                spec.num_params, spec.num_coords - actual_total_coords,
                dtype=dtype, device='cpu',
            )
            P = torch.cat([P, pad], dim=1)
        elif actual_total_coords > spec.num_coords:
            P = P[:, :spec.num_coords]

        return P.to(device=device)

    # ------------------------------------------------------------------
    # Rotation coordination
    # ------------------------------------------------------------------

    def rotate_all(
        self,
        projections: Dict[str, torch.Tensor],
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Rotate all layer projections according to the configured mode.

        All layers rotate together on the same schedule (synchronized rotation).
        For 'random' mode, regenerates all projections with new seeds.
        For 'displacement' mode, uses SVD of the layer's portion of displacement
        history to keep productive directions.

        Args:
            projections: Dict of current projection matrices {entry_key: P_layer}.
            step: Current optimization step (0-indexed).
            total_steps: Total number of optimization steps.
            displacement_history: Optional tensor of shape
                (history_len, subspace_dim) with recent displacement vectors
                in global subspace coordinates. Required for displacement mode.

        Returns:
            Dict of new projection matrices {entry_key: P_layer}.

        Example::

            # Random rotation
            projections = hybrid.rotate_all(projections, step=1, total_steps=100)

            # Displacement-biased rotation
            displacement = torch.randn(5, hybrid.subspace_dim) * 0.1
            projections = hybrid.rotate_all(
                projections, step=1, total_steps=100,
                displacement_history=displacement
            )
        """
        # Skip rotation if rotation is disabled (interval <= 0) or not on interval
        if self.rotation_interval <= 0:
            return projections
        if self.rotation_interval > 1 and step % self.rotation_interval != 0:
            return projections

        # Get device/dtype from first dense projection (sparse tensors lack .device/.dtype)
        device, dtype = torch.device('cpu'), torch.float32  # safe defaults
        for p in projections.values():
            if isinstance(p, torch.Tensor):
                device, dtype = p.device, p.dtype
                break

        # Determine rotation mode
        use_random = (
            self.rotation_mode == "random"
            or displacement_history is None
            or displacement_history.shape[0] == 0
        )

        if not use_random:
            # Check if displacement history has meaningful magnitude and is finite.
            # Squared-norm avoids sqrt; explicit .item() makes GPU-CPU sync visible.
            if not torch.isfinite(displacement_history).all():
                use_random = True
            elif (displacement_history * displacement_history).sum().item() < 1e-20:
                use_random = True

        if use_random:
            return self._rotate_all_random(device, dtype, step)
        else:
            svd_ratio = self.get_svd_ratio(step, total_steps)
            return self._rotate_all_displacement(
                projections, displacement_history, svd_ratio,
                device, dtype, step,
            )

    def _rotate_all_random(
        self,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> Dict[str, Union[torch.Tensor, SparseRandomProjection]]:
        """Regenerate all projections with new seeds (random mode).

        For sparse layers, creates a new SparseRandomProjection with an
        updated seed. For dense layers, regenerates the dense projection.

        Args:
            device: Target device.
            dtype: Target dtype.
            step: Current step (used for seed).

        Returns:
            Dict of new projection matrices.
        """
        new_projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}
        for spec in self.specs:
            dense_bytes = spec.num_params * spec.num_coords * 4
            if spec.is_projected and dense_bytes > self.sparse_threshold_bytes:
                entry_seed = hash((self.seed, spec.entry_key, step)) % (2**31)
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
        """Rotate all projections using displacement-biased SVD (displacement mode).

        For each layer, extracts that layer's portion of the displacement history,
        computes SVD to find productive directions, and combines with random
        directions for the new projection. Sparse layers fall back to random
        rotation since SVD-based rotation requires dense matmul.

        Args:
            projections: Current projection matrices (dense or sparse).
            displacement_history: Shape (history_len, subspace_dim).
            svd_ratio: Fraction of num_coords to fill with SVD directions.
            device: Target device.
            dtype: Target dtype.
            step: Current step.

        Returns:
            Dict of new projection matrices.
        """
        new_projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}

        for spec in self.specs:
            P = projections[spec.entry_key]
            if isinstance(P, SparseRandomProjection):
                # Sparse layers: fall back to random rotation (new seed)
                entry_seed = hash((self.seed, spec.entry_key, step)) % (2**31)
                new_projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                # Dense layers: displacement-biased SVD rotation
                layer_disp = displacement_history[:, spec.flat_start:spec.flat_end]
                new_P = self._rotate_layer_displacement(
                    P, spec, layer_disp, svd_ratio, device, dtype, step,
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
        """Rotate a single layer's projection using displacement-biased SVD.

        Projects the layer's displacement history to parameter space, computes
        SVD to find productive directions, keeps top k_svd directions, and
        fills remainder with random directions. Final matrix is NOT QR-orthonormalized
        (uses 1/sqrt(N) scaling like LinearSubspace).

        Args:
            P_old: Current projection matrix for this layer, shape (num_params, num_coords).
            spec: LayerProjectionSpec for this layer.
            layer_displacement: Shape (history_len, num_coords) displacement history.
            svd_ratio: Fraction of num_coords to fill with SVD directions.
            device: Target device.
            dtype: Target dtype.
            step: Current step.

        Returns:
            New projection matrix of shape (num_params, num_coords).
        """
        # 1D params: return identity
        if spec.num_params == spec.num_coords:
            return torch.eye(spec.num_params, dtype=dtype, device=device)

        k_svd = max(1, int(svd_ratio * spec.num_coords))
        k_random = spec.num_coords - k_svd

        # Project displacement history to full parameter space
        # layer_displacement: (history_len, num_coords)
        # P_old: (num_params, num_coords)
        # D_full: (num_params, history_len)
        D_full = P_old @ layer_displacement.T

        # Guard against non-finite values
        if not torch.isfinite(D_full).all():
            return self._get_projection(spec, device, dtype, step=step)

        # SVD of the full-space displacement matrix
        if k_svd < min(D_full.shape) // 2 and min(D_full.shape) > 6:
            # Randomized SVD: faster when k_svd << rank
            U_top, S_top, V_top = torch.pca_lowrank(D_full, q=k_svd, niter=2)
        else:
            U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
            k_svd = min(k_svd, U.shape[1])
            U_top = U[:, :k_svd]
        k_random = spec.num_coords - k_svd

        # Generate random directions for the remainder directly on target device
        entry_seed = hash((self.seed, spec.entry_key, step, "random")) % (2**31)
        gen = torch.Generator(device=device)
        gen.manual_seed(entry_seed)

        Z_random = torch.randn(
            spec.num_params, k_random,
            generator=gen, dtype=dtype, device=device,
        )

        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        # Concatenate SVD directions + random, QR-orthogonalize then scale
        combined = torch.cat([U_top, Z_random], dim=1)
        # QR requires float32 on CPU (BF16 unsupported); upcast and convert back
        orig_dtype = combined.dtype
        if combined.dtype == torch.bfloat16 and combined.device.type == 'cpu':
            combined = combined.float()
        Q, R = torch.linalg.qr(combined)
        if Q.dtype != orig_dtype:
            Q = Q.to(orig_dtype)
        # QR preserves relative importance of SVD vs random directions
        # Apply 1/sqrt(N) scaling for LinearSubspace convention
        combined = Q * (1.0 / math.sqrt(spec.num_coords))

        return combined

    def get_svd_ratio(self, step: int, total_steps: int) -> float:
        """Compute SVD ratio at the given step via linear interpolation.

        The ratio starts at ``svd_ratio_init`` (step 0) and linearly
        increases to ``svd_ratio_final`` (step = total_steps).

        Args:
            step: Current optimization step.
            total_steps: Total number of steps.

        Returns:
            SVD ratio in [svd_ratio_init, svd_ratio_final].
        """
        progress = min(1.0, step / max(1, total_steps))
        return self.svd_ratio_init + progress * (self.svd_ratio_final - self.svd_ratio_init)

    # ------------------------------------------------------------------
    # Absorb coordination
    # ------------------------------------------------------------------

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Check whether an absorb-and-rotate should be triggered.

        Matches AdaptiveSubspace's absorb logic: triggers on stagnation
        (stagnation_count >= absorb_patience) or periodically
        (iteration % absorb_interval == 0).

        Args:
            stagnation_count: Number of consecutive steps without improvement.
            iteration: Current iteration number (0-indexed).

        Returns:
            True if absorb should be triggered based on the configured mode.

        Example::

            if hybrid.should_absorb(stagnation_count=25, iteration=100):
                # Trigger absorb and rotate
                pass
        """
        if self.absorb_mode == "stagnation":
            return stagnation_count >= self.absorb_patience
        elif self.absorb_mode == "periodic" and self.absorb_interval > 0:
            return iteration > 0 and iteration % self.absorb_interval == 0
        return False

    # ------------------------------------------------------------------
    # Core reconstruction methods (matching LinearSubspace contract)
    # ------------------------------------------------------------------

    def apply_perturbation(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct full state_dict from base params + flat subspace vector.

        For each layer, slices the subspace coordinates, applies the per-layer
        projection matrix P_layer, reshapes to original parameter shape, and
        adds to the base parameter. For 1D params (biases), adds chunk directly
        without projection. Supports both dense tensors and SparseRandomProjection.

        Args:
            projections: Dict mapping entry_key to per-layer projection matrices
                (dense Tensor or SparseRandomProjection for large layers).
            base_sd: Base state_dict with original parameter values.
            flat_subspace: 1D tensor of shape (subspace_dim,).

        Returns:
            New state_dict with perturbed parameters.

        Example::

            projections = hybrid.init_projections(device, dtype)
            base_sd = model.state_dict()
            coords = torch.randn(hybrid.subspace_dim) * 0.1
            perturbed_sd = hybrid.apply_perturbation(projections, base_sd, coords)
            model.load_state_dict(perturbed_sd)
        """
        result = {}
        for spec in self.specs:
            chunk = flat_subspace[spec.flat_start:spec.flat_end]
            base = base_sd[spec.entry_key]

            if spec.is_projected:
                P = projections[spec.entry_key]
                if isinstance(P, SparseRandomProjection):
                    # Sparse path: project coords to full param space, add to base
                    delta = P.project(chunk)  # (num_params,)
                    result[spec.entry_key] = (base.reshape(-1) + delta).reshape(spec.original_shape)
                else:
                    # Dense path: fused add + projection via addmm
                    result_flat = torch.addmm(
                        base.reshape(1, -1),
                        chunk.unsqueeze(0),
                        P.t(),
                    )
                    result[spec.entry_key] = result_flat.reshape(spec.original_shape)
            else:
                # 1D param (bias): add coords directly
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape)

        return result

    def build_fused_projection(
        self,
        projections: Dict[str, Union[torch.Tensor, "SparseRandomProjection"]],
    ) -> None:
        """Build a fused block-diagonal projection matrix from per-layer projections.

        Combines all dense per-layer projection matrices into a single
        block-diagonal matrix for efficient single-matmul reconstruction.
        Only used when the fused matrix is small enough (<256 MB) to avoid
        memory regression from dense storage of a sparse block-diagonal.

        Called once after each rotation (in optimizer.py after rotate_all).
        The fused matrix is cached in ``self._fused_P`` and reused across
        all ``reconstruct_batch`` calls until the next rotation.
        """
        dense_blocks = []
        self._fused_dense_specs = []  # specs participating in fused matmul
        self._fused_sparse_specs = []  # specs needing per-layer sparse path
        self._fused_bias_specs = []  # 1D params (identity, no projection)
        total_params = 0
        total_coords = 0

        for spec in self.specs:
            P = projections.get(spec.entry_key)
            if not spec.is_projected:
                self._fused_bias_specs.append(spec)
            elif isinstance(P, SparseRandomProjection):
                self._fused_sparse_specs.append((spec, P))
            else:
                self._fused_dense_specs.append((spec, total_params))
                dense_blocks.append(P)
                total_params += spec.num_params
                total_coords += spec.num_coords

        # Only fuse if the dense block-diagonal matrix is < 256 MB.
        # Otherwise the zero-padding wastes more memory than it saves in
        # kernel launch overhead.
        fused_bytes = total_params * total_coords * 4
        max_fused_bytes = 256 * 1024 * 1024  # 256 MB

        if dense_blocks and fused_bytes <= max_fused_bytes:
            self._fused_P = torch.block_diag(*dense_blocks)
            self._fused_total_dense_params = total_params
        else:
            self._fused_P = None
            self._fused_total_dense_params = 0

    def apply_perturbation_inplace(
        self,
        projections: Dict[str, Union[torch.Tensor, "SparseRandomProjection"]],
        model: "nn.Module",
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
        param_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Apply perturbation directly to model weights in-place.

        EGGROLL-inspired: instead of materializing a full perturbed state_dict
        and loading it, this modifies model parameters' ``.data`` tensors
        directly. Avoids allocating a new dict and reduces memory to one
        addmm result per layer (immediately written into the param tensor).

        Args:
            projections: Per-layer projection matrices.
            model: The model whose parameters will be modified in-place.
            base_sd: Base (unperturbed) state_dict values.
            flat_subspace: 1D tensor of shape (subspace_dim,).
            param_dict: Optional pre-built dict from ``model.named_parameters()``.
                Avoids re-traversing the module tree when called in a loop.
        """
        if param_dict is None:
            param_dict = dict(model.named_parameters())
        for spec in self.specs:
            key = spec.entry_key
            if key not in param_dict:
                continue
            chunk = flat_subspace[spec.flat_start:spec.flat_end]
            base = base_sd[key]
            param = param_dict[key]

            if spec.is_projected:
                P = projections[key]
                if isinstance(P, SparseRandomProjection):
                    delta = P.project(chunk)
                    param.data.copy_((base.reshape(-1) + delta).reshape(spec.original_shape))
                else:
                    # Fused: base + coords @ P^T, written directly into param.data
                    torch.addmm(
                        base.reshape(1, -1), chunk.unsqueeze(0), P.t(),
                        out=param.data.reshape(1, -1),
                    )
            else:
                param.data.copy_(base + chunk.reshape(spec.original_shape))

    def reconstruct_batch(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points.

        Uses a fused block-diagonal matmul for all dense layers (single cuBLAS
        call instead of per-layer calls), with fallback to per-layer for sparse
        projections and 1D biases.

        If ``build_fused_projection`` has not been called, falls back to the
        per-layer loop for all entries.

        Args:
            projections: Dict mapping entry_key to per-layer projection matrices
                (dense Tensor or SparseRandomProjection for large layers).
            base_sd: Base state_dict with original parameter values.
            flat_subspace_batch: 2D tensor of shape (N, subspace_dim).

        Returns:
            Dict ``{key: (N, *original_shape)}`` with batched perturbed params.
        """
        N = flat_subspace_batch.shape[0]
        result = {}

        # Fast path: fused block-diagonal matmul for all dense layers
        if getattr(self, '_fused_P', None) is not None and self._fused_dense_specs:
            # Gather subspace coords for all dense layers.
            # If dense specs are laid out contiguously (common case), use a
            # single slice instead of a 72-iteration Python loop + torch.cat.
            first_start = self._fused_dense_specs[0][0].flat_start
            last_spec = self._fused_dense_specs[-1][0]
            last_end = last_spec.flat_end
            total_dense_coords = sum(s.num_coords for s, _ in self._fused_dense_specs)
            if last_end - first_start == total_dense_coords:
                # Contiguous layout - single slice
                fused_coords = flat_subspace_batch[:, first_start:last_end]
            else:
                # Non-contiguous - fallback to cat
                coord_slices = []
                for spec, _ in self._fused_dense_specs:
                    coord_slices.append(flat_subspace_batch[:, spec.flat_start:spec.flat_end])
                fused_coords = torch.cat(coord_slices, dim=1)  # (N, total_dense_coords)

            # Single matmul: (N, total_dense_coords) @ (total_dense_coords, total_dense_params)
            fused_delta = fused_coords @ self._fused_P.t()  # (N, total_dense_params)

            # Split into per-layer deltas and add to base
            for spec, param_offset in self._fused_dense_specs:
                base = base_sd[spec.entry_key]
                delta = fused_delta[:, param_offset:param_offset + spec.num_params]
                result_flat = base.reshape(1, -1) + delta
                result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)

            # Sparse layers: per-layer fallback
            for spec, P in self._fused_sparse_specs:
                chunk = flat_subspace_batch[:, spec.flat_start:spec.flat_end]
                base = base_sd[spec.entry_key]
                delta = P.project(chunk)
                result_flat = base.reshape(1, -1) + delta
                result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)

            # 1D biases: direct add
            for spec in self._fused_bias_specs:
                chunk = flat_subspace_batch[:, spec.flat_start:spec.flat_end]
                base = base_sd[spec.entry_key]
                delta = chunk.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base.unsqueeze(0) + delta
        else:
            # Fallback: per-layer loop (no fused projection built yet)
            for spec in self.specs:
                chunk = flat_subspace_batch[:, spec.flat_start:spec.flat_end]
                base = base_sd[spec.entry_key]

                if spec.is_projected:
                    P = projections[spec.entry_key]
                    if isinstance(P, SparseRandomProjection):
                        delta = P.project(chunk)
                        result_flat = base.reshape(1, -1) + delta
                        result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)
                    else:
                        result_flat = base.reshape(1, -1) + chunk @ P.t()
                        result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)
                else:
                    delta = chunk.reshape(N, *spec.original_shape)
                    result[spec.entry_key] = base.unsqueeze(0) + delta

        return result

    def reconstruct_base(
        self,
        projections: Dict[str, Union[torch.Tensor, "SparseRandomProjection"]],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct full params from subspace coords (single config, no batch dim).

        Returns unbatched tensors suitable as a base for delta reconstruction.
        """
        result = {}
        for spec in self.specs:
            chunk = flat_subspace[spec.flat_start:spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_projected:
                P = projections[spec.entry_key]
                if isinstance(P, SparseRandomProjection):
                    delta = P.project(chunk.unsqueeze(0)).squeeze(0)
                    result[spec.entry_key] = (base.reshape(-1) + delta).reshape(spec.original_shape)
                else:
                    result[spec.entry_key] = (base.reshape(-1) + chunk @ P.t()).reshape(spec.original_shape)
            else:
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape)
        return result

    def reconstruct_batch_delta(
        self,
        projections: Dict[str, Union[torch.Tensor, "SparseRandomProjection"]],
        base_reconstructed: Dict[str, torch.Tensor],
        flat_subspace_base: torch.Tensor,
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Delta-based batched reconstruction: only recompute layers affected by changes.

        Instead of reconstructing all layers from scratch for each probe,
        computes delta = (probe_coords - base_coords) and applies only to
        layers whose coord ranges overlap with the changed positions.

        This is much faster when probe configs differ from base in only a few
        contiguous coords (e.g., one particle's pdim-dimensional coordinates).

        Args:
            projections: Per-layer projection matrices.
            base_reconstructed: Pre-computed full reconstruction from reconstruct_base().
            flat_subspace_base: 1D tensor (subspace_dim,) - the base subspace coords.
            flat_subspace_batch: 2D tensor (N, subspace_dim) - the probed configs.

        Returns:
            Dict {key: (N, *shape)} with batched perturbed params.
        """
        N = flat_subspace_batch.shape[0]
        # Compute delta coords: where batch differs from base
        delta_coords = flat_subspace_batch - flat_subspace_base.unsqueeze(0)  # (N, sub_dim)

        result = {}
        for spec in self.specs:
            delta_chunk = delta_coords[:, spec.flat_start:spec.flat_end]  # (N, num_coords)

            # Check if any probe actually modifies this layer's coords
            # Use a fast check: if all deltas are zero, just broadcast base
            has_delta = delta_chunk.any()

            base_val = base_reconstructed[spec.entry_key]  # (*shape)

            if not has_delta:
                # No change in this layer - broadcast base
                result[spec.entry_key] = base_val.unsqueeze(0).expand(N, *spec.original_shape)
            elif spec.is_projected:
                P = projections[spec.entry_key]
                if isinstance(P, SparseRandomProjection):
                    delta_params = P.project(delta_chunk)  # (N, num_params)
                    result_flat = base_val.reshape(1, -1) + delta_params
                    result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)
                else:
                    delta_params = delta_chunk @ P.t()  # (N, num_params)
                    result_flat = base_val.reshape(1, -1) + delta_params
                    result[spec.entry_key] = result_flat.reshape(N, *spec.original_shape)
            else:
                delta = delta_chunk.reshape(N, *spec.original_shape)
                result[spec.entry_key] = base_val.unsqueeze(0) + delta

        return result

    def absorb(
        self,
        projections: Dict[str, torch.Tensor],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fold subspace perturbation into base weights and zero the subspace.

        This operation "absorbs" the current perturbation into the base parameters,
        allowing exploration of a new region of parameter space on subsequent steps.
        After absorb, the projection matrices should be regenerated via rotate_all
        to ensure diverse exploration.

        Args:
            projections: Dict mapping entry_key to per-layer projection matrices.
            base_sd: Base state_dict to absorb perturbation into.
            flat_subspace: Current subspace vector of shape (subspace_dim,).

        Returns:
            Tuple of (new_base_sd, zeroed_subspace_vector).

        Example::

            if hybrid.should_absorb(stagnation_count, iteration):
                new_base, zero_coords = hybrid.absorb(projections, base_sd, coords)
                # Regenerate all projections for fresh exploration
                projections = hybrid.init_projections(device, dtype)
        """
        new_sd = self.apply_perturbation(projections, base_sd, flat_subspace)
        return new_sd, torch.zeros_like(flat_subspace)


# ------------------------------------------------------------------
# Block creation for per-layer OT decomposition
# ------------------------------------------------------------------


def create_hybrid_blocks(
    hybrid: HybridSubspace,
    particle_dim: int = 8,
) -> "List[BlockConfig]":
    """Create blocks for per-layer OT decomposition in hybrid mode.

    Creates one BlockConfig per LayerProjectionSpec, allowing independent
    OT solves per layer while maintaining global cost evaluation through
    the shared projection matrices.

    Unlike create_subspace_blocks (which evenly divides a global subspace),
    this function respects layer boundaries -- each block corresponds exactly
    to one layer's subspace coordinates.

    Note: Block flat ranges are computed with contiguous offsets (accounting for
    padding between layers) and may differ from the spec flat ranges. When
    splitting/reassembling, use the block's flat_start/flat_end, not the spec's.

    Args:
        hybrid: HybridSubspace instance with per-layer specs.
        particle_dim: Dimension of each particle within a block (default 8).
            Higher values give more polytope vertices (2*dim for orthoplex)
            but fewer particles per block.

    Returns:
        List of BlockConfig, one per layer in hybrid.specs.

    Example::

        hybrid = HybridSubspace.from_layout(layout, rank=4)
        blocks = create_hybrid_blocks(hybrid, particle_dim=8)
        # Use blocks for per-layer OT decomposition

    See Also:
        ``blockwise.create_subspace_blocks`` for uniform subspace division.
        ``blockwise.BlockConfig`` for block configuration details.
    """
    from .blockwise import BlockConfig

    blocks = []
    offset = 0  # Track contiguous offset accounting for padding

    for i, spec in enumerate(hybrid.specs):
        # Pad layer's num_coords to be divisible by particle_dim
        num_coords = spec.flat_end - spec.flat_start
        padded_coords = num_coords + (-num_coords % particle_dim)
        num_particles = padded_coords // particle_dim

        # Use contiguous offset, not spec.flat_start, to avoid gaps/overlaps
        blocks.append(BlockConfig(
            name=spec.entry_key,
            leaf_indices=(i,),  # Index into hybrid.specs
            flat_start=offset,
            flat_end=offset + padded_coords,
            num_particles=num_particles,
            particle_dim=particle_dim,
        ))
        offset += padded_coords

    # Note: total offset (sum of padded block dims) may exceed hybrid.subspace_dim
    # due to per-block padding. Callers must allocate vectors of size `offset`,
    # not hybrid.subspace_dim, when using these blocks.

    return blocks
