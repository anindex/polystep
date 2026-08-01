"""Per-layer projections with coordinated rotation.

:class:`HybridSubspace` combines :class:`LinearSubspace`'s per-layer
projections with :class:`AdaptiveSubspace`'s synchronized rotation:

- **Per-layer projections.** Each parameter entry has its own
  ``P_layer`` of shape ``(num_params, num_coords)``. Per-layer
  projections cover more parameters per step than a single global one
  (empirically ~4.3% vs ~0.25% on MNIST MLPs).
- **Synchronized rotation.** All layer projections rotate on the same
  schedule (every ``rotation_interval`` steps; ``0`` disables rotation).
- **QR-orthonormal per-layer columns**, giving isotropic unit-norm perturbations
  (``||delta||^2 ~ num_coords`` for unit-variance coords). This differs from
  :class:`LinearSubspace`'s scaled-Gaussian columns (``||delta||^2 ~ num_params``):
  switching ``LinearSubspace`` -> ``HybridSubspace`` at the *same* ``step_radius``
  changes the actual perturbation magnitude by ``sqrt(num_coords / num_params)``
  (often 10-100x smaller), so retune ``step_radius`` when switching.
- **Displacement-biased rotation.** In ``'displacement'`` mode each layer
  rotates toward its own slice of the displacement history.

Example::

    from polystep import HybridSubspace, ParamLayout
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(784, 128), nn.Linear(128, 10))
    layout = ParamLayout.from_module(model)
    hybrid = HybridSubspace.auto_from_layout(layout)
    projections = hybrid.init_projections(torch.device('cpu'), torch.float32)
    projections = hybrid.rotate_all(projections, step=i, total_steps=100)

See Also:
    :class:`LinearSubspace`: fixed per-layer projections (no rotation).
    :class:`AdaptiveSubspace`: single global rotating projection.
"""

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


# Same seven fields as the canonical spec; kept as a name, not a second class.
LayerProjectionSpec = ProjectionSpec


def _mixed_entry_dtypes(layout: "ParamLayout") -> "Optional[Dict[str, torch.dtype]]":
    """Per-entry dtypes, or None when they already agree.

    Only heterogeneity needs recording. A uniform model must keep deferring to the dtype
    the caller asks for, which is how ``mixed_precision`` casts the projections to BF16
    and how a device move re-materializes them.
    """
    dtypes = {e.key: e.dtype for e in layout.entries}
    floats = {d for d in dtypes.values() if d.is_floating_point}
    return dtypes if len(floats) > 1 else None


def _require_positive_rank(rank: int, name: str) -> None:
    """Reject a rank of zero, which yields a spec with no coordinates.

    QR of an ``(n, 0)`` and the ``(1, 0) x (0, n)`` addmm both succeed, so every
    reconstruction adds exactly zeros and the layer never moves.
    """
    if rank < 1:
        raise ValueError(f"{name} must be >= 1, got {rank}.")


def _require_tall(spec: "LayerProjectionSpec") -> None:
    """Every spec builder caps ``num_coords`` at ``num_params``.

    A wide spec has no orthonormal basis, so QR would drop columns and a Gaussian
    fallback would sample at gain ``sqrt(num_params / num_coords)`` where the step
    radii assume 1. Reject it rather than move at the wrong scale.
    """
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
    """Scale projected specs so the total stays within ``max_dim``.

    Unprojected specs keep their full width: they are perturbed directly and have no
    rank to trade away. The rest share what is left, proportionally, against a running
    budget so per-spec rounding cannot accumulate past the cap.

    Returns ``(new_specs, new_total_dim)``. No-op when ``max_dim`` is None or the
    current total already fits.
    """
    if max_dim is None or current_dim <= max_dim:
        return specs, current_dim

    unprojected_dim = sum(s.num_coords for s in specs if not s.is_projected)
    n_projected = sum(1 for s in specs if s.is_projected)
    target_projected = max_dim - unprojected_dim

    if target_projected < n_projected:
        # Unprojected width alone exceeds the budget, so the cap is unreachable.
        # Floor every projected spec at one coordinate and say what it cost.
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
            # At full width the projection is the identity, so perturb the parameter
            # directly rather than allocating an eye(num_params).
            spec = replace(spec, num_coords=num_coords, is_projected=num_coords < spec.num_params)
        new_specs.append(replace(spec, flat_start=new_offset, flat_end=new_offset + spec.num_coords))
        new_offset += spec.num_coords
    return new_specs, new_offset


@dataclass
class HybridSubspace(ProjectedAbsorbMixin, SvdRatioMixin):
    """Hybrid subspace compression with per-layer projections and global rotation.

    Combines LinearSubspace's per-layer structure with AdaptiveSubspace's
    synchronized rotation coordination. Each layer has its own projection
    matrix, but all projections rotate together on the same schedule.

    Two rotation modes are supported:

    - ``'random'``: Regenerates all layer projections with new seeds.
      Cheap, and adequate for exploration.

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

        projections = hybrid.init_projections(torch.device('cpu'), torch.float32)

        # Rotate all projections together
        projections = hybrid.rotate_all(projections, step=1, total_steps=100)

    Attributes:
        specs: Per-layer projection specifications.
        subspace_dim: Total subspace dimension (sum of all layer num_coords).
        compression_ratio: subspace_dim / total_params.
        seed: Base seed for deterministic projection generation.
        rotation_mode: 'random' or 'displacement' (default 'displacement').
        rotation_interval: Rotate every N steps. ``0`` disables rotation
            (default; rotating hurts accuracy on small MLPs because it
            re-randomizes already-discovered descent directions).
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
    # Regenerate at the absorb boundary by displacement-SVD instead of a random redraw.
    # Safe there because the duals are already reset, so it avoids the per-step-rotation
    # degradation that keeps rotation_interval=0. Lets a smaller rank track descent
    # across absorbs, or is a no-op if the local descent is not low-rank.
    absorb_aligned_active: bool = False

    _total_params: int = 0
    # Budget control settings (preserved across rank transitions)
    _max_subspace_dim: Optional[int] = None
    # Per-entry parameter dtype. Coordinates carry the layout's dominant dtype, so on a
    # mixed-dtype model a minority entry's projection has to be built at its own dtype or
    # the reconstruction matmul mixes Double and Float. Empty means uniform.
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
                # Same formula as LinearSubspace for drop-in compatibility, capped at
                # num_params. A wide (num_params, num_coords) matrix has rank at most
                # num_params, so the extra coordinates are redundant, and QR cannot
                # orthonormalize it: the fallback would be a scaled Gaussian of gain
                # sqrt(num_params / num_coords) where the step radii assume 1.
                num_coords = min(d_out * effective_rank + effective_rank * d_in, num_params)
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                        # At full width the projection is the identity, so carry the
                        # parameter directly rather than allocating an eye(num_params).
                        is_projected=num_coords < num_params,
                    )
                )
                offset += num_coords
            else:
                # 1D param (bias, LayerNorm): full perturbation, no projection
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
                # Capped at num_params for the same reason as from_layout above.
                num_coords = min(d_out * effective_rank + effective_rank * d_in, num_params)
                specs.append(
                    LayerProjectionSpec(
                        entry_key=entry.key,
                        original_shape=shape,
                        num_params=num_params,
                        num_coords=num_coords,
                        flat_start=offset,
                        flat_end=offset + num_coords,
                        # At full width the projection is the identity, so carry the
                        # parameter directly rather than allocating an eye(num_params).
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
        """Per-layer projection matrices, one ``(num_params, num_coords)`` per layer.

        Seeded at step 0, so every call for a given ``(device, dtype)`` rebuilds the same
        basis. Memoized and returned by identity, which skips the per-layer QR on absorb.
        Callers must not write into the result: rotations and rank transitions build new
        projections rather than mutating these.

        Layers whose dense projection would exceed ``sparse_threshold_bytes`` get a
        ``SparseRandomProjection`` instead.

        Returns a dict keyed by entry, values dense or sparse per layer.
        """
        cached = getattr(self, "_init_projections_cache", None)
        if cached is not None and cached[0] == (device, dtype):
            return cached[1]
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]] = {}
        for spec in self.specs:
            if not spec.is_projected:
                continue  # 1D params add coords directly; storing an eye wastes O(n^2)
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
        """The dtype this entry's projection is built at.

        Its own parameter dtype, matching :class:`LinearSubspace`, which resolves it from
        ``base.dtype`` at reconstruct time. Coordinates stay at the layout's dominant
        dtype, so a minority entry's chunk is cast at the matmul rather than the whole
        projection being rebuilt.
        """
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
        """Generate the projection matrix for a parameter entry.

        Generates P of shape (num_params, num_coords) deterministically from
        self.seed, spec.entry_key and step: a dense Gaussian, QR-orthogonalized.

        Args:
            spec: LayerProjectionSpec for this parameter.
            device: Target device.
            dtype: Target dtype.
            step: Current step (used for rotation seed).

        Returns:
            Projection matrix P of shape (num_params, num_coords).
        """
        dtype = self.entry_dtype(spec, dtype)
        # Dense Gaussian with QR-orthogonal columns: lower-variance perturbations than
        # i.i.d. scaled columns at the same cost (SubZero, ICCV 2025).
        entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(entry_seed)

        # Generate on CPU then move (Generator doesn't support CUDA)
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
        # Same guard as FactoredSubspace.rotate_all: step 0 is the freshly seeded basis,
        # so rotating it discards a basis nothing has been evaluated against yet.
        if self.rotation_interval <= 0 or step <= 0 or step % self.rotation_interval != 0:
            return projections

        # Get device/dtype from first dense projection (sparse tensors lack .device/.dtype)
        device, dtype = torch.device("cpu"), torch.float32  # safe defaults
        for p in projections.values():
            if isinstance(p, torch.Tensor):
                device, dtype = p.device, p.dtype
                break

        # Determine rotation mode
        use_random = (
            self.rotation_mode == "random" or displacement_history is None or displacement_history.shape[0] == 0
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
            if not spec.is_projected:
                continue  # 1D params add coords directly; storing an eye wastes O(n^2)
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
            if not spec.is_projected:
                continue  # 1D params have no stored projection
            P = projections[spec.entry_key]
            if isinstance(P, SparseRandomProjection):
                # Sparse layers: fall back to random rotation (new seed)
                entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step)
                new_projections[spec.entry_key] = SparseRandomProjection(
                    full_dim=spec.num_params,
                    subspace_dim=spec.num_coords,
                    seed=entry_seed,
                )
            else:
                # Dense layers: displacement-biased SVD rotation
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
        """Rotate a single layer's projection using displacement-biased SVD.

        Projects the layer's displacement history to parameter space, computes
        SVD to find productive directions, keeps top k_svd directions, and fills
        the remainder with random directions. Columns come back unit-norm, matching
        _get_projection.

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
        _require_tall(spec)
        # The rotated basis has to come back at the same dtype as the one it replaces,
        # and P_old carries this entry's, which is not the coordinates' on a mixed model.
        dtype = self.entry_dtype(spec, dtype)
        # Asking for no SVD directions means a fresh random basis, and skips the SVD.
        if svd_ratio <= 0.0:
            return self._get_projection(spec, device, dtype, step=step)

        k_svd = max(1, int(svd_ratio * spec.num_coords))
        k_random = spec.num_coords - k_svd

        # Displacement history to full parameter space: D_full is
        # (num_params, history_len).
        D_full = P_old @ layer_displacement.T.to(P_old.dtype)

        # Guard against non-finite values
        if not torch.isfinite(D_full).all():
            return self._get_projection(spec, device, dtype, step=step)

        svd_dtype = D_full.dtype
        D_full = D_full.to(decomposition_dtype(svd_dtype))

        # SVD of the full-space displacement matrix. The history is short, so the
        # full decomposition costs about the same as a randomized one and stays
        # deterministic under a caller-supplied generator.
        U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
        k_svd = min(k_svd, U.shape[1])
        U_top = U[:, :k_svd]
        k_random = spec.num_coords - k_svd
        if U_top.dtype != svd_dtype:
            U_top = U_top.to(svd_dtype)

        # Generate random directions for the remainder directly on target device
        entry_seed = _stable_entry_seed(self.seed, spec.entry_key, step, "random")
        gen = torch.Generator(device=device)
        gen.manual_seed(entry_seed)

        Z_random = torch.randn(
            spec.num_params,
            k_random,
            generator=gen,
            dtype=dtype,
            device=device,
        )

        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        # Unit-norm QR, matching _get_projection so magnitude is continuous across
        # the first rotation. Tall by the same invariant _get_projection enforces.
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

    # Core reconstruction methods (matching LinearSubspace contract)

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
            chunk = flat_subspace[spec.flat_start : spec.flat_end]
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
                        chunk.unsqueeze(0).to(P.dtype),
                        P.t(),
                    )
                    result[spec.entry_key] = result_flat.reshape(spec.original_shape)
            else:
                # 1D param (bias): add coords directly
                result[spec.entry_key] = base + chunk.reshape(spec.original_shape).to(base.dtype)

        return result

    def _gather_fused_coords(self, flat_subspace: torch.Tensor) -> torch.Tensor:
        """Slice out the coordinates the fused dense blocks consume, in block order.

        Works for a single candidate ``(subspace_dim,)`` and for a batch
        ``(N, subspace_dim)``; the dense specs are contiguous in the common case,
        so this is one view rather than a per-layer gather.

        Cast to the fused block's dtype: coordinates carry the layout's dominant dtype
        over every entry, the block fuses the majority dtype among projected dense specs
        alone, and on a mixed-dtype model those disagree.
        """
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
        """Cache the concatenated dense base row for a run of in-place candidates.

        The base is constant across the candidates in a chunk, so building it once
        here keeps :meth:`apply_perturbation_inplace` down to two kernel launches
        per candidate. Call :meth:`release_inplace` when the chunk is done: a
        stale row would perturb around the wrong point.
        """
        if getattr(self, "_fused_P", None) is None or not self._fused_dense_specs:
            return
        self._fused_base_row_cache = torch.cat(
            [base_sd[spec.entry_key].reshape(-1) for spec, _ in self._fused_dense_specs]
        ).unsqueeze(0)

    def release_inplace(self) -> None:
        """Drop the cached base row and scratch buffer from :meth:`prepare_inplace`."""
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
        """Build a fused block-diagonal projection matrix from per-layer projections.

        Combines the dense per-layer projections into one block-diagonal matrix so
        reconstruction is a single matmul instead of one per layer, at the cost of
        reading the zero padding on every candidate.

        Two guards. A single block is just a copy, so skip it. Past ``max_fused_bytes``
        the padding read costs more than the launches saved, so the fuse declines.
        Measured crossover in docs/performance.md.

        Called once after each rotation (in optimizer.py after rotate_all).
        The fused matrix is cached in ``self._fused_P`` and reused across
        all ``reconstruct_batch`` calls until the next rotation.
        """
        dense_blocks = []
        self._fused_dense_specs = []  # specs participating in fused matmul
        self._fused_sparse_specs = []  # specs needing per-layer sparse path
        self._fused_bias_specs = []  # 1D params (identity, no projection)
        self._fused_odd_dtype_specs = []  # dense, but not the fused block's dtype
        total_params = 0
        total_coords = 0

        # block_diag needs one dtype. On a mixed-dtype model fuse the majority and leave
        # the rest on the per-layer dense path, which is where they already were.
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
        # The cached base row describes the old block layout.
        self.release_inplace()

    def apply_perturbation_inplace(
        self,
        projections: Dict[str, Union[torch.Tensor, SparseRandomProjection]],
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

        # Fused path: one addmm covers every dense layer, then a single grouped
        # copy scatters the result into the parameter tensors. Two kernel
        # launches per candidate regardless of layer count.
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
                    # addmm writes straight into the parameter storage.
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

        One fused block-diagonal matmul covers every dense layer, then the base
        weights are added into that buffer in place and each layer is handed out
        as a strided view of it. Sparse projections and 1D biases still go
        per-layer. Falls back to the per-layer loop entirely when
        ``build_fused_projection`` has not been called.

        The dense entries alias one buffer, so callers must treat them as
        read-only. Every evaluator does: vmap/functional_call and bmm read them,
        and the in-place path copies out of them.

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

        if getattr(self, "_fused_P", None) is not None and self._fused_dense_specs:
            fused_coords = self._gather_fused_coords(flat_subspace_batch)
            # One matmul covers every dense layer at once.
            fused_delta = fused_coords @ self._fused_P.t()

            for spec, param_offset in self._fused_dense_specs:
                block = fused_delta[:, param_offset : param_offset + spec.num_params]
                block.add_(base_sd[spec.entry_key].reshape(1, -1))
                # unflatten, not reshape: the slice is contiguous inside each row,
                # so this is a view. reshape would copy the whole (N, num_params).
                result[spec.entry_key] = block.unflatten(1, spec.original_shape)

            for spec, P in self._fused_sparse_specs:
                chunk = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
                # project() returns a transposed sparse-mm result, so the reshape
                # below already copies. Adding first fuses the two passes into one.
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
                    # project() returns a transposed sparse-mm result, so the reshape
                    # already copies. Adding first fuses the two passes into one.
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

    Respects layer boundaries, where ``blockwise.create_subspace_blocks`` divides a
    global subspace evenly. Block flat ranges use contiguous offsets that account for
    inter-layer padding, so they can differ from the spec flat ranges: split and
    reassemble with the block's ``flat_start``/``flat_end``, not the spec's.

    Args:
        hybrid: HybridSubspace instance with per-layer specs.
        particle_dim: Dimension of each particle within a block. Higher values give
            more polytope vertices (``2 * dim`` for orthoplex) but fewer particles.
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
        blocks.append(
            BlockConfig(
                name=spec.entry_key,
                leaf_indices=(i,),  # Index into hybrid.specs
                flat_start=offset,
                flat_end=offset + padded_coords,
                num_particles=num_particles,
                particle_dim=particle_dim,
            )
        )
        offset += padded_coords

    # total offset (sum of padded block dims) may exceed hybrid.subspace_dim
    # due to per-block padding. Callers must allocate vectors of size `offset`,
    # not hybrid.subspace_dim, when using these blocks.

    return blocks
