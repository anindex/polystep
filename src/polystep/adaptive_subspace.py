"""Rotating orthogonal projection subspace.

:class:`AdaptiveSubspace` replaces :class:`LinearSubspace`'s fixed random
per-layer projection with a single *global* QR-orthogonal projection
``P`` of shape ``(full_dim, subspace_dim)`` that can be re-drawn each
iteration, optionally biased toward productive directions.

Differences from :class:`LinearSubspace`:

- One projection matrix covers all parameters (cross-layer mixing).
- ``P`` is stored in ``SolverState.projection`` and passed in to every
  method, so rotation does not require rebuilding the subspace object.
- Columns are QR-orthonormal (``P^T P = I``), so ``||P @ coords|| =
  ||coords||``. There is no ``1/sqrt(N)`` dilation, so the right
  ``step_radius`` is typically smaller than for :class:`LinearSubspace`.

Rotation modes:

- ``'random'`` - fresh QR-orthogonal basis every call.
- ``'displacement'`` - SVD of recent displacements keeps productive
  directions; the SVD share grows linearly from ``svd_ratio_init`` to
  ``svd_ratio_final`` over the schedule.
``absorb()`` folds the current subspace perturbation into the base
weights and zeros the subspace vector. Combined with rotation each
iteration explores a fresh subspace centered on the current best.
"""

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
    """Mirror a non-CPU generator onto CPU, seeding from a fresh draw.

    CPU tensors cannot use a CUDA generator, so CPU-side QR needs a CPU one.
    Seeding it from initial_seed() reuses the same constant every call and
    freezes every basis to be identical; drawing a new integer advances the
    source generator so successive calls differ. Returns the generator unchanged
    when it is already CPU or None.
    """
    if generator is None or generator.device.type == "cpu":
        return generator
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)
    return cpu_gen


def _draw_basis_gaussian(rows, cols, device, dtype, generator):
    """Draw a (rows, cols) Gaussian for basis construction.

    Seeded callers go through CPU so the random stream is identical whatever the
    target device. Unseeded there is no such contract, so draw on the target and skip
    a host allocation plus transfer that reaches gigabytes per step at large full_dim.
    """
    target = torch.device(device)
    if generator is None:
        return torch.randn(rows, cols, device=target, dtype=dtype)
    Z = torch.randn(rows, cols, generator=_spawn_cpu_generator(generator), device="cpu", dtype=dtype)
    return Z.to(device=target) if target.type != "cpu" else Z


@dataclass(frozen=True)
class EntrySpec:
    """Describes the mapping of a single parameter entry in the flat vector.

    Attributes:
        entry_key: state_dict key (e.g., "fc1.weight").
        original_shape: Original parameter shape.
        num_params: Total number of elements in this parameter.
        flat_start: Start offset into the global flat vector.
        flat_end: End offset into the global flat vector.
    """

    entry_key: str
    original_shape: Tuple[int, ...]
    num_params: int
    flat_start: int
    flat_end: int


@dataclass
class AdaptiveSubspace(ProjectedAbsorbMixin, SvdRatioMixin):
    """Adaptive subspace compression with rotating orthogonal projection.

    Stores static configuration only. The projection matrix P lives in
    optimizer state (``SolverState.projection``) and is passed as an
    argument to all methods.

    Two rotation modes are supported:

    - ``'random'``: Draws entirely new QR-orthogonalized basis each call
      to ``rotate()``. Cheap, and equivalent to random search
      in a new subspace each iteration.

    - ``'displacement'``: Uses SVD of recent displacement history to retain
      productive directions. The fraction of SVD-derived directions increases
      linearly from ``svd_ratio_init`` to ``svd_ratio_final`` over optimization.

    Example::

        from polystep import AdaptiveSubspace, ParamLayout

        model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))
        sub = AdaptiveSubspace.auto_from_params(model)

        # Initialize projection (store in optimizer state)
        P = sub.init_projection(device='cuda')

        # Each iteration: rotate, apply perturbation, absorb
        P = sub.rotate(P, step=i, total_steps=100, displacement_history=disp)
        perturbed_sd = sub.apply_perturbation(P, base_sd, coords)
        base_sd, coords = sub.absorb(P, base_sd, coords)

    Attributes:
        full_dim: Total flattened parameter count.
        subspace_dim: Number of subspace coordinates (rank).
        compression_ratio: subspace_dim / full_dim.
        rotation_mode: 'random' or 'displacement' (default 'displacement').
        svd_ratio_init: Starting SVD ratio for displacement mode (default 0.0).
        svd_ratio_final: Ending SVD ratio for displacement mode (default 0.5).
        displacement_history_size: Rolling window size for displacement
            history (default 5).
        absorb_mode: 'stagnation' or 'periodic' (default 'stagnation').
        absorb_patience: Steps of stagnation before absorb (default 20).
        absorb_interval: Periodic absorb interval; 0 = disabled (default 0).
        rotation_interval: Steps between basis rotations (default 1, every step).
            Each rotation costs an absorb plus a QR/SVD over ``(full_dim, subspace_dim)``,
            which dominates the step at large ``full_dim``. Raising it trades basis
            freshness for that cost.
    """

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
        """Generate initial random orthogonal projection matrix.

        Creates P of shape ``(full_dim, subspace_dim)`` with orthonormal
        columns via QR decomposition of a random Gaussian matrix. The sign
        ambiguity of QR is resolved by making the diagonal of R positive.

        Args:
            generator: Optional torch.Generator for reproducibility.
            device: Target device for the projection matrix. If None, uses CPU.
            dtype: Optional dtype for projection matrix. If None, uses float32.
                Use bfloat16 for mixed precision mode to reduce memory.

        Returns:
            Projection matrix P with shape ``(full_dim, subspace_dim)``
            satisfying ``P.T @ P = I``.
        """
        # Default to float32 if not specified
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
        """Create an orthogonal matrix via QR with sign correction.

        Args:
            rows: Number of rows (full_dim).
            cols: Number of columns (subspace_dim). Must be <= rows.
            device: Target device.
            dtype: Target dtype.
            generator: Optional PRNG generator.

        Returns:
            Orthogonal matrix of shape (rows, cols).
        """
        # QR of the tall (rows, cols) matrix dominates the per-step cost, so keep it on
        # the GPU when the target is CUDA. bf16 QR is unsupported: decompose in fp32.
        target_device = torch.device(device)
        qr_device = target_device if target_device.type == "cuda" else torch.device("cpu")
        Z = _draw_basis_gaussian(rows, cols, qr_device, torch.float32, generator)
        P, R = thin_qr(Z)
        # Fix sign ambiguity: positive diagonal in R (replace zeros with 1).
        d = torch.sign(torch.diagonal(R))
        d[d == 0] = 1.0
        P = (P * d)[:, :cols]  # slice in case QR returned a full square Q
        if dtype != torch.float32:
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
        """Rotate the projection basis according to the configured mode.

        For ``'random'`` mode, draws an entirely new QR-orthogonalized basis.
        For ``'displacement'`` mode, uses SVD of the displacement history to
        keep productive directions and fills the remainder with random.

        Args:
            projection: Current projection matrix P of shape
                ``(full_dim, subspace_dim)``.
            step: Current optimization step (0-indexed).
            total_steps: Total number of optimization steps.
            displacement_history: Optional tensor of shape
                ``(history_len, subspace_dim)`` with recent displacement
                vectors in subspace coordinates. Required for displacement
                mode; if None, falls back to random rotation.
            generator: Optional torch.Generator for reproducibility.
            history_is_full: True when ``displacement_history`` is already in full
                parameter space ``(history_len, full_dim)``, so each row keeps the
                basis it was measured in instead of being reprojected.

        Returns:
            New projection matrix P_new of shape ``(full_dim, subspace_dim)``
            with orthonormal columns.
        """
        device = projection.device
        dtype = projection.dtype

        # Fall back to random if displacement mode lacks history
        use_random = (
            self.rotation_mode == "random" or displacement_history is None or displacement_history.shape[0] == 0
        )

        if not use_random:
            # Check if displacement history has meaningful magnitude and is finite
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
        """Draw entirely new QR-orthogonalized basis.

        Args:
            device: Target device.
            dtype: Target dtype.
            generator: Optional PRNG generator.

        Returns:
            New orthogonal projection of shape (full_dim, subspace_dim).
        """
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
        """Rotate basis using SVD of displacement history.

        Projects displacement history to full parameter space, computes SVD
        to find productive directions, keeps the top ``k_svd`` singular
        vectors, and fills the remaining ``k_random`` directions with random
        vectors. The combined matrix is QR-orthogonalized.

        Args:
            projection: Current projection P of shape (full_dim, subspace_dim).
            displacement_history: Shape (history_len, subspace_dim).
            svd_ratio: Fraction of subspace_dim to fill with SVD directions.
            device: Target device.
            dtype: Target dtype.
            generator: Optional PRNG generator.

        Returns:
            New orthogonal projection of shape (full_dim, subspace_dim).
        """
        # Asking for no SVD directions means a fresh random basis, and skips the SVD.
        if svd_ratio <= 0.0:
            return self._rotate_random(device, dtype, generator)

        k_svd = max(1, int(svd_ratio * self.subspace_dim))
        k_random = self.subspace_dim - k_svd

        # D_full: (full_dim, history_len). A history already in full parameter space
        # is used as-is; each row then carries the basis it was measured in, instead of
        # being re-projected through whichever basis happens to be current. The caller
        # states which frame it holds: inferring it from the shape picks the wrong
        # branch whenever subspace_dim == full_dim.
        if history_is_full:
            D_full = displacement_history.T
        else:
            D_full = projection @ displacement_history.T

        # Guard against non-finite values from numerical issues
        if not torch.isfinite(D_full).all():
            return self._rotate_random(device, dtype, generator)

        compute_dtype = decomposition_dtype(dtype)
        D_full = D_full.to(compute_dtype)

        # pca_lowrank draws from the global RNG and takes no generator, so a seeded
        # run would still depend on torch.manual_seed; the history is short enough that
        # the full SVD costs the same. center=False: centering each displacement is a
        # different operator, not an approximation of the else-branch.
        if generator is None and k_svd < min(D_full.shape) // 2 and min(D_full.shape) > 6:
            # Randomized SVD: faster when k_svd << rank
            U_top, S_top, V_top = torch.pca_lowrank(D_full, q=k_svd, center=False, niter=2)
        else:
            U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
            k_svd = min(k_svd, U.shape[1])
            U_top = U[:, :k_svd]
        k_random = self.subspace_dim - k_svd

        # Random directions for the remainder
        Z_random = _draw_basis_gaussian(self.full_dim, k_random, device, compute_dtype, generator)
        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        # Concatenate SVD directions + random, then QR-orthogonalize
        combined = torch.cat([U_top, Z_random], dim=1)
        P_new, R = thin_qr(combined)
        # Fix sign ambiguity
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
        """Reconstruct full state_dict from base params + subspace coords.

        Computes ``delta_flat = P @ coords`` in full parameter space, then
        slices and reshapes per entry to reconstruct individual parameters.

        Args:
            projection: Projection matrix P of shape (full_dim, subspace_dim),
                or SparseRandomProjection instance for sparse mode.
            base_sd: Base state_dict with original parameter values.
            flat_subspace: 1D subspace coordinate vector of shape (subspace_dim,).

        Returns:
            New state_dict with perturbed parameters.
        """
        # Handle sparse projection
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
        """Vectorized reconstruction for N probe points.

        Computes ``delta_batch = batch @ P.T`` to get (N, full_dim) deltas, then
        adds each entry's base into its slice in place and hands out a view of
        it, so the entries alias one buffer and callers must not write to them.

        Args:
            projection: Projection matrix P of shape (full_dim, subspace_dim),
                or SparseRandomProjection instance for sparse mode.
            base_sd: Base state_dict with original parameter values.
            flat_subspace_batch: 2D tensor of shape (N, subspace_dim).

        Returns:
            Dict ``{key: (N, *original_shape)}`` with batched perturbed params.
        """
        # Handle sparse projection
        from .projection import SparseRandomProjection

        if isinstance(projection, SparseRandomProjection):
            # project() returns a transposed sparse-mm result. One copy here beats
            # handing the evaluator a tensor strided by N along the parameter axis.
            delta_batch = projection.project(flat_subspace_batch).contiguous()  # (N, full_dim)
        else:
            # (N, subspace_dim) @ (subspace_dim, full_dim) -> (N, full_dim)
            delta_batch = flat_subspace_batch @ projection.T  # (N, full_dim)
        result: Dict[str, torch.Tensor] = {}
        for spec in self._entry_specs:
            delta_chunk = delta_batch[:, spec.flat_start : spec.flat_end]
            delta_chunk.add_(base_sd[spec.entry_key].reshape(1, -1))
            # unflatten, not reshape: the slice is contiguous inside each row, so
            # this is a view. reshape would copy the whole (N, num_params).
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
        """Create an AdaptiveSubspace from an nn.Module with auto rank.

        Computes the total parameter count, selects a subspace rank based
        on the compression target (clamped to [min_rank, max_rank]), and
        builds entry specs from the model's ``ParamLayout``.

        Args:
            model: Any PyTorch module.
            compression_target: Target ratio of subspace_dim / full_dim.
            min_rank: Minimum subspace dimension.
            max_rank: Maximum subspace dimension.
            **kwargs: Additional keyword arguments passed to the constructor
                (e.g., rotation_mode, svd_ratio_init, svd_ratio_final).

        Returns:
            AdaptiveSubspace configured for the model.
        """
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
        """Create an AdaptiveSubspace from a ParamLayout with explicit rank.

        Args:
            layout: ParamLayout describing the model's parameter structure.
            rank: Subspace dimension (will be clamped to full_dim).
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            AdaptiveSubspace configured for the layout.
        """
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
        """Build EntrySpec list from a ParamLayout.

        Maps each parameter entry to a contiguous slice of the global flat
        vector of size full_dim (= layout.total_params).

        Args:
            layout: ParamLayout with entry metadata.

        Returns:
            List of EntrySpec for each parameter entry.
        """
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
