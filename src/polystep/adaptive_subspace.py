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
- ``'ot_bias'`` - biases a fraction of columns toward high-transport
  directions extracted from the OT plan (falls back to random when the
  full-dim layout doesn't match the particle layout).

``absorb()`` folds the current subspace perturbation into the base
weights and zeros the subspace vector. Combined with rotation each
iteration explores a fresh subspace centered on the current best.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .transform import ParamLayout


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
class AdaptiveSubspace:
    """Adaptive subspace compression with rotating orthogonal projection.

    Stores static configuration only. The projection matrix P lives in
    optimizer state (``SolverState.projection``) and is passed as an
    argument to all methods.

    Two rotation modes are supported:

    - ``'random'``: Draws entirely new QR-orthogonalized basis each call
      to ``rotate()``. Simple but effective -- equivalent to random search
      in a new subspace each iteration.

    - ``'displacement'``: Uses SVD of recent displacement history to retain
      productive directions. The fraction of SVD-derived directions increases
      linearly from ``svd_ratio_init`` to ``svd_ratio_final`` over optimization.

    - ``'ot_bias'``: Biases projection toward high-transport directions from
      the OT plan. The fraction of OT-derived directions is controlled by
      ``ot_bias_ratio``. Falls back to random when OT info is not available.

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
        rotation_mode: 'random', 'displacement', or 'ot_bias' (default 'displacement').
        svd_ratio_init: Starting SVD ratio for displacement mode (default 0.0).
        svd_ratio_final: Ending SVD ratio for displacement mode (default 0.5).
        ot_bias_ratio: Fraction of subspace from OT directions in 'ot_bias' mode
            (default 0.3).
        displacement_history_size: Rolling window size for displacement
            history (default 5).
        absorb_mode: 'stagnation' or 'periodic' (default 'stagnation').
        absorb_patience: Steps of stagnation before absorb (default 20).
        absorb_interval: Periodic absorb interval; 0 = disabled (default 0).
    """

    full_dim: int
    subspace_dim: int
    compression_ratio: float = 0.0
    rotation_mode: str = "displacement"
    svd_ratio_init: float = 0.0
    svd_ratio_final: float = 0.5
    ot_bias_ratio: float = 0.3
    displacement_history_size: int = 5
    absorb_mode: str = "stagnation"
    absorb_patience: int = 20
    absorb_interval: int = 0
    _entry_specs: Tuple[EntrySpec, ...] = ()

    def __post_init__(self) -> None:
        if self.compression_ratio == 0.0 and self.full_dim > 0:
            object.__setattr__(
                self, "compression_ratio", self.subspace_dim / self.full_dim
            )

    # ------------------------------------------------------------------
    # Projection initialization
    # ------------------------------------------------------------------

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
            self.full_dim, self.subspace_dim, device=projection_device, dtype=projection_dtype,
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
        # Generate on CPU (torch.linalg.qr is faster on CPU for moderate sizes)
        # Handle generator device mismatch: CUDA generator can't be used with CPU tensor.
        # Create a CPU generator seeded from the CUDA generator's state for reproducibility.
        gen_device = "cpu"
        if generator is not None and generator.device.type != "cpu":
            cpu_gen = torch.Generator(device="cpu")
            cpu_gen.manual_seed(generator.initial_seed())
            generator = cpu_gen
        # QR decomposition requires FP32 on CPU (BF16 not supported for geqrf_cpu)
        # Generate in FP32, compute QR, then convert to target dtype
        Z = torch.randn(rows, cols, generator=generator, device=gen_device, dtype=torch.float32)
        P, R = torch.linalg.qr(Z)
        # Fix sign ambiguity: ensure positive diagonal in R
        d = torch.sign(torch.diagonal(R))
        # Replace zeros with 1 (degenerate case)
        d[d == 0] = 1.0
        P = P * d
        # Slice to requested columns (QR may produce full Q for square input)
        P = P[:, :cols]
        # Convert to target dtype (e.g., BF16 for mixed precision)
        if dtype != torch.float32:
            P = P.to(dtype=dtype)
        if str(device) != gen_device:
            P = P.to(device=device)
        return P

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def rotate(
        self,
        projection: torch.Tensor,
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        # OT-bias mode inputs
        transport_matrix: Optional[torch.Tensor] = None,
        X_vertices: Optional[torch.Tensor] = None,
        X_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate the projection basis according to the configured mode.

        For ``'random'`` mode, draws an entirely new QR-orthogonalized basis.
        For ``'displacement'`` mode, uses SVD of the displacement history to
        keep productive directions and fills the remainder with random.
        For ``'ot_bias'`` mode, uses high-transport directions from the OT
        plan to bias a fraction of the projection basis.

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
            transport_matrix: OT transport plan, shape (num_particles, num_vertices).
                Required for ot_bias mode; if None, falls back to random.
            X_vertices: Polytope vertex positions for each particle,
                shape (num_particles, num_vertices, particle_dim).
                Required for ot_bias mode.
            X_current: Current particle positions, shape (num_particles, particle_dim).
                Required for ot_bias mode.

        Returns:
            New projection matrix P_new of shape ``(full_dim, subspace_dim)``
            with orthonormal columns.
        """
        device = projection.device
        dtype = projection.dtype

        # Handle ot_bias mode
        if self.rotation_mode == "ot_bias":
            has_ot_info = (
                transport_matrix is not None
                and X_vertices is not None
                and X_current is not None
            )
            if has_ot_info:
                return self._rotate_ot_bias(
                    transport_matrix, X_vertices, X_current,
                    device, dtype, generator,
                )
            else:
                # Fallback to random when OT info not available
                return self._rotate_random(device, dtype, generator)

        # Fall back to random if displacement mode lacks history
        use_random = (
            self.rotation_mode == "random"
            or displacement_history is None
            or displacement_history.shape[0] == 0
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
                projection, displacement_history, svd_ratio,
                device, dtype, generator,
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
            self.full_dim, self.subspace_dim, device=device, dtype=dtype,
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
        k_svd = max(1, int(svd_ratio * self.subspace_dim))
        k_random = self.subspace_dim - k_svd

        # Project displacement history to full parameter space
        # displacement_history: (history_len, subspace_dim)
        # projection: (full_dim, subspace_dim)
        # D_full: (full_dim, history_len)
        D_full = projection @ displacement_history.T

        # Guard against non-finite values from numerical issues
        if not torch.isfinite(D_full).all():
            return self._rotate_random(device, dtype, generator)

        # SVD of the full-space displacement matrix
        if k_svd < min(D_full.shape) // 2 and min(D_full.shape) > 6:
            # Randomized SVD: faster when k_svd << rank
            U_top, S_top, V_top = torch.pca_lowrank(D_full, q=k_svd, niter=2)
        else:
            U, S, Vh = torch.linalg.svd(D_full, full_matrices=False)
            k_svd = min(k_svd, U.shape[1])
            U_top = U[:, :k_svd]
        k_random = self.subspace_dim - k_svd

        # Generate random directions for the remainder
        gen_device = "cpu"
        # Handle generator device mismatch: create CPU generator for reproducibility
        gen_to_use = generator
        if generator is not None and generator.device.type != "cpu":
            gen_to_use = torch.Generator(device="cpu")
            gen_to_use.manual_seed(generator.initial_seed())
        Z_random = torch.randn(
            self.full_dim, k_random,
            generator=gen_to_use, device=gen_device, dtype=dtype,
        )
        if str(device) != gen_device:
            Z_random = Z_random.to(device=device)
        if U_top.device != Z_random.device:
            U_top = U_top.to(device=Z_random.device)

        # Concatenate SVD directions + random, then QR-orthogonalize
        combined = torch.cat([U_top, Z_random], dim=1)
        P_new, R = torch.linalg.qr(combined)
        # Fix sign ambiguity
        d = torch.sign(torch.diagonal(R))
        d[d == 0] = 1.0
        P_new = P_new * d
        P_new = P_new[:, :self.subspace_dim]

        if P_new.device != device:
            P_new = P_new.to(device=device)

        return P_new

    @torch.inference_mode()
    def _rotate_ot_bias(
        self,
        transport_matrix: torch.Tensor,
        X_vertices: torch.Tensor,
        X_current: torch.Tensor,
        device: str | torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Rotate basis using OT-informed high-transport directions.

        Extracts directions where particles moved the most mass according to
        the transport plan, projects these particle-space directions to full
        parameter space (by tiling to match full_dim), then combines with
        random directions via QR orthogonalization.

        The fraction of directions from OT bias is controlled by ot_bias_ratio.

        Args:
            transport_matrix: OT transport plan, shape (num_particles, num_vertices).
            X_vertices: Polytope vertices, shape (num_particles, num_vertices, particle_dim).
            X_current: Current particles, shape (num_particles, particle_dim).
            device: Target device.
            dtype: Target dtype.
            generator: Optional PRNG generator.

        Returns:
            New orthogonal projection of shape (full_dim, subspace_dim).
        """
        from .cma import compute_ot_bias_directions

        # Number of directions to allocate to OT bias vs random.
        k_ot = max(1, int(self.ot_bias_ratio * self.subspace_dim))

        # Get high-transport directions in particle space.
        ot_dirs = compute_ot_bias_directions(
            transport_matrix, X_vertices, X_current, top_k=k_ot,
        )
        # ot_dirs: (k_ot_actual, particle_dim)
        k_ot_actual = ot_dirs.shape[0]

        # Lifting particle-space directions to ``full_dim`` only makes sense
        # when the full vector is a concatenation of equal-size particle
        # slots (full_dim == num_particles * particle_dim). For real
        # parameter layouts ``full_dim`` is the total number of trainable
        # parameters; tiling would map an OT direction to arbitrary layer
        # weights and lose its meaning. Fall back to pure random in that
        # case.
        num_particles = X_current.shape[0]
        particle_dim = ot_dirs.shape[1]
        if num_particles * particle_dim == self.full_dim:
            ot_dirs_full = ot_dirs.unsqueeze(1).expand(-1, num_particles, -1)
            ot_dirs_full = ot_dirs_full.reshape(k_ot_actual, -1)
            ot_cols = ot_dirs_full.T  # (full_dim, k_ot_actual)
        else:
            ot_cols = None
            k_ot_actual = 0
        k_random = self.subspace_dim - k_ot_actual

        # Generate random directions for the remainder
        gen_device = "cpu"
        # Handle generator device mismatch: create CPU generator for reproducibility
        gen_to_use = generator
        if generator is not None and generator.device.type != "cpu":
            gen_to_use = torch.Generator(device="cpu")
            gen_to_use.manual_seed(generator.initial_seed())
        Z_random = torch.randn(
            self.full_dim, k_random,
            generator=gen_to_use, device=gen_device, dtype=dtype,
        )
        if str(device) != gen_device:
            Z_random = Z_random.to(device=device)
        if ot_cols is not None:
            if ot_cols.device != Z_random.device:
                ot_cols = ot_cols.to(device=Z_random.device)
            combined = torch.cat([ot_cols, Z_random], dim=1)
        else:
            combined = Z_random
        P_new, R = torch.linalg.qr(combined)
        # Fix sign ambiguity
        d = torch.sign(torch.diagonal(R))
        d[d == 0] = 1.0
        P_new = P_new * d
        P_new = P_new[:, :self.subspace_dim]

        if P_new.device != device:
            P_new = P_new.to(device=device)

        return P_new

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
    # Core interface methods (match LinearSubspace contract)
    # ------------------------------------------------------------------

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
            delta_chunk = delta_flat[spec.flat_start:spec.flat_end]
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

        Computes ``delta_batch = batch @ P.T`` to get (N, full_dim) deltas,
        then slices and reshapes per entry.

        Args:
            projection: Projection matrix P of shape (full_dim, subspace_dim),
                or SparseRandomProjection instance for sparse mode.
            base_sd: Base state_dict with original parameter values.
            flat_subspace_batch: 2D tensor of shape (N, subspace_dim).

        Returns:
            Dict ``{key: (N, *original_shape)}`` with batched perturbed params.
        """
        N = flat_subspace_batch.shape[0]
        # Handle sparse projection
        from .projection import SparseRandomProjection
        if isinstance(projection, SparseRandomProjection):
            # Sparse projection: use project() method for batch
            delta_batch = projection.project(flat_subspace_batch)  # (N, full_dim)
        else:
            # (N, subspace_dim) @ (subspace_dim, full_dim) -> (N, full_dim)
            delta_batch = flat_subspace_batch @ projection.T  # (N, full_dim)
        result: Dict[str, torch.Tensor] = {}
        for spec in self._entry_specs:
            delta_chunk = delta_batch[:, spec.flat_start:spec.flat_end]
            base = base_sd[spec.entry_key]
            result[spec.entry_key] = (
                base.unsqueeze(0) + delta_chunk.reshape(N, *spec.original_shape)
            )
        return result

    def absorb(
        self,
        projection,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fold subspace perturbation into base weights and zero the subspace.

        This is the standard absorb operation: apply the current perturbation
        to base weights, then return zeroed subspace coordinates. Combined
        with rotation, this recenters the subspace around updated parameters.

        Args:
            projection: Projection matrix P of shape (full_dim, subspace_dim),
                or SparseRandomProjection instance for sparse mode.
            base_sd: Base state_dict to absorb perturbation into.
            flat_subspace: Current subspace vector of shape (subspace_dim,).

        Returns:
            Tuple of (new_base_sd, zeroed_subspace_vector).
        """
        new_sd = self.apply_perturbation(projection, base_sd, flat_subspace)
        return new_sd, torch.zeros_like(flat_subspace)

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Check whether an absorb-and-rotate should be triggered.

        Args:
            stagnation_count: Number of consecutive steps without improvement.
            iteration: Current iteration number (0-indexed).

        Returns:
            True if absorb should be triggered based on the configured mode.
        """
        if self.absorb_mode == "stagnation":
            return stagnation_count >= self.absorb_patience
        elif self.absorb_mode == "periodic" and self.absorb_interval > 0:
            return iteration > 0 and iteration % self.absorb_interval == 0
        return False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

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
            specs.append(EntrySpec(
                entry_key=entry.key,
                original_shape=entry.shape,
                num_params=entry.numel,
                flat_start=entry.offset,
                flat_end=entry.offset + entry.numel,
            ))
        return specs
