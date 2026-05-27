"""CMAAdaptiveSubspace: CMA-ES covariance adaptation on top of AdaptiveSubspace.

Wraps an ``AdaptiveSubspace`` with the separable CMA-ES (sep-CMA-ES) variant
of the Covariance Matrix Adaptation Evolution Strategy. sep-CMA-ES uses a
diagonal covariance instead of a full one, reducing memory from O(n^2) to O(n)
and compute from O(n^3) to O(n) per update -- which is what makes it usable
in the high-dimensional subspaces that show up in neural network training.

Key CMA-ES concepts integrated:

1. **Evolution Paths (p_c, p_sigma)**: Cumulative sums of displacement directions
   over generations. These track the "momentum" of the search and are used to
   update covariance and step-size.

2. **Diagonal Covariance (C_diag)**: Per-dimension scaling of the search distribution.
   Dimensions that consistently show movement get higher variance, allowing the
   optimizer to stretch the search ellipsoid along productive directions.

3. **Cumulative Step-size Adaptation (CSA)**: Uses p_sigma to detect if steps are
   too short (should increase sigma) or too long (should decrease sigma) relative
   to what would be expected under random selection.

4. **Covariance Scaling**: The projection matrix P can be scaled by sqrt(C_diag)
   to modify the effective search distribution in parameter space.

Design Decision:
    CMA state (p_c, p_sigma, C_diag, sigma, generation) lives in ``SolverState``,
    not in this class. This maintains JIT compatibility and enables checkpointing
    of the full optimizer state. CMAAdaptiveSubspace only stores static config
    (hyperparameters) and provides methods that operate on the state.

Example::

    from polystep import AdaptiveSubspace, CMAAdaptiveSubspace, SolverState
    import torch.nn as nn

    model = nn.Linear(100, 10)
    base = AdaptiveSubspace.auto_from_params(model)
    cma_sub = CMAAdaptiveSubspace.from_adaptive_subspace(base)

    # Initialize CMA state (store in SolverState)
    cma_state = cma_sub.init_cma_state(device='cuda')
    # cma_state = {'p_c': ..., 'p_sigma': ..., 'C_diag': ...}

    # Apply covariance scaling to projection
    P_scaled = cma_sub.apply_covariance_scaling(P, cma_state['C_diag'])
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch

from .adaptive_subspace import AdaptiveSubspace

from .cma import compute_cma_hyperparameters

if TYPE_CHECKING:
    from .transform import ParamLayout


@dataclass
class CMAAdaptiveSubspace:
    """CMA-ES enhanced adaptive subspace via composition.

    Wraps an ``AdaptiveSubspace`` instance and adds CMA-ES hyperparameters
    and covariance-related methods. The underlying AdaptiveSubspace handles
    projection initialization, rotation, and core reconstruction operations.

    CMA-ES hyperparameters are auto-computed from subspace_dim using the
    standard Hansen formulas (see ``compute_cma_hyperparameters``).

    Attributes:
        base: The wrapped AdaptiveSubspace instance.
        c_c: Learning rate for covariance evolution path update.
        c_sigma: Learning rate for step-size evolution path update.
        c_1: Learning rate for rank-one covariance update.
        c_mu: Learning rate for rank-mu covariance update.
        d_sigma: Damping factor for step-size adaptation.
        expected_norm: Expected length of N(0,I) random vector (chi_n).
        mu_eff: Effective population size for weighted recombination.
        cov_min: Minimum allowed value for C_diag entries.
        cov_max: Maximum allowed value for C_diag entries.
    """

    base: AdaptiveSubspace
    # CMA-ES hyperparameters (auto-computed from subspace_dim)
    c_c: float = 0.0
    c_sigma: float = 0.0
    c_1: float = 0.0
    c_mu: float = 0.0
    d_sigma: float = 0.0
    expected_norm: float = 0.0
    mu_eff: float = 1.0
    # Numerical stability bounds
    cov_min: float = 1e-6
    cov_max: float = 1e6

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def full_dim(self) -> int:
        """Total flattened parameter count (delegated to base)."""
        return self.base.full_dim

    @property
    def subspace_dim(self) -> int:
        """Subspace dimension / rank (delegated to base)."""
        return self.base.subspace_dim

    @property
    def compression_ratio(self) -> float:
        """Compression ratio: subspace_dim / full_dim (delegated to base)."""
        return self.base.compression_ratio

    @property
    def rotation_mode(self) -> str:
        """Rotation mode: 'random' or 'displacement' (delegated to base)."""
        return self.base.rotation_mode

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def init_projection(
        self,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Initialize projection matrix (delegated to base).

        Args:
            generator: Optional torch.Generator for reproducibility.
            device: Target device for the projection matrix. If None, uses CPU.
            dtype: Optional dtype for projection matrix. If None, uses float32.
                Use bfloat16 for mixed precision mode to reduce memory.

        Returns:
            Projection matrix P with shape (full_dim, subspace_dim).
        """
        return self.base.init_projection(generator=generator, device=device, dtype=dtype)

    def rotate(
        self,
        projection: torch.Tensor,
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        # OT-bias mode inputs (forwarded to base.rotate; ignored otherwise)
        transport_matrix: Optional[torch.Tensor] = None,
        X_vertices: Optional[torch.Tensor] = None,
        X_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate projection basis (delegated to base).

        Args:
            projection: Current projection matrix P.
            step: Current optimization step.
            total_steps: Total number of optimization steps.
            displacement_history: Optional displacement history tensor.
            generator: Optional torch.Generator for reproducibility.
            transport_matrix: OT transport plan (only used when the wrapped
                AdaptiveSubspace is in ``'ot_bias'`` rotation mode).
            X_vertices: Polytope vertex positions (ot_bias mode only).
            X_current: Current particle positions (ot_bias mode only).

        Returns:
            New projection matrix P_new.
        """
        return self.base.rotate(
            projection, step, total_steps, displacement_history, generator,
            transport_matrix=transport_matrix,
            X_vertices=X_vertices,
            X_current=X_current,
        )

    def apply_perturbation(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct full state_dict from base + subspace coords (delegated to base).

        Args:
            projection: Projection matrix P.
            base_sd: Base state_dict.
            flat_subspace: Subspace coordinate vector.

        Returns:
            New state_dict with perturbed parameters.
        """
        return self.base.apply_perturbation(projection, base_sd, flat_subspace)

    def reconstruct_batch(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized reconstruction for N probe points (delegated to base).

        Args:
            projection: Projection matrix P.
            base_sd: Base state_dict.
            flat_subspace_batch: Batch of subspace coordinates (N, subspace_dim).

        Returns:
            Dict with batched perturbed params {key: (N, *shape)}.
        """
        return self.base.reconstruct_batch(projection, base_sd, flat_subspace_batch)

    def absorb(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fold subspace perturbation into base weights (delegated to base).

        Args:
            projection: Projection matrix P.
            base_sd: Base state_dict.
            flat_subspace: Current subspace vector.

        Returns:
            Tuple of (new_base_sd, zeroed_subspace_vector).
        """
        return self.base.absorb(projection, base_sd, flat_subspace)

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Check whether absorb should be triggered (delegated to base).

        Args:
            stagnation_count: Consecutive steps without improvement.
            iteration: Current iteration number.

        Returns:
            True if absorb should be triggered.
        """
        return self.base.should_absorb(stagnation_count, iteration)

    # ------------------------------------------------------------------
    # CMA-specific methods
    # ------------------------------------------------------------------

    def apply_covariance_scaling(
        self,
        projection: torch.Tensor,
        C_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Scale projection columns by sqrt(C_diag) for covariance-adapted sampling.

        In CMA-ES, the search distribution is N(m, sigma^2 * C). With diagonal
        covariance C = diag(C_diag), sampling x ~ N(m, sigma^2 * C) is equivalent
        to sampling z ~ N(0, I) and computing x = m + sigma * C^{1/2} * z.

        This method applies the C^{1/2} scaling to the projection matrix, so
        that sampling in the original subspace coordinates and then projecting
        gives the covariance-scaled effect in full parameter space.

        Args:
            projection: Projection matrix P of shape (full_dim, subspace_dim).
            C_diag: Diagonal covariance entries of shape (subspace_dim,).

        Returns:
            Scaled projection P_scaled = P @ diag(sqrt(C_diag)).
        """
        # Clamp C_diag for numerical stability
        C_diag_clamped = torch.clamp(C_diag, min=self.cov_min, max=self.cov_max)
        # Scale columns: P_scaled[:, i] = P[:, i] * sqrt(C_diag[i])
        sqrt_C = torch.sqrt(C_diag_clamped)
        return projection * sqrt_C.unsqueeze(0)

    def init_cma_state(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Initialize CMA-ES state tensors.

        Creates the initial evolution paths and diagonal covariance for a fresh
        CMA-ES optimization run. These should be stored in SolverState.

        Initial values:
        - p_c: zeros (no accumulated covariance direction yet)
        - p_sigma: zeros (no accumulated step-size direction yet)
        - C_diag: ones (isotropic initial covariance)

        Args:
            device: Target device for state tensors.
            dtype: Target dtype for state tensors.

        Returns:
            Dict with keys 'p_c', 'p_sigma', 'C_diag', each of shape (subspace_dim,).
        """
        subspace_dim = self.base.subspace_dim
        return {
            "p_c": torch.zeros(subspace_dim, device=device, dtype=dtype),
            "p_sigma": torch.zeros(subspace_dim, device=device, dtype=dtype),
            "C_diag": torch.ones(subspace_dim, device=device, dtype=dtype),
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_adaptive_subspace(
        cls,
        base: AdaptiveSubspace,
        mu_eff: Optional[float] = None,
        cov_min: float = 1e-6,
        cov_max: float = 1e6,
    ) -> "CMAAdaptiveSubspace":
        """Create CMAAdaptiveSubspace by wrapping an existing AdaptiveSubspace.

        CMA-ES hyperparameters (c_c, c_sigma, c_1, c_mu, d_sigma, expected_norm)
        are automatically computed from the subspace dimension using the standard
        Hansen formulas.

        Args:
            base: The AdaptiveSubspace to wrap.
            mu_eff: Effective population size. If None, defaults to
                subspace_dim / 4 (heuristic for OT-based selection).
            cov_min: Minimum allowed C_diag entry (numerical stability).
            cov_max: Maximum allowed C_diag entry (numerical stability).

        Returns:
            CMAAdaptiveSubspace wrapping the base instance.
        """
        n = base.subspace_dim

        # Default mu_eff: fraction of subspace dim, capped per Hansen CMA tutorial.
        # Hansen recommends mu_eff ~ mu ~ lambda/2, which scales as O(sqrt(n)).
        # Cap at 5*sqrt(n) to avoid over-aggressive step-size adaptation.
        if mu_eff is None:
            mu_eff = max(1.0, min(n / 4.0, 5.0 * math.sqrt(n)))

        # Compute CMA hyperparameters from dimension and mu_eff
        hyperparams = compute_cma_hyperparameters(n, mu_eff)

        return cls(
            base=base,
            c_c=hyperparams["c_c"],
            c_sigma=hyperparams["c_sigma"],
            c_1=hyperparams["c_1"],
            c_mu=hyperparams["c_mu"],
            d_sigma=hyperparams["d_sigma"],
            expected_norm=hyperparams["expected_norm"],
            mu_eff=mu_eff,
            cov_min=cov_min,
            cov_max=cov_max,
        )

    @classmethod
    def auto_from_params(
        cls,
        model: torch.nn.Module,
        compression_target: float = 0.05,
        min_rank: int = 64,
        max_rank: int = 4096,
        mu_eff: Optional[float] = None,
        cov_min: float = 1e-6,
        cov_max: float = 1e6,
        **kwargs,
    ) -> "CMAAdaptiveSubspace":
        """Create CMAAdaptiveSubspace directly from an nn.Module.

        Convenience factory that first creates an AdaptiveSubspace with
        ``auto_from_params``, then wraps it with CMA-ES functionality.

        Args:
            model: Any PyTorch module.
            compression_target: Target ratio of subspace_dim / full_dim.
            min_rank: Minimum subspace dimension.
            max_rank: Maximum subspace dimension.
            mu_eff: Effective population size for CMA (see from_adaptive_subspace).
            cov_min: Minimum allowed C_diag entry.
            cov_max: Maximum allowed C_diag entry.
            **kwargs: Additional arguments passed to AdaptiveSubspace.auto_from_params.

        Returns:
            CMAAdaptiveSubspace configured for the model.
        """
        base = AdaptiveSubspace.auto_from_params(
            model,
            compression_target=compression_target,
            min_rank=min_rank,
            max_rank=max_rank,
            **kwargs,
        )
        return cls.from_adaptive_subspace(
            base, mu_eff=mu_eff, cov_min=cov_min, cov_max=cov_max
        )
