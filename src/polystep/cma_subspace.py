"""CMAAdaptiveSubspace: sep-CMA-ES covariance adaptation on top of AdaptiveSubspace.

The separable variant keeps a diagonal covariance instead of a full one, which drops
memory from O(n^2) to O(n) and the update from O(n^3) to O(n). That is what makes it
usable at the subspace dimensions neural network training produces.

State (``p_c``, ``p_sigma``, ``C_diag``, ``sigma``, ``generation``) lives in
``SolverState``, not on this class, so it checkpoints with the rest of the optimizer
and stays JIT-compatible. This class holds the static hyperparameters and the methods
that operate on that state.

Example::

    base = AdaptiveSubspace.auto_from_params(model)
    cma_sub = CMAAdaptiveSubspace.from_adaptive_subspace(base)
    cma_state = cma_sub.init_cma_state(device="cuda")
    P_scaled = cma_sub.apply_covariance_scaling(P, cma_state["C_diag"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch

from .adaptive_subspace import AdaptiveSubspace

from .cma import compute_cma_hyperparameters

if TYPE_CHECKING:
    pass


def default_mu_eff(subspace_dim: int) -> float:
    """Fallback effective population size when the polytope size is unknown.

    ``mu_eff`` is ``1 / sum(w^2)`` over the recombination weights, which are the OT
    transport row, so its ceiling is the vertex count rather than the subspace
    dimension. ``PolyStepOptimizer`` knows the vertex count and overrides this; the
    dimension-derived value here only applies to a standalone subspace object.

    Args:
        subspace_dim: Subspace dimension ``n``.

    Returns:
        ``mu_eff`` capped per the Hansen tutorial's ``mu ~ lambda/2 ~ O(sqrt(n))``.
    """
    return max(1.0, min(subspace_dim / 4.0, 5.0 * math.sqrt(subspace_dim)))


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
    # 0.0 is a "derive me" sentinel, like the learning rates above. A literal 1.0
    # would be a legal-looking value that silently zeroes c_mu.
    mu_eff: float = 0.0
    # Numerical stability bounds
    cov_min: float = 1e-6
    cov_max: float = 1e6

    def __post_init__(self) -> None:
        """Fill unset CMA hyperparameters from the Hansen formulas.

        Direct construction leaves them at their ``0.0`` sentinel, which makes CMA
        inert (``c_1 = c_mu = c_c = 0`` never updates the covariance) and divides by
        zero in the CSA update. Filling them here makes every construction path
        behave like :meth:`from_adaptive_subspace`.
        """
        # Recorded so PolyStepOptimizer knows whether to override mu_eff with the
        # polytope vertex count or respect a value the caller chose.
        self._mu_eff_explicit = self.mu_eff > 0.0
        if self.d_sigma == 0.0:
            n = self.base.subspace_dim
            mu_eff = self.mu_eff if self.mu_eff > 0.0 else default_mu_eff(n)
            hyperparams = compute_cma_hyperparameters(n, mu_eff)
            self.c_c = self.c_c or hyperparams["c_c"]
            self.c_sigma = self.c_sigma or hyperparams["c_sigma"]
            self.c_1 = self.c_1 or hyperparams["c_1"]
            self.c_mu = self.c_mu or hyperparams["c_mu"]
            self.d_sigma = hyperparams["d_sigma"]
            self.expected_norm = self.expected_norm or hyperparams["expected_norm"]
            self.mu_eff = mu_eff

    @property
    def full_dim(self) -> int:
        """Total flattened parameter count (delegated to base)."""
        return self.base.full_dim

    @property
    def displacement_history_size(self) -> int:
        """Rolling displacement-history length (delegated to base)."""
        return self.base.displacement_history_size

    @property
    def absorb_mode(self) -> str:
        """Absorb trigger mode (delegated to base)."""
        return self.base.absorb_mode

    @property
    def absorb_patience(self) -> int:
        """Stagnation steps before a stagnation absorb (delegated to base)."""
        return self.base.absorb_patience

    @property
    def absorb_interval(self) -> int:
        """Steps between periodic absorbs (delegated to base)."""
        return self.base.absorb_interval

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
    ) -> torch.Tensor:
        """Rotate projection basis (delegated to base).

        Args:
            projection: Current projection matrix P.
            step: Current optimization step.
            total_steps: Total number of optimization steps.
            displacement_history: Optional displacement history tensor.
            generator: Optional torch.Generator for reproducibility.

        Returns:
            New projection matrix P_new.
        """
        return self.base.rotate(projection, step, total_steps, displacement_history, generator)

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
        C_diag_clamped = torch.clamp(C_diag, min=self.cov_min, max=self.cov_max)
        sqrt_C = torch.sqrt(C_diag_clamped)
        # Write into a cached buffer: this runs once per step and the result is a full
        # (full_dim, subspace_dim) tensor, 1 GB in fp32 at full_dim=500K, subspace_dim=512.
        out = getattr(self, "_scaled_projection_buf", None)
        if (
            out is None
            or out.shape != projection.shape
            or out.dtype != projection.dtype
            or out.device != projection.device
        ):
            out = torch.empty_like(projection)
            self._scaled_projection_buf = out
        return torch.mul(projection, sqrt_C.unsqueeze(0), out=out)

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
            mu_eff: Effective population size. If None, falls back to
                :func:`default_mu_eff`; ``PolyStepOptimizer`` overrides that with the
                polytope vertex count, which is the real ceiling on ``1/sum(w^2)``.
            cov_min: Minimum allowed C_diag entry (numerical stability).
            cov_max: Maximum allowed C_diag entry (numerical stability).

        Returns:
            CMAAdaptiveSubspace wrapping the base instance.
        """
        # 0.0 lets __post_init__ resolve mu_eff and record it as not caller-chosen, so
        # PolyStepOptimizer is free to substitute the polytope vertex count.
        return cls(base=base, mu_eff=0.0 if mu_eff is None else mu_eff, cov_min=cov_min, cov_max=cov_max)

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
        return cls.from_adaptive_subspace(base, mu_eff=mu_eff, cov_min=cov_min, cov_max=cov_max)
