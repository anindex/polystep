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
from typing import Dict, Optional, Tuple

import torch

from .adaptive_subspace import AdaptiveSubspace
from .cma import compute_cma_hyperparameters


def default_mu_eff(subspace_dim: int) -> float:
    """Fallback effective population size when the polytope size is unknown.

    ``mu_eff`` is ``1 / sum(w^2)`` over the recombination weights, which are the OT
    transport row, so its ceiling is the vertex count rather than the subspace
    dimension. ``PolyStepOptimizer`` knows the vertex count and overrides this; the
    dimension-derived value here only applies to a standalone subspace object. Capped per
    the Hansen tutorial's ``mu ~ lambda/2 ~ O(sqrt(n))``.
    """
    return max(1.0, min(subspace_dim / 4.0, 5.0 * math.sqrt(subspace_dim)))


_RATE_KEYS = ("c_c", "c_sigma", "c_1", "c_mu")


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
    # 0.0 is a "derive me" sentinel, like the learning rates above. A literal 1.0
    # would be a legal-looking value that silently zeroes c_mu.
    mu_eff: float = 0.0
    # Numerical stability bounds
    cov_min: float = 1e-6
    cov_max: float = 1e6

    def __post_init__(self) -> None:
        """Fill unset CMA hyperparameters from the Hansen formulas.

        Direct construction leaves them at their ``0.0`` sentinel, which makes CMA
        inert: ``c_1 = c_mu = c_c = 0`` never updates the covariance and ``c_sigma = 0``
        freezes the evolution path. Filling them here makes every construction path
        behave like :meth:`from_adaptive_subspace`.
        """
        # Recorded so PolyStepOptimizer knows which values the caller chose and
        # which it may derive itself.
        self._mu_eff_explicit = self.mu_eff > 0.0
        self._explicit_rates = {name: value for name in _RATE_KEYS if (value := getattr(self, name)) != 0.0}
        n = self.base.subspace_dim
        mu_eff = self.mu_eff if self._mu_eff_explicit else default_mu_eff(n)
        hyperparams = compute_cma_hyperparameters(n, mu_eff)
        # Each coefficient fills independently: gating the whole block on one of them
        # left the others at their sentinel.
        for name in _RATE_KEYS:
            setattr(self, name, getattr(self, name) or hyperparams[name])
        self.mu_eff = mu_eff

    # Everything below the CMA state forwards to ``base`` unchanged. Written out
    # rather than routed through __getattr__ so the subspace interface stays
    # greppable and type-checkable.

    full_dim = property(lambda self: self.base.full_dim)
    subspace_dim = property(lambda self: self.base.subspace_dim)
    compression_ratio = property(lambda self: self.base.compression_ratio)
    rotation_mode = property(lambda self: self.base.rotation_mode)
    rotation_interval = property(lambda self: self.base.rotation_interval)
    displacement_history_size = property(lambda self: self.base.displacement_history_size)
    absorb_mode = property(lambda self: self.base.absorb_mode)
    absorb_patience = property(lambda self: self.base.absorb_patience)
    absorb_interval = property(lambda self: self.base.absorb_interval)

    def init_projection(
        self,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self.base.init_projection(generator=generator, device=device, dtype=dtype)

    def rotate(
        self,
        projection: torch.Tensor,
        step: int,
        total_steps: int,
        displacement_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        history_is_full: bool = False,
    ) -> torch.Tensor:
        return self.base.rotate(projection, step, total_steps, displacement_history, generator, history_is_full)

    def apply_perturbation(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.base.apply_perturbation(projection, base_sd, flat_subspace)

    def reconstruct_batch(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.base.reconstruct_batch(projection, base_sd, flat_subspace_batch)

    def absorb(
        self,
        projection: torch.Tensor,
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.base.absorb(projection, base_sd, flat_subspace)

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
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
        # Cached buffer: runs once per step and the result is a full
        # (full_dim, subspace_dim) tensor, too large to reallocate.
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
        """Fresh ``p_c``, ``p_sigma``, ``C_diag``, each ``(subspace_dim,)``, for SolverState.

        Both evolution paths start at zero and the covariance isotropic at one.
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
        """Wrap an ``AdaptiveSubspace``, deriving the CMA constants from its dimension.

        ``c_c``, ``c_sigma``, ``c_1`` and ``c_mu`` follow the Hansen formulas.
        ``mu_eff=None`` falls back to :func:`default_mu_eff`;
        ``PolyStepOptimizer`` overrides that with the polytope vertex count, the real
        ceiling on ``1/sum(w^2)``. ``cov_min``/``cov_max`` clamp ``C_diag``.
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
