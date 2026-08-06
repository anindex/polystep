"""CMAAdaptiveSubspace: sep-CMA-ES covariance adaptation on top of AdaptiveSubspace."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, Optional

import torch

from .adaptive_subspace import AdaptiveSubspace
from .cma import compute_cma_hyperparameters


def default_mu_eff(subspace_dim: int) -> float:
    """Fallback effective population size when the polytope size is unknown."""
    return max(1.0, min(subspace_dim / 4.0, 5.0 * math.sqrt(subspace_dim)))


_RATE_KEYS = ("c_c", "c_sigma", "c_1", "c_mu")


@dataclass
class CMAAdaptiveSubspace(AdaptiveSubspace):
    """AdaptiveSubspace plus a diagonal covariance the sampler scales its columns by."""

    # CMA-ES hyperparameters (auto-computed from subspace_dim)
    c_c: float = 0.0
    c_sigma: float = 0.0
    c_1: float = 0.0
    c_mu: float = 0.0
    # 0.0 is a "derive me" sentinel; a literal 1.0 would silently zero c_mu.
    mu_eff: float = 0.0
    # Numerical stability bounds
    cov_min: float = 1e-6
    cov_max: float = 1e6

    def __post_init__(self) -> None:
        """Fill unset CMA hyperparameters from the Hansen formulas."""
        super().__post_init__()
        # Record which values the caller chose and which may be derived.
        self._mu_eff_explicit = self.mu_eff > 0.0
        self._explicit_rates = {name: value for name in _RATE_KEYS if (value := getattr(self, name)) != 0.0}
        n = self.subspace_dim
        mu_eff = self.mu_eff if self._mu_eff_explicit else default_mu_eff(n)
        hyperparams = compute_cma_hyperparameters(n, mu_eff)
        # Fill each independently; gating on one would leave the rest at their sentinel.
        for name in _RATE_KEYS:
            setattr(self, name, getattr(self, name) or hyperparams[name])
        self.mu_eff = mu_eff

    def apply_covariance_scaling(
        self,
        projection: torch.Tensor,
        C_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Scale projection columns by sqrt(C_diag) for covariance-adapted sampling."""
        C_diag_clamped = torch.clamp(C_diag, min=self.cov_min, max=self.cov_max)
        sqrt_C = torch.sqrt(C_diag_clamped)
        # Cached buffer: the result is a full (full_dim, subspace_dim) tensor, too large to reallocate.
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
        """Fresh ``p_c``, ``p_sigma``, ``C_diag``, each ``(subspace_dim,)``, for SolverState."""
        return {
            "p_c": torch.zeros(self.subspace_dim, device=device, dtype=dtype),
            "p_sigma": torch.zeros(self.subspace_dim, device=device, dtype=dtype),
            "C_diag": torch.ones(self.subspace_dim, device=device, dtype=dtype),
        }

    @classmethod
    def from_adaptive_subspace(
        cls,
        base: AdaptiveSubspace,
        mu_eff: Optional[float] = None,
        cov_min: float = 1e-6,
        cov_max: float = 1e6,
    ) -> "CMAAdaptiveSubspace":
        """Rebuild an ``AdaptiveSubspace`` as a CMA one, deriving the constants from its dimension."""
        carried = {f.name: getattr(base, f.name) for f in dataclass_fields(AdaptiveSubspace)}
        # 0.0 marks mu_eff as not caller-chosen, so the optimizer may substitute the vertex count.
        return cls(**carried, mu_eff=0.0 if mu_eff is None else mu_eff, cov_min=cov_min, cov_max=cov_max)
