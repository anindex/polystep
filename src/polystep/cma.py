"""CMA-ES-inspired update functions for adaptive optimization.

Diagonal (sep-CMA-ES) covariance only; particle weights come from OT transport masses.
References: Hansen, arXiv:1604.00772; Ros & Hansen, PPSN 2008.
"""

import math
from typing import Dict, Optional

import torch

__all__ = [
    "compute_cma_hyperparameters",
    "update_evolution_path_c",
    "update_evolution_path_sigma",
    "update_covariance_diagonal",
    "compute_heaviside_sigma",
]


def compute_cma_hyperparameters(n: int, mu_eff: float = 2.0) -> Dict[str, float]:
    """Cumulation factors and learning rates for dimension ``n`` (Hansen, arXiv:1604.00772)."""
    # Eq. 3; the +3 (vs the tutorial's +5) is a deliberate tuning choice.
    c_sigma = (mu_eff + 2) / (n + mu_eff + 3)

    # Eq. 4, mu_eff-aware form.
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)

    # Eq. 5.
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)

    # Eq. 6, capped so c_1 + c_mu <= 1; exactly 0 at mu_eff = 1.
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

    return {"c_sigma": c_sigma, "c_c": c_c, "c_1": c_1, "c_mu": c_mu}


@torch.inference_mode()
def update_evolution_path_sigma(
    p_sigma: torch.Tensor,
    displacement: torch.Tensor,
    C_diag: Optional[torch.Tensor],
    c_sigma: float,
    mu_eff: float,
    cov_min: float = 1e-6,
) -> torch.Tensor:
    """Step-size evolution path (Hansen Eq. 3).

    ``cov_min`` must match the floor used when clamping ``C_diag``, or the whitening is wrong at the floor.
    """
    sqrt_factor = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
    if C_diag is None:
        return (1 - c_sigma) * p_sigma + sqrt_factor * displacement
    # For diagonal C: C^(-1/2) = 1/sqrt(C_diag).
    C_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(C_diag, min=cov_min))
    return (1 - c_sigma) * p_sigma + sqrt_factor * C_inv_sqrt * displacement


@torch.inference_mode()
def update_evolution_path_c(
    p_c: torch.Tensor,
    displacement: torch.Tensor,
    h_sigma: bool,
    c_c: float,
    mu_eff: float,
) -> torch.Tensor:
    """Covariance evolution path (Hansen Eq. 4); ``h_sigma=False`` freezes the path."""
    sqrt_factor = math.sqrt(c_c * (2 - c_c) * mu_eff)
    h_sigma_float = 1.0 if h_sigma else 0.0
    return (1 - c_c) * p_c + h_sigma_float * sqrt_factor * displacement


@torch.inference_mode()
def compute_heaviside_sigma(
    p_sigma_norm: float,
    expected_norm: float,
    n: int,
    c_sigma: float,
    generation: int,
) -> bool:
    """Heaviside flag (Hansen below Eq. 4); true when p_sigma is healthy."""
    # generation is 0-based and incremented after the update, so use generation + 1.
    gen = generation + 1
    cumulation_factor = math.sqrt(1 - (1 - c_sigma) ** (2 * gen))
    threshold = (1.4 + 2 / (n + 1)) * expected_norm * cumulation_factor
    return p_sigma_norm < threshold


@torch.inference_mode()
def update_covariance_diagonal(
    C_diag: torch.Tensor,
    p_c: torch.Tensor,
    rank_mu: torch.Tensor,
    c_1: float,
    c_mu: float,
    h_sigma: bool,
    c_c: float,
    trace_scale: float = 1.0,
    cov_min: float = 1e-6,
    cov_max: float = 1e6,
    trace: Optional[float] = None,
) -> torch.Tensor:
    """sep-CMA-ES diagonal covariance update (Ros & Hansen, PPSN 2008).

    ``trace_scale`` compensates for ``p_c`` built from unit-norm directions; ``trace`` rescales to that total before clamping.
    """
    rank_one = trace_scale * p_c**2

    # When h_sigma is 0, the missing rank-one mass is added back to the old-C weight.
    old_coeff = (1 - c_1 - c_mu) + (0.0 if h_sigma else c_1 * c_c * (2 - c_c))

    C_new = old_coeff * C_diag + c_1 * rank_one + c_mu * rank_mu

    if trace is not None:
        C_new = C_new * (trace / C_new.sum().clamp(min=1e-12))

    return torch.clamp(C_new, min=cov_min, max=cov_max)
