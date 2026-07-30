"""CMA-ES-inspired update functions for adaptive optimization.

Pure functions computing diagonal (sep-CMA-ES) covariance updates adapted for
OT-based optimization:

- Particle weights come from OT transport masses, not truncation selection
- Diagonal covariance only, O(n) rather than O(n^2)
- Evolution paths track cumulative information across optimization steps

Sampling scales the projection columns by sqrt(C_diag), so the offspring step
in covariance-metric coordinates is y = sqrt(C_diag) * z (z is the raw subspace
step). The evolution paths and rank-mu are defined on y. These updates are wired
into the monolithic step only; blockwise disables CMA (see PolyStepOptimizer).

Step size comes from ``use_adaptive_radius``, not from CSA.

References:
- Hansen, N. "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772)
- Ros, R. & Hansen, N. "A Simple Modification in CMA-ES Achieving Linear
  Time and Space Complexity" (PPSN 2008) for sep-CMA-ES validation
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
    """Cumulation factors and learning rates for dimension ``n``.

    Follows Hansen's tutorial (arXiv:1604.00772, Section 3 Table 1) except for
    ``c_sigma``, where the deviation is documented below.

    Args:
        n: Subspace dimension (number of parameters being optimized).
        mu_eff: Variance-effectiveness of weights. For OT-based optimization,
            a value around 2.0 is typical since transport masses provide
            soft weighting rather than hard truncation selection.

    Returns:
        ``c_sigma`` (step-size path), ``c_c`` (covariance path), ``c_1``
        (rank-one rate), ``c_mu`` (rank-mu rate).
    """
    # Cumulation factor for the step-size path (Eq. 3). Hansen's tutorial
    # (arXiv:1604.00772, Table 1) writes (mu_eff + 2) / (n + mu_eff + 5); the +3 here
    # is a tuning choice for faster adaptation, not a published variant. At mu_eff = 1
    # it gives 3/(n+4) against the tutorial's 3/(n+6): mu_eff sits in the denominator
    # too, so the constant is not the whole difference.
    c_sigma = (mu_eff + 2) / (n + mu_eff + 3)

    # Cumulation factor for covariance path (Eq. 4). The mu_eff-aware form from the
    # current tutorial; the older 4/(n+4) also feeds the c_c*(2-c_c) make-up term.
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)

    # Rank-one learning rate (Eq. 5)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)

    # Rank-mu learning rate (Eq. 6)
    # Ensure c_1 + c_mu <= 1
    # Exactly 0 at mu_eff = 1: one recombination point carries no second-moment
    # information. The integrated optimizer runs at mu_eff = 1 by default, so
    # covariance adaptation there is rank-one only (see LIMITATIONS.md).
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
    """Step-size evolution path, Hansen arXiv:1604.00772 Eq. 3.

        p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) C^-1/2 d

    ``mu_eff`` is 1.0 when ``displacement`` is already standardized to unit expected
    squared norm. ``cov_min`` must match the floor used when clamping ``C_diag``, or
    the whitening is wrong at the floor. Pass ``C_diag=None`` when ``displacement`` is
    already in whitened coordinates, which skips a sqrt and two elementwise passes.
    """
    sqrt_factor = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
    if C_diag is None:
        return (1 - c_sigma) * p_sigma + sqrt_factor * displacement
    # For diagonal C: C^(-1/2) = 1/sqrt(C_diag) element-wise
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
    """Covariance evolution path, Hansen arXiv:1604.00772 Eq. 4.

        p_c <- (1 - c_c) p_c + h_sigma sqrt(c_c (2 - c_c) mu_eff) d

    Feeds the rank-one covariance update. ``h_sigma=False`` freezes the path, which
    is what stops the covariance exploding while the step size is shrinking.
    """
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
    """Heaviside flag, Hansen arXiv:1604.00772 below Eq. 4. True when p_sigma is healthy.

        h_sigma = ||p_sigma|| < (1.4 + 2/(n+1)) expected_norm sqrt(1 - (1-c_sigma)^2g)

    The sqrt term corrects for early generations, where cumulation has not built up.

    ``expected_norm`` is the stationary norm of p_sigma under no selection:
    ``E||N(0,I)|| ~ sqrt(n)`` for Gaussian mutations, 1.0 for the standardized unit
    direction the OT step supplies. Passing a running mean of ``p_sigma_norm`` itself
    makes the test almost always true, disabling stall detection.
    """
    # Caller passes a 0-based count of completed generations and increments it
    # after the update, so the current generation is generation + 1.
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

        C[j] <- old_coeff C[j] + c_1 p_c[j]^2 + c_mu rank_mu[j]
        old_coeff = (1 - c_1 - c_mu) + (0 if h_sigma else c_1 c_c (2 - c_c))

    ``rank_mu`` is the weighted mean of squared offspring steps
    ``sum_k w_k y_{k,j}^2``, with ``y = sqrt(C_diag) z`` so it sits on the same
    covariance-metric scale as ``p_c``.

    ``trace_scale`` multiplies the rank-one term. ``p_c`` built from unit-norm
    directions has total squared norm 1 while ``C_diag`` sums to the dimension, so
    without a matching trace convention the rank-one term contributes nothing and
    ``C_diag`` decays every generation.

    ``trace`` rescales to that total before clamping, so C carries shape and the step
    radius carries scale. Both here so the order cannot be got wrong outside.
    """
    # Rank-one update from the covariance evolution path.
    rank_one = trace_scale * p_c**2

    # sep-CMA-ES old-covariance coefficient. When the Heaviside h_sigma is 0
    # (step-size path stalled), the missing rank-one mass c_1*c_c*(2-c_c) is
    # added back to the old-C weight (additive make-up), matching Hansen's form,
    # rather than multiplied in.
    old_coeff = (1 - c_1 - c_mu) + (0.0 if h_sigma else c_1 * c_c * (2 - c_c))

    C_new = old_coeff * C_diag + c_1 * rank_one + c_mu * rank_mu

    if trace is not None:
        C_new = C_new * (trace / C_new.sum().clamp(min=1e-12))

    return torch.clamp(C_new, min=cov_min, max=cov_max)
