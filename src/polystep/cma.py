"""CMA-ES-inspired update functions for adaptive optimization.

Pure functions that compute CMA-ES (Covariance Matrix Adaptation Evolution
Strategy) updates adapted for OT-based optimization. These functions
implement diagonal (sep-CMA-ES) covariance updates and cumulative step-size
adaptation (CSA) as described in Hansen's CMA-ES Tutorial.

Key adaptations for OT-based optimization:
- Particle weights from OT transport masses instead of truncation selection
- Diagonal covariance only (O(n) complexity, not full O(n^2))
- Evolution paths track cumulative information across optimization steps

Sampling scales the projection columns by sqrt(C_diag), so the offspring step
in covariance-metric coordinates is y = sqrt(C_diag) * z (z is the raw subspace
step). The evolution paths and rank-mu are defined on y. These updates are wired
into the monolithic step only; blockwise disables CMA (see PolyStepOptimizer).

Warning - CSA Instability with OT:
    CSA (Cumulative Step-size Adaptation) may be unstable for OT-based
    optimization. In standard CMA-ES, mutations are `sigma * N(0, C)` with
    displacement ≈ 100% of the step size. In OT-based optimization:

    1. Actual displacement is typically 1-5% of polytope size (not 100%)
    2. The displacement-sigma relationship is nonlinear
    3. This breaks the CSA feedback loop, causing sigma to collapse or explode

    **Recommendation**: Use `use_adaptive_radius=True` instead of `use_csa=True`
    for stable step-size adaptation in OT-based optimization.

References:
- Hansen, N. "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772)
- Ros, R. & Hansen, N. "A Simple Modification in CMA-ES Achieving Linear
  Time and Space Complexity" (PPSN 2008) for sep-CMA-ES validation
"""

import math
from typing import Dict

import torch

__all__ = [
    "compute_cma_hyperparameters",
    "update_evolution_path_c",
    "update_evolution_path_sigma",
    "update_step_size_csa",
    "update_covariance_diagonal",
    "compute_heaviside_sigma",
]


def compute_cma_hyperparameters(n: int, mu_eff: float = 2.0) -> Dict[str, float]:
    """Compute default CMA-ES hyperparameters for dimension n.

    Uses the standard CMA-ES formulas from Hansen's tutorial for computing
    cumulation factors, learning rates, and damping parameters.

    Args:
        n: Subspace dimension (number of parameters being optimized).
        mu_eff: Variance-effectiveness of weights. For OT-based optimization,
            a value around 2.0 is typical since transport masses provide
            soft weighting rather than hard truncation selection.

    Returns:
        Dictionary with keys:
            - c_sigma: Cumulation factor for step-size evolution path p_sigma.
            - c_c: Cumulation factor for covariance evolution path p_c.
            - c_1: Learning rate for rank-one covariance update.
            - c_mu: Learning rate for rank-mu covariance update.
            - d_sigma: Damping factor for step-size adaptation.
            - expected_norm: Expected norm of N(0,I) in n dimensions.

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Section 3 Table 1.
    """
    # Cumulation factor for the step-size path (Eq. 3). Hansen's tutorial
    # (arXiv:1604.00772, Table 1) writes (mu_eff + 2) / (n + mu_eff + 5); the +3 here
    # is a tuning choice for faster adaptation, not a published variant. At mu_eff = 1
    # it gives 3/(n+3) against the tutorial's 3/(n+6).
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

    # Damping factor for step-size (Eq. 7)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

    # Expected norm of N(0,I) in n dimensions
    # E[||N(0,I)||] ~ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
    expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    return {
        "c_sigma": c_sigma,
        "c_c": c_c,
        "c_1": c_1,
        "c_mu": c_mu,
        "d_sigma": d_sigma,
        "expected_norm": expected_norm,
    }


@torch.inference_mode()
def update_evolution_path_sigma(
    p_sigma: torch.Tensor,
    displacement: torch.Tensor,
    C_diag: torch.Tensor,
    c_sigma: float,
    mu_eff: float,
    cov_min: float = 1e-6,
) -> torch.Tensor:
    """Update step-size evolution path using CMA-ES Tutorial Eq. 3.

    The evolution path p_sigma accumulates normalized step directions over
    multiple generations. Its norm is used by CSA to adapt the step-size.

    Formula:
        p_sigma^(g+1) = (1 - c_sigma) * p_sigma^(g)
                      + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^(-1/2) * displacement

    For diagonal covariance: C^(-1/2) = 1/sqrt(C_diag) element-wise.

    Args:
        p_sigma: Previous step-size evolution path, shape (subspace_dim,).
        displacement: Weighted mean displacement from current step, shape (subspace_dim,).
        C_diag: Diagonal covariance values, shape (subspace_dim,).
        c_sigma: Cumulation factor (typically from compute_cma_hyperparameters).
        mu_eff: Variance-effectiveness of weights. Pass 1.0 when ``displacement`` is
            already standardized to unit expected squared norm.
        cov_min: Floor applied to C_diag inside the whitening. Must match the ``cov_min``
            used when clamping C_diag, or the whitening is wrong at the floor.

    Returns:
        Updated evolution path p_sigma, shape (subspace_dim,).

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 3.
    """
    sqrt_factor = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
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
    """Update covariance evolution path using CMA-ES Tutorial Eq. 4.

    The evolution path p_c accumulates step directions for the rank-one
    covariance update. It is dampened by h_sigma when p_sigma stalls.

    Formula:
        p_c^(g+1) = (1 - c_c) * p_c^(g)
                  + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * displacement

    Args:
        p_c: Previous covariance evolution path, shape (subspace_dim,).
        displacement: Weighted mean displacement from current step, shape (subspace_dim,).
        h_sigma: Heaviside flag (1 if p_sigma is healthy, 0 if stalled).
            When False/0, the path is not updated to prevent covariance
            explosion during step-size reduction.
        c_c: Cumulation factor for covariance path.
        mu_eff: Variance-effectiveness of weights.

    Returns:
        Updated evolution path p_c, shape (subspace_dim,).

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 4.
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
    """Compute Heaviside function h_sigma for stall detection.

    h_sigma is used to dampen the covariance path update when the step-size
    evolution path p_sigma is much smaller than expected, indicating that
    the step-size is being reduced.

    Formula:
        h_sigma = 1 if ||p_sigma|| < threshold else 0

    where:
        threshold = (1.4 + 2/(n+1)) * expected_norm * sqrt(1 - (1-c_sigma)^(2*generation))

    The threshold accounts for early generations where ||p_sigma|| is
    naturally smaller due to cumulation buildup.

    Args:
        p_sigma_norm: Current norm of p_sigma.
        expected_norm: Stationary norm of p_sigma under no selection. This is
            ``E||N(0,I)|| ~ sqrt(n)`` for Gaussian mutations, but 1.0 for a
            standardized OT step (see :func:`update_step_size_csa`). Passing a running
            mean of ``p_sigma_norm`` itself makes the test almost always true, which
            disables stall detection.
        n: Subspace dimension.
        c_sigma: Cumulation factor for step-size path.
        generation: Count of completed generations (0-based); the formula uses
            generation + 1 as the current generation index.

    Returns:
        True (h_sigma=1) if p_sigma is healthy, False (h_sigma=0) if stalled.

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), below Equation 4.
    """
    # Caller passes a 0-based count of completed generations and increments it
    # after the update, so the current generation is generation + 1.
    gen = generation + 1

    # Compute threshold with cumulation correction
    # (1 - (1-c_sigma)^(2*g)) accounts for early-generation buildup
    cumulation_factor = math.sqrt(1 - (1 - c_sigma) ** (2 * gen))

    # Threshold from Hansen tutorial
    threshold = (1.4 + 2 / (n + 1)) * expected_norm * cumulation_factor

    return p_sigma_norm < threshold


@torch.inference_mode()
def update_step_size_csa(
    sigma: float,
    p_sigma: torch.Tensor,
    c_sigma: float,
    d_sigma: float,
    n: int,
    p_sigma_norm: float | None = None,
    expected_norm: float | None = None,
) -> float:
    """Update step-size using CSA (Cumulative Step-size Adaptation) formula.

    CSA adapts the step-size based on the length of the evolution path p_sigma.
    If ||p_sigma|| exceeds its expected norm the step-size is increased (not
    exploring enough); below it, the step-size is decreased.

    Formula (CMA-ES Tutorial Eq. 7):
        sigma^(g+1) = sigma^(g) * exp((c_sigma / d_sigma) * (||p_sigma|| / E - 1))

    Canonical CMA-ES uses ``E = E[||N(0,I)||] = sqrt(n)`` because a Gaussian mutation has
    norm ~sqrt(n). An OT step over an orthoplex has no such fixed norm (antithetic vertex
    pairs cancel toward zero), so pass 1.0 and feed ``p_sigma`` unit-norm innovations:
    their stationary path norm is exactly 1 in any dimension.

    Do not pass a running mean of ``||p_sigma||``. The ratio is then pinned near 1, and
    while the norm trends it exceeds its own trailing mean by construction, so sigma grows
    until it clamps.

    Args:
        sigma: Current step-size.
        p_sigma: Step-size evolution path, shape (subspace_dim,).
        c_sigma: Cumulation factor for step-size path.
        d_sigma: Damping factor for step-size adaptation.
        n: Subspace dimension.
        p_sigma_norm: Pre-computed ``||p_sigma||`` as a Python float. When
            provided, skips the internal ``torch.norm(...).item()`` sync.
        expected_norm: Norm to compare ||p_sigma|| against. Defaults to the
            Gaussian ``sqrt(n)`` value when None.

    Returns:
        Updated step-size, clamped to [1e-6, 100.0] for numerical stability.

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 7.
    """
    if expected_norm is None:
        expected_norm = math.sqrt(n) * (1.0 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
    expected_norm = max(expected_norm, 1e-8)
    if p_sigma_norm is None:
        p_sigma_norm = torch.norm(p_sigma).item()

    # CSA update
    exponent = (c_sigma / d_sigma) * (p_sigma_norm / expected_norm - 1)
    # Clamp exponent to prevent overflow (exp(10) ≈ 22000, exp(-10) ≈ 0.00005)
    exponent = max(-10.0, min(exponent, 10.0))
    sigma_new = sigma * math.exp(exponent)

    # Clamp to reasonable bounds for numerical stability
    return max(1e-6, min(sigma_new, 100.0))


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
) -> torch.Tensor:
    """Update diagonal covariance with rank-one and rank-mu terms.

    Implements the sep-CMA-ES diagonal covariance update (Ros & Hansen,
    PPSN 2008). The caller supplies ``rank_mu``, the weighted mean of squared
    offspring steps ``sum_k w_k * y_{k,j}^2`` per coordinate, where the offspring
    step ``y = sqrt(C_diag) * z`` is on the same covariance-metric scale as the
    rank-one path ``p_c``.

    Formula:
        C_diag[j] = old_coeff * C_diag[j] + c_1 * p_c[j]^2 + c_mu * rank_mu[j]
        old_coeff = (1 - c_1 - c_mu) + (0 if h_sigma else c_1 * c_c * (2 - c_c))

    Args:
        C_diag: Current diagonal covariance, shape ``(subspace_dim,)``.
        p_c: Covariance evolution path, shape ``(subspace_dim,)``.
        rank_mu: Weighted squared offspring steps ``sum_k w_k * y_{k,j}^2`` per
            coordinate, with ``y = sqrt(C_diag) * z``, shape ``(subspace_dim,)``.
        c_1: Learning rate for rank-one update.
        c_mu: Learning rate for rank-mu update.
        h_sigma: Heaviside flag (``True`` if ``p_sigma`` healthy).
        c_c: Cumulation factor for covariance path (used in ``h_factor``).
        trace_scale: Multiplier on the rank-one term. ``p_c`` built from unit-norm
            directions has total squared norm 1, while ``C_diag`` sums to the dimension,
            so the rank-one term needs the same trace convention as ``rank_mu`` or it
            contributes nothing and ``C_diag`` decays every generation.

    Returns:
        Updated diagonal covariance, clamped to ``[1e-6, 1e6]``,
        shape ``(subspace_dim,)``.

    References:
        Hansen, "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772).
        Ros & Hansen, "A Simple Modification in CMA-ES Achieving Linear
        Time and Space Complexity", PPSN 2008.
    """
    # Rank-one update from the covariance evolution path.
    rank_one = trace_scale * p_c**2

    # sep-CMA-ES old-covariance coefficient. When the Heaviside h_sigma is 0
    # (step-size path stalled), the missing rank-one mass c_1*c_c*(2-c_c) is
    # added back to the old-C weight (additive make-up), matching Hansen's form,
    # rather than multiplied in.
    old_coeff = (1 - c_1 - c_mu) + (0.0 if h_sigma else c_1 * c_c * (2 - c_c))

    C_new = old_coeff * C_diag + c_1 * rank_one + c_mu * rank_mu

    return torch.clamp(C_new, min=cov_min, max=cov_max)
