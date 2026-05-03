"""CMA-ES-inspired update functions for adaptive optimization.

Pure functions that compute CMA-ES (Covariance Matrix Adaptation Evolution
Strategy) updates adapted for OT-based optimization. These functions
implement diagonal (sep-CMA-ES) covariance updates and cumulative step-size
adaptation (CSA) as described in Hansen's CMA-ES Tutorial.

Key adaptations for OT-based optimization:
- Particle weights from OT transport masses instead of truncation selection
- Diagonal covariance only (O(n) complexity, not full O(n^2))
- Evolution paths track cumulative information across optimization steps

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
    'compute_cma_hyperparameters',
    'update_evolution_path_c',
    'update_evolution_path_sigma',
    'update_step_size_csa',
    'update_covariance_diagonal',
    'compute_ot_weights',
    'compute_heaviside_sigma',
    'compute_ot_bias_directions',
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
    # Cumulation factor for step-size path (Eq. 3)
    # c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    # Note: Tutorial uses +5, some variants use +3; we use +3 for faster adaptation
    c_sigma = (mu_eff + 2) / (n + mu_eff + 3)

    # Cumulation factor for covariance path (Eq. 4)
    c_c = 4.0 / (n + 4)

    # Rank-one learning rate (Eq. 5)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)

    # Rank-mu learning rate (Eq. 6)
    # Ensure c_1 + c_mu <= 1
    c_mu = min(
        1 - c_1,
        2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff)
    )

    # Damping factor for step-size (Eq. 7)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

    # Expected norm of N(0,I) in n dimensions
    # E[||N(0,I)||] ~ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
    expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    return {
        'c_sigma': c_sigma,
        'c_c': c_c,
        'c_1': c_1,
        'c_mu': c_mu,
        'd_sigma': d_sigma,
        'expected_norm': expected_norm,
    }


@torch.no_grad()
def update_evolution_path_sigma(
    p_sigma: torch.Tensor,
    displacement: torch.Tensor,
    C_diag: torch.Tensor,
    c_sigma: float,
    mu_eff: float,
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
        mu_eff: Variance-effectiveness of weights.

    Returns:
        Updated evolution path p_sigma, shape (subspace_dim,).

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 3.
    """
    sqrt_factor = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
    # For diagonal C: C^(-1/2) = 1/sqrt(C_diag) element-wise
    # Add small constant for numerical stability
    C_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(C_diag, min=1e-8))
    return (1 - c_sigma) * p_sigma + sqrt_factor * C_inv_sqrt * displacement


@torch.no_grad()
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


@torch.no_grad()
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
        expected_norm: Expected norm of N(0,I) in n dimensions.
        n: Subspace dimension.
        c_sigma: Cumulation factor for step-size path.
        generation: Current generation/iteration count (1-indexed for formula).

    Returns:
        True (h_sigma=1) if p_sigma is healthy, False (h_sigma=0) if stalled.

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), below Equation 4.
    """
    # Use generation + 1 to handle generation=0 case
    gen = max(1, generation)

    # Compute threshold with cumulation correction
    # (1 - (1-c_sigma)^(2*g)) accounts for early-generation buildup
    cumulation_factor = math.sqrt(1 - (1 - c_sigma) ** (2 * gen))

    # Threshold from Hansen tutorial
    threshold = (1.4 + 2 / (n + 1)) * expected_norm * cumulation_factor

    return p_sigma_norm < threshold


@torch.no_grad()
def update_step_size_csa(
    sigma: float,
    p_sigma: torch.Tensor,
    c_sigma: float,
    d_sigma: float,
    n: int,
) -> float:
    """Update step-size using CSA (Cumulative Step-size Adaptation) formula.

    CSA adapts the step-size based on the length of the evolution path p_sigma.
    If ||p_sigma|| > E[||N(0,I)||], the step-size is increased (not exploring
    enough). If ||p_sigma|| < E[||N(0,I)||], the step-size is decreased
    (exploring too much/backtracking).

    Formula (CMA-ES Tutorial Eq. 7):
        sigma^(g+1) = sigma^(g) * exp((c_sigma / d_sigma) * (||p_sigma|| / E[||N(0,I)||] - 1))

    where E[||N(0,I)||] = sqrt(n) * (1 - 1/(4n) + 1/(21n^2)).

    Args:
        sigma: Current step-size.
        p_sigma: Step-size evolution path, shape (subspace_dim,).
        c_sigma: Cumulation factor for step-size path.
        d_sigma: Damping factor for step-size adaptation.
        n: Subspace dimension.

    Returns:
        Updated step-size, clamped to [1e-6, 100.0] for numerical stability.

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 7.
    """
    # Expected norm of N(0,I) in n dimensions
    expected_norm = math.sqrt(n) * (1.0 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
    p_sigma_norm = torch.norm(p_sigma).item()

    # CSA update
    exponent = (c_sigma / d_sigma) * (p_sigma_norm / expected_norm - 1)
    # Clamp exponent to prevent overflow (exp(10) ≈ 22000, exp(-10) ≈ 0.00005)
    exponent = max(-10.0, min(exponent, 10.0))
    sigma_new = sigma * math.exp(exponent)

    # Clamp to reasonable bounds for numerical stability
    return max(1e-6, min(sigma_new, 100.0))


@torch.no_grad()
def update_covariance_diagonal(
    C_diag: torch.Tensor,
    p_c: torch.Tensor,
    displacements: torch.Tensor,
    weights: torch.Tensor,
    c_1: float,
    c_mu: float,
    h_sigma: bool,
    c_c: float,
) -> torch.Tensor:
    """Update diagonal covariance with rank-one and rank-mu terms.

    Implements the sep-CMA-ES diagonal covariance update. The full CMA-ES
    covariance update is O(n^2), but for diagonal-only we achieve O(n).

    Formula (adapted from CMA-ES for diagonal):
        C_diag[j] = h_factor * (1 - c_1 - c_mu) * C_diag[j]
                  + c_1 * p_c[j]^2                           [rank-one]
                  + c_mu * sum(w_i * z_i[j]^2)               [rank-mu]

    where:
        h_factor = 1.0 if h_sigma else (1 - c_1 * c_c * (2 - c_c))

    The h_factor dampens the update when p_sigma is stalled to prevent
    covariance explosion during step-size reduction.

    Args:
        C_diag: Current diagonal covariance, shape (subspace_dim,).
        p_c: Covariance evolution path, shape (subspace_dim,).
        displacements: Per-particle displacements, shape (num_particles, subspace_dim).
        weights: Particle weights from OT masses, shape (num_particles,).
            Should sum to approximately 1.0.
        c_1: Learning rate for rank-one update.
        c_mu: Learning rate for rank-mu update.
        h_sigma: Heaviside flag (True if p_sigma healthy, False if stalled).
        c_c: Cumulation factor for covariance path (used in h_factor).

    Returns:
        Updated diagonal covariance, clamped to [1e-6, 1e6] for stability,
        shape (subspace_dim,).

    References:
        Hansen CMA-ES Tutorial (arXiv:1604.00772), Equation 5.
        Ros & Hansen PPSN 2008 for sep-CMA-ES diagonal adaptation.
    """
    # Rank-one update: p_c^2 element-wise
    rank_one = p_c ** 2

    # Rank-mu update: weighted sum of squared displacements
    # displacements: (num_particles, subspace_dim)
    # weights: (num_particles,)
    rank_mu = (weights[:, None] * (displacements ** 2)).sum(dim=0)

    # Heaviside correction factor for stalled p_sigma
    h_factor = 1.0 if h_sigma else (1 - c_1 * c_c * (2 - c_c))

    # Combined update
    C_new = h_factor * (1 - c_1 - c_mu) * C_diag + c_1 * rank_one + c_mu * rank_mu

    # Clamp to reasonable bounds for numerical stability
    cov_min = 1e-6
    cov_max = 1e6
    return torch.clamp(C_new, min=cov_min, max=cov_max)


@torch.no_grad()
def compute_ot_weights(
    transport_matrix: torch.Tensor,
    source_marginal: torch.Tensor,
) -> torch.Tensor:
    """Compute particle weights from OT transport plan for rank-mu update.

    In standard CMA-ES, weights come from ranking (best mu out of lambda).
    In OT-CMA-ES, we use the transport masses: particles that move more
    mass contribute more to the covariance update.

    Args:
        transport_matrix: OT transport plan, shape (num_particles, num_vertices).
            Entry T[i,v] indicates mass transported from particle i to vertex v.
        source_marginal: Source marginal (particle masses), shape (num_particles,).
            Not used in current implementation but kept for API consistency.

    Returns:
        Normalized weights, shape (num_particles,), summing to approximately 1.0.

    Note:
        The source_marginal parameter is kept for future extensions where
        we might want to weight by relative transport (mass_moved / source_mass).
    """
    # Total mass moved by each particle
    mass_moved = transport_matrix.sum(dim=1)  # (num_particles,)

    # Normalize to weights that sum to 1
    weights = mass_moved / (mass_moved.sum() + 1e-10)

    return weights


@torch.no_grad()
def compute_ot_bias_directions(
    transport_matrix: torch.Tensor,
    X_vertices: torch.Tensor,
    X_current: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Extract high-transport directions from the OT plan for subspace bias.

    Computes weighted displacement directions based on the transport plan,
    where particles that moved more mass to vertices contribute more strongly
    to the bias directions. These directions can be used to bias the random
    subspace projection toward productive regions of parameter space.

    The algorithm:
    1. For each particle i, compute the weighted vertex centroid based on
       transport weights: centroid_i = sum_v(T[i,v] * X_vertices[i,v])
    2. Compute displacement: direction_i = centroid_i - X_current[i]
    3. Weight each direction by total mass moved by that particle
    4. Take top-k directions by weight, normalize them

    Args:
        transport_matrix: OT transport plan, shape (num_particles, num_vertices).
            Entry T[i,v] indicates mass transported from particle i to vertex v.
        X_vertices: Polytope vertex positions for each particle,
            shape (num_particles, num_vertices, particle_dim).
        X_current: Current particle positions, shape (num_particles, particle_dim).
        top_k: Number of top directions to return.

    Returns:
        Tensor of shape (top_k, particle_dim) containing normalized high-transport
        directions. If fewer than top_k particles have meaningful transport,
        returns as many directions as available.

    Example:
        >>> T = torch.rand(10, 4)  # 10 particles, 4 vertices
        >>> T = T / T.sum()  # normalize
        >>> X_verts = torch.randn(10, 4, 8)  # vertices in 8D particle space
        >>> X_curr = torch.randn(10, 8)
        >>> dirs = compute_ot_bias_directions(T, X_verts, X_curr, top_k=5)
        >>> dirs.shape
        torch.Size([5, 8])
    """
    P, V = transport_matrix.shape

    # Normalize transport per particle (row-wise)
    row_sums = transport_matrix.sum(dim=1, keepdim=True)
    T_norm = transport_matrix / (row_sums + 1e-10)  # (P, V)

    # Compute weighted centroid for each particle
    # T_norm: (P, V) -> (P, V, 1) for broadcasting with X_vertices: (P, V, pdim)
    weighted_vertices = T_norm.unsqueeze(-1) * X_vertices  # (P, V, pdim)
    centroids = weighted_vertices.sum(dim=1)  # (P, pdim)

    # Displacement from current position to centroid
    displacements = centroids - X_current  # (P, pdim)

    # Weight by total mass moved (particles that moved more mass are more important)
    mass_moved = transport_matrix.sum(dim=1)  # (P,)
    weights = mass_moved / (mass_moved.sum() + 1e-10)

    # Get top-k by weight
    k_actual = min(top_k, P)
    topk_indices = torch.topk(weights, k_actual).indices

    # Extract top-k displacements
    top_displacements = displacements[topk_indices]  # (k_actual, pdim)

    # Normalize each direction (avoid zero division)
    norms = torch.norm(top_displacements, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    normalized_dirs = top_displacements / norms  # (k_actual, pdim)

    return normalized_dirs
