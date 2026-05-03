"""Shared helpers for PolyStepOptimizer step methods.

Deduplicates common patterns that appear across _step_monolithic,
_step_blockwise, and _step_subspace_blockwise: radius resolution,
biased rotation (Gram-Schmidt), cost matrix sanitization, NaN-safe
state revert, diagnostics update, transport direction capture, and
CMA-ES evolution path updates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .solver import SolverState

__all__ = [
    "resolve_radii",
    "apply_biased_rotation",
    "sanitize_cost_matrix",
    "nan_safe_revert_monolithic",
    "nan_safe_revert_blockwise",
    "update_diagnostics",
    "capture_transport_direction",
    "update_cma_es",
]


# ---------------------------------------------------------------------------
# Radius resolution
# ---------------------------------------------------------------------------


def resolve_radii(
    optimizer,
    iteration: int,
    state: "SolverState",
) -> Tuple[float, float, float, float]:
    """Resolve step_radius, probe_radius, current_eps, and radius_mult.

    Returns:
        (step_r, probe_r, current_eps, radius_mult)
    """
    current_eps = optimizer._get_epsilon(iteration)

    # Use CSA sigma or heuristic radius_multiplier
    if optimizer.use_csa and getattr(state, 'use_csa', False):
        radius_mult = state.sigma
    elif optimizer.use_adaptive_radius:
        radius_mult = state.radius_multiplier
    else:
        radius_mult = 1.0

    _sr = optimizer._get_step_radius(iteration)
    _pr = optimizer._get_probe_radius(iteration)
    _sr_scheduled = hasattr(optimizer.step_radius, 'at')
    _pr_scheduled = hasattr(optimizer.probe_radius, 'at')

    if optimizer.trust_region:
        step_r = _sr * optimizer._trust_region_multiplier * (1.0 if _sr_scheduled else current_eps) * radius_mult
    else:
        step_r = _sr * (1.0 if _sr_scheduled else current_eps) * radius_mult
    probe_r = _pr * (1.0 if _pr_scheduled else current_eps) * radius_mult

    # Apply probe-radius jitter (Fubini transversality, Thm. 4.2 condition (iv)).
    # No-op when optimizer.probe_radius_jitter == 0 (default).
    probe_r = optimizer._apply_probe_radius_jitter(probe_r)

    return step_r, probe_r, current_eps, radius_mult


# ---------------------------------------------------------------------------
# Biased rotation (Gram-Schmidt orthogonalization)
# ---------------------------------------------------------------------------


def apply_biased_rotation(
    rot_mats: torch.Tensor,
    bias_dir: torch.Tensor,
    pdim: int,
) -> torch.Tensor:
    """Apply transport-biased rotation: replace first column with bias
    direction and re-orthogonalize remaining columns via Gram-Schmidt.

    Args:
        rot_mats: (P, pdim, pdim) rotation matrices (modified in-place).
        bias_dir: (P, pdim) bias direction vectors.
        pdim: Particle dimension.

    Returns:
        Modified rotation matrices with proper SO(n) structure.
    """
    bias_norms = torch.norm(bias_dir, dim=-1, keepdim=True).clamp(min=1e-10)
    bias_dir_norm = bias_dir / bias_norms
    # Save original for Gram-Schmidt fallback
    rot_mats_orig = rot_mats.clone()
    # Replace column 0 with normalized bias direction
    rot_mats[:, :, 0] = bias_dir_norm
    # Re-orthogonalize remaining columns via Gram-Schmidt
    for col in range(1, pdim):
        v = rot_mats[:, :, col].clone()
        for prev_col in range(col):
            proj = (v * rot_mats[:, :, prev_col]).sum(dim=-1, keepdim=True)
            v = v - proj * rot_mats[:, :, prev_col]
        raw_norm = torch.norm(v, dim=-1, keepdim=True)
        norms_v = raw_norm.clamp(min=1e-10)
        # Only replace column if sufficiently independent
        mask = (raw_norm > 1e-6).float()
        rot_mats[:, :, col] = mask * (v / norms_v) + (1 - mask) * rot_mats_orig[:, :, col]

    # Fix determinant: Gram-Schmidt can flip det from +1 to -1.
    dets = torch.det(rot_mats)
    flip = (dets < 0).unsqueeze(-1)  # (P, 1)
    rot_mats[:, :, -1] = torch.where(flip, -rot_mats[:, :, -1], rot_mats[:, :, -1])

    return rot_mats


# ---------------------------------------------------------------------------
# Cost matrix sanitization
# ---------------------------------------------------------------------------


def sanitize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values in cost matrix with a large penalty.

    Pure-tensor path, no GPU-CPU sync.

    Args:
        cost_matrix: (P, V) cost matrix.

    Returns:
        Sanitized cost matrix with non-finite values replaced.
    """
    if not torch.isfinite(cost_matrix).all():
        finite_mask = cost_matrix.isfinite()
        max_val = cost_matrix.where(finite_mask, torch.zeros_like(cost_matrix)).abs().amax()
        penalty = torch.clamp(max_val * 2.0 + 1.0, min=1e6)
        cost_matrix = cost_matrix.where(finite_mask, penalty)
    return cost_matrix


# ---------------------------------------------------------------------------
# NaN-safe state revert
# ---------------------------------------------------------------------------


def nan_safe_revert_monolithic(
    optimizer,
    state: "SolverState",
    X_old: torch.Tensor,
) -> bool:
    """Revert state.X, velocity, and clear cached directions if NaN detected.

    Returns True if a revert occurred.
    """
    if torch.isfinite(state.X).all():
        return False

    state.X = X_old.clone()
    # Reset velocity to prevent NaN propagation through momentum
    if optimizer.use_momentum and state.velocity is not None:
        state.velocity = torch.zeros_like(state.velocity)

    # Clear biased rotation descent direction and Newton direction
    if optimizer.biased_rotation:
        optimizer._prev_descent_direction = None
        optimizer._prev_descent_direction_finite = False
    optimizer._newton_direction = None

    return True


def nan_safe_revert_blockwise(
    optimizer,
    state: "SolverState",
    X_old: torch.Tensor,
    blocks,
) -> bool:
    """Revert state.X and clear blockwise-specific cached state if NaN detected.

    Returns True if a revert occurred.
    """
    if torch.isfinite(state.X).all():
        return False

    state.X = X_old.clone()
    state.block_duals = [(None, None) for _ in blocks]
    # Reset velocity to prevent NaN propagation through momentum
    if optimizer.use_momentum and state.velocity is not None:
        state.velocity = torch.zeros_like(state.velocity)
    # Clear cached state
    optimizer._transport_direction_ema = None
    optimizer._prev_descent_direction = None
    optimizer._prev_descent_direction_finite = False
    if optimizer._dual_momentum_beta > 0.0:
        state._prev_prev_block_duals = None
    if optimizer.biased_rotation:
        optimizer._prev_block_descent_directions = None

    return True


# ---------------------------------------------------------------------------
# Diagnostics update
# ---------------------------------------------------------------------------


def update_diagnostics(
    state: "SolverState",
    cost: float,
    converged: bool,
    disp_sqnorm: float,
) -> None:
    """Append cost, convergence, and displacement to state diagnostics."""
    state.costs.append(cost)
    state.linear_convergence.append(converged)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.iteration_count += 1


# ---------------------------------------------------------------------------
# Transport direction capture (for amortized OT)
# ---------------------------------------------------------------------------


def capture_transport_direction(
    optimizer,
    state: "SolverState",
    X_old: torch.Tensor,
    nan_reverted: bool,
) -> None:
    """Capture and EMA-smooth the transport direction for amortized momentum.

    Only active when ``optimizer.amortize_steps > 1``.
    """
    if optimizer.amortize_steps <= 1:
        return

    if nan_reverted:
        optimizer._transport_direction = None
        optimizer._transport_direction_ema = None
    else:
        raw_direction = (state.X - X_old).detach()
        optimizer._transport_direction = raw_direction
        alpha = optimizer.amortize_ema
        if optimizer._transport_direction_ema is None:
            optimizer._transport_direction_ema = raw_direction
        else:
            optimizer._transport_direction_ema = (
                alpha * optimizer._transport_direction_ema + (1.0 - alpha) * raw_direction
            )


# ---------------------------------------------------------------------------
# Adaptive radius update
# ---------------------------------------------------------------------------


def update_adaptive_radius_state(
    optimizer,
    state: "SolverState",
    cost_mean: float,
) -> None:
    """Update adaptive radius multiplier based on cost stagnation."""
    if not optimizer.use_adaptive_radius:
        return

    from .dynamics import update_adaptive_radius

    rm, sc, pl = update_adaptive_radius(
        cost_mean,
        state.prev_loss,
        state.stagnation_count,
        state.radius_multiplier,
        stagnation_threshold=optimizer.stagnation_threshold,
        stagnation_patience=optimizer.stagnation_patience,
        radius_increase=optimizer.radius_increase,
        radius_decrease=optimizer.radius_decrease,
        radius_min=optimizer.radius_min,
        radius_max=optimizer.radius_max,
    )
    state.radius_multiplier = rm
    state.stagnation_count = sc
    state.prev_loss = pl


# ---------------------------------------------------------------------------
# CMA-ES updates (extracted from monolithic step)
# ---------------------------------------------------------------------------


def update_cma_es(
    optimizer,
    state: "SolverState",
    _pre_step_sub_coords: torch.Tensor,
    _pre_step_particle_coords: Optional[torch.Tensor],
    ot_result,
    step_r: float,
    probes: torch.Tensor,
    probe_r: float,
    pdim: int,
) -> None:
    """Update CMA-ES evolution paths, covariance, and step-size.

    Handles p_sigma, p_c, C_diag, sigma updates, and rank-mu covariance
    using per-particle OT-weighted displacements.

    Args:
        optimizer: PolyStepOptimizer instance.
        state: Current solver state.
        _pre_step_sub_coords: Subspace coords before barycentric projection.
        _pre_step_particle_coords: Per-particle coords before step (for rank-mu).
        ot_result: OT solve result (for rank-mu weights).
        step_r: Current step radius.
        probes: Probe positions tensor.
        probe_r: Current probe radius.
        pdim: Particle dimension.
    """
    from .cma import (
        update_evolution_path_sigma,
        compute_heaviside_sigma,
        update_evolution_path_c,
        update_covariance_diagonal,
        compute_ot_weights,
        update_step_size_csa,
    )

    cma_sub = optimizer.subspace  # CMAAdaptiveSubspace
    sub_dim = cma_sub.subspace_dim

    # Compute mean displacement in subspace coordinates
    post_step_coords = state.X.reshape(-1)[:sub_dim]
    raw_displacement = post_step_coords - _pre_step_sub_coords

    # Normalize displacement by sigma (CMA-ES convention)
    normalized_displacement = raw_displacement / state.sigma

    # 1. Update p_sigma (step-size evolution path)
    state.p_sigma = update_evolution_path_sigma(
        p_sigma=state.p_sigma,
        displacement=normalized_displacement,
        C_diag=state.C_diag,
        c_sigma=optimizer._cma_params['c_sigma'],
        mu_eff=optimizer._cma_params['mu_eff'],
    )

    # 2. Compute Heaviside for stall detection
    p_sigma_norm = torch.norm(state.p_sigma).item()
    h_sigma = compute_heaviside_sigma(
        p_sigma_norm=p_sigma_norm,
        expected_norm=optimizer._cma_params['expected_norm'],
        n=sub_dim,
        c_sigma=optimizer._cma_params['c_sigma'],
        generation=state.generation,
    )

    # 3. Update p_c (covariance evolution path)
    state.p_c = update_evolution_path_c(
        p_c=state.p_c,
        displacement=normalized_displacement,
        h_sigma=h_sigma,
        c_c=optimizer._cma_params['c_c'],
        mu_eff=optimizer._cma_params['mu_eff'],
    )

    # 4. Update diagonal covariance with full rank-mu (if enabled)
    if optimizer.use_covariance_adaptation:
        P_count = state.X.shape[0]
        pdim_local = state.X.shape[1]

        # Per-particle displacement: (P, particle_dim)
        particle_displacements = state.X - _pre_step_particle_coords

        # Project per-particle displacements to subspace dimension
        if sub_dim >= P_count * pdim_local:
            per_particle_sub_disp_full = torch.zeros(
                P_count, sub_dim, device=state.X.device, dtype=state.X.dtype
            )
            row_idx = torch.arange(P_count, device=state.X.device)
            col_offsets = torch.arange(pdim_local, device=state.X.device).unsqueeze(0)
            col_idx = row_idx.unsqueeze(1) * pdim_local + col_offsets
            valid_mask = col_idx < sub_dim
            col_idx = col_idx.clamp(max=sub_dim - 1)
            src = particle_displacements[:, :pdim_local] * valid_mask
            per_particle_sub_disp_full.scatter_(1, col_idx, src)
        else:
            per_particle_sub_disp_full = torch.zeros(
                P_count, sub_dim, device=state.X.device, dtype=state.X.dtype
            )
            dims_per_particle = max(1, sub_dim // P_count)
            row_idx = torch.arange(P_count, device=state.X.device)
            col_offsets = torch.arange(dims_per_particle, device=state.X.device).unsqueeze(0)
            col_idx = row_idx.unsqueeze(1) * dims_per_particle + col_offsets
            valid_mask = col_idx < sub_dim
            col_idx = col_idx.clamp(max=sub_dim - 1)
            src = particle_displacements[:, :dims_per_particle] * valid_mask
            per_particle_sub_disp_full.scatter_(1, col_idx, src)

        # Normalize per-particle displacements by sigma
        normalized_per_particle_disp = per_particle_sub_disp_full / state.sigma

        # Compute OT-informed weights for rank-mu update
        ot_weights = compute_ot_weights(ot_result.matrix, state.a)

        state.C_diag = update_covariance_diagonal(
            C_diag=state.C_diag,
            p_c=state.p_c,
            displacements=normalized_per_particle_disp,
            weights=ot_weights,
            c_1=optimizer._cma_params['c_1'],
            c_mu=optimizer._cma_params['c_mu'],
            h_sigma=h_sigma,
            c_c=optimizer._cma_params['c_c'],
        )
        # Enforce bounds
        state.C_diag = torch.clamp(
            state.C_diag,
            optimizer._cma_params['cov_min'],
            optimizer._cma_params['cov_max'],
        )

    # 5. Update step-size via CSA (if enabled)
    if optimizer.use_csa:
        state.sigma = update_step_size_csa(
            sigma=state.sigma,
            p_sigma=state.p_sigma,
            c_sigma=optimizer._cma_params['c_sigma'],
            d_sigma=optimizer._cma_params['d_sigma'],
            n=sub_dim,
        )
        # Floor sigma to prevent collapse to zero
        state.sigma = max(state.sigma, 1e-6)

    # 6. Increment generation
    state.generation += 1


# ---------------------------------------------------------------------------
# Invalidate cached state (after absorb or rank transition)
# ---------------------------------------------------------------------------


def invalidate_cached_state(optimizer) -> None:
    """Clear all cached probe/cost/direction state.

    Called after absorb, rank transition, or any event that changes the
    cost landscape geometry.
    """
    optimizer._prev_cost_matrix = None
    optimizer._prev_losses_3d = None
    optimizer._prev_displacement_sqnorms = None
    optimizer._prev_k_eff = None
    optimizer._prev_step_r = None
    optimizer._newton_direction = None
    optimizer._prev_descent_direction = None
    optimizer._prev_descent_direction_finite = False
    optimizer._transport_direction_ema = None
    optimizer._transport_direction = None
