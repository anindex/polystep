"""Quadratic model extraction from orthoplex cost evaluations.

Extracts finite-difference gradient, Hessian diagonal, and Newton step
from the (P, V, K) loss tensor produced by orthoplex probe evaluations.
All functions are pure tensor operations with no side effects.

Vertex ordering convention (orthoplex):
  Vertices 0..d-1: +e_0, +e_1, ..., +e_{d-1}
  Vertices d..2d-1: -e_0, -e_1, ..., -e_{d-1}
  Pair for direction i: vertex i (+) and vertex i+d (-)
"""

import torch


def extract_fd_gradient(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
) -> torch.Tensor:
    """Extract finite-difference gradient from orthoplex cost evaluations.

    Central difference at each probe scale, averaged for robustness:
        g_i = mean_k[(L(+s_k*d_i) - L(-s_k*d_i)) / (2 * s_k * r)]

    Args:
        losses_3d: Loss values of shape (P, V, K) where V = 2*pdim.
        scales: Probe scale factors of shape (K,).
        probe_radius: Probe distance multiplier.
        pdim: Particle dimension (number of orthoplex directions).

    Returns:
        Gradient in rotated frame of shape (P, pdim).
    """
    assert losses_3d.shape[1] == 2 * pdim, (
        f"FD gradient needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}"
    )
    fwd = losses_3d[:, :pdim, :]  # (P, pdim, K) at +directions
    bwd = losses_3d[:, pdim:, :]  # (P, pdim, K) at -directions

    # Central difference at each scale
    denom = (2.0 * scales * probe_radius).unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    grad_per_scale = (fwd - bwd) / denom.clamp(min=1e-10)  # (P, pdim, K)

    return grad_per_scale.mean(dim=-1)  # (P, pdim)


def extract_fd_hessian_diag(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
) -> torch.Tensor:
    """Extract diagonal Hessian from orthoplex cost evaluations.

    Uses the symmetry property of the quadratic model:
        L(+s) + L(-s) = 2*L(0) + H_ii * s^2
    Regresses the symmetric sum on s^2 to estimate H_ii.

    Args:
        losses_3d: Loss values of shape (P, V, K) where V = 2*pdim.
        scales: Probe scale factors of shape (K,).
        probe_radius: Probe distance multiplier.
        pdim: Particle dimension.

    Returns:
        Diagonal Hessian in rotated frame of shape (P, pdim).
    """
    assert losses_3d.shape[1] == 2 * pdim, (
        f"FD Hessian needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}"
    )
    fwd = losses_3d[:, :pdim, :]  # (P, pdim, K)
    bwd = losses_3d[:, pdim:, :]  # (P, pdim, K)

    # Symmetric sum: L(+s) + L(-s) = 2a + H*s^2
    sym_sum = fwd + bwd  # (P, pdim, K)

    s_sq = (scales * probe_radius) ** 2  # (K,)

    # Linear regression of sym_sum on s_sq to get slope = H
    # Center for numerical stability
    s_sq_mean = s_sq.mean()
    sym_mean = sym_sum.mean(dim=-1, keepdim=True)  # (P, pdim, 1)

    s_centered = s_sq - s_sq_mean  # (K,)
    y_centered = sym_sum - sym_mean  # (P, pdim, K)

    # slope = sum(s_centered * y_centered) / sum(s_centered^2)
    numerator = (s_centered.unsqueeze(0).unsqueeze(0) * y_centered).sum(dim=-1)  # (P, pdim)
    denominator = (s_centered**2).sum().clamp(min=1e-10)

    return numerator / denominator  # (P, pdim) = Hessian diagonal


def compute_newton_step(
    gradient: torch.Tensor,
    hessian_diag: torch.Tensor,
    max_step_norm: float = 10.0,
    hessian_reg: float = 1e-4,
) -> torch.Tensor:
    """Compute a diagonal Newton step in the rotated frame.

    Where curvature is above ``hessian_reg`` returns ``-g_i / H_i``; where it
    is at or below ``hessian_reg`` (small-positive or nonpositive) the Hessian
    is floored to ``hessian_reg`` (a small gradient step, never an ascent step).
    The full step is then clipped to ``max_step_norm``.

    Args:
        gradient: FD gradient of shape ``(P, pdim)``.
        hessian_diag: Diagonal Hessian of shape ``(P, pdim)``.
        max_step_norm: Maximum step norm (trust region bound).
        hessian_reg: Floor applied to curvature entries at or below it.

    Returns:
        Newton step in rotated frame of shape ``(P, pdim)``.
    """
    # Where curvature is positive use H_i directly; otherwise fall back to
    # the regulariser so the direction stays a descent step.
    H_safe = torch.where(
        hessian_diag > hessian_reg,
        hessian_diag,
        torch.full_like(hessian_diag, hessian_reg),
    )
    delta = -gradient / H_safe  # (P, pdim)

    # Clamp step norm
    norms = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-10)
    scale = torch.clamp(max_step_norm / norms, max=1.0)
    return delta * scale


def compute_predicted_improvement(
    gradient: torch.Tensor,
    hessian_diag: torch.Tensor,
    step: torch.Tensor,
) -> torch.Tensor:
    """Predict improvement from quadratic model: dL = g.delta + 0.5*delta.H.delta.

    Args:
        gradient: FD gradient of shape (P, pdim).
        hessian_diag: Diagonal Hessian of shape (P, pdim).
        step: Step vector in rotated frame of shape (P, pdim).

    Returns:
        Predicted loss change per particle of shape (P,). Negative = improvement.
    """
    linear = (gradient * step).sum(dim=-1)
    quadratic = 0.5 * (hessian_diag * step**2).sum(dim=-1)
    return linear + quadratic


def apply_newton_refinement(
    X_bary: torch.Tensor,
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
    rot_mats: torch.Tensor,
    X_current: torch.Tensor,
    alpha: float = 0.3,
    max_step_norm: float = 1.0,
    hessian_reg: float = 1e-4,
) -> torch.Tensor:
    """Apply post-OT Newton refinement using the quadratic model from probe evaluations.

    After the OT barycentric projection gives a coarse update direction, this
    function uses second-order information already available from the probe
    evaluations to compute a Newton correction step, improving per-step
    convergence without additional forward passes.

    Steps:
    1. Extract FD gradient and diagonal Hessian from losses_3d (rotated frame)
    2. Compute Newton step in rotated frame: delta_rot = -g / (H + reg)
    3. Transform Newton step to original space: delta_orig = rot_mats @ delta_rot
    4. Newton-predicted minimum from the probe center: X_newton = X_current + delta_orig
    5. Blend: X_refined = (1 - alpha) * X_bary + alpha * X_newton

    Args:
        X_bary: Post-OT barycentric position, shape (P, pdim).
        losses_3d: Probe loss values, shape (P, V, K) where V = 2*pdim.
        scales: Probe scale factors, shape (K,).
        probe_radius: Probe distance multiplier.
        pdim: Particle dimension (number of orthoplex directions).
        rot_mats: Rotation matrices, shape (P, pdim, pdim).
        X_current: Probe-center position the quadratic model is built around,
            shape (P, pdim). The Newton step is measured from here, so the
            predicted minimum is X_current + delta_orig.
        alpha: Blending weight. 0 = pure OT, 1 = pure Newton correction.
        max_step_norm: Maximum Newton correction norm (trust region bound).
        hessian_reg: Regularization for near-zero Hessian entries.

    Returns:
        Refined position of shape (P, pdim).
    """
    # 1. Extract gradient and Hessian in rotated frame
    gradient = extract_fd_gradient(losses_3d, scales, probe_radius, pdim)
    hessian_diag = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)

    # 2. Compute Newton step in rotated frame
    delta_rot = compute_newton_step(
        gradient,
        hessian_diag,
        max_step_norm=max_step_norm,
        hessian_reg=hessian_reg,
    )

    # 3. Transform Newton step to original space: delta_orig = rot_mats @ delta_rot
    # rot_mats: (P, pdim, pdim), delta_rot: (P, pdim)
    delta_orig = torch.einsum("bij,bj->bi", rot_mats, delta_rot)

    # 4. Newton-predicted minimum, anchored at the probe center X_current (the
    # point the quadratic is built around). X_bary already carries the transport
    # step, so anchoring the Newton step there would double-count that move.
    X_newton = X_current + delta_orig

    # 5. Blend the OT result with the Newton prediction
    X_refined = (1.0 - alpha) * X_bary + alpha * X_newton

    return X_refined


def update_trust_region(
    predicted_improvement: torch.Tensor,
    actual_improvement: torch.Tensor,
    current_radius: float,
    expand_threshold: float = 0.75,
    shrink_threshold: float = 0.25,
    expand_factor: float = 1.5,
    shrink_factor: float = 0.5,
    min_radius: float = 0.1,
    max_radius: float = 3.0,
) -> float:
    """Update trust region multiplier based on predicted vs actual improvement.

    Both predicted and actual improvement use the same sign convention:
    negative = loss decreased (improvement). The ratio actual/predicted
    should be positive and near 1.0 when the quadratic model is accurate.

    Args:
        predicted_improvement: Predicted loss change (P,). Negative = improvement.
        actual_improvement: Actual loss change scalar or (1,). Negative = improvement.
        current_radius: Current trust region multiplier.
        expand_threshold: Ratio above which to expand.
        shrink_threshold: Ratio below which to shrink.
        expand_factor: Radius expansion multiplier.
        shrink_factor: Radius shrink multiplier.
        min_radius: Minimum allowed multiplier.
        max_radius: Maximum allowed multiplier.

    Returns:
        Updated trust region multiplier.
    """
    pred = predicted_improvement.mean().item()
    actual = actual_improvement.mean().item()

    if abs(pred) < 1e-10:
        return current_radius

    ratio = actual / pred

    # Clamp ratio to prevent extreme updates from noisy estimates
    ratio = max(-2.0, min(ratio, 5.0))

    # Negative ratio: actual went opposite direction from prediction - aggressive shrink
    if ratio < 0:
        return max(current_radius * shrink_factor * 0.5, min_radius)

    # Only expand when the model predicted an improvement (pred < 0) and reality
    # matched it. An accurate but worsening step (pred > 0, actual > 0) also gives
    # ratio ~1 and must not grow the region.
    if ratio > expand_threshold and pred < 0:
        return min(current_radius * expand_factor, max_radius)
    elif ratio < shrink_threshold:
        return max(current_radius * shrink_factor, min_radius)
    return current_radius
