"""Quadratic model extraction from orthoplex cost evaluations.

Extracts finite-difference gradient, Hessian diagonal, and Newton step
from the (P, V, K) loss tensor produced by orthoplex probe evaluations.
All functions are pure tensor operations with no side effects.

Vertex ordering convention (orthoplex):
  Vertices 0..d-1: +e_0, +e_1, ..., +e_{d-1}
  Vertices d..2d-1: -e_0, -e_1, ..., -e_{d-1}
  Pair for direction i: vertex i (+) and vertex i+d (-)

The extractors see only the loss tensor, so they can check ``V == 2*pdim`` but not the
ordering behind it. Callers must pass orthoplex losses built on a unit-radius template,
which the ``2*s*r`` denominator also assumes. A cube at ``pdim=2`` satisfies the shape
check while pairing vertex 0 with a non-antipodal partner, so it would return a wrong
gradient; the optimizer gates every call on ``polytope_type == "orthoplex"``.
"""

import torch


def extract_fd_gradient(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
) -> torch.Tensor:
    """Extract finite-difference gradient from orthoplex cost evaluations.

    Central difference at each probe scale, averaged over scales:
        g_i = mean_k[(L(+s_k*d_i) - L(-s_k*d_i)) / (2 * s_k * r)]

    ``losses_3d`` is ``(P, V, K)`` with ``V == 2*pdim``, ``scales`` is ``(K,)``. Returns
    the gradient in the rotated frame, ``(P, pdim)``.
    """
    if not losses_3d.shape[1] == 2 * pdim:
        raise ValueError(f"FD gradient needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}")
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
    """Regress the symmetric cost sum on s^2 over orthoplex probe scales.

        L(+s) + L(-s) = 2*L(0) + H_ii * s^2

    What comes back is the least-squares slope of the loss's even part across the probe
    scales, which equals the diagonal Hessian only where the loss is locally quadratic.
    On a piecewise-constant objective it is a secant slope and can be negative where the
    true local curvature is zero; ``compute_newton_step`` floors it, so the step stays
    non-ascent but its size is set by ``hessian_reg``, not by real curvature.

    The regression runs on the unit-radius scales and the slope is divided by
    ``probe_radius**2`` afterwards. Regressing on the absolute offsets instead puts
    ``probe_radius**4`` in the denominator, which falls under the guard below once
    epsilon anneals: at ``probe_radius=2e-3`` the estimate collapsed to 0.9% of the
    true curvature.

    Non-finite regressions return zero curvature rather than NaN, so an overflowed
    denominator cannot reach ``state.X``.

    Shapes match :func:`extract_fd_gradient`.
    """
    if not losses_3d.shape[1] == 2 * pdim:
        raise ValueError(f"FD Hessian needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}")
    fwd = losses_3d[:, :pdim, :]  # (P, pdim, K)
    bwd = losses_3d[:, pdim:, :]  # (P, pdim, K)

    # Symmetric sum: L(+s) + L(-s) = 2a + H*(s*r)^2
    sym_sum = fwd + bwd  # (P, pdim, K)

    s_sq = scales**2  # (K,), O(1) regardless of probe_radius

    # Least-squares slope of sym_sum on s_sq, centered for stability.
    s_centered = s_sq - s_sq.mean()  # (K,)
    y_centered = sym_sum - sym_sum.mean(dim=-1, keepdim=True)  # (P, pdim, K)

    numerator = (s_centered.unsqueeze(0).unsqueeze(0) * y_centered).sum(dim=-1)  # (P, pdim)
    denominator = (s_centered**2).sum().clamp(min=1e-10)
    # Slope is in units of s^2; convert to units of (s*r)^2.
    curvature = numerator / denominator / max(probe_radius**2, 1e-30)  # (P, pdim)
    return torch.where(torch.isfinite(curvature), curvature, torch.zeros_like(curvature))


def extract_fd_hessian_diag_centered(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
    center_loss: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Hessian from one probe scale plus a shared centre evaluation.

        H_ii = (L(+s) + L(-s) - 2 L(0)) / (s*r)^2

    The regression above needs ``K >= 2`` scales per vertex; this needs one, plus a
    single ``L(0)`` per particle that every coordinate shares. On the orthoplex that
    turns ``P*V*K`` candidate evaluations into ``P*V + P``.

    ``center_loss`` is ``(P,)``. Averaged over ``K`` when more than one scale is given.
    """
    if not losses_3d.shape[1] == 2 * pdim:
        raise ValueError(f"FD Hessian needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}")
    fwd = losses_3d[:, :pdim, :]  # (P, pdim, K)
    bwd = losses_3d[:, pdim:, :]

    second_diff = fwd + bwd - 2.0 * center_loss.reshape(-1, 1, 1)  # (P, pdim, K)
    denom = (scales**2).reshape(1, 1, -1) * max(probe_radius**2, 1e-30)
    curvature = (second_diff / denom.clamp(min=1e-30)).mean(dim=-1)  # (P, pdim)
    return torch.where(torch.isfinite(curvature), curvature, torch.zeros_like(curvature))


def _floor_curvature(hessian_diag: torch.Tensor, hessian_reg: float) -> torch.Tensor:
    """Raise nonpositive and near-flat curvature to ``hessian_reg``.

    A floor, not ``H + reg`` and not ``|H|``: the step direction stays ``-sign(g)``
    in every regime, so no coordinate can produce an ascent step.
    """
    return torch.where(
        hessian_diag > hessian_reg,
        hessian_diag,
        torch.full_like(hessian_diag, hessian_reg),
    )


def compute_newton_step(
    gradient: torch.Tensor,
    hessian_diag: torch.Tensor,
    max_step_norm: float = 10.0,
    hessian_reg: float = 1e-4,
) -> torch.Tensor:
    """Compute a diagonal Newton step in the rotated frame.

    Where curvature is above ``hessian_reg`` returns ``-g_i / H_i``; where it is at or
    below ``hessian_reg`` (small-positive or nonpositive) the step falls back to a
    gradient step whose length is capped per coordinate, never an ascent step. The full
    step is then clipped to ``max_step_norm``. All tensors are ``(P, pdim)``.
    """
    delta = -gradient / _floor_curvature(hessian_diag, hessian_reg)  # (P, pdim)

    # Cap per coordinate before the global clip. A flat coordinate divides by hessian_reg,
    # giving a 1e4x step that dominates the norm, and the clip is a single rescale, so it
    # would shrink every well-conditioned coordinate by that same factor.
    delta = delta.clamp(-max_step_norm, max_step_norm)

    norms = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-10)
    scale = torch.clamp(max_step_norm / norms, max=1.0)
    return delta * scale


def compute_predicted_improvement(
    gradient: torch.Tensor,
    hessian_diag: torch.Tensor,
    step: torch.Tensor,
    hessian_reg: float = 1e-4,
) -> torch.Tensor:
    """``dL = g.delta + 0.5 * delta.H.delta`` from ``(P, pdim)`` inputs.

    Returns ``(P,)``; negative is an improvement.

    Curvature is floored the same way :func:`compute_newton_step` floors it, so the
    trust-region ratio scores the step against the model it was built from. Under raw
    negative curvature the model rewards displacement without bound and a longer step
    always predicts a larger gain, which is not a quantity any step minimized.
    """
    linear = (gradient * step).sum(dim=-1)
    quadratic = 0.5 * (_floor_curvature(hessian_diag, hessian_reg) * step**2).sum(dim=-1)
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
    """Blend the OT step with a Newton correction read off the probe evaluations.

    The probes already carry second-order information, so the correction costs no
    extra forward passes. The Newton step is taken in the rotated frame, mapped back,
    anchored at ``X_current`` (the point the quadratic is built around), and blended
    ``(1 - alpha) * X_bary + alpha * X_newton``.

    Args:
        X_bary: Post-OT barycentric position, ``(P, pdim)``.
        losses_3d: Probe losses, ``(P, V, K)`` with ``V = 2*pdim``.
        scales: Probe scale factors, ``(K,)``.
        probe_radius: Probe distance multiplier.
        pdim: Particle dimension.
        rot_mats: Rotation matrices, ``(P, pdim, pdim)``.
        X_current: Probe center, ``(P, pdim)``. ``X_bary`` already carries the transport
            step, so anchoring the Newton step there would double-count that move.
        alpha: Blending weight; 0 is pure OT, 1 is pure Newton.
        max_step_norm: Trust-region bound on the correction.
        hessian_reg: Curvature floor for flat or nonpositive coordinates.

    Returns:
        Refined position, ``(P, pdim)``.
    """
    gradient = extract_fd_gradient(losses_3d, scales, probe_radius, pdim)
    hessian_diag = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)

    delta_rot = compute_newton_step(
        gradient,
        hessian_diag,
        max_step_norm=max_step_norm,
        hessian_reg=hessian_reg,
    )

    # Transform Newton step to original space: delta_orig = rot_mats @ delta_rot
    # rot_mats: (P, pdim, pdim), delta_rot: (P, pdim)
    delta_orig = torch.einsum("bij,bj->bi", rot_mats, delta_rot)

    X_refined = (1.0 - alpha) * X_bary + alpha * (X_current + delta_orig)

    # Keep the blend only where the model predicts it is no worse than the pure-OT
    # step. The model is diagonal in the rotated frame, so score both steps there.
    rot_mats_t = rot_mats.transpose(-1, -2)
    ot_rot = torch.einsum("bij,bj->bi", rot_mats_t, X_bary - X_current)
    refined_rot = torch.einsum("bij,bj->bi", rot_mats_t, X_refined - X_current)
    pred_ot = compute_predicted_improvement(gradient, hessian_diag, ot_rot, hessian_reg)
    pred_refined = compute_predicted_improvement(gradient, hessian_diag, refined_rot, hessian_reg)
    accept = (pred_refined <= pred_ot).unsqueeze(-1)
    return torch.where(accept, X_refined, X_bary)


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

    # Model and reality disagree in sign. Shrink twice as hard only when the model
    # promised a gain and the loss rose; the reverse case still improved the loss, so
    # it takes the ordinary shrink.
    if ratio < 0:
        factor = shrink_factor * 0.5 if pred < 0 else shrink_factor
        return max(current_radius * factor, min_radius)

    # Only expand when the model predicted an improvement (pred < 0) and reality
    # matched it. An accurate but worsening step (pred > 0, actual > 0) also gives
    # ratio ~1 and must not grow the region.
    if ratio > expand_threshold and pred < 0:
        return min(current_radius * expand_factor, max_radius)
    elif ratio < shrink_threshold:
        return max(current_radius * shrink_factor, min_radius)
    return current_radius
