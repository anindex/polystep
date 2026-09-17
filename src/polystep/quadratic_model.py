"""Quadratic model extraction from the cost evaluations of a centred tight frame.

``sum_v v = 0`` and ``sum_v v v^T = (V/d) I`` hold up to floating-point rounding
for every template, giving closed-form fits.
"""

import math

import torch


def extract_fd_gradient(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
    verts: torch.Tensor = None,
) -> torch.Tensor:
    """``g = (d / (V s r)) sum_v L_v v``, the rotated-frame least-squares gradient.

    ``verts=None`` uses central differences on orthoplex pairs. These recover the
    gradient of a quadratic up to rounding. Without antipodal pairs, a quadratic's
    third-moment contribution can bias the estimate by ``O(probe_radius)``.
    """
    V = losses_3d.shape[1]

    if verts is None:
        if V != 2 * pdim:
            raise ValueError(f"FD gradient needs `verts`, or orthoplex losses (V == 2*pdim); got V={V}, pdim={pdim}")
        # sum_v L_v v is L(+e_i) - L(-e_i) per coordinate, and d/V is 1/2.
        denom = (2.0 * scales * probe_radius).clamp(min=1e-10)  # (K,)
        return ((losses_3d[:, :pdim, :] - losses_3d[:, pdim:, :]) / denom).mean(dim=-1)

    # (P, V, K) against (V, pdim) -> (P, pdim, K)
    # The frame is centred mathematically, not bit-for-bit. Remove a common loss
    # before taking moments or a large constant objective invents a gradient.
    centered = losses_3d - losses_3d[:, :1, :]
    moment = torch.einsum("pvk,vd->pdk", centered, verts.to(dtype=losses_3d.dtype, device=losses_3d.device))
    denom = ((V / pdim) * scales * probe_radius).clamp(min=1e-10)  # (K,)
    return (moment / denom).mean(dim=-1)  # (P, pdim)


def extract_iso_curvature(
    losses_3d: torch.Tensor,
    center_loss: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
) -> torch.Tensor:
    """Isotropic curvature ``tr(H)/d`` as ``(P, 1)``, from the vertex mean and one ``f(X)``.

    ``mean_v v^T H v = tr(H sum_v v v^T)/V = tr(H)/d``, so this is exact at any radius for a quadratic objective.
    The frame is Haar-random each step, so ``E[H_jj] = tr(H)/d`` for every ``j``.
    """
    mean_over_v = losses_3d.mean(dim=1)  # (P, K)
    denom = (scales * probe_radius).pow(2).clamp(min=1e-30)  # (K,)
    curvature = (2.0 * (mean_over_v - center_loss.reshape(-1, 1)) / denom).mean(dim=-1, keepdim=True)  # (P, 1)
    return torch.where(torch.isfinite(curvature), curvature, torch.zeros_like(curvature))


def extract_fd_hessian_diag(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
) -> torch.Tensor:
    """Diagonal Hessian from the symmetric cost sum across probe scales.

    Regresses on unit-radius scales; non-finite regressions return zero curvature.
    """
    if not losses_3d.shape[1] == 2 * pdim:
        raise ValueError(f"FD Hessian needs orthoplex vertices (V == 2*pdim); got V={losses_3d.shape[1]}, pdim={pdim}")
    fwd = losses_3d[:, :pdim, :]  # (P, pdim, K)
    bwd = losses_3d[:, pdim:, :]  # (P, pdim, K)

    # Symmetric sum: L(+s) + L(-s) = 2a + H*(s*r)^2.
    sym_sum = fwd + bwd  # (P, pdim, K)

    s_sq = scales**2  # (K,), O(1) regardless of probe_radius

    # Least-squares slope of sym_sum on s_sq, centered for stability.
    s_centered = s_sq - s_sq.mean()  # (K,)
    y_centered = sym_sum - sym_sum.mean(dim=-1, keepdim=True)  # (P, pdim, K)

    numerator = (s_centered.unsqueeze(0).unsqueeze(0) * y_centered).sum(dim=-1)  # (P, pdim)
    denominator = (s_centered**2).sum().clamp(min=1e-10)
    # Convert from s^2 units to (s*r)^2.
    curvature = numerator / denominator / max(probe_radius**2, 1e-30)  # (P, pdim)
    return torch.where(torch.isfinite(curvature), curvature, torch.zeros_like(curvature))


def extract_fd_hessian_diag_centered(
    losses_3d: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
    pdim: int,
    center_loss: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Hessian from one probe scale plus a shared centre evaluation."""
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

    A floor, not additive regularization: ``H + reg`` leaves negative curvature
    negative, which flips the Newton step into an ascent direction.
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
    """Compute a diagonal Newton step in the rotated frame, clipped to ``max_step_norm``."""
    delta = -gradient / _floor_curvature(hessian_diag, hessian_reg)  # (P, pdim)

    # Cap per coordinate first; a flat coordinate would otherwise dominate the global clip.
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
    """``dL = g.delta + 0.5 * delta.H.delta`` from ``(P, pdim)`` inputs; negative is an improvement."""
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
    """Blend the OT step with a Newton correction read off the probe evaluations."""
    gradient = extract_fd_gradient(losses_3d, scales, probe_radius, pdim)
    hessian_diag = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)

    delta_rot = compute_newton_step(
        gradient,
        hessian_diag,
        max_step_norm=max_step_norm,
        hessian_reg=hessian_reg,
    )

    # Back to original space: delta_orig = rot_mats @ delta_rot.
    delta_orig = torch.einsum("bij,bj->bi", rot_mats, delta_rot)

    X_refined = (1.0 - alpha) * X_bary + alpha * (X_current + delta_orig)

    # Score both steps in the rotated frame and keep the blend only where it is no worse.
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
    """Update the trust-region multiplier from predicted vs actual improvement (negative means improvement)."""
    pred = predicted_improvement.mean().item()
    actual = actual_improvement.mean().item()

    if not (math.isfinite(pred) and math.isfinite(actual)):
        return max(current_radius * shrink_factor, min_radius)
    if abs(pred) < 1e-10:
        return current_radius

    # The model predicted the loss would rise, so the step is bad at any ratio.
    if pred > 0:
        return max(current_radius * shrink_factor, min_radius)

    ratio = max(-2.0, min(actual / pred, 5.0))

    # Promised a gain and the loss rose: shrink twice as hard.
    if ratio < 0:
        return max(current_radius * shrink_factor * 0.5, min_radius)
    if ratio > expand_threshold:
        return min(current_radius * expand_factor, max_radius)
    if ratio < shrink_threshold:
        return max(current_radius * shrink_factor, min_radius)
    return current_radius
