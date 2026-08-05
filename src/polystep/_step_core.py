"""State reset shared by the step drivers."""

from __future__ import annotations

import warnings

import torch


def warn_all_nonfinite(opt) -> None:
    """Warn once when every candidate of a step evaluated to NaN or inf."""
    if getattr(opt, "_nonfinite_warned", False):
        return
    opt._nonfinite_warned = True
    warnings.warn(
        "every candidate evaluated to NaN or inf this step, so the cost matrix "
        "carries no ranking information: the plan is uniform, the step is exactly "
        "zero, and the reported cost is the sanitize penalty rather than a loss.",
        RuntimeWarning,
        stacklevel=3,
    )


def record_saturation(opt, state, transport, step_radius) -> None:
    """Log how far each transport row is from uniform; off unless asked for."""
    log = getattr(opt, "_saturation_log", None)
    if log is None:
        return
    V = transport.shape[-1]
    p = transport / transport.sum(-1, keepdim=True).clamp_min(1e-30)
    omega = (p - 1.0 / V).abs().amax(-1) * V
    # At lambda = 0 the row is softmax(-C/eps), so log(p_max/p_min) is the exact cost-row spread.
    # Rows whose smallest weight underflows float32 are counted, not trusted: their spread is a floor.
    p_min = p.amin(-1)
    p_max = p.amax(-1)
    spread = torch.log(p_max.clamp_min(1e-30)) - torch.log(p_min.clamp_min(1e-30))
    underflow = p_min <= 1e-30
    log.append(
        {
            "iteration": int(getattr(state, "iteration_count", len(log))),
            "step_radius": float(step_radius),
            "particles": int(omega.numel()),
            "saturated_frac": float((omega > 1.0).to(torch.float32).mean()),
            "omega_mean": float(omega.mean()),
            "omega_max": float(omega.amax()),
            "spread_mean": float(spread.mean()),
            "spread_max": float(spread.amax()),
            "spread_underflow_frac": float(underflow.to(torch.float32).mean()),
            "spread_sat_frac": float((spread > 1.0).to(torch.float32).mean()),
        }
    )


def update_stagnation_and_radius(opt, state, loss: float) -> None:
    """Advance the stagnation counter and, if enabled, the adaptive radius."""
    from .dynamics import update_radius_multiplier, update_stagnation

    prev_loss = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        loss,
        state.prev_loss,
        state.stagnation_count,
        stagnation_threshold=opt.stagnation_threshold,
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            loss,
            prev_loss,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )


def update_amortized_direction(opt, raw_direction, nan_reverted: bool) -> None:
    """EMA the coasting direction an amortized step reuses, or drop it after a revert."""
    if opt.amortize_steps <= 1:
        return
    if nan_reverted:
        opt._transport_direction_ema = None
        return
    alpha = opt.amortize_ema
    prev = opt._transport_direction_ema
    opt._transport_direction_ema = raw_direction if prev is None else alpha * prev + (1.0 - alpha) * raw_direction


def invalidate_for_basis_change(opt, state) -> None:
    """Drop every cached direction and dual that indexes the outgoing basis; call after an absorb or rotation."""
    state.f = None
    state.g = None
    state.prev_prev_f = None
    state.prev_prev_g = None
    if getattr(state, "block_duals", None) is not None:
        state.block_duals = [(None, None) for _ in state.block_duals]
    state._prev_prev_block_duals = None

    opt._transport_direction_ema = None
    opt._newton_direction = None
    opt._prev_descent_direction = None
    opt._prev_descent_direction_finite = False
    opt._prev_block_descent_directions = None
    opt._center_loss = None
    opt._prev_predicted_improvement = None
    opt._prev_pre_step_loss = None
    opt._invalidate_reuse_cache()

    if opt.use_momentum and state.velocity is not None:
        state.velocity = torch.zeros_like(state.velocity)
