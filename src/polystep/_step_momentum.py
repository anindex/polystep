"""Amortized momentum step and related helpers."""

from __future__ import annotations

from typing import Callable

import torch


def step_momentum(opt, closure: Callable) -> float:
    """Cheap step: reapply the last direction with decay, no forward passes."""
    state = opt._state
    opt._update_sampling_projection()
    if opt._transport_direction_ema is None:
        return state.costs[-1] if state.costs else float("inf")

    # Linear decay across the cheap steps.
    phase = (opt._amortize_counter % opt.amortize_steps) / opt.amortize_steps
    decay = 1.0 - phase

    # No clone: the update below is out-of-place, so this reference stays valid.
    X_old = state.X

    if opt.use_quadratic_model and opt._newton_direction is not None:
        direction = opt._newton_direction
    else:
        direction = opt._transport_direction_ema

    state.X = state.X + decay * direction

    # NaN check: revert to pre-step state and reuse the previous cost.
    if not torch.isfinite(state.X).all():
        state.X = X_old
        opt._transport_direction_ema = None
        opt._newton_direction = None
        # X_old is a momentum position after the first step, so the cached rows still describe somewhere the particles left.
        opt._invalidate_reuse_cache()
        opt._sync_model()
        prev_cost = state.costs[-1] if state.costs else float("inf")
        state.costs.append(prev_cost)
        # No solve ran; True would read as "converged" to an early-stop callback.
        state.linear_convergence.append(False)
        state.displacement_sqnorms.append(0.0)
        state.record_solver_health(evals=0)
        state.iteration_count += 1
        return prev_cost

    # The particles moved, so the adaptive-probe cache is stale; its reuse guards wouldn't catch it.
    opt._invalidate_reuse_cache()

    opt._sync_model()

    # Reuse the last cost; no validation forward.
    cost = state.costs[-1] if state.costs else float("inf")

    disp_sqnorm = torch.mean(torch.sum((state.X - X_old) ** 2, dim=-1)).item()
    state.costs.append(cost)
    state.linear_convergence.append(False)
    state.displacement_sqnorms.append(disp_sqnorm)
    # No OT solve and no forwards, so ess/rho carry forward.
    state.record_solver_health(evals=0)
    state.iteration_count += 1

    return cost
