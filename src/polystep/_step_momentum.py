"""Amortized momentum step and related helpers."""

from __future__ import annotations

import math
from typing import Callable

import torch


def step_momentum(opt, closure: Callable) -> float:
    """Cheap momentum step: apply EMA transport direction with decay.

    Applies the EMA-smoothed transport direction without a validation
    forward pass.  NaN safety is guaranteed by reverting to the
    pre-step position if non-finite values appear.

    Args:
        closure: ``closure(batched_params) -> losses`` (unused - kept
            for API compatibility with other ``_step_*`` methods).

    Returns:
        Reused cost from the last OT step.
    """
    state = opt._state
    opt._update_sampling_projection()
    if opt._transport_direction_ema is None:
        return state.costs[-1] if state.costs else float("inf")

    # Linear decay: first momentum step gets full strength, last gets minimal
    phase = (opt._amortize_counter % opt.amortize_steps) / opt.amortize_steps
    decay = 1.0 - phase

    # Store pre-step X for NaN safety
    X_old = state.X.clone()

    # Choose momentum direction: Newton (if available) or EMA transport
    if opt.use_quadratic_model and opt._newton_direction is not None:
        direction = opt._newton_direction
    else:
        direction = opt._transport_direction_ema

    # Apply direction with decay.
    state.X = state.X + decay * direction

    # NaN check - revert to pre-step state and use previous cost
    if not torch.isfinite(state.X).all():
        state.X = X_old
        opt._transport_direction = None
        opt._transport_direction_ema = None
        opt._newton_direction = None
        # X_old is where the last OT step left the particles only on the first
        # momentum step of a run; after that it is itself a momentum position, so
        # the cached cost rows still describe somewhere the particles have left.
        opt._invalidate_reuse_cache()
        opt._sync_model()
        prev_cost = state.costs[-1] if state.costs else float("inf")
        state.costs.append(prev_cost)
        state.linear_convergence.append(True)
        state.displacement_sqnorms.append(0.0)
        state.iteration_count += 1
        return prev_cost

    # The particles moved, so the adaptive-probe cache now describes a position
    # they have left. Its reuse guards check K_eff and both radii, none of which
    # change on a momentum step, so nothing else would catch it.
    opt._invalidate_reuse_cache()

    # Sync model parameters from updated particles
    opt._sync_model()

    # Reuse last cost (no validation forward pass)
    cost = state.costs[-1] if state.costs else float("inf")

    # Update diagnostics
    disp_sqnorm = torch.mean(torch.sum((state.X - X_old) ** 2, dim=-1)).item()
    state.costs.append(cost)
    state.linear_convergence.append(True)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.iteration_count += 1

    return cost
