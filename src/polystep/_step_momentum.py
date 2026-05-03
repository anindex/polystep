"""Amortized momentum step and related helpers.

Extracted from optimizer.py for maintainability.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

logger = __import__("logging").getLogger(__name__)


def amortize_should_revert(opt, amortized: float, last: float) -> bool:
    """Decide whether the amortized momentum step regressed badly.

    The natural threshold ``amortized > 1.5 * last`` only makes
    sense for positive losses. Once losses go negative -- e.g.
    entropic OT cost, log-likelihood, or any objective with a
    symmetry-breaking constant -- the multiplicative test inverts
    (``1.5 * -10 = -15`` is BETTER than -10). Fall back to an
    additive tolerance whenever either value is negative or the
    reference is too close to zero to anchor a relative test.

    Args:
        amortized: Loss after the cheap momentum step.
        last: Reference loss from the previous fresh OT step.

    Returns:
        True if the optimizer should revert to the pre-momentum
        state and recompute a fresh OT step.
    """
    if math.isnan(amortized) or amortized == float("inf"):
        return True
    threshold = opt.amortize_loss_gate_threshold
    floor = opt.amortize_loss_gate_floor
    if amortized < 0 or last < 0 or abs(last) < floor:
        # Additive tolerance: revert when amortized exceeds last
        # by more than (threshold - 1) * floor.
        return amortized > last + (threshold - 1.0) * floor
    return amortized > threshold * last



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
    if opt._transport_direction_ema is None:
        return state.costs[-1] if state.costs else float('inf')

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
        opt._sync_model()
        prev_cost = state.costs[-1] if state.costs else float('inf')
        state.costs.append(prev_cost)
        state.linear_convergence.append(True)
        state.displacement_sqnorms.append(0.0)
        state.iteration_count += 1
        return prev_cost

    # Sync model parameters from updated particles
    opt._sync_model()

    # Reuse last cost (no validation forward pass)
    cost = state.costs[-1] if state.costs else float('inf')

    # Update diagnostics
    disp_sqnorm = torch.mean(torch.sum((state.X - X_old) ** 2, dim=-1)).item()
    state.costs.append(cost)
    state.linear_convergence.append(True)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.iteration_count += 1

    return cost


def evaluate_current_loss(opt, closure: Callable) -> float:
    """Evaluate model loss at current particle position via single forward pass.

    Not called by the core optimization loop (momentum validation was
    removed), but retained as a public diagnostic utility.

    Builds a batched param config (batch=1) from current state.X and
    calls the closure. Handles both subspace and full-space modes.

    Returns:
        Scalar loss value (float).
    """
    state = opt._state
    with torch.no_grad():
        if opt.subspace is not None:
            flat_sub = state.X.reshape(-1)[:state.subspace.subspace_dim].unsqueeze(0)
            if opt._mixed_precision and state.projection is not None:
                flat_sub = flat_sub.to(dtype=state.projection.dtype)
            if opt._adaptive or opt._cma_subspace:
                val_params = state.subspace.reconstruct_batch(
                    state.projection, state.base_params, flat_sub,
                )
            elif opt._hybrid:
                val_params = state.subspace.reconstruct_batch(
                    state.hybrid_projections, state.base_params, flat_sub,
                )
            else:
                val_params = state.subspace.reconstruct_batch(
                    state.base_params, flat_sub,
                )
        else:
            flat_config = state.X.reshape(1, -1)
            layout_flat = opt.layout.padded_size
            if flat_config.shape[1] >= layout_flat:
                flat_for_layout = flat_config[:, :layout_flat]
            else:
                flat_for_layout = torch.nn.functional.pad(
                    flat_config, (0, layout_flat - flat_config.shape[1]),
                )
            val_params = opt.layout.batch_unflatten(flat_for_layout)

        val_loss_tensor = closure(val_params)
        val_loss = val_loss_tensor.mean().item()
        # Guard against NaN propagation - treat as infinite loss
        if math.isnan(val_loss):
            return float('inf')
        return val_loss
