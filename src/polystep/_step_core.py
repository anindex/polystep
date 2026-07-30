"""State reset shared by the step drivers."""

from __future__ import annotations

import torch


def invalidate_for_basis_change(opt, state) -> None:
    """Drop every cached direction and dual that indexes the outgoing basis.

    Call after an absorb or a rotation, once ``state.X`` is re-anchored. Duals index
    vertices of the old chart; the transport EMA, the Newton direction and the momentum
    velocity are coordinates in it.
    """
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
