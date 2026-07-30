"""Momentum and adaptive radius dynamics, as pure functions.

Momentum accumulates a velocity across steps in OT particle space, blending it with
the displacement the barycentric projection gives. Adaptive radius grows the step on
stagnation and shrinks it on improvement; see ``update_radius_multiplier`` for why
that direction is deliberate. ``PolyStepOptimizer`` composes both through
``use_momentum`` and ``use_adaptive_radius``.
"""

import math
from typing import Tuple

import torch


def compute_momentum_coefficient(
    iteration: int,
    max_iterations: int,
    momentum_init: float = 0.5,
    momentum_final: float = 0.95,
) -> float:
    """Linear warm-up from ``momentum_init`` to ``momentum_final`` over the run."""
    progress = min(1.0, iteration / max(1, max_iterations - 1))
    return momentum_init + progress * (momentum_final - momentum_init)


@torch.inference_mode()
def apply_momentum(
    X_old: torch.Tensor,
    X_barycentric: torch.Tensor,
    velocity: torch.Tensor,
    beta: float,
    velocity_lr: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend the barycentric displacement into the velocity, all tensors ``(N, D)``.

    Returns ``(X_new, velocity_new)``. ``beta`` runs 0 (no momentum) to 1 (full).
    """
    displacement = X_barycentric - X_old
    velocity_new = beta * velocity + displacement
    X_new = X_old + velocity_lr * velocity_new
    return X_new, velocity_new


@torch.inference_mode()
def update_stagnation(
    current_loss: float,
    prev_loss: float,
    stagnation_count: int,
    stagnation_threshold: float = 1e-4,
) -> Tuple[int, float]:
    """Count consecutive near-flat steps.

    Separate from radius adaptation because the subspace absorb trigger
    (``absorb_mode="stagnation"``) also reads ``state.stagnation_count``, and that
    counter must advance even when ``use_adaptive_radius`` is False.

    Returns ``(stagnation_count, current_loss)``; the loss comes back so the caller can
    store it as ``prev_loss``.
    """
    # No progress signal from a non-finite loss, so leave the counter alone.
    if not math.isfinite(current_loss) or not math.isfinite(prev_loss):
        return (stagnation_count, current_loss)

    rel_change = abs(current_loss - prev_loss) / (abs(prev_loss) + 1e-10)
    return ((stagnation_count + 1) if rel_change < stagnation_threshold else 0, current_loss)


@torch.inference_mode()
def update_radius_multiplier(
    current_loss: float,
    prev_loss: float,
    stagnation_count: int,
    radius_multiplier: float,
    stagnation_patience: int = 10,
    radius_increase: float = 1.5,
    radius_decrease: float = 0.9,
    radius_min: float = 0.5,
    radius_max: float = 3.0,
) -> Tuple[float, int]:
    """Adjust the radius multiplier from an already-updated stagnation count.

    Boosts the radius after ``stagnation_patience`` flat steps, decays it on
    improvement. A boost resets ``stagnation_count``, so with
    ``stagnation_patience <= absorb_patience`` it fires before a stagnation absorb can.
    ``PolyStepOptimizer`` warns on that combination.

    The direction is the opposite of a trust region, which grows the
    step on a successful iteration. Here every improving step multiplies the radius by
    ``radius_decrease``, so a healthy run contracts toward ``radius_min`` and only
    stagnation restores reach. The intent is to settle into the current basin and to keep
    exploration for when progress stops, not to accelerate down a slope.

    Returns:
        (radius_multiplier, stagnation_count).
    """
    if not math.isfinite(current_loss) or not math.isfinite(prev_loss):
        return (radius_multiplier, stagnation_count)

    if stagnation_count >= stagnation_patience:
        radius_multiplier = min(radius_multiplier * radius_increase, radius_max)
        stagnation_count = 0
    elif current_loss < prev_loss:
        radius_multiplier = max(radius_multiplier * radius_decrease, radius_min)

    return (radius_multiplier, stagnation_count)
