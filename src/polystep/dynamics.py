"""Momentum and adaptive radius dynamics, as pure functions."""

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
    """Blend the barycentric displacement into the velocity; returns ``(X_new, velocity_new)``.

    Heavy-ball, as in torch SGD: no ``(1-beta)`` on the increment, so the steady-state move
    is ``velocity_lr/(1-beta)`` times the displacement. Only the displacement is scaled by
    ``step_radius``, the trust region and the radius controller.
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
    """Count consecutive near-flat steps; returns ``(stagnation_count, current_loss)``."""
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
    """Adjust the radius multiplier from the updated stagnation count.

    Boosts after ``stagnation_patience`` flat steps, decays on improvement. The
    direction is the opposite of a trust region: improving steps shrink toward
    ``radius_min`` so the run settles into the current basin.
    """
    if not math.isfinite(current_loss) or not math.isfinite(prev_loss):
        return (radius_multiplier, stagnation_count)

    if stagnation_count >= stagnation_patience:
        radius_multiplier = min(radius_multiplier * radius_increase, radius_max)
        stagnation_count = 0
    elif current_loss < prev_loss:
        radius_multiplier = max(radius_multiplier * radius_decrease, radius_min)

    return (radius_multiplier, stagnation_count)
