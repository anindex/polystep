"""Momentum and adaptive radius dynamics for Sinkhorn Step optimization.

Pure functions that compute momentum coefficients, apply velocity updates,
and adapt the step radius based on loss stagnation. These are composed by
``PolyStepOptimizer`` (via ``use_momentum`` and ``use_adaptive_radius``
parameters) but kept standalone for testability.

**Momentum** accumulates a velocity vector across optimization steps, similar
to SGD momentum but applied in the OT particle space. The OT barycentric
projection gives a displacement; momentum blends this with accumulated
velocity to smooth the trajectory and escape local minima.

**Adaptive radius** automatically adjusts the step size based on optimization
progress. When the loss stagnates (small relative change for several steps),
the radius is increased to explore a wider region. When the loss is improving,
the radius is decreased to exploit the current basin.
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
    """Linearly warm up momentum coefficient from init to final.

    Args:
        iteration: Current iteration index (0-based).
        max_iterations: Total number of iterations.
        momentum_init: Starting momentum value.
        momentum_final: Final momentum value.

    Returns:
        Interpolated momentum coefficient.
    """
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
    """Apply momentum update to particles.

    Computes the displacement from the OT barycentric projection, blends it
    with the accumulated velocity, and returns the updated positions and
    velocity.

    Args:
        X_old: Particle positions before the OT step, shape (N, D).
        X_barycentric: Particle positions from barycentric projection, shape (N, D).
        velocity: Previous velocity tensor, shape (N, D).
        beta: Momentum coefficient (0 = no momentum, 1 = full momentum).
        velocity_lr: Learning rate applied to the velocity update.

    Returns:
        Tuple of (X_new, velocity_new) where X_new are the updated positions
        and velocity_new is the updated velocity tensor.
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

    Args:
        current_loss: Loss at current iteration.
        prev_loss: Loss at previous iteration.
        stagnation_count: Current consecutive stagnation count.
        stagnation_threshold: Relative change below which an iteration is stagnating.

    Returns:
        (stagnation_count, current_loss). The loss is returned so the caller can store
        it as prev_loss.
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

    Note the direction is deliberately the opposite of a trust region, which grows the
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


def update_adaptive_radius(
    current_loss: float,
    prev_loss: float,
    stagnation_count: int,
    radius_multiplier: float,
    stagnation_threshold: float = 1e-4,
    stagnation_patience: int = 10,
    radius_increase: float = 1.5,
    radius_decrease: float = 0.9,
    radius_min: float = 0.5,
    radius_max: float = 3.0,
) -> Tuple[float, int, float]:
    """Update adaptive radius based on loss progress.

    Tracks stagnation (small relative loss change) and adjusts the radius
    multiplier: increases on prolonged stagnation to explore more, decreases
    on improvement to exploit.

    Args:
        current_loss: Loss at current iteration.
        prev_loss: Loss at previous iteration.
        stagnation_count: Current consecutive stagnation count.
        radius_multiplier: Current radius multiplier.
        stagnation_threshold: Relative change below which iteration is stagnating.
        stagnation_patience: Stagnation iterations before radius boost.
        radius_increase: Multiplicative factor for radius boost.
        radius_decrease: Multiplicative factor for radius decay on improvement.
        radius_min: Minimum allowed radius multiplier.
        radius_max: Maximum allowed radius multiplier.

    Returns:
        Tuple of (radius_multiplier, stagnation_count, current_loss) where
        current_loss is returned so the caller can store it as prev_loss.
    """
    # Composition of the two halves above, kept for backward compatibility. Both no-op
    # on non-finite losses, so NaN cannot reset the counter or corrupt the multiplier,
    # and the first step (prev_loss=inf) is skipped.
    stagnation_count, prev = update_stagnation(current_loss, prev_loss, stagnation_count, stagnation_threshold)
    radius_multiplier, stagnation_count = update_radius_multiplier(
        current_loss,
        prev_loss,
        stagnation_count,
        radius_multiplier,
        stagnation_patience=stagnation_patience,
        radius_increase=radius_increase,
        radius_decrease=radius_decrease,
        radius_min=radius_min,
        radius_max=radius_max,
    )
    return (radius_multiplier, stagnation_count, prev)
