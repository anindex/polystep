"""Epsilon schedulers for entropic regularization decay.

Entropic regularization (epsilon) controls the smoothness of the optimal
transport plan. Higher epsilon gives a smoother, more diffuse plan (easier
to solve but less precise). Lower epsilon gives a sharper plan (closer to
exact OT but harder to solve numerically). Annealing from high to low
epsilon allows coarse-to-fine optimization.

Provides two schedulers:

- ``LinearEpsilon``: Fixed linear decay: eps_t = max(init - decay * t, target).
- ``ProgressiveEpsilon``: Feedback-driven auto-adjustment based on Sinkhorn solver
  convergence behavior, inspired by ProgOT (Kassab & Thornton, 2025).
"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LinearEpsilon:
    """Linearly decaying epsilon scheduler.

    Computes epsilon at step t as:
        epsilon(t) = max(init - decay * t, target)

    The schedule starts at ``init`` and decreases by ``decay`` per iteration
    until it reaches the ``target`` floor. This enables coarse-to-fine
    optimization: early iterations use high epsilon for broad exploration,
    later iterations use low epsilon for precise refinement.

    Attributes:
        target: Floor value for epsilon.
        init: Starting value at iteration 0.
        decay: Amount to subtract per iteration.
    """

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01

    def at(self, iteration: Optional[int] = 1) -> float:
        """Compute epsilon at the given iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Epsilon value (float).
        """
        if iteration is None:
            # Not yet started - return initial epsilon, not target
            return self.init
        eps = self.init - (self.decay * iteration)
        return max(eps, self.target)


@dataclass
class ProgressiveEpsilon:
    """Auto-adjusting epsilon based on Sinkhorn solver feedback.

    Inspired by ProgOT (Kassab & Thornton, 2025): epsilon is adjusted
    based on actual solver convergence behavior rather than a fixed schedule.

    When the solver converges quickly (few iterations), epsilon is decreased
    to sharpen the transport plan. When the solver struggles (many iterations
    or fails to converge), epsilon is increased to stabilize solving.

    The ``at()`` method is a drop-in replacement for ``LinearEpsilon.at()``,
    but ignores the ``iteration`` parameter. Instead, epsilon is driven by
    explicit ``update()`` calls from the optimizer after each OT solve.

    Attributes:
        init: Starting epsilon value.
        target: Floor value (minimum epsilon).
        max_epsilon: Ceiling value (maximum epsilon).
        increase_factor: Multiplicative increase when solver struggles.
        decrease_factor: Multiplicative decrease when solver converges fast.
        fast_threshold: n_iters below this fraction of max_iterations triggers decrease.
        slow_threshold: n_iters above this fraction of max_iterations triggers increase.
        ema_alpha: EMA smoothing for epsilon changes (0 = no smoothing, 1 = no change).
    """

    init: float = 1.0
    target: float = 0.01
    max_epsilon: float = 5.0
    increase_factor: float = 1.2
    decrease_factor: float = 0.95
    fast_threshold: float = 0.1
    slow_threshold: float = 0.5
    ema_alpha: float = 0.7

    # Internal state (not part of constructor signature for users)
    _current: float = field(init=False, repr=False)
    _smoothed: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._current = self.init
        self._smoothed = self.init

    def at(self, iteration: Optional[int] = None) -> float:
        """Return current epsilon value. Compatible with LinearEpsilon interface.

        The ``iteration`` parameter is accepted for API compatibility with
        ``LinearEpsilon`` but is ignored. ProgressiveEpsilon is driven by
        explicit ``update()`` calls, not iteration count.

        Args:
            iteration: Ignored. Present for API compatibility.

        Returns:
            Current smoothed epsilon value (float).
        """
        return self._smoothed

    def update(self, n_iters: int, max_iterations: int, converged: bool) -> None:
        """Update epsilon based on solver feedback.

        Args:
            n_iters: Number of Sinkhorn iterations used in the last solve.
            max_iterations: Maximum iterations the solver was allowed.
            converged: Whether the solver converged.
        """
        ratio = n_iters / max(max_iterations, 1)

        if not converged or ratio > self.slow_threshold:
            # Solver struggling: increase epsilon
            self._current = min(
                self._current * self.increase_factor, self.max_epsilon
            )
        elif ratio < self.fast_threshold:
            # Solver converging fast: decrease epsilon
            self._current = max(
                self._current * self.decrease_factor, self.target
            )
        # else: ratio in [fast_threshold, slow_threshold] -- keep current

        # EMA smooth
        self._smoothed = (
            self.ema_alpha * self._smoothed
            + (1.0 - self.ema_alpha) * self._current
        )
        self._smoothed = max(self._smoothed, self.target)
        self._smoothed = min(self._smoothed, self.max_epsilon)


@dataclass
class CosineEpsilon:
    """Cosine-annealed epsilon scheduler.

    Computes epsilon at step t using cosine annealing:
        epsilon(t) = target + 0.5 * (init - target) * (1 + cos(pi * t / T))

    where T = total_steps (computed from init/target/decay for API compatibility
    with LinearEpsilon, or set explicitly via total_steps).

    Cosine annealing keeps epsilon higher for longer in the middle of training
    (encouraging exploration) then decays rapidly at the end (encouraging
    exploitation). This typically improves convergence versus linear decay.

    Supports optional warm restarts (SGDR-style): epsilon periodically resets
    to init, with each period multiplied by ``restart_mult``.

    Attributes:
        target: Floor value for epsilon.
        init: Starting value at iteration 0.
        decay: Per-step linear decay rate (used to infer total_steps if
            total_steps is not set: T = (init - target) / decay).
        total_steps: Explicit total step count. Overrides decay-based inference.
        restart_mult: Period multiplier for warm restarts (1.0 = no restarts).
    """

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01
    total_steps: int = 0
    restart_mult: float = 1.0

    def at(self, iteration: Optional[int] = 1) -> float:
        """Compute epsilon at the given iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Epsilon value (float).
        """
        if iteration is None:
            return self.init

        T = self.total_steps if self.total_steps > 0 else max(
            1, int((self.init - self.target) / max(self.decay, 1e-12))
        )

        if self.restart_mult > 1.0:
            # Warm restart: find which period we're in
            period = T
            t = iteration
            # Guard: limit iterations to prevent unbounded loop when
            # restart_mult is very close to 1.0 or period is tiny
            max_restarts = 100
            restarts = 0
            while t >= period and period > 0 and restarts < max_restarts:
                t -= period
                period = int(period * self.restart_mult)
                restarts += 1
            T_local = max(period, 1)
            t_local = t
        else:
            T_local = T
            t_local = min(iteration, T)

        cos_val = math.cos(math.pi * t_local / max(T_local, 1))
        return self.target + 0.5 * (self.init - self.target) * (1.0 + cos_val)
