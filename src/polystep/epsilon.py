"""Epsilon schedulers for entropic regularization decay."""

import math
from dataclasses import dataclass, field
from typing import Optional


def resolve_radius(radius, iteration: int, epsilon: float) -> float:
    """The physical radius at ``iteration``: a scalar is ``radius * epsilon``, a schedule is its own value."""
    base = radius.at(iteration) if hasattr(radius, "at") else radius
    return float(base) * radius_epsilon_factor(radius, epsilon)


def radius_epsilon_factor(radius, epsilon: float) -> float:
    """``1.0`` for a scheduled radius, ``epsilon`` for a scalar one. Keep the existing multiplication order."""
    return 1.0 if hasattr(radius, "at") else float(epsilon)


@dataclass
class LinearEpsilon:
    """``epsilon(t) = max(init - decay * t, target)``."""

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01

    def __post_init__(self) -> None:
        # A negative decay grows epsilon without bound and stalls training.
        if self.decay < 0:
            raise ValueError(f"decay must be >= 0, got {self.decay}")

    def at(self, iteration: Optional[int] = None) -> float:
        if iteration is None:
            # Not yet started: initial epsilon, not target.
            return self.init
        eps = self.init - (self.decay * iteration)
        return max(eps, self.target)


@dataclass
class PowerDecay:
    """``r(t) = max(init * (t + 1) ** -(0.5 + gamma), target)``, the step-radius schedule the convergence analysis assumes."""

    init: float = 1.0
    gamma: float = 0.1
    target: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0 for a square-summable schedule, got {self.gamma}")

    def at(self, iteration: Optional[int] = None) -> float:
        t = 0 if iteration is None else max(int(iteration), 0)
        return max(self.init * (t + 1) ** -(0.5 + self.gamma), self.target)


@dataclass
class ProgressiveEpsilon:
    """Epsilon driven by Sinkhorn convergence, after ProgOT (arXiv:2406.05061)."""

    init: float = 1.0
    target: float = 0.01
    max_epsilon: float = 5.0
    increase_factor: float = 1.2
    decrease_factor: float = 0.95
    fast_threshold: float = 0.1
    slow_threshold: float = 0.5
    ema_alpha: float = 0.7

    # Internal state (not part of constructor signature for users).
    _current: float = field(init=False, repr=False)
    _smoothed: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._current = self.init
        self._smoothed = self.init

    def at(self, iteration: Optional[int] = None) -> float:
        """``iteration`` is ignored; ``update()`` drives this scheduler."""
        return self._smoothed

    def update(self, n_iters: int, max_iterations: int, converged: bool) -> None:
        """Move epsilon from the last solve's iteration count and converged flag."""
        ratio = n_iters / max(max_iterations, 1)

        if not converged or ratio > self.slow_threshold:
            self._current = min(self._current * self.increase_factor, self.max_epsilon)
        elif ratio < self.fast_threshold:
            self._current = max(self._current * self.decrease_factor, self.target)
        # Between the thresholds epsilon holds.

        self._smoothed = self.ema_alpha * self._smoothed + (1.0 - self.ema_alpha) * self._current
        self._smoothed = max(self._smoothed, self.target)
        self._smoothed = min(self._smoothed, self.max_epsilon)


def feed_solver_stats(scheduler, solver, n_iters: int, converged: bool) -> None:
    """Feed a solve's stats to ``scheduler``; no-op for schedulers that ignore them.

    Skipped at ``threshold <= 0``: fixed-iteration Sinkhorn always reports converged with
    ``n_iters == max_iterations``, a ratio of 1.0, which would raise epsilon to
    ``max_epsilon`` every step.
    """
    if not isinstance(scheduler, ProgressiveEpsilon) or getattr(solver, "threshold", 1.0) <= 0:
        return
    scheduler.update(
        n_iters=n_iters,
        max_iterations=getattr(solver, "max_iterations", 1),
        converged=converged,
    )


@dataclass
class CosineEpsilon:
    """``epsilon(t) = target + 0.5 * (init - target) * (1 + cos(pi * t / T))``."""

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01
    total_steps: int = 0
    restart_mult: float = 1.0

    def at(self, iteration: Optional[int] = None) -> float:
        if iteration is None:
            return self.init

        # ceil, not truncation, so the floor lands where LinearEpsilon's would.
        T = (
            self.total_steps
            if self.total_steps > 0
            else max(1, math.ceil((self.init - self.target) / max(self.decay, 1e-12)))
        )

        if self.restart_mult > 1.0:
            # Walk to the current restart period; the bound stops a near-1.0 restart_mult looping forever.
            period = T
            t = iteration
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

        # Clamp so the restart loop can't push cos past pi.
        t_local = min(max(t_local, 0), T_local)
        cos_val = math.cos(math.pi * t_local / max(T_local, 1))
        return self.target + 0.5 * (self.init - self.target) * (1.0 + cos_val)
