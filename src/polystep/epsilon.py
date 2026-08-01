"""Epsilon schedulers for entropic regularization decay.

High epsilon gives a smooth, diffuse transport plan that is easy to solve; low epsilon
gives a sharp one closer to exact OT but harder numerically. Annealing high to low is
coarse-to-fine.

- ``LinearEpsilon``: ``eps_t = max(init - decay * t, target)``.
- ``CosineEpsilon``: cosine annealing, optional SGDR-style warm restarts.
- ``ProgressiveEpsilon``: driven by Sinkhorn convergence rather than by ``t``, after
  ProgOT (Kassraie et al., NeurIPS 2024, arXiv:2406.05061).
- ``PowerDecay``: ``r_t = init * (t + 1)^-(1/2 + gamma)``, the step-radius schedule
  Theorem 4.2 assumes.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LinearEpsilon:
    """``epsilon(t) = max(init - decay * t, target)``."""

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01

    def __post_init__(self) -> None:
        # A negative decay grows epsilon without bound: the plan flattens to uniform, the
        # step becomes the mean of the polytope vertices, and training stalls with no error.
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
    """``r(t) = max(init * (t + 1) ** -(0.5 + gamma), target)``.

    The step-radius schedule the convergence analysis assumes: square-summable but
    not summable, which is what makes the noise term vanish while the iterates can
    still travel an unbounded distance. ``gamma`` around 0.1 is the usual choice.
    Same ``at(iteration)`` interface as the epsilon schedulers, so it drops into
    ``PolyStepOptimizer(step_radius=...)`` unchanged.

    Attributes:
        init: ``r_0``.
        gamma: Extra decay beyond ``t^-1/2``; must be > 0 for square-summability.
        target: Floor, so a long run does not shrink the step into fp32 noise.
    """

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
    """Epsilon driven by Sinkhorn convergence, after ProgOT (arXiv:2406.05061).

    A fast solve sharpens the plan by decreasing epsilon; a slow or failed one raises it
    to stay solvable. ``at()`` ignores its ``iteration`` argument, matching
    ``LinearEpsilon``'s interface; the optimizer calls ``update()`` after each solve.

    Attributes:
        fast_threshold: ``n_iters/max_iterations`` below this decreases epsilon.
        slow_threshold: the same ratio above this increases it.
        ema_alpha: smoothing on the change; 0 none, 1 freezes epsilon.
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

    Called by every step driver, so ``ProgressiveEpsilon`` advances in monolithic,
    block-wise and standalone ``PolyStep`` runs alike.

    Skipped at ``threshold <= 0``: fixed-iteration Sinkhorn always reports converged
    with ``n_iters == max_iterations``, a ratio of 1.0, which would raise epsilon to
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
    """``epsilon(t) = target + 0.5 * (init - target) * (1 + cos(pi * t / T))``.

    Holds epsilon high through the middle of the run and drops it late, where linear
    decay is already at the floor.

    Attributes:
        decay: infers ``T = (init - target) / decay`` when ``total_steps`` is unset,
            so a ``LinearEpsilon`` config transfers unchanged.
        total_steps: explicit ``T``, overriding that inference.
        restart_mult: SGDR period multiplier; 1.0 disables warm restarts.
    """

    target: float = 1e-3
    init: float = 1.0
    decay: float = 0.01
    total_steps: int = 0
    restart_mult: float = 1.0

    def at(self, iteration: Optional[int] = None) -> float:
        if iteration is None:
            return self.init

        # ceil, not truncation: a fractional ratio (the defaults give 99.9) would
        # otherwise reach the floor one step before the LinearEpsilon it transfers from.
        T = (
            self.total_steps
            if self.total_steps > 0
            else max(1, math.ceil((self.init - self.target) / max(self.decay, 1e-12)))
        )

        if self.restart_mult > 1.0:
            # Walk to the current restart period. Bounded: a restart_mult near 1.0 or a
            # tiny period would otherwise loop unbounded.
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

        # Clamp so a maxed-out restart loop can't push cos past pi (which would
        # drift epsilon outside [target, init]).
        t_local = min(max(t_local, 0), T_local)
        cos_val = math.cos(math.pi * t_local / max(T_local, 1))
        return self.target + 0.5 * (self.init - self.target) * (1.0 + cos_val)
