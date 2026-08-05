"""Epsilon schedules: the value they return at each step, and their bounds.

A schedule that silently ignores its floor or ceiling retunes every run built on it,
with no error. ProgressiveEpsilon is driven by solver feedback rather than by step.
"""

import math

import pytest

from polystep.epsilon import CosineEpsilon, LinearEpsilon, ProgressiveEpsilon


class TestLinearEpsilon:
    def test_decays_linearly_from_init_and_holds_at_target(self):
        s = LinearEpsilon(init=1.0, decay=0.01, target=0.2)
        assert s.at(0) == pytest.approx(1.0)
        assert s.at(50) == pytest.approx(0.5)
        # 80 steps reaches the floor exactly; past it the value must not keep sliding.
        assert s.at(80) == pytest.approx(0.2)
        assert s.at(10_000) == pytest.approx(0.2)

    def test_unstarted_returns_init_not_target(self):
        """``at(None)`` means no step has run, which is the top of the schedule."""
        assert LinearEpsilon(init=1.0, decay=0.01, target=0.2).at(None) == 1.0

    def test_rejects_negative_decay(self):
        """A growing epsilon flattens the plan to uniform: training stalls with no error."""
        with pytest.raises(ValueError, match="decay"):
            LinearEpsilon(init=1.0, decay=-0.01)


class TestCosineEpsilon:
    def test_spans_init_to_target_over_the_inferred_horizon(self):
        # decay infers T = (init - target) / decay = 80, so a LinearEpsilon config transfers.
        s = CosineEpsilon(init=1.0, decay=0.01, target=0.2)
        assert s.at(0) == pytest.approx(1.0)
        assert s.at(40) == pytest.approx(0.6)  # half of T sits at the midpoint
        assert s.at(80) == pytest.approx(0.2)
        assert s.at(500) == pytest.approx(0.2), "past the horizon the value must clamp, not wrap"

    def test_inferred_horizon_matches_linear_on_a_fractional_ratio(self):
        """``(init - target) / decay`` need not be integral; truncating it ends the
        cosine before the LinearEpsilon it transfers from reaches the floor."""
        cos = CosineEpsilon(init=1.0, decay=0.01, target=1e-3)
        lin = LinearEpsilon(init=1.0, decay=0.01, target=1e-3)
        first_floor = next(t for t in range(200) if lin.at(t) == pytest.approx(lin.target))
        assert cos.at(first_floor) == pytest.approx(cos.target)
        assert cos.at(first_floor - 1) > cos.target

    def test_unstarted_returns_init_not_target(self):
        assert CosineEpsilon(init=1.0, decay=0.01, target=0.2).at(None) == 1.0

    def test_stays_above_linear_through_the_middle(self):
        """The reason to pick cosine: it holds epsilon high while linear is already low."""
        cos = CosineEpsilon(init=1.0, decay=0.01, target=0.0)
        lin = LinearEpsilon(init=1.0, decay=0.01, target=0.0)
        assert all(cos.at(t) > lin.at(t) for t in range(5, 50))

    def test_warm_restarts_return_to_init(self):
        s = CosineEpsilon(init=1.0, decay=0.01, target=0.2, total_steps=10, restart_mult=2.0)
        assert s.at(10) == pytest.approx(1.0, rel=0.05), "a restart must climb back toward init"
        assert all(math.isfinite(s.at(t)) and 0.2 <= s.at(t) <= 1.0 for t in range(200))


class TestProgressiveEpsilon:
    def test_at_returns_init_value(self):
        """ProgressiveEpsilon.at(0) returns init value."""
        pe = ProgressiveEpsilon(init=1.0, target=0.01)
        assert pe.at(0) == 1.0

    def test_fast_convergence_decreases_epsilon(self):
        """After update(n_iters=5, converged=True) (fast), next epsilon decreases."""
        pe = ProgressiveEpsilon(
            init=1.0,
            target=0.01,
            max_epsilon=5.0,
            fast_threshold=0.1,
            slow_threshold=0.5,
            decrease_factor=0.95,
            increase_factor=1.2,
            ema_alpha=0.0,  # no smoothing for clear test
        )
        initial = pe.at()
        pe.update(n_iters=5, max_iterations=1000, converged=True)
        after = pe.at()
        assert after < initial, f"Expected decrease: {after} < {initial}"

    def test_slow_convergence_increases_epsilon(self):
        """After update(n_iters=500, converged=False) (slow), next epsilon increases."""
        pe = ProgressiveEpsilon(
            init=1.0,
            target=0.01,
            max_epsilon=5.0,
            fast_threshold=0.1,
            slow_threshold=0.5,
            decrease_factor=0.95,
            increase_factor=1.2,
            ema_alpha=0.0,  # no smoothing for clear test
        )
        initial = pe.at()
        pe.update(n_iters=500, max_iterations=1000, converged=False)
        after = pe.at()
        assert after > initial, f"Expected increase: {after} > {initial}"

    def test_epsilon_never_below_target(self):
        """Epsilon never goes below target floor."""
        pe = ProgressiveEpsilon(
            init=0.05,
            target=0.01,
            max_epsilon=5.0,
            decrease_factor=0.5,
            ema_alpha=0.0,
        )
        for _ in range(100):
            pe.update(n_iters=1, max_iterations=1000, converged=True)
        assert pe.at() >= 0.01

    def test_epsilon_never_above_max(self):
        """Epsilon never goes above max_epsilon ceiling."""
        pe = ProgressiveEpsilon(
            init=1.0,
            target=0.01,
            max_epsilon=5.0,
            increase_factor=2.0,
            ema_alpha=0.0,
        )
        for _ in range(100):
            pe.update(n_iters=999, max_iterations=1000, converged=False)
        assert pe.at() <= 5.0

    def test_ema_smoothing_prevents_oscillation(self):
        """Multiple update() calls with EMA smoothing create a smooth trajectory."""
        pe = ProgressiveEpsilon(
            init=1.0,
            target=0.01,
            max_epsilon=5.0,
            fast_threshold=0.1,
            slow_threshold=0.5,
            decrease_factor=0.8,
            increase_factor=1.5,
            ema_alpha=0.7,  # heavy smoothing
        )
        values = [pe.at()]
        for i in range(10):
            if i % 2 == 0:
                pe.update(n_iters=1, max_iterations=1000, converged=True)  # fast
            else:
                pe.update(n_iters=999, max_iterations=1000, converged=False)  # slow
            values.append(pe.at())

        # EMA=0.7 caps each step at 30% of the raw change, so steps stay small.
        max_change = max(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))
        assert max_change < 0.5, (
            f"EMA smoothing failed: max step change = {max_change:.4f}, values = {[f'{v:.4f}' for v in values]}"
        )


class TestProgressiveEpsilonIsDrivenByEveryStepDriver:
    """``at()`` moves only when a driver feeds it solver stats, so every driver must."""

    @staticmethod
    def _run(**kwargs):
        import torch
        import torch.nn as nn
        from torch.func import functional_call, vmap

        from polystep.optimizer import PolyStepOptimizer

        model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
        kwargs.setdefault("auto_epsilon", True)
        opt = PolyStepOptimizer(model, seed=0, compile=False, **kwargs)
        x = torch.randn(8, 6, generator=torch.Generator().manual_seed(0))

        def closure(batched_params):
            return vmap(lambda p: functional_call(model, p, (x,)).pow(2).mean())(batched_params)

        seen = []
        for _ in range(6):
            opt.step(closure)
            seen.append(opt._progressive_epsilon.at(None))
        return seen

    def test_scheduler_passed_as_epsilon_is_driven(self):
        """A user-supplied ``epsilon=ProgressiveEpsilon(...)`` must advance, not just
        the one built by ``auto_epsilon=True``."""
        seen = self._run(auto_epsilon=False, epsilon=ProgressiveEpsilon(init=1.0, target=0.01, max_epsilon=5.0))
        assert len(set(seen)) > 1, f"epsilon pinned at {seen[0]}"

    @pytest.mark.parametrize("strategy", ["monolithic", "per_layer", "grouped"])
    def test_epsilon_moves_under_every_block_strategy(self, strategy):
        seen = self._run(block_strategy=strategy)
        assert len(set(seen)) > 1, f"{strategy}: epsilon pinned at {seen[0]}"

    def test_standalone_polystep_advances_epsilon(self):
        import torch

        from polystep.solver import PolyStep

        scheduler = ProgressiveEpsilon(init=1.0, target=0.01, max_epsilon=5.0)
        solver = PolyStep(
            objective_fn=lambda points: (points**2).sum(dim=-1),
            dim=2,
            epsilon=scheduler,
            compile=False,
        )
        state = solver.init_state(torch.randn(6, 2, generator=torch.Generator().manual_seed(0)))

        seen = []
        for _ in range(6):
            state = solver.step(state)
            seen.append(scheduler.at(None))
        assert len(set(seen)) > 1, f"epsilon pinned at {seen[0]}"


class TestDefaults:
    """The defaults are what an unconfigured run gets, so pin the behaviour they set."""

    def test_linear_defaults_decay_from_one_to_the_target_floor(self):
        s = LinearEpsilon()
        assert s.at(0) == pytest.approx(1.0)
        assert s.at(50) == pytest.approx(0.5)
        assert s.at(10_000) == pytest.approx(1e-3)

    def test_progressive_defaults_raise_on_a_slow_solve_and_cap(self):
        pe = ProgressiveEpsilon()
        assert pe.at() == pytest.approx(1.0)
        # ratio 1.0 is past slow_threshold, so epsilon climbs by increase_factor
        # and the EMA lags it.
        pe.update(n_iters=100, max_iterations=100, converged=True)
        assert pe.at() == pytest.approx(0.7 * 1.0 + 0.3 * 1.2)
        for _ in range(200):
            pe.update(n_iters=100, max_iterations=100, converged=True)
        assert pe.at() == pytest.approx(5.0)

    def test_progressive_defaults_lower_on_a_fast_solve_and_floor(self):
        pe = ProgressiveEpsilon()
        pe.update(n_iters=5, max_iterations=100, converged=True)
        assert pe.at() == pytest.approx(0.7 * 1.0 + 0.3 * 0.95)
        for _ in range(500):
            pe.update(n_iters=5, max_iterations=100, converged=True)
        assert pe.at() == pytest.approx(0.01)

    @pytest.mark.parametrize(
        "n_iters, expected",
        [
            (9, 0.7 + 0.3 * 0.95),  # ratio 0.09, just under fast_threshold: decrease
            (11, 1.0),  # ratio 0.11, just over it: hold
            (49, 1.0),  # ratio 0.49, just under slow_threshold: still hold
            (51, 0.7 + 0.3 * 1.2),  # ratio 0.51, just over it: increase
        ],
    )
    def test_progressive_defaults_straddle_both_thresholds(self, n_iters, expected):
        pe = ProgressiveEpsilon()
        pe.update(n_iters=n_iters, max_iterations=100, converged=True)
        assert pe.at() == pytest.approx(expected)


def test_power_decay_is_square_summable_but_not_summable():
    """r_t = r_0 (t+1)^-(1/2+gamma): the schedule the convergence analysis assumes."""
    from polystep.epsilon import PowerDecay

    s = PowerDecay(init=2.0, gamma=0.1)
    assert s.at(None) == s.at(0) == 2.0
    assert s.at(1) == pytest.approx(2.0 * 2**-0.6)
    vals = [s.at(t) for t in range(50)]
    assert all(a > b > 0 for a, b in zip(vals, vals[1:]))
    assert PowerDecay(init=2.0, gamma=0.1, target=1.0).at(1000) == 1.0
    with pytest.raises(ValueError):
        PowerDecay(init=1.0, gamma=0.0)


class TestRadiusEpsilonSemantics:
    """A scalar radius is a multiplier on epsilon; a scheduled one is a distance.

    Both entry points must apply this rule the same way.
    """

    def test_scalar_radius_is_a_multiplier_on_epsilon(self):
        from polystep.epsilon import radius_epsilon_factor, resolve_radius

        assert resolve_radius(2.0, 0, 0.5) == 1.0
        assert resolve_radius(2.0, 999, 0.5) == 1.0  # no iteration dependence
        assert radius_epsilon_factor(2.0, 0.5) == 0.5

    def test_scheduled_radius_is_the_physical_distance(self):
        from polystep.epsilon import radius_epsilon_factor, resolve_radius

        sched = CosineEpsilon(init=32.0, target=8.0, decay=(32.0 - 8.0) / 100)
        assert resolve_radius(sched, 0, 0.5) == 32.0
        assert resolve_radius(sched, 10**6, 0.5) == pytest.approx(8.0)
        assert radius_epsilon_factor(sched, 0.5) == 1.0
        # Epsilon must not enter: a different temperature gives the same distance.
        assert resolve_radius(sched, 7, 0.5) == resolve_radius(sched, 7, 50.0)

    @pytest.mark.parametrize("radius", [2.0, CosineEpsilon(init=4.0, target=1.0, decay=0.03)])
    def test_both_entry_points_agree(self, radius):
        """PolyStepOptimizer's expression and PolyStep's must give the same distance."""
        from polystep.epsilon import radius_epsilon_factor, resolve_radius

        eps, iteration = 0.5, 13
        resolved = radius.at(iteration) if hasattr(radius, "at") else radius
        optimizer_side = resolved * radius_epsilon_factor(radius, eps) * 1.0
        solver_side = resolve_radius(radius, iteration, eps)
        assert optimizer_side == solver_side

    def test_solver_accepts_a_scheduled_radius_without_raising(self):
        """A scheduled radius must not raise a TypeError in the solver."""
        import torch

        from polystep.solver import PolyStep

        solver = PolyStep.create(
            lambda x: (x**2).sum(dim=-1),
            dim=2,
            epsilon=0.5,
            step_radius=CosineEpsilon(init=1.0, target=0.25, decay=0.05),
            probe_radius=CosineEpsilon(init=2.0, target=0.5, decay=0.05),
            max_iterations=3,
        )
        state = solver.run(torch.zeros(4, 2))
        assert len(state.costs) == 3
        assert all(c == c for c in state.costs)  # no NaN
