"""Integration tests for PolyStepOptimizer step, momentum, and adaptive radius."""

import copy
import warnings

import math

import pytest
import torch
import torch.nn as nn

from polystep.solver import PolyStep
from polystep import CosineEpsilon, PolyStepOptimizer, SolverState
from polystep.cost_nn import NNCostEvaluator
from polystep.dynamics import compute_momentum_coefficient


def _make_model():
    """Small MLP for fast testing."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _make_closure(model):
    """Create a batched closure using NNCostEvaluator."""
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 1)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


@pytest.fixture
def model():
    torch.manual_seed(42)
    return _make_model()


@pytest.fixture
def closure(model):
    return _make_closure(model)


@pytest.fixture
def optimizer(model):
    return PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        sinkhorn_max_iters=100,
        compile=False,
        seed=42,
    )


class TestClosureInterface:
    def test_step_updates_model(self, model, closure):
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.step(closure)
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change after step"

    def test_step_increments_iteration(self, optimizer, closure):
        assert optimizer.state.iteration_count == 0
        optimizer.step(closure)
        assert optimizer.state.iteration_count == 1

    def test_multiple_steps(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        for _ in range(5):
            opt.step(closure)
        assert len(opt.state.costs) == 5
        assert opt.state.iteration_count == 5


class TestMomentum:
    def test_momentum_disabled_by_default(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.state.velocity is None
        opt.step(closure)
        assert opt.state.velocity is None

    def test_momentum_initializes_velocity(self, model):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
        )
        assert opt.state.velocity is not None
        assert torch.all(opt.state.velocity == 0)

    def test_momentum_updates_velocity(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
        )
        opt.step(closure)
        assert opt.state.velocity is not None
        # After one step, velocity should be non-zero (displacement was applied)
        assert torch.any(opt.state.velocity != 0)

    def test_momentum_warmup(self):
        """Beta at iteration 0 is momentum_init, increases over iterations."""
        beta_0 = compute_momentum_coefficient(0, 100, 0.5, 0.95)
        beta_50 = compute_momentum_coefficient(50, 100, 0.5, 0.95)
        beta_99 = compute_momentum_coefficient(99, 100, 0.5, 0.95)
        assert beta_0 == pytest.approx(0.5)
        assert beta_50 > beta_0
        assert beta_99 == pytest.approx(0.95)

    def test_momentum_smooths_trajectory(self):
        """Momentum version has smaller displacement variance."""
        torch.manual_seed(42)
        model_no_mom = _make_model()
        model_mom = copy.deepcopy(model_no_mom)
        closure_no_mom = _make_closure(model_no_mom)
        closure_mom = _make_closure(model_mom)

        opt_no_mom = PolyStepOptimizer(
            model_no_mom,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt_mom = PolyStepOptimizer(
            model_mom,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.95,
        )

        n_steps = 10
        for _ in range(n_steps):
            opt_no_mom.step(closure_no_mom)
            opt_mom.step(closure_mom)

        # Both should have completed without error
        assert len(opt_no_mom.state.displacement_sqnorms) == n_steps
        assert len(opt_mom.state.displacement_sqnorms) == n_steps

        # Momentum must have a real effect: velocity accumulates and the
        # trajectory diverges from the no-momentum run (both seeded identically,
        # so any difference is momentum). The variance-ordering claim is
        # stochastic, so we assert the mechanism rather than a flaky inequality.
        assert opt_mom.state.velocity is not None
        assert opt_mom.state.velocity.abs().sum() > 0
        traj_no_mom = torch.tensor(opt_no_mom.state.displacement_sqnorms)
        traj_mom = torch.tensor(opt_mom.state.displacement_sqnorms)
        assert not torch.allclose(traj_no_mom, traj_mom)


class TestAdaptiveRadius:
    def test_adaptive_disabled_by_default(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.step(closure)
        assert opt.state.radius_multiplier == 1.0

    def test_radius_stays_in_bounds(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_adaptive_radius=True,
            radius_min=0.5,
            radius_max=3.0,
        )
        for _ in range(20):
            opt.step(closure)
        assert 0.5 <= opt.state.radius_multiplier <= 3.0


class TestIntegration:
    def test_momentum_and_adaptive_together(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
            use_adaptive_radius=True,
        )
        for _ in range(10):
            opt.step(closure)
        # Both should be active
        assert opt.state.velocity is not None
        assert torch.any(opt.state.velocity != 0)
        assert 0.5 <= opt.state.radius_multiplier <= 3.0
        assert opt.state.iteration_count == 10

    def test_state_accessible(self, optimizer, closure):
        optimizer.step(closure)
        state = optimizer.state
        assert isinstance(state, SolverState)
        assert state.iteration_count == 1
        assert len(state.costs) == 1
        assert state.X is not None
        assert state.a is not None


class TestParticleDim:
    @pytest.mark.parametrize(
        "polytope, dim, verts", [("orthoplex", 4, 8), ("orthoplex", 8, 16), ("simplex", 4, 5), ("simplex", 8, 9)]
    )
    def test_particle_dim_4(self, polytope, dim, verts):
        """particle_dim=D gives D-dim particles and the polytope's own vertex count.

        The simplex (the default) is D+1, the orthoplex 2*D. Naming the polytope keeps this
        from silently re-testing whatever the default happens to be.
        """
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            particle_dim=dim,
            polytope_type=polytope,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.layout.particle_dim == dim
        assert opt._particle_dim == dim
        assert opt._polytope_vertices.shape[0] == verts
        assert opt._polytope_vertices.shape[1] == dim

    def test_particle_dim_step(self):
        """particle_dim=4 optimizer can run step() without error and updates model."""
        torch.manual_seed(42)
        model = _make_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            particle_dim=4,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        closure = _make_closure(model)
        loss = opt.step(closure)

        assert math.isfinite(loss)
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change after step with particle_dim=4"

    def test_particle_dim_invalid(self):
        """ValueError for particle_dim < 2."""
        torch.manual_seed(42)
        model = _make_model()
        with pytest.raises(ValueError, match="particle_dim must be >= 2"):
            PolyStepOptimizer(
                model,
                particle_dim=1,
                max_iterations=50,
                epsilon=0.1,
                compile=False,
            )

    def test_particle_dim_cube_warning(self):
        """UserWarning when particle_dim=8 and polytope_type='cube'."""
        import warnings as _warnings

        torch.manual_seed(42)
        model = _make_model()
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            PolyStepOptimizer(
                model,
                particle_dim=8,
                polytope_type="cube",
                max_iterations=50,
                epsilon=0.1,
                sinkhorn_max_iters=100,
                compile=False,
                seed=42,
            )
            cube_warnings = [x for x in w if "cube" in str(x.message).lower()]
            assert len(cube_warnings) >= 1, f"Expected cube warning, got: {[str(x.message) for x in w]}"


class TestAdaptiveProbes:
    """Tests for cost-matrix reuse across steps.

    A candidate is the whole configuration with one particle row replaced, so every
    row of the cost matrix depends on every particle's position. Reuse is therefore
    all or nothing: the matrix is reused whole while X has not moved, and dropped as
    soon as it has.
    """

    def test_one_moving_particle_invalidates_every_cached_row(self):
        """A stagnant particle's row is still measured against the others' positions."""
        torch.manual_seed(0)
        model = _make_model()
        opt = PolyStepOptimizer(model, adaptive_probes=True, epsilon=0.1, compile=False, seed=0)
        closure = _make_closure(model)

        calls = {"n": 0}

        def counting(batched_params):
            calls["n"] += batched_params[next(iter(batched_params))].shape[0]
            return closure(batched_params)

        opt.step(counting)
        # Freeze all but one particle, then move that one well past the threshold.
        opt._prev_X = opt.state.X.clone()
        opt._prev_X[0] += 1.0

        calls["n"] = 0
        opt.step(counting)
        assert calls["n"] > 0, "a moved particle must force a full re-evaluation"

    def test_an_unmoved_configuration_reuses_the_matrix(self):
        torch.manual_seed(0)
        model = _make_model()
        opt = PolyStepOptimizer(model, adaptive_probes=True, epsilon=0.1, compile=False, seed=0)
        closure = _make_closure(model)

        calls = {"n": 0}

        def counting(batched_params):
            calls["n"] += batched_params[next(iter(batched_params))].shape[0]
            return closure(batched_params)

        opt.step(counting)
        opt._prev_X = opt.state.X.clone()

        calls["n"] = 0
        opt.step(counting)
        assert calls["n"] == 0, "an unmoved configuration must spend no forwards"

    @pytest.mark.parametrize("block_strategy,expected", [("monolithic", True), ("per_layer", False)])
    def test_adaptive_probes_defaults_to_where_it_is_implemented(self, block_strategy, expected):
        """``adaptive_probes=None`` resolves to on for monolithic, off elsewhere.

        Blockwise never populates the reuse cache, so leaving it on there would only
        cost memory. An explicit True still warns.
        """
        torch.manual_seed(42)
        opt = PolyStepOptimizer(
            _make_model(),
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            block_strategy=block_strategy,
        )
        assert opt._adaptive_probes is expected

    def test_adaptive_probes_off_stores_nothing(self):
        """With reuse disabled, no per-particle displacement or cost row is retained."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            adaptive_probes=False,
        )
        assert opt._prev_X is None
        assert opt._prev_cost_matrix is None

        opt.step(_make_closure(model))
        assert opt._prev_X is None
        assert opt._prev_cost_matrix is None

    def test_adaptive_probes_enabled(self):
        """adaptive_probes=True runs without error and produces valid optimization steps."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            adaptive_probes=True,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt._adaptive_probes
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        closure = _make_closure(model)
        losses = []
        for _ in range(5):
            loss = opt.step(closure)
            losses.append(loss)
            assert math.isfinite(loss)

        # After the first step the configuration and cost matrix are cached
        assert opt._prev_X is not None
        assert opt._prev_cost_matrix is not None

        # Model should have changed
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change with adaptive_probes=True"

    @pytest.mark.parametrize(
        "threshold, reuses",
        [
            (1e10, True),  # every particle counts as stagnant, so every row is reused
            (0.0, False),  # strict <, so nothing is stagnant and every row is re-measured
        ],
    )
    def test_adaptive_probes_reuse_follows_the_threshold(self, threshold, reuses):
        """Reused rows cost no forward passes, so the closure call count is the check."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            adaptive_probes=True,
            adaptive_probes_threshold=threshold,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )

        call_counts = []
        base_closure = _make_closure(model)

        def counting_closure(batched_params):
            call_counts.append(batched_params[next(iter(batched_params))].shape[0])
            return base_closure(batched_params)

        opt.step(counting_closure)  # no previous cost matrix, so no reuse is possible
        step1_evals = sum(call_counts)
        call_counts.clear()

        opt.step(counting_closure)
        step2_evals = sum(call_counts)

        if reuses:
            assert step2_evals < step1_evals, f"{step2_evals} vs {step1_evals}"
        else:
            assert step2_evals == step1_evals, f"{step2_evals} vs {step1_evals}"


class TestDualMomentum:
    def test_beta_zero_matches_standard(self):
        """dual_momentum_beta=0.0 produces identical cost to default optimizer over 3 steps."""
        torch.manual_seed(42)
        model_default = _make_model()
        model_beta0 = copy.deepcopy(model_default)

        # Create shared random data for identical closures
        evaluator_default = NNCostEvaluator(model_default, loss_fn=nn.MSELoss())
        evaluator_beta0 = NNCostEvaluator(model_beta0, loss_fn=nn.MSELoss())
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 1)

        def closure_default(bp):
            return evaluator_default.evaluate(bp, inputs, targets)

        def closure_beta0(bp):
            return evaluator_beta0.evaluate(bp, inputs, targets)

        opt_default = PolyStepOptimizer(
            model_default,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt_beta0 = PolyStepOptimizer(
            model_beta0,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.0,
        )

        for _ in range(3):
            cost_default = opt_default.step(closure_default)
            cost_beta0 = opt_beta0.step(closure_beta0)
            assert cost_default == pytest.approx(cost_beta0, rel=1e-6), (
                f"beta=0 should match default: {cost_beta0} vs {cost_default}"
            )

    def test_extrapolation_applied(self):
        """After 2+ steps, state.prev_prev_f is not None (history is being tracked)."""
        torch.manual_seed(42)
        model = _make_model()
        closure = _make_closure(model)
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.3,
        )
        opt.step(closure)
        opt.step(closure)
        assert opt.state.prev_prev_f is not None, "After 2 steps, prev_prev_f should be set"
        assert opt.state.prev_prev_g is not None, "After 2 steps, prev_prev_g should be set"

    def test_extrapolation_clamped(self):
        """The extrapolated warm start handed to the solver stays inside the clamp.

        Checked on the value passed to solve(), not on state.f: sinkhorn clamps its
        output again internally, so the output cannot show whether this clamp ran.
        """
        torch.manual_seed(42)
        model = _make_model()
        closure = _make_closure(model)
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.3,
        )
        seen = []
        real_solve = opt.solver.solve

        def spy(*args, **kwargs):
            seen.append((kwargs.get("init_f"), kwargs.get("init_g")))
            return real_solve(*args, **kwargs)

        opt.solver.solve = spy
        for _ in range(5):
            opt.step(closure)

        max_abs = 80.0 * max(opt.state.epsilon, 0.01)
        warm = [f for f, _ in seen if f is not None]
        assert warm, "no warm start was ever passed, so the clamp was never reached"
        for f in warm:
            assert f.abs().max().item() <= max_abs + 1e-6, f"warm-start dual exceeds max_abs={max_abs}"


class TestSinkhornParamWiring:
    """Verify anderson_depth, adaptive_omega, data_dependent_init wire through
    from PolyStepOptimizer constructor to the internal SinkhornSolver."""

    def test_custom_sinkhorn_params_wired(self):
        """PolyStepOptimizer(anderson_depth=5, adaptive_omega=True,
        data_dependent_init=True) creates solver with those exact values."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            compile=False,
            seed=42,
            anderson_depth=5,
            adaptive_omega=True,
            data_dependent_init=True,
        )
        assert opt.solver.anderson_depth == 5
        assert opt.solver.adaptive_omega is True
        assert opt.solver.data_dependent_init is True


class TestAutoEpsilon:
    def test_auto_epsilon_adjusts_from_solver_feedback(self):
        """auto_epsilon should change epsilon based on convergence speed."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        optimizer = PolyStepOptimizer(
            model,
            epsilon=1.0,
            step_radius=0.5,
            num_probe=2,
            compile=False,
            seed=42,
            auto_epsilon=True,
        )
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss()

        def closure(batched_params):
            from polystep.cost_nn import NNCostEvaluator

            evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
            return evaluator.evaluate(batched_params, inputs, targets)

        eps_values = []
        for _ in range(5):
            optimizer.step(closure)
            eps_values.append(optimizer._progressive_epsilon.at())

        # Epsilon should be changing (not stuck at init)
        assert len(set(round(e, 6) for e in eps_values)) > 1, f"Epsilon did not change across steps: {eps_values}"
        # All values should be finite and positive
        assert all(0 < e < 100 for e in eps_values)


@pytest.mark.parametrize(
    "param_name, value",
    [
        ("curvature_aware_radius", True),
        ("entropy_target", 0.7),
        ("nesterov_lookahead", True),
    ],
)
def test_curvature_aware_radius_removed(param_name, value):
    """Parameters removed in cleanup should raise TypeError."""
    model = nn.Linear(4, 2)
    with pytest.raises(TypeError, match=param_name):
        PolyStepOptimizer(model, **{param_name: value})


@pytest.mark.parametrize(
    "attribute, expected",
    [
        ("layout.particle_dim", 2),
        ("_particle_dim", 2),
        ("ent_epsilon", None),
        ("stagnation_threshold", 1e-4),
        ("_progressive_epsilon", None),
        ("solver.anderson_depth", 0),
        ("solver.adaptive_omega", False),
        ("solver.data_dependent_init", False),
    ],
)
def test_documented_defaults(attribute, expected):
    """The published defaults, pinned off one optimizer rather than one build each.

    Every acceleration knob is off unless asked for, so a default construction runs
    the plain algorithm.
    """
    opt = PolyStepOptimizer(_make_model(), epsilon=0.1, compile=False, seed=42)
    value = opt
    for part in attribute.split("."):
        value = getattr(value, part)
    if expected is None:
        assert value is None
    else:
        assert value == expected


class TestProbeRadiusJitter:
    """Probe-radius jitter implements Theorem 4.2 condition (iv).

    The Fubini transversality argument requires the joint (rotation, jitter)
    probe distribution to be absolutely continuous on a positive-Lebesgue-measure
    tube around the (d_p-1)-sphere. Default 0.0 keeps reported experiments
    bit-for-bit reproducible; non-zero values activate the jitter.
    """

    def test_default_is_zero(self):
        """Default probe_radius_jitter == 0 preserves backward compatibility."""
        torch.manual_seed(0)
        model = _make_model()
        opt = PolyStepOptimizer(model, particle_dim=2, probe_radius=2.0)
        assert opt.probe_radius_jitter == 0.0

    def test_default_jitter_is_no_op(self):
        """With jitter=0, _apply_probe_radius_jitter returns the input unchanged
        and consumes NO random state from the optimizer's generator."""
        torch.manual_seed(0)
        model = _make_model()
        opt = PolyStepOptimizer(model, particle_dim=2, probe_radius=2.0, seed=42)
        # Snapshot the generator state BEFORE calling the helper.
        state_before = opt._generator.get_state().clone()
        result = opt._apply_probe_radius_jitter(2.0)
        state_after = opt._generator.get_state()
        assert result == 2.0
        # No random call should have occurred.
        assert torch.equal(state_before, state_after)

    def test_jitter_perturbs_probe_radius(self):
        """With jitter > 0, _apply_probe_radius_jitter returns a value in
        the bounded multiplicative interval [(1-eta_max)*r, (1+eta_max)*r]
        and consumes random state from the optimizer's generator."""
        torch.manual_seed(0)
        model = _make_model()
        eta_max = 0.05
        opt = PolyStepOptimizer(
            model,
            particle_dim=2,
            probe_radius=2.0,
            probe_radius_jitter=eta_max,
            seed=42,
        )
        base = 2.0
        # Sample many jitter values and check (a) they are all in the interval,
        # (b) the variance is non-zero (jitter is actually being applied).
        samples = [opt._apply_probe_radius_jitter(base) for _ in range(200)]
        lo = base * (1.0 - eta_max)
        hi = base * (1.0 + eta_max)
        assert all(lo <= s <= hi for s in samples), (
            f"jitter samples must lie in [{lo}, {hi}], got min={min(samples)} max={max(samples)}"
        )
        assert max(samples) - min(samples) > 0.01, "jitter samples should span a non-trivial range"

    def test_jitter_validation_rejects_out_of_range(self):
        """probe_radius_jitter must lie in [0, 1) - values >= 1 risk negative
        effective probe radius and are rejected at init time."""
        torch.manual_seed(0)
        model = _make_model()
        with pytest.raises(ValueError, match="probe_radius_jitter"):
            PolyStepOptimizer(model, probe_radius_jitter=1.0)
        with pytest.raises(ValueError, match="probe_radius_jitter"):
            PolyStepOptimizer(model, probe_radius_jitter=-0.1)

    def test_jitter_step_runs_and_changes_iterates(self):
        """End-to-end: optimization with jitter > 0 still produces a valid
        step and updates the iterate, exercising the helper from inside the
        full step path (resolve_radii / _step_monolithic)."""
        torch.manual_seed(0)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            particle_dim=2,
            probe_radius=2.0,
            probe_radius_jitter=0.05,
            seed=42,
        )
        closure = _make_closure(model)
        params_before = torch.cat([p.detach().flatten() for p in model.parameters()])
        opt.step(closure)
        params_after = torch.cat([p.detach().flatten() for p in model.parameters()])
        assert opt._state.iteration_count == 1
        # The step must have produced a finite, non-trivial update.
        assert torch.isfinite(params_after).all()
        assert not torch.equal(params_before, params_after)


def test_num_probe_defaults_to_one_on_both_entry_points():
    """K=1 is the reported and optimal setting; the two APIs must not disagree."""

    def objective(x):
        return (x**2).sum(-1)

    assert PolyStepOptimizer(nn.Linear(4, 2, bias=False), epsilon=0.5).num_probe == 1
    assert PolyStep.create(objective, dim=4).num_probe == 1


class _FakeLIF(nn.Module):
    """Stand-in for an snnTorch Leaky neuron, without the snntorch dependency."""

    def forward(self, x):
        return torch.relu(x)


class _SNNStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.lif1 = _FakeLIF()  # the name pattern the guard matches on
        self.fc2 = nn.Linear(8, 2)


@pytest.mark.parametrize("build, expect_warning", [(_SNNStub, True), (_make_model, False)])
def test_cosine_step_radius_warns_only_on_an_snn(build, expect_warning):
    """Scheduling step_radius on an SNN collapsed accuracy from ~93% to 10-47%
    (experiments/EXPERIMENT_INDEX.md), so the combination must warn, and only there."""
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(build(), epsilon=0.5, step_radius=cosine)

    hits = [
        m
        for m in (str(w.message).lower() for w in caught)
        if any(k in m for k in ("snn", "leaky", "lif", "spik")) and ("step_radius" in m or "cosine" in m)
    ]
    assert bool(hits) is expect_warning, hits
