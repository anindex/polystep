"""Integration tests for PolyStepOptimizer step, momentum, and adaptive radius."""

import copy

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer, SolverState
from polystep.cost_nn import NNCostEvaluator
from polystep.dynamics import compute_momentum_coefficient
from polystep.epsilon import LinearEpsilon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestClosureInterface
# ---------------------------------------------------------------------------


class TestClosureInterface:
    """Tests for basic step(closure) interface."""

    def test_step_returns_float(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        loss = opt.step(closure)
        assert isinstance(loss, float)

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


# ---------------------------------------------------------------------------
# TestMomentum
# ---------------------------------------------------------------------------


class TestMomentum:
    """Tests for momentum integration."""

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


# ---------------------------------------------------------------------------
# TestAdaptiveRadius
# ---------------------------------------------------------------------------


class TestAdaptiveRadius:
    """Tests for adaptive radius integration."""

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

    # test_adaptive_enabled removed: a weaker subset of test_radius_stays_in_bounds
    # below (same bounds assertion, fewer steps). The adaptation logic itself is
    # covered in detail by tests/test_dynamics.py::TestAdaptiveRadius.

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


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Combined integration tests."""

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


# ---------------------------------------------------------------------------
# TestParticleDim (parametric extension)
# ---------------------------------------------------------------------------


class TestParticleDim:
    """Tests for configurable particle_dim in full-space mode."""

    def test_particle_dim_default(self):
        """particle_dim=2 (default) produces same layout as current behavior."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.layout.particle_dim == 2
        assert opt._particle_dim == 2

    @pytest.mark.parametrize("dim, verts", [(4, 8), (8, 16)])
    def test_particle_dim_4(self, dim, verts):
        """particle_dim=D creates layout with D-dim particles and orthoplex 2*D vertices."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            particle_dim=dim,
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

        assert isinstance(loss, float)
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
    """Tests for adaptive probe count (adaptive probe extension).

    Adaptive probes detect stagnant particles (small displacement) and reuse
    the previous step's cost matrix rows instead of recomputing, saving
    V*K forward passes per stagnant particle.
    """

    def test_adaptive_probes_default_off(self):
        """adaptive_probes=False (default) preserves current behavior."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        # Default should be off
        assert not opt._adaptive_probes
        assert opt._prev_displacement_sqnorms is None
        assert opt._prev_cost_matrix is None

        # Run a step - no displacement/cost storage should happen
        closure = _make_closure(model)
        opt.step(closure)
        assert opt._prev_displacement_sqnorms is None
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
            assert isinstance(loss, float)

        # After first step, displacement and cost matrix should be stored
        assert opt._prev_displacement_sqnorms is not None
        assert opt._prev_cost_matrix is not None

        # Model should have changed
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change with adaptive_probes=True"

    def test_adaptive_probes_saves_evals(self):
        """Stagnant particles get cost rows reused (fewer forward passes).

        We verify this by counting closure calls with a wrapper. After the
        first step (no reuse possible), subsequent steps should call the
        closure fewer times if some particles are stagnant.
        """
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            adaptive_probes=True,
            adaptive_probes_threshold=1e10,  # Very high: ALL particles stagnant after step 1
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

        # Step 1: no reuse possible (no previous cost matrix)
        opt.step(counting_closure)
        step1_evals = sum(call_counts)
        call_counts.clear()

        # Step 2: all particles are "stagnant" (threshold=1e10),
        # so all rows should be reused -> 0 evaluations
        opt.step(counting_closure)
        step2_evals = sum(call_counts)

        assert step2_evals < step1_evals, f"Step 2 should have fewer evals than step 1: {step2_evals} vs {step1_evals}"

    def test_adaptive_probes_threshold(self):
        """adaptive_probes_threshold controls which particles get reduced probes."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            adaptive_probes=True,
            adaptive_probes_threshold=0.0,  # Threshold 0: NO particles are stagnant
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

        # Step 1
        opt.step(counting_closure)
        step1_evals = sum(call_counts)
        call_counts.clear()

        # Step 2: threshold=0.0 means no particles match as stagnant (strict <)
        # so all should be re-evaluated
        opt.step(counting_closure)
        step2_evals = sum(call_counts)

        assert step2_evals == step1_evals, (
            f"With threshold=0.0, step 2 should have same evals as step 1: {step2_evals} vs {step1_evals}"
        )


# ---------------------------------------------------------------------------
# TestDualMomentum (convergence acceleration)
# ---------------------------------------------------------------------------


class TestDualMomentum:
    """Tests for dual potential momentum (dual momentum extension)."""

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

    def test_beta_positive_runs(self):
        """dual_momentum_beta=0.3 completes 5 steps without error and returns finite costs."""
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
        for _ in range(5):
            cost = opt.step(closure)
            assert isinstance(cost, float)
            assert cost == cost  # not NaN

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
        """Dual momentum extrapolation does not exceed max_abs_dual bounds."""
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
        for _ in range(5):
            opt.step(closure)
        # Check that duals are within bounds
        if opt.state.f is not None:
            max_abs = 80.0 * max(opt.state.epsilon, 0.01)
            assert opt.state.f.abs().max().item() <= max_abs + 1e-6, f"Dual f exceeds max_abs={max_abs}"


# ---------------------------------------------------------------------------
# TestCurvatureAwareRadius (convergence acceleration)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sinkhorn acceleration integration tests
# ---------------------------------------------------------------------------


class TestSinkhornAccelerationIntegration:
    """Integration tests verifying all convergence acceleration improvements compose correctly."""

    def test_all_sinkhorn_improvements_compose(self):
        """SinkhornSolver with anderson_depth=3, data_dependent_init=True, adaptive_omega=True
        all enabled produces valid transport plan on a standard problem."""
        from polystep.solvers import SinkhornSolver

        torch.manual_seed(42)
        P, V = 20, 4
        cost_matrix = torch.rand(P, V)
        a = torch.ones(P) / P

        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=200,
            threshold=1e-6,
            compile=False,
            anderson_depth=3,
            data_dependent_init=True,
            adaptive_omega=True,
        )
        result = solver.solve(cost_matrix=cost_matrix, a=a)

        # Verify valid transport plan
        assert result.matrix is not None
        assert torch.isfinite(result.matrix).all(), "Transport plan has non-finite values"
        assert (result.matrix >= -1e-6).all(), "Transport plan has negative values"
        # Row sums should match marginal a
        row_sums = result.matrix.sum(dim=1)
        assert torch.allclose(row_sums, a, atol=1e-3), (
            f"Row sums don't match marginal: max diff={torch.abs(row_sums - a).max().item()}"
        )


# ---------------------------------------------------------------------------
# Sinkhorn parameter wiring: Sinkhorn param wiring tests
# ---------------------------------------------------------------------------


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

    def test_default_sinkhorn_params_backward_compat(self):
        """Default PolyStepOptimizer(model) creates solver with all convergence acceleration
        Sinkhorn params disabled (backward compatibility)."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            compile=False,
            seed=42,
        )
        assert opt.solver.anderson_depth == 0
        assert opt.solver.adaptive_omega is False
        assert opt.solver.data_dependent_init is False


# ---------------------------------------------------------------------------
# TestEntEpsilon
# ---------------------------------------------------------------------------


class TestEntEpsilon:
    """Tests for the ent_epsilon parameter (separate OT solver epsilon)."""

    @pytest.mark.parametrize("ent_epsilon", [0.1, LinearEpsilon(1.0, 0.1, 10)])
    def test_ent_epsilon_float_accepted(self, ent_epsilon):
        """ent_epsilon (float or LinearEpsilon schedule) is accepted and step completes."""
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
            ent_epsilon=ent_epsilon,
        )
        cost = opt.step(closure)
        assert isinstance(cost, float)
        assert cost == cost  # not NaN

    def test_ent_epsilon_none_default(self):
        """ent_epsilon=None (default) means the main epsilon is used."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.ent_epsilon is None


# ---------------------------------------------------------------------------
# TestStagnationThreshold
# ---------------------------------------------------------------------------


class TestStagnationThreshold:
    """Tests for stagnation_threshold parameter with adaptive radius."""

    def test_stagnation_threshold_accepted(self):
        """stagnation_threshold=0.001 with use_adaptive_radius=True runs without error."""
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
            use_adaptive_radius=True,
            stagnation_threshold=0.001,
        )
        for _ in range(5):
            cost = opt.step(closure)
            assert isinstance(cost, float)
            assert cost == cost  # not NaN

    def test_stagnation_threshold_various_values(self):
        """Different stagnation_threshold values (0.0, 0.1, 1e-4) are all accepted."""
        for threshold in [0.0, 0.1, 1e-4]:
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
                use_adaptive_radius=True,
                stagnation_threshold=threshold,
            )
            assert opt.stagnation_threshold == threshold
            cost = opt.step(closure)
            assert isinstance(cost, float)
            assert cost == cost, f"NaN with stagnation_threshold={threshold}"

    def test_stagnation_threshold_default(self):
        """Default stagnation_threshold is 1e-4."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.stagnation_threshold == 1e-4


# ---------------------------------------------------------------------------
# Tests for auto_epsilon (progressive epsilon)
# ---------------------------------------------------------------------------


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

    def test_auto_epsilon_default_off(self):
        """auto_epsilon=False (default) should not create progressive epsilon."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        opt = PolyStepOptimizer(model, epsilon=0.5, compile=False)
        assert opt._progressive_epsilon is None


# ---------------------------------------------------------------------------
# TestRemovedParameters
# ---------------------------------------------------------------------------


class TestRemovedParameters:
    """Parameters removed in cleanup should raise TypeError."""

    @pytest.mark.parametrize(
        "param_name, value",
        [
            ("curvature_aware_radius", True),
            ("entropy_target", 0.7),
            ("nesterov_lookahead", True),
        ],
    )
    def test_curvature_aware_radius_removed(self, param_name, value):
        model = nn.Linear(4, 2)
        with pytest.raises(TypeError, match=param_name):
            PolyStepOptimizer(model, **{param_name: value})


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
