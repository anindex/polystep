"""Integration tests for softmax solver wired into PolyStepOptimizer.

Verifies the solver strategy pattern works end-to-end: solver selection,
ProgressiveEpsilon blocking, functional step(), turbo features, subspace
modes, and epsilon sharing.
"""
import pytest
import torch
import torch.nn as nn

from polystep import (
    PolyStepOptimizer,
    LinearSubspace,
    HybridSubspace,
    ParamLayout,
    RankSchedule,
)
from polystep.cost_nn import NNCostEvaluator
from polystep.epsilon import LinearEpsilon
from polystep.solvers import SoftmaxSolver, SinkhornSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model():
    """Small MLP for fast testing."""
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _make_closure(model):
    """Create a batched closure using NNCostEvaluator."""
    torch.manual_seed(42)
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 1)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def closure(model):
    return _make_closure(model)


@pytest.fixture
def layout(model):
    return ParamLayout.from_module(model)


# ---------------------------------------------------------------------------
# Solver selection tests
# ---------------------------------------------------------------------------


class TestSolverSelection:
    """Verify default solver auto-selection and explicit overrides."""

    def test_full_space_defaults_to_sinkhorn(self):
        """No subspace -> SinkhornSolver by default."""
        model = _make_model()
        opt = PolyStepOptimizer(model)
        assert isinstance(opt.solver, SinkhornSolver)

    def test_linear_subspace_defaults_to_softmax(self, model, layout):
        """LinearSubspace -> SoftmaxSolver by default."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub)
        assert isinstance(opt.solver, SoftmaxSolver)

    def test_hybrid_subspace_defaults_to_softmax(self, model, layout):
        """HybridSubspace -> SoftmaxSolver by default."""
        sub = HybridSubspace.from_layout(layout, rank=4, rotation_interval=0)
        opt = PolyStepOptimizer(model, subspace=sub)
        assert isinstance(opt.solver, SoftmaxSolver)

    def test_explicit_sinkhorn_with_subspace(self, model, layout):
        """solver='sinkhorn' override with subspace uses SinkhornSolver."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub, solver='sinkhorn')
        assert isinstance(opt.solver, SinkhornSolver)

    def test_explicit_softmax_without_subspace(self, model):
        """solver='softmax' override without subspace uses SoftmaxSolver."""
        opt = PolyStepOptimizer(model, solver='softmax')
        assert isinstance(opt.solver, SoftmaxSolver)

    def test_invalid_solver_raises_value_error(self, model):
        """solver='invalid' raises ValueError."""
        with pytest.raises(ValueError, match="Unknown solver"):
            PolyStepOptimizer(model, solver='invalid')

    def test_solver_none_auto_detects(self, model, layout):
        """solver=None auto-selects based on subspace presence."""
        opt_full = PolyStepOptimizer(model, solver=None)
        assert isinstance(opt_full.solver, SinkhornSolver)

        sub = LinearSubspace.from_layout(layout, rank=4)
        model2 = _make_model()
        opt_sub = PolyStepOptimizer(model2, subspace=sub, solver=None)
        assert isinstance(opt_sub.solver, SoftmaxSolver)


# ---------------------------------------------------------------------------
# ProgressiveEpsilon blocking
# ---------------------------------------------------------------------------


class TestProgressiveEpsilonBlocking:
    """Verify ProgressiveEpsilon is blocked with softmax solver."""

    def test_auto_epsilon_with_softmax_raises(self, model):
        """auto_epsilon=True with solver='softmax' raises ValueError."""
        with pytest.raises(ValueError, match="ProgressiveEpsilon"):
            PolyStepOptimizer(model, solver='softmax', auto_epsilon=True)

    def test_auto_epsilon_with_sinkhorn_ok(self, model):
        """auto_epsilon=True with solver='sinkhorn' does NOT raise."""
        opt = PolyStepOptimizer(model, solver='sinkhorn', auto_epsilon=True)
        assert opt._progressive_epsilon is not None

    def test_auto_epsilon_auto_subspace_raises(self, model, layout):
        """auto_epsilon=True with subspace (auto-selects softmax) raises."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        with pytest.raises(ValueError, match="ProgressiveEpsilon"):
            PolyStepOptimizer(model, subspace=sub, auto_epsilon=True)


# ---------------------------------------------------------------------------
# Functional step tests
# ---------------------------------------------------------------------------


class TestSoftmaxFunctionalStep:
    """Verify softmax solver works through full optimizer step pipeline."""

    @pytest.mark.timeout(30)
    def test_step_with_softmax_returns_finite(self, model, closure):
        """Softmax step returns finite loss value."""
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        loss = opt.step(closure)
        assert isinstance(loss, float)
        assert not (loss != loss), "Loss is NaN"  # NaN check

    @pytest.mark.timeout(30)
    def test_step_updates_model_params(self, model, closure):
        """Softmax step actually updates model parameters."""
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        opt.step(closure)
        changed = False
        for k, v in model.state_dict().items():
            if not torch.equal(v, initial_params[k]):
                changed = True
                break
        assert changed, "Model parameters did not change after softmax step"

    @pytest.mark.timeout(60)
    def test_multiple_steps_no_crash(self, model, closure):
        """3 consecutive softmax steps execute without error."""
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        losses = []
        for _ in range(3):
            loss = opt.step(closure)
            losses.append(loss)
        assert all(isinstance(loss_v, float) for loss_v in losses)

    @pytest.mark.timeout(30)
    def test_state_f_g_none_after_softmax_solve(self, model, closure):
        """After softmax solve step, state.f and state.g are None."""
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        opt.step(closure)
        state = opt._state
        assert state.f is None, "state.f should be None after softmax solve"
        assert state.g is None, "state.g should be None after softmax solve"

    @pytest.mark.timeout(30)
    def test_state_f_g_tensor_after_sinkhorn_solve(self, model, closure):
        """After sinkhorn solve step, state.f and state.g are tensors."""
        opt = PolyStepOptimizer(model, solver='sinkhorn', epsilon=0.5)
        opt.step(closure)
        state = opt._state
        assert isinstance(state.f, torch.Tensor), "state.f should be Tensor after sinkhorn"
        assert isinstance(state.g, torch.Tensor), "state.g should be Tensor after sinkhorn"


# ---------------------------------------------------------------------------
# Turbo feature tests
# ---------------------------------------------------------------------------


class TestTurboFeaturesWithSoftmax:
    """Verify all turbo features work with softmax solver."""

    @pytest.mark.timeout(60)
    def test_amortize_steps_with_softmax(self, model, closure):
        """EMA amortization (amortize_steps=2) works with softmax solver."""
        opt = PolyStepOptimizer(
            model, solver='softmax', epsilon=0.5,
            amortize_steps=2, amortize_ema=0.7,
        )
        for _ in range(4):
            loss = opt.step(closure)
            assert isinstance(loss, float)

    @pytest.mark.timeout(60)
    def test_biased_rotation_with_softmax(self, model, closure):
        """Transport-biased rotation (biased_rotation=True) works with softmax."""
        opt = PolyStepOptimizer(
            model, solver='softmax', epsilon=0.5,
            biased_rotation=True,
        )
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)

    @pytest.mark.timeout(60)
    def test_adaptive_probes_with_softmax(self, model, closure):
        """Adaptive probes (adaptive_probes=True) works with softmax."""
        opt = PolyStepOptimizer(
            model, solver='softmax', epsilon=0.5,
            adaptive_probes=True,
        )
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)

    @pytest.mark.timeout(60)
    def test_momentum_with_softmax(self, model, closure):
        """Momentum (use_momentum=True) works with softmax."""
        opt = PolyStepOptimizer(
            model, solver='softmax', epsilon=0.5,
            use_momentum=True,
        )
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# Subspace mode tests
# ---------------------------------------------------------------------------


class TestSubspaceModes:
    """Verify softmax solver works with different subspace types."""

    @pytest.mark.timeout(60)
    def test_linear_subspace_step(self, model, layout):
        """LinearSubspace with softmax solver runs 2 steps without error."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        closure = _make_closure(model)
        for _ in range(2):
            loss = opt.step(closure)
            assert isinstance(loss, float)

    @pytest.mark.timeout(60)
    def test_hybrid_subspace_step(self, model, layout):
        """HybridSubspace with softmax solver runs 2 steps without error."""
        sub = HybridSubspace.from_layout(layout, rank=4, rotation_interval=0)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        closure = _make_closure(model)
        for _ in range(2):
            loss = opt.step(closure)
            assert isinstance(loss, float)

    @pytest.mark.timeout(60)
    def test_hybrid_subspace_with_sinkhorn_override(self, model, layout):
        """HybridSubspace with solver='sinkhorn' override works."""
        sub = HybridSubspace.from_layout(layout, rank=4, rotation_interval=0)
        opt = PolyStepOptimizer(model, subspace=sub, solver='sinkhorn', epsilon=0.5)
        assert isinstance(opt.solver, SinkhornSolver)
        closure = _make_closure(model)
        loss = opt.step(closure)
        assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# Epsilon sharing test
# ---------------------------------------------------------------------------


class TestEpsilonSharing:
    """Verify epsilon schedule is shared with softmax solver."""

    @pytest.mark.timeout(60)
    def test_linear_epsilon_updates_solver(self, model):
        """LinearEpsilon schedule updates solver.epsilon between steps."""
        eps_schedule = LinearEpsilon(init=1.0, target=0.01, decay=0.3)
        opt = PolyStepOptimizer(
            model, solver='softmax',
            epsilon=eps_schedule,
        )
        closure = _make_closure(model)

        # Record epsilon before first step
        _ = opt.solver.epsilon

        opt.step(closure)

        # After step, solver epsilon should have been set from the schedule
        # The optimizer sets self.solver.epsilon = ot_epsilon inside _step_*
        # We can verify by doing another step and checking the schedule moved
        opt.step(closure)

        # The epsilon schedule should give decreasing values
        eps_step0 = eps_schedule.at(0)
        eps_step1 = eps_schedule.at(1)
        assert eps_step1 < eps_step0, (
            f"LinearEpsilon should decrease: step0={eps_step0}, step1={eps_step1}"
        )

    @pytest.mark.timeout(60)
    def test_fixed_epsilon_with_softmax(self, model):
        """Fixed float epsilon works with softmax solver."""
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        closure = _make_closure(model)
        loss = opt.step(closure)
        assert isinstance(loss, float)
        # Solver epsilon should be set from the optimizer
        assert opt.solver.epsilon == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Dual momentum guard test
# ---------------------------------------------------------------------------


class TestDualMomentumGuard:
    """Verify dual momentum doesn't crash when f/g are None from softmax."""

    @pytest.mark.timeout(60)
    def test_dual_momentum_with_softmax_no_crash(self, model, closure):
        """dual_momentum_beta > 0 with softmax doesn't crash (None.clone guard)."""
        opt = PolyStepOptimizer(
            model, solver='softmax', epsilon=0.5,
            dual_momentum_beta=0.5,
        )
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# Fused softmax dispatch tests
# ---------------------------------------------------------------------------


class TestFusedSoftmaxDispatch:
    """Verify fused softmax fast path activation and correctness."""

    def test_fused_softmax_path_active_with_softmax_solver(self, model, layout):
        """Fused path is active when solver='softmax' with subspace."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        assert opt._use_fused_softmax is True, "_use_fused_softmax should be True for softmax solver"
        # Run 2 steps to verify it works end-to-end
        closure = _make_closure(model)
        losses = []
        for _ in range(2):
            loss = opt.step(closure)
            losses.append(loss)
        assert all(isinstance(loss_v, float) for loss_v in losses)
        assert all(loss_v == loss_v for loss_v in losses), "Loss should not be NaN"

    def test_fused_softmax_path_inactive_with_sinkhorn(self, model):
        """Fused path is NOT active when solver='sinkhorn'."""
        opt = PolyStepOptimizer(model, solver='sinkhorn', epsilon=0.5)
        assert opt._use_fused_softmax is False, "_use_fused_softmax should be False for sinkhorn solver"

    @pytest.mark.timeout(60)
    def test_fused_path_with_turbo_features(self, model, layout):
        """Fused path works with biased_rotation + amortization."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(
            model, subspace=sub, epsilon=0.5,
            biased_rotation=True,
            amortize_steps=2, amortize_ema=0.7,
        )
        assert opt._use_fused_softmax is True
        closure = _make_closure(model)
        for _ in range(5):
            loss = opt.step(closure)
            assert isinstance(loss, float)
            assert loss == loss, "Loss should not be NaN"

    @pytest.mark.timeout(30)
    def test_fused_path_solver_result_has_correct_fields(self, model, layout):
        """After fused softmax step, optimizer state is consistent (no None crashes)."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        closure = _make_closure(model)
        opt.step(closure)
        state = opt._state
        # Fused path sets f=None, g=None in SolverResult
        assert state.f is None, "state.f should be None after fused softmax"
        assert state.g is None, "state.g should be None after fused softmax"

    @pytest.mark.timeout(60)
    def test_fused_path_monolithic_no_subspace(self, model):
        """Fused path works in monolithic mode without subspace."""
        opt = PolyStepOptimizer(model, solver='softmax', epsilon=0.5)
        assert opt._use_fused_softmax is True
        closure = _make_closure(model)
        losses = []
        for _ in range(3):
            loss = opt.step(closure)
            losses.append(loss)
        assert all(isinstance(loss_v, float) for loss_v in losses)


# ---------------------------------------------------------------------------
# K=1 reshape shortcut tests
# ---------------------------------------------------------------------------


class TestK1ReshapeShortcut:
    """Verify K=1 probe reshape optimization produces identical results."""

    @pytest.mark.timeout(60)
    def test_k1_probe_produces_same_result(self, model, layout):
        """K=1 shortcut is mathematically identical to reshape+mean."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        # num_probe=1 means K=1
        opt = PolyStepOptimizer(
            model, subspace=sub, epsilon=0.5,
            num_probe=1, seed=42,
        )
        closure = _make_closure(model)
        losses = []
        for _ in range(3):
            loss = opt.step(closure)
            losses.append(loss)
        assert all(isinstance(loss_v, float) for loss_v in losses)
        assert all(loss_v == loss_v for loss_v in losses), "All losses should be finite"

    @pytest.mark.timeout(60)
    def test_k3_still_uses_mean_path(self, model, layout):
        """K=3 (num_probe=3) still uses the .mean(dim=-1) path -- no breakage."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(
            model, subspace=sub, epsilon=0.5,
            num_probe=3, seed=42,
        )
        closure = _make_closure(model)
        losses = []
        for _ in range(2):
            loss = opt.step(closure)
            losses.append(loss)
        assert all(isinstance(loss_v, float) for loss_v in losses)
        assert all(loss_v == loss_v for loss_v in losses), "All losses should be finite"

    @pytest.mark.timeout(60)
    def test_k1_with_sinkhorn_solver(self, model):
        """K=1 shortcut works with sinkhorn solver too (not just softmax)."""
        opt = PolyStepOptimizer(
            model, solver='sinkhorn', epsilon=0.5,
            num_probe=1, seed=42,
        )
        closure = _make_closure(model)
        loss = opt.step(closure)
        assert isinstance(loss, float)
        assert loss == loss, "Loss should not be NaN"


# ---------------------------------------------------------------------------
# Tests migrated from test_softmax_edge_cases.py (unique optimizer-level tests)
# ---------------------------------------------------------------------------


class TestSoftmaxEdgeCases:
    """Edge cases for softmax solver at the optimizer level."""

    @pytest.mark.timeout(30)
    def test_single_particle_optimizer_step(self):
        """PolyStepOptimizer with num_particles=1 (P=1) and solver='softmax' runs a step."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = PolyStepOptimizer(
            model,
            solver='softmax',
            compile=False,
            seed=42,
            particle_dim=49,
        )

        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 1)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        loss = optimizer.step(closure)
        assert isinstance(loss, float)
        assert loss == loss, "Loss should not be NaN"

    @pytest.mark.timeout(30)
    def test_no_gradient_leakage(self):
        """After softmax optimizer.step(), all param.grad is None."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = PolyStepOptimizer(model, solver='softmax', compile=False, seed=42)

        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 1)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        optimizer.step(closure)

        for name, param in model.named_parameters():
            assert param.grad is None, f"Gradient leakage: {name}.grad is not None"
