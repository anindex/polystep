"""Integration tests for softmax solver wired into PolyStepOptimizer.

Verifies the solver strategy pattern works end-to-end: solver selection,
ProgressiveEpsilon blocking, functional step(), amortization, subspace
modes, and epsilon sharing.
"""

import math

import pytest
import torch
import torch.nn as nn

from polystep import (
    PolyStepOptimizer,
    LinearSubspace,
    HybridSubspace,
    ParamLayout,
)
from polystep.cost_nn import NNCostEvaluator
from polystep.solvers import SoftmaxSolver, SinkhornSolver


def _make_model():
    """Small MLP for fast testing."""
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


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


class TestSolverSelection:
    """Verify default solver auto-selection and explicit overrides."""

    def test_full_space_defaults_to_sinkhorn(self):
        """No subspace -> SinkhornSolver by default."""
        model = _make_model()
        opt = PolyStepOptimizer(model)
        assert isinstance(opt.solver, SinkhornSolver)

    @pytest.mark.parametrize(
        "make_subspace",
        [
            lambda layout: LinearSubspace.from_layout(layout, rank=4),
            lambda layout: HybridSubspace.from_layout(layout, rank=4, rotation_interval=0),
        ],
    )
    def test_linear_subspace_defaults_to_softmax(self, model, layout, make_subspace):
        """Subspace -> SoftmaxSolver by default."""
        sub = make_subspace(layout)
        opt = PolyStepOptimizer(model, subspace=sub)
        assert isinstance(opt.solver, SoftmaxSolver)


class TestProgressiveEpsilonBlocking:
    """Verify ProgressiveEpsilon is blocked with softmax solver."""

    def test_auto_epsilon_with_softmax_raises(self, model):
        """auto_epsilon=True with solver='softmax' raises ValueError."""
        with pytest.raises(ValueError, match="ProgressiveEpsilon"):
            PolyStepOptimizer(model, solver="softmax", auto_epsilon=True)

    def test_auto_epsilon_with_sinkhorn_ok(self, model):
        """auto_epsilon=True with solver='sinkhorn' does NOT raise."""
        opt = PolyStepOptimizer(model, solver="sinkhorn", auto_epsilon=True)
        assert opt._progressive_epsilon is not None

    def test_auto_epsilon_auto_subspace_raises(self, model, layout):
        """auto_epsilon=True with subspace (auto-selects softmax) raises."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        with pytest.raises(ValueError, match="ProgressiveEpsilon"):
            PolyStepOptimizer(model, subspace=sub, auto_epsilon=True)


class TestSoftmaxFunctionalStep:
    """Verify softmax solver works through full optimizer step pipeline."""

    def test_step_with_softmax_returns_finite(self, model, closure):
        """Softmax step returns finite loss value."""
        opt = PolyStepOptimizer(model, solver="softmax", epsilon=0.5)
        loss = opt.step(closure)
        assert math.isfinite(loss)
        assert not (loss != loss), "Loss is NaN"  # NaN check

    def test_step_updates_model_params(self, model, closure):
        """Softmax step actually updates model parameters."""
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}
        opt = PolyStepOptimizer(model, solver="softmax", epsilon=0.5)
        opt.step(closure)
        changed = False
        for k, v in model.state_dict().items():
            if not torch.equal(v, initial_params[k]):
                changed = True
                break
        assert changed, "Model parameters did not change after softmax step"

    def test_state_f_g_none_after_softmax_solve(self, model, closure):
        """After softmax solve step, state.f and state.g are None."""
        opt = PolyStepOptimizer(model, solver="softmax", epsilon=0.5)
        opt.step(closure)
        state = opt._state
        assert state.f is None, "state.f should be None after softmax solve"
        assert state.g is None, "state.g should be None after softmax solve"

    def test_state_f_g_tensor_after_sinkhorn_solve(self, model, closure):
        """After sinkhorn solve step, state.f and state.g are tensors."""
        opt = PolyStepOptimizer(model, solver="sinkhorn", epsilon=0.5)
        opt.step(closure)
        state = opt._state
        assert isinstance(state.f, torch.Tensor), "state.f should be Tensor after sinkhorn"
        assert isinstance(state.g, torch.Tensor), "state.g should be Tensor after sinkhorn"


@pytest.mark.parametrize(
    "feature_kwargs",
    [
        {"amortize_steps": 2, "amortize_ema": 0.7},
        {"biased_rotation": True},
        {"adaptive_probes": True},
        {"use_momentum": True},
    ],
)
def test_amortize_steps_with_softmax(model, closure, feature_kwargs):
    """amortize_steps + adaptive_probes + biased_rotation, all on softmax."""
    opt = PolyStepOptimizer(
        model,
        solver="softmax",
        epsilon=0.5,
        **feature_kwargs,
    )
    for _ in range(4):
        loss = opt.step(closure)
        assert math.isfinite(loss)


class TestSubspaceModes:
    """Verify softmax solver works with different subspace types."""

    def test_hybrid_subspace_step(self, model, layout):
        """HybridSubspace with softmax solver runs 2 steps without error."""
        sub = HybridSubspace.from_layout(layout, rank=4, rotation_interval=0)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        closure = _make_closure(model)
        for _ in range(2):
            loss = opt.step(closure)
            assert math.isfinite(loss)

    def test_hybrid_subspace_with_sinkhorn_override(self, model, layout):
        """HybridSubspace with solver='sinkhorn' override works."""
        sub = HybridSubspace.from_layout(layout, rank=4, rotation_interval=0)
        opt = PolyStepOptimizer(model, subspace=sub, solver="sinkhorn", epsilon=0.5)
        assert isinstance(opt.solver, SinkhornSolver)
        closure = _make_closure(model)
        loss = opt.step(closure)
        assert math.isfinite(loss)


def test_fixed_epsilon_with_softmax(model):
    """Fixed float epsilon works with softmax solver."""
    opt = PolyStepOptimizer(model, solver="softmax", epsilon=0.5)
    closure = _make_closure(model)
    loss = opt.step(closure)
    assert math.isfinite(loss)
    # Solver epsilon should be set from the optimizer
    assert opt.solver.epsilon == pytest.approx(0.5, abs=0.01)


def test_dual_momentum_with_softmax_keeps_stepping(model, closure):
    """softmax has no duals to extrapolate, so dual_momentum_beta must be inert."""
    opt = PolyStepOptimizer(
        model,
        solver="softmax",
        epsilon=0.5,
        dual_momentum_beta=0.5,
    )
    losses = [opt.step(closure) for _ in range(3)]
    assert all(math.isfinite(v) for v in losses)
    assert opt.state.f is None, "softmax produced duals for the momentum to extrapolate"


class TestFusedSoftmaxDispatch:
    """Verify fused softmax fast path activation and correctness."""

    def test_fused_softmax_path_active_with_softmax_solver(self, model, layout):
        """Fused path is active when solver='softmax' with subspace."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.5)
        assert opt._use_fused_softmax is True, "_use_fused_softmax should be True for softmax solver"
        closure = _make_closure(model)
        before = _flat_params(model)
        losses = [opt.step(closure) for _ in range(2)]
        # A bound and a moved model, not `loss == loss`: that rules out NaN and nothing
        # else, so an inf or a frozen optimizer passes it.
        assert all(0.0 < loss_v < 10.0 for loss_v in losses), losses
        assert not torch.equal(before, _flat_params(model))

    def test_fused_softmax_path_inactive_with_sinkhorn(self, model):
        """Fused path is NOT active when solver='sinkhorn'."""
        opt = PolyStepOptimizer(model, solver="sinkhorn", epsilon=0.5)
        assert opt._use_fused_softmax is False, "_use_fused_softmax should be False for sinkhorn solver"

    def test_fused_path_with_turbo_features(self, model, layout):
        """Fused path works with biased_rotation + amortization."""
        sub = LinearSubspace.from_layout(layout, rank=4)
        opt = PolyStepOptimizer(
            model,
            subspace=sub,
            epsilon=0.5,
            biased_rotation=True,
            amortize_steps=2,
            amortize_ema=0.7,
        )
        assert opt._use_fused_softmax is True
        closure = _make_closure(model)
        before = _flat_params(model)
        losses = [opt.step(closure) for _ in range(5)]
        assert all(0.0 < loss_v < 10.0 for loss_v in losses), losses
        assert not torch.equal(before, _flat_params(model))

    def test_fused_path_monolithic_no_subspace(self, model):
        """Fused path works in monolithic mode without subspace."""
        opt = PolyStepOptimizer(model, solver="softmax", epsilon=0.5)
        assert opt._use_fused_softmax is True
        closure = _make_closure(model)
        before = _flat_params(model)
        losses = [opt.step(closure) for _ in range(3)]
        assert all(0.0 < loss_v < 10.0 for loss_v in losses), losses
        assert not torch.equal(before, _flat_params(model))


@pytest.mark.parametrize("use_subspace,solver", [(True, None), (False, "sinkhorn")])
def test_k1_shortcut_matches_the_averaging_path(model, layout, use_subspace, solver):
    """The K=1 cost-matrix shortcut must equal the general averaging path.

    ``_step_monolithic`` skips ``losses.reshape(P, V, K).mean(-1)`` when ``K_eff == 1``
    and reshapes straight to ``(P, V)``. The two must agree exactly; averaging over a
    length-1 axis is the identity."""
    # adaptive_probes is what populates _prev_cost_matrix; use_quadratic_model is
    # what populates _losses_3d. Both are needed to compare the two paths.
    kwargs = {
        "epsilon": 0.5,
        "num_probe": 1,
        "seed": 42,
        "use_quadratic_model": True,
        "adaptive_probes": True,
    }
    if use_subspace:
        kwargs["subspace"] = LinearSubspace.from_layout(layout, rank=4)
    if solver is not None:
        kwargs["solver"] = solver
    opt = PolyStepOptimizer(model, **kwargs)
    closure = _make_closure(model)

    for _ in range(3):
        opt.step(closure)

    losses_3d = opt._losses_3d
    assert losses_3d is not None and losses_3d.shape[-1] == 1, "expected a K=1 probe buffer"
    torch.testing.assert_close(opt._prev_cost_matrix, losses_3d.mean(dim=-1), rtol=0, atol=0)


class TestSoftmaxEdgeCases:
    """Edge cases for softmax solver at the optimizer level."""

    def test_single_particle_optimizer_step(self):
        """PolyStepOptimizer with num_particles=1 (P=1) and solver='softmax' runs a step."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = PolyStepOptimizer(
            model,
            solver="softmax",
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
        assert math.isfinite(loss)
        assert loss == loss, "Loss should not be NaN"

    def test_no_gradient_leakage(self):
        """After softmax optimizer.step(), all param.grad is None."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = PolyStepOptimizer(model, solver="softmax", compile=False, seed=42)

        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 1)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        optimizer.step(closure)

        for name, param in model.named_parameters():
            assert param.grad is None, f"Gradient leakage: {name}.grad is not None"
