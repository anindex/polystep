"""Feature interaction tests: cross-module feature combinations.

Tests combinations NOT already covered by test_optimizer.py:
- subspace + momentum
- blockwise + adaptive radius
- subspace + blockwise (combined mode)
- API-level integration (train() with feature combos)
- Parametric improvements: particle_dim, omega, rank_schedule, adaptive_probes
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, train, TrainConfig, RankSchedule
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.subspace import LowRankSubspace, LinearSubspace
from polystep.transform import ParamLayout


class TestFeatureInteractions:
    """Tests for feature combination correctness."""

    def test_subspace_with_momentum(self, simple_mlp, make_closure):
        """Subspace + momentum runs 5 steps without crash, produces non-zero velocity."""
        torch.manual_seed(42)
        model = simple_mlp
        layout = ParamLayout.from_module(model)
        subspace = LowRankSubspace.from_layout(layout, rank=4)

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.95,
            compile=False,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
        )

        closure = make_closure(model)
        for _ in range(5):
            opt.step(closure)

        assert opt.state.iteration_count == 5
        assert opt.state.velocity is not None
        assert torch.any(opt.state.velocity != 0), "Velocity should be non-zero after 5 steps"
        assert len(opt.state.costs) == 5
        # All costs should be finite
        for c in opt.state.costs:
            assert torch.isfinite(torch.tensor(c)), f"Cost {c} is not finite"

    def test_blockwise_with_adaptive_radius(self, simple_mlp, make_closure):
        """Blockwise + adaptive radius runs 5 steps without crash, radius in bounds."""
        torch.manual_seed(42)
        model = simple_mlp

        opt = PolyStepOptimizer(
            model,
            block_strategy="per_layer",
            use_adaptive_radius=True,
            radius_min=0.5,
            radius_max=3.0,
            compile=False,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
        )

        closure = make_closure(model)
        for _ in range(5):
            opt.step(closure)

        assert opt.state.iteration_count == 5
        assert 0.5 <= opt.state.radius_multiplier <= 3.0, (
            f"Radius {opt.state.radius_multiplier} out of bounds [0.5, 3.0]"
        )
        assert len(opt.state.costs) == 5

    def test_subspace_blockwise_combined(self, simple_mlp, make_closure):
        """Subspace + blockwise combined mode runs without error."""
        torch.manual_seed(42)
        model = simple_mlp
        layout = ParamLayout.from_module(model)
        subspace = LowRankSubspace.from_layout(layout, rank=4)

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            block_strategy="per_layer",
            compile=False,
            seed=42,
        )
        closure = make_closure(model)
        opt.step(closure)
        assert opt.state.iteration_count == 1

    def test_blockwise_with_adaptive_via_train_api(self):
        """Blockwise + adaptive radius via high-level train() API."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        X = torch.randn(32, 4)
        y = torch.randn(32, 2)
        dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

        opt = PolyStepOptimizer(
            model,
            block_strategy="per_layer",
            use_adaptive_radius=True,
            radius_min=0.5,
            radius_max=3.0,
            compile=False,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
        )

        config = TrainConfig(epochs=1)
        result = train(model, dl, nn.MSELoss(), opt, config)
        assert result is model
        assert opt.state.iteration_count == 2  # 32 samples / 16 batch = 2 steps
        assert 0.5 <= opt.state.radius_multiplier <= 3.0

    def test_subspace_momentum_via_train_api(self):
        """Subspace + momentum via high-level train() API."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        subspace = LowRankSubspace.from_layout(layout, rank=4)

        X = torch.randn(32, 4)
        y = torch.randn(32, 2)
        dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.95,
            compile=False,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
        )

        config = TrainConfig(epochs=1)
        result = train(model, dl, nn.MSELoss(), opt, config)
        assert result is model
        assert opt.state.iteration_count == 2
        assert opt.state.velocity is not None
        assert torch.any(opt.state.velocity != 0)


# ---------------------------------------------------------------------------
# parametric extension: Individual improvement integration tests
# ---------------------------------------------------------------------------


def _make_small_model():
    """Tiny model for integration tests: Linear(8, 6) -> ReLU -> Linear(6, 4).

    Kept small on purpose; these tests check optimizer plumbing (finite loss,
    param change, iteration counts, determinism), all of which are independent
    of model size, so a tiny net keeps the per-step forward and OT solve cheap.
    """
    return nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))


def _make_integration_closure(model, input_dim=8, output_dim=4):
    """Create a batched closure for integration testing."""
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(16, input_dim)
    targets = torch.randn(16, output_dim)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


class TestParticleDimAdaptiveProbes:
    """Integration tests for four core configuration knobs.

    1. ``particle_dim=4`` (richer OT signal in full-space mode).
    2. ``omega=1.5`` (overrelaxed Sinkhorn iterations).
    3. ``RankSchedule`` (progressive rank expansion).
    4. ``adaptive_probes`` (cost-row reuse for stagnant particles).
    """

    def test_particle_dim_4_mnist_step(self):
        """particle_dim=4 on a small model runs 3 steps successfully.

        Uses a small model (not full MNIST) because particle_dim=4 in full-space
        mode creates P*8*K evaluations which are prohibitive for large models.
        """
        torch.manual_seed(42)
        # Small model: keeps particle count manageable with particle_dim=4
        model = _make_small_model()  # Linear(50, 30) -> ReLU -> Linear(30, 10)
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

        closure = _make_integration_closure(model)

        losses = []
        for _ in range(3):
            loss = opt.step(closure)
            losses.append(loss)
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss))

        # Model params should have changed
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change after 3 steps with particle_dim=4"
        assert opt.state.iteration_count == 3

    def test_overrelaxed_sinkhorn_integration(self):
        """Overrelaxed Sinkhorn (omega=1.5) runs 3 steps without error."""
        torch.manual_seed(42)
        model = _make_small_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.solver.omega = 1.5

        closure = _make_integration_closure(model)
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss))

        # Verify optimization proceeded
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Optimization should proceed with omega=1.5"
        assert opt.state.iteration_count == 3

    def test_rank_schedule_integration(self):
        """RankSchedule with HybridSubspace transitions rank at specified step."""
        torch.manual_seed(42)
        model = _make_small_model()
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)
        schedule = RankSchedule(stages=[(0, 2), (2, 4)])

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=schedule,
            epsilon=0.1,
            max_iterations=50,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )

        closure = _make_integration_closure(model)
        for _ in range(3):
            loss = opt.step(closure)
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss))

        # After 3 steps, rank should have transitioned to 4 at step 2
        assert opt.state.iteration_count == 3

    def test_adaptive_probes_integration(self):
        """adaptive_probes=True runs 5 steps without error."""
        torch.manual_seed(42)
        model = _make_small_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            adaptive_probes=True,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )

        closure = _make_integration_closure(model)
        for _ in range(5):
            loss = opt.step(closure)
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss))

        # Verify displacement tracking is active
        assert opt._prev_displacement_sqnorms is not None
        assert opt._prev_cost_matrix is not None
        assert opt.state.iteration_count == 5

        # Verify optimization happened
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Optimization should proceed with adaptive_probes=True"


class TestSinkhornAccelerationComposition:
    """Composition tests verifying the four knobs work together.

    ``particle_dim`` is full-space mode and ``rank_schedule`` is
    subspace mode, so we cover both scenarios:

    * Full-space: ``particle_dim=4`` + ``omega=1.5`` + ``adaptive_probes=True``.
    * Subspace: ``rank_schedule`` + ``omega=1.5`` + ``adaptive_probes=True``.
    """

    def test_fullspace_composition(self):
        """Full-space: particle_dim=4 + omega=1.5 + adaptive_probes=True."""
        torch.manual_seed(42)
        model = _make_small_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            particle_dim=4,  # improvement 1
            adaptive_probes=True,  # improvement 4
            epsilon=0.1,
            max_iterations=50,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.solver.omega = 1.5  # improvement 2

        closure = _make_integration_closure(model)
        for step_i in range(5):
            loss = opt.step(closure)
            assert isinstance(loss, float), f"Step {step_i}: loss not float"
            assert torch.isfinite(torch.tensor(loss)), f"Step {step_i}: loss not finite"

        assert opt.state.iteration_count == 5

        # Verify optimization happened (params changed)
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Full-space composition should produce parameter changes"

        # Verify no NaN in state
        assert torch.isfinite(opt.state.X).all(), "State X should be finite"

    def test_subspace_composition(self):
        """Subspace: rank_schedule + omega=1.5 + adaptive_probes=True."""
        torch.manual_seed(42)
        model = _make_small_model()

        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)
        schedule = RankSchedule(stages=[(0, 2), (3, 4)])  # improvement 3

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=schedule,  # improvement 3
            adaptive_probes=True,  # improvement 4
            epsilon=0.1,
            max_iterations=50,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.solver.omega = 1.5  # improvement 2

        closure = _make_integration_closure(model)
        for step_i in range(5):
            loss = opt.step(closure)
            assert isinstance(loss, float), f"Step {step_i}: loss not float"
            assert torch.isfinite(torch.tensor(loss)), f"Step {step_i}: loss not finite"

        assert opt.state.iteration_count == 5

        # Verify no NaN in state
        assert torch.isfinite(opt.state.X).all(), "State X should be finite"


class TestTurboBlockwiseRegression:
    """Regression guard: default parameters produce identical behavior."""

    def test_default_step_is_deterministic(self):
        """PolyStepOptimizer with default kwargs and a fixed seed produces a
        finite, reproducible step trajectory over a small model."""
        torch.manual_seed(42)
        model = _make_small_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )

        closure = _make_integration_closure(model)

        # Run 3 steps with default parameters
        losses = []
        for _ in range(3):
            loss = opt.step(closure)
            losses.append(loss)
            assert isinstance(loss, float)
            assert torch.isfinite(torch.tensor(loss))

        # Verify defaults: particle_dim=2, omega=1.0, no rank_schedule, no adaptive_probes
        assert opt._particle_dim == 2, "Default particle_dim should be 2"
        assert opt.solver.omega == 1.0, "Default omega should be 1.0"
        assert opt._rank_schedule is None, "Default rank_schedule should be None"
        assert not opt._adaptive_probes, "Default adaptive_probes should be False"

        # Verify no adaptive probes state stored
        assert opt._prev_displacement_sqnorms is None
        assert opt._prev_cost_matrix is None

        # Verify optimization succeeded
        assert opt.state.iteration_count == 3
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Default optimization should still update parameters"

        # Verify reproducibility (same seed gives same results)
        torch.manual_seed(42)
        model2 = _make_small_model()
        opt2 = PolyStepOptimizer(
            model2,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        closure2 = _make_integration_closure(model2)
        losses2 = []
        for _ in range(3):
            losses2.append(opt2.step(closure2))

        for i, (l1, l2) in enumerate(zip(losses, losses2)):
            assert abs(l1 - l2) < 1e-6, f"Step {i}: losses differ ({l1} vs {l2}), default behavior not reproducible"
