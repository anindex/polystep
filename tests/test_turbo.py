"""Tests for turbo mode speedups (vectorized loop, cost_batch_size, amortize_steps)."""

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import (
    PolyStepOptimizer as _PSO,
    PolyStep,
    CosineEpsilon,
)
from polystep.optimizer import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def test_vectorized_step_produces_finite_cost():
    """Basic sanity: optimizer step works and produces finite cost."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(model, epsilon=0.5, step_radius=0.5, num_probe=2, compile=False, seed=42)

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    def closure(batched_params):
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        return evaluator.evaluate(batched_params, inputs, targets)

    cost = optimizer.step(closure)
    assert cost is not None
    assert isinstance(cost, float)
    assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"


# test_vectorized_step_updates_model removed: identical to the canonical
# tests/test_optimizer.py::TestClosureInterface::test_step_updates_model.


def test_amortize_steps_alternates_ot_and_momentum():
    """With amortize_steps=3, only 1 in 3 steps should run full OT."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        amortize_steps=3,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    def closure(batched_params):
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        return evaluator.evaluate(batched_params, inputs, targets)

    # Step 1: full OT step (counter=0, 0 % 3 == 0)
    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"
    assert cost1 != 0.0, "First step should be a full OT step with nonzero cost"

    # Step 2: momentum step (counter=1, 1 % 3 != 0)
    cost2 = optimizer.step(closure)
    assert isinstance(cost2, float), "Second step should return a float"
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"

    # Step 3: momentum step (counter=2, 2 % 3 != 0)
    cost3 = optimizer.step(closure)
    assert isinstance(cost3, float), "Third step should return a float"
    assert torch.isfinite(torch.tensor(cost3)), f"Cost is not finite: {cost3}"

    # Step 4: full OT step again (counter=3, 3 % 3 == 0)
    cost4 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost4)), f"Cost is not finite: {cost4}"
    assert cost4 != 0.0, "Fourth step should be a full OT step"


def test_amortize_steps_model_updates_on_momentum():
    """Model params should change on momentum steps too."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        amortize_steps=2,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    def closure(batched_params):
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        return evaluator.evaluate(batched_params, inputs, targets)

    # First step (full OT)
    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"
    params_after_ot = {k: v.clone() for k, v in model.named_parameters()}

    # Second step (momentum)
    cost2 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"
    params_after_momentum = {k: v.clone() for k, v in model.named_parameters()}

    # Params should be different after momentum step
    changed = False
    for k in params_after_ot:
        if not torch.equal(params_after_ot[k], params_after_momentum[k]):
            changed = True
            break
    assert changed, "Model params should change on momentum step"


def test_ema_transport_direction_stored():
    """EMA transport direction should blend recent and historical directions."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        amortize_steps=2,
        amortize_ema=0.7,
    )
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    # Step 1: full OT - should set _transport_direction_ema
    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"
    assert optimizer._transport_direction_ema is not None

    # Step 2: full OT again - EMA should be blended, not just replaced
    dir_after_1 = optimizer._transport_direction_ema.clone()
    # Force full OT by resetting counter
    optimizer._amortize_counter = 0
    cost2 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"
    dir_after_2 = optimizer._transport_direction_ema
    # Should be different from first (blended)
    assert not torch.equal(dir_after_1, dir_after_2)


def test_amortize_with_momentum_stays_finite():
    """Momentum and amortized coasting together must not compound: the coasting
    direction is the pure OT step, so the trajectory stays finite and descends."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        compile=False,
        seed=0,
        amortize_steps=3,
        use_momentum=True,
    )
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)
    loss_fn = nn.MSELoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    losses = [optimizer.step(closure) for _ in range(24)]
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert min(losses) < losses[0]


def test_momentum_step_reuses_last_cost():
    """Momentum step should apply EMA direction and reuse last OT cost (no forward pass)."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        amortize_steps=3,
        amortize_ema=0.7,
    )
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    # Do initial OT step - sets EMA direction and records cost
    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"

    # Momentum step - should reuse last cost, not call closure
    X_before = optimizer._state.X.clone()
    cost2 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"
    # Cost should be reused from OT step
    assert cost2 == cost1
    # Particles should have moved
    assert not torch.equal(X_before, optimizer._state.X)


def test_adaptive_probe_reduces_probes():
    """On a steadily-improving objective the decreasing-loss counter must
    increment after warmup -- that counter is what drops K_eff to 1."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=3,
        compile=False,
        seed=42,
        adaptive_num_probe=True,
        adaptive_probe_warmup=2,
    )

    # Convex bowl ||params||^2: the optimizer descends it monotonically, so the
    # "3 consecutive decreasing costs" trigger reliably fires (random-data
    # cross-entropy did not decrease, so the old >= 0 check was tautological).
    def closure(batched_params):
        n = next(iter(batched_params.values())).shape[0]
        total = torch.zeros(n)
        for v in batched_params.values():
            total = total + (v.reshape(n, -1) ** 2).sum(dim=1)
        return total

    for _ in range(12):
        optimizer.step(closure)

    assert optimizer.state.costs[-1] < optimizer.state.costs[0], "objective did not decrease"
    assert optimizer._loss_decreasing_count >= 1, "decreasing-loss tracking never fired"


@pytest.mark.parametrize("num_probe", [3, 1])
def test_adaptive_probe_step_works(num_probe):
    """Optimizer step should work with adaptive_num_probe=True."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=num_probe,
        compile=False,
        seed=42,
        adaptive_num_probe=True,
        adaptive_probe_warmup=0,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    initial_cost = None
    for _ in range(5):
        cost = optimizer.step(closure)
        assert isinstance(cost, float)
        assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"
        if initial_cost is None:
            initial_cost = cost
    # Verify optimizer isn't diverging wildly (10x tolerance for stochastic optimizer)
    assert abs(cost) < abs(initial_cost) * 10 + 1, f"Cost exploded: {cost} vs initial {initial_cost}"


def test_biased_rotation_stores_descent_direction():
    """After an OT step, _prev_descent_direction should be populated."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        biased_rotation=True,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    assert optimizer._prev_descent_direction is None
    optimizer.step(closure)
    assert optimizer._prev_descent_direction is not None
    # Shape should be (P, pdim) matching particle dimensions
    assert optimizer._prev_descent_direction.dim() == 2


def test_biased_rotation_step_works():
    """Multiple steps with biased_rotation should work without error."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=1,
        compile=False,
        seed=42,
        biased_rotation=True,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    costs = []
    for _ in range(5):
        c = optimizer.step(closure)
        costs.append(c)
        assert isinstance(c, float)
        assert torch.isfinite(torch.tensor(c)), f"Cost is not finite: {c}"
    # Verify optimizer isn't diverging wildly (10x tolerance for stochastic optimizer)
    assert abs(costs[-1]) < abs(costs[0]) * 10 + 1, f"Cost exploded: {costs[-1]} vs initial {costs[0]}"


def test_all_three_features_compose():
    """All three features (EMA amortize, adaptive probes, biased rotation) compose."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=3,
        compile=False,
        seed=42,
        amortize_steps=2,
        amortize_ema=0.7,
        adaptive_num_probe=True,
        adaptive_probe_warmup=2,
        biased_rotation=True,
    )

    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    # Run 10 steps - should not error
    initial_cost = None
    for _ in range(10):
        cost = optimizer.step(closure)
        assert isinstance(cost, float)
        assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"
        if initial_cost is None:
            initial_cost = cost

    # Model should have changed
    assert optimizer._state.iteration_count == 10
    # Verify optimizer isn't diverging wildly (10x tolerance for stochastic optimizer)
    assert abs(cost) < abs(initial_cost) * 10 + 1, f"Cost exploded: {cost} vs initial {initial_cost}"


def test_newton_refinement_runs_without_error():
    """Newton refinement should not crash and produce finite params."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=3,
        compile=False,
        seed=42,
        newton_refinement=True,
        newton_refinement_alpha=0.3,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    def closure(batched_params):
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        return evaluator.evaluate(batched_params, inputs, targets)

    # First step populates _prev_losses_3d, second step uses it
    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"
    cost = optimizer.step(closure)
    assert isinstance(cost, float)
    assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_structured_projection_subspace_step():
    """HybridSubspace with structured projections should run without errors."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    from polystep.transform import ParamLayout
    from polystep.hybrid_subspace import HybridSubspace

    layout = ParamLayout.from_module(model)
    hybrid = HybridSubspace.from_layout(
        layout,
        rank=2,
        rotation_interval=0,
        projection_mode="structured",
    )

    optimizer = PolyStepOptimizer(
        model,
        subspace=hybrid,
        epsilon=0.5,
        step_radius=2.0,
        num_probe=2,
        compile=False,
        seed=42,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()

    def closure(batched_params):
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        return evaluator.evaluate(batched_params, inputs, targets)

    # Run 3 steps to verify stability
    initial_cost = None
    for _ in range(3):
        cost = optimizer.step(closure)
        assert isinstance(cost, float)
        assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"
        if initial_cost is None:
            initial_cost = cost
    # Verify optimizer isn't diverging wildly (10x tolerance for stochastic optimizer)
    assert abs(cost) < abs(initial_cost) * 10 + 1, f"Cost exploded: {cost} vs initial {initial_cost}"

    # Verify model params are finite
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_structured_vs_random_projections_differ():
    """Structured and random projections should have different sparsity patterns."""
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    from polystep.transform import ParamLayout
    from polystep.hybrid_subspace import HybridSubspace

    layout = ParamLayout.from_module(model)

    hybrid_random = HybridSubspace.from_layout(layout, rank=2, projection_mode="random")
    hybrid_struct = HybridSubspace.from_layout(layout, rank=2, projection_mode="structured")

    proj_random = hybrid_random.init_projections(torch.device("cpu"), torch.float32)
    proj_struct = hybrid_struct.init_projections(torch.device("cpu"), torch.float32)

    # Get the first projected layer's projection matrix
    key = [k for k in proj_random.keys() if hybrid_random.specs[list(proj_random.keys()).index(k)].is_projected][0]
    P_rand = proj_random[key]
    P_struct = proj_struct[key]

    # Same shape
    assert P_rand.shape == P_struct.shape

    # Structured should have zeros in off-diagonal blocks
    zero_count_struct = (P_struct == 0).sum().item()
    zero_count_random = (P_rand.abs() < 1e-10).sum().item()
    assert zero_count_struct > zero_count_random, "Structured projection should have more zeros (block-diagonal)"


def test_amortize_steps_with_momentum():
    """amortize_steps=2 composes with use_momentum=True.

    Both features should work together: amortized steps use EMA direction
    while momentum smooths the overall trajectory.
    """
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    params_before = {k: v.clone() for k, v in model.named_parameters()}

    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=0.5,
        num_probe=2,
        compile=False,
        seed=42,
        amortize_steps=2,
        amortize_ema=0.7,
        use_momentum=True,
        momentum_init=0.5,
        momentum_final=0.95,
    )

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    # Run several steps (mix of OT and momentum-amortized steps)
    initial_cost = None
    for _ in range(6):
        cost = optimizer.step(closure)
        assert isinstance(cost, float)
        assert torch.isfinite(torch.tensor(cost)), f"Cost is not finite: {cost}"
        if initial_cost is None:
            initial_cost = cost
    # Verify optimizer isn't diverging wildly (10x tolerance for stochastic optimizer)
    assert abs(cost) < abs(initial_cost) * 10 + 1, f"Cost exploded: {cost} vs initial {initial_cost}"

    # Params should have changed
    changed = False
    for k, v in model.named_parameters():
        if not torch.equal(v, params_before[k]):
            changed = True
            break
    assert changed, "Model params should change with amortize_steps + momentum"

    # Velocity should be active (momentum is on)
    assert optimizer._state.velocity is not None
    assert torch.any(optimizer._state.velocity != 0)


class TestTurboIntegration:
    """Test all 3 turbo features combined."""

    def test_all_turbo_features_combined(self):
        """amortize_steps + cost_batch_size + biased_rotation together."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
        loss_fn = nn.CrossEntropyLoss()
        inputs = torch.randn(32, 10)
        targets = torch.randint(0, 2, (32,))
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

        optimizer = PolyStepOptimizer(
            model,
            epsilon=1.0,
            amortize_steps=3,
            cost_batch_size=16,
            biased_rotation=True,
            compile=False,
            seed=42,
        )

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        costs = []
        for _ in range(6):  # 2 full amortization cycles
            cost = optimizer.step(closure)
            assert torch.isfinite(torch.tensor(cost)), "Cost must be finite"
            costs.append(cost)

        # After 6 steps, cost should not be stuck at initial value
        assert not all(c == costs[0] for c in costs), "Cost should change over steps"


# ===========================================================================
# Turbo-mode features and defaults (merged from test_turbo_features.py)
# ===========================================================================


# ---------------------------------------------------------------------------
# K=1 vs K=3 default unification
# ---------------------------------------------------------------------------


def test_poly_step_optimizer_num_probe_default_is_1():
    """Headline runners use K=1; the optimizer default must match."""
    model = nn.Linear(4, 2, bias=False)
    opt = _PSO(model, epsilon=0.5)
    assert opt.num_probe == 1


def test_poly_step_low_level_num_probe_default_is_1():
    """The low-level PolyStep (synthetic objectives) used to default to
    K=5; the default is unified to K=1 to match
    PolyStepOptimizer and the paper's optimal value."""

    def dummy_obj(x):
        return (x**2).sum(-1)

    solver = PolyStep.create(dummy_obj, dim=4)
    assert solver.num_probe == 1, (
        f"PolyStep.num_probe default should match PolyStepOptimizer (K=1 per paper); got {solver.num_probe}"
    )


# ---------------------------------------------------------------------------
# SNN cosine epsilon guard
# ---------------------------------------------------------------------------


class _FakeLIF(nn.Module):
    """Stand-in for a snnTorch.Leaky neuron without the snntorch dep."""

    def forward(self, x):
        return torch.relu(x)


class _SNNStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.lif1 = _FakeLIF()  # name pattern matches LIF/Leaky cells
        self.fc2 = nn.Linear(8, 2)


def test_snn_with_cosine_step_radius_warns():
    """Per experiments/EXPERIMENT_INDEX.md, scheduling step_radius on an SNN model
    collapses accuracy from ~93% to 10-47%. The optimizer must warn the
    caller when this combination is detected.
    """
    model = _SNNStub()
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _PSO(model, epsilon=0.5, step_radius=cosine)

    msgs = [str(w.message).lower() for w in caught]
    assert any(
        ("snn" in m or "leaky" in m or "lif" in m or "spik" in m) and ("step_radius" in m or "cosine" in m)
        for m in msgs
    ), f"expected a warning about CosineEpsilon on step_radius for an SNN-like model; got: {msgs}"


def test_non_snn_with_cosine_step_radius_no_warn():
    """The guard must NOT fire on an MLP-only model."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _PSO(model, epsilon=0.5, step_radius=cosine)
    msgs = [str(w.message).lower() for w in caught]
    snn_warnings = [m for m in msgs if "snn" in m or "leaky" in m or "lif" in m]
    assert not snn_warnings, f"guard fired on a non-SNN MLP: {snn_warnings}"
