"""Amortized steps: run the OT solve every N steps and move on the cached transport
direction in between, plus the probe reuse that makes a repeated step free."""

import torch
import torch.nn as nn

from polystep.optimizer import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def test_amortize_steps_alternates_ot_and_momentum():
    """With amortize_steps=3, only steps 0 and 3 evaluate the objective.

    The saving is what amortization is for, so the test counts closure calls
    rather than checking the returned costs are finite, which they are either way.
    """
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
    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    calls = []

    def closure(batched_params):
        calls.append(1)
        return evaluator.evaluate(batched_params, inputs, targets)

    evaluated_on = []
    for step in range(4):
        before = len(calls)
        optimizer.step(closure)
        if len(calls) > before:
            evaluated_on.append(step)

    assert evaluated_on == [0, 3], f"expected OT on steps 0 and 3, got {evaluated_on}"


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
    # A bare `< losses[0]` passes on a one-ULP dip,
    # which an amortized step that stopped descending would still satisfy.
    assert min(losses) < losses[0] * 0.90, f"loss barely moved: {losses[0]:.4f} -> {min(losses):.4f}"


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
    increment after warmup: that counter is what drops K_eff to 1."""
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
    # "3 consecutive decreasing costs" trigger fires. Random-data cross-entropy does
    # not decrease, which leaves nothing to assert.
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


def test_amortize_cost_batch_and_biased_rotation_compose():
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
