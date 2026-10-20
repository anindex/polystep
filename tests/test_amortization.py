"""Amortized steps: run the OT solve every N steps and move on the cached transport
direction in between."""

import torch
import torch.nn as nn

from polystep import AdaptiveSubspace, ParamLayout
from polystep.optimizer import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def test_amortize_steps_alternates_ot_and_momentum():
    """With amortize_steps=3, only steps 0 and 3 call the closure."""
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

    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"
    assert optimizer._transport_direction_ema is not None

    dir_after_1 = optimizer._transport_direction_ema.clone()
    # Reset the counter to force a full OT step.
    optimizer._amortize_counter = 0
    cost2 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"
    dir_after_2 = optimizer._transport_direction_ema
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
    # A bare `< losses[0]` would pass on a one-ULP dip.
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

    cost1 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost1)), f"Cost is not finite: {cost1}"

    X_before = optimizer._state.X.clone()
    params_before = [p.detach().clone() for p in model.parameters()]
    cost2 = optimizer.step(closure)
    assert torch.isfinite(torch.tensor(cost2)), f"Cost is not finite: {cost2}"
    assert cost2 == cost1
    assert not torch.equal(X_before, optimizer._state.X)
    assert any(not torch.equal(a, b) for a, b in zip(params_before, model.parameters()))


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


def test_amortize_cost_batch_biased_rotation_compose():
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

    assert not all(c == costs[0] for c in costs), "Cost should change over steps"


def test_noamort_never_takes_momentum_step():
    """With amortize_steps=1, every step should be a full OT step."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    optimizer = PolyStepOptimizer(
        model,
        epsilon=0.5,
        step_radius=1.0,
        num_probe=1,
        sinkhorn_max_iters=20,
        amortize_steps=1,
    )
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn)
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4,))

    def closure(bp):
        return evaluator.evaluate(bp, inputs, targets)

    n_steps = 5
    for _ in range(n_steps):
        optimizer.step(closure)

    assert optimizer._transport_direction_ema is None, "amortize_steps=1 should never populate _transport_direction_ema"
    assert optimizer._state.iteration_count == n_steps, (
        f"Expected {n_steps} OT iterations, got {optimizer._state.iteration_count}"
    )


def test_resync_forces_next_step_evaluation():
    """resync_from_model clears the amortized state."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PolyStepOptimizer(model, seed=0, amortize_steps=2)
    calls = [0]

    def counting(params):
        calls[0] += 1
        v = next(iter(params.values()))
        return v.reshape(v.shape[0], -1).pow(2).sum(1)

    opt.step(counting)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    opt.resync_from_model()
    # Both must clear, or the next step coasts along a direction measured at the
    # old anchor.
    assert opt._transport_direction_ema is None
    assert opt._amortize_counter == 0

    calls[0] = 0
    opt.step(counting)
    assert calls[0] > 0


def test_resync_clears_displacement_history():
    """resync_from_model drops moves recorded along the abandoned trajectory."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1))
    inputs, targets = torch.randn(64, 8), torch.randn(64, 1)
    sub = AdaptiveSubspace.from_layout(ParamLayout.from_module(model), rank=20, rotation_interval=4)
    opt = PolyStepOptimizer(model, subspace=sub, compile=False, seed=0)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    for _ in range(3):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets))
    opt.resync_from_model()
    assert opt._state.displacement_history_count == 0
    assert opt._state.displacement_history_idx == 0
    assert opt._state.displacement_history_full is None
    assert not opt._state.displacement_history.any()


def test_probe_count_drops_on_negative_descent():
    """K reduction keys off descent, not sign.

    RL returns and margin losses are routinely negative. Gating the trigger on
    ``all(c > 0)`` would make those objectives pay the full ``K`` forward passes forever.
    """
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PolyStepOptimizer(
        model,
        compile=False,
        num_probe=3,
        adaptive_num_probe=True,
        adaptive_probe_warmup=0,
        seed=0,
    )

    # A quadratic offset below zero: minimizable, and always negative.
    def closure(batched):
        return sum(v.flatten(1).pow(2).sum(dim=1) for v in batched.values()) - 1000.0

    for _ in range(8):
        opt.step(closure)

    assert all(c < 0 for c in opt._ot_step_costs), "objective was meant to stay negative"
    assert opt._loss_decreasing_count >= 3, "descent on a negative objective never registered"


def test_amortized_step_reports_no_convergence():
    """No solve runs, so True would mislead early-stop callbacks."""
    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, amortize_steps=2)
    for _ in range(3):
        opt.step(lambda p: torch.full((next(iter(p.values())).shape[0],), 1.0))
    assert not all(opt.state.linear_convergence)
