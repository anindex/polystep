"""Solver-health diagnostics: ess, rho, and the evaluation count.

Anything that zips these against costs needs them equal length and recorded on every
step, including the amortized ones that run no OT solve.
"""

import pytest
import torch
import torch.nn as nn

from polystep.api import get_diagnostics
from polystep.cost_nn import NNCostEvaluator
from polystep.optimizer import PolyStepOptimizer


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 4))


def test_ess_and_rho_are_recorded():
    """ESS/V reads 1 at uniform weights, and rho is a ratio in [0, 1]."""
    torch.manual_seed(0)
    model = _model()
    x, y = torch.randn(32, 12), torch.randint(0, 4, (32,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    opt = PolyStepOptimizer(model=model, particle_dim=2, epsilon=1.0, seed=0, solver="softmax")
    for _ in range(3):
        opt.step(lambda bp: ev.evaluate(bp, x, y))

    diag = get_diagnostics(opt)
    assert len(diag["ess"]) == 3 and len(diag["rho"]) == 3
    assert all(0.0 < e <= 1.0 + 1e-6 for e in diag["ess"])
    assert all(0.0 <= r <= 1.0 + 1e-6 for r in diag["rho"])


def test_greedy_solver_moves_all_the_way_to_a_vertex():
    """A selection solver moves to a vertex, so rho = 1 by construction."""
    torch.manual_seed(0)
    model = _model()
    x, y = torch.randn(32, 12), torch.randint(0, 4, (32,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    opt = PolyStepOptimizer(model=model, particle_dim=2, epsilon=1.0, seed=0, solver="min_cost_greedy")
    for _ in range(2):
        opt.step(lambda bp: ev.evaluate(bp, x, y))
    assert all(r == pytest.approx(1.0, abs=1e-5) for r in get_diagnostics(opt)["rho"])


def test_evals_records_what_the_step_actually_cost():
    """P*V*K per step, and 0 on a reused-probe step. The denominator for any design
    comparison."""
    torch.manual_seed(0)
    model = _model()
    x, y = torch.randn(32, 12), torch.randint(0, 4, (32,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    opt = PolyStepOptimizer(model=model, particle_dim=4, epsilon=1.0, seed=0, num_probe=1, adaptive_probes=False)
    for _ in range(3):
        opt.step(lambda bp: ev.evaluate(bp, x, y))

    # 276 parameters at particle_dim=4 give 69 particles, and a 4-D simplex has 5
    # vertices. Reading P and V back off the optimizer would pass if both drifted.
    assert get_diagnostics(opt)["evals"] == [69 * 5] * 3


@pytest.mark.parametrize("solver", ["min_cost_greedy", "top_k_mean", "softmax", None])
def test_all_lists_stay_aligned_with_costs(solver):
    model = _model()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(12, 12), torch.randint(0, 4, (12,))
    opt = PolyStepOptimizer(
        model, particle_dim=2, seed=0, solver=solver, max_iterations=20, amortize_steps=3, use_momentum=True
    )
    opt.register_evaluator(evaluator, inputs, targets)
    for i in range(7):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets), objective_token=i)

    diagnostics = get_diagnostics(opt)
    lengths = {k: len(v) for k, v in diagnostics.items() if isinstance(v, list)}
    assert set(lengths.values()) == {diagnostics["iteration_count"]}, lengths


@pytest.mark.parametrize("strategy", ["per_layer", "grouped"])
def test_blockwise_records_them_too(strategy):
    model = _model()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(12, 12), torch.randint(0, 4, (12,))
    opt = PolyStepOptimizer(model, particle_dim=2, seed=0, block_strategy=strategy, max_iterations=20)
    opt.register_evaluator(evaluator, inputs, targets)
    for i in range(4):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets), objective_token=i)

    diagnostics = get_diagnostics(opt)
    lengths = {k: len(v) for k, v in diagnostics.items() if isinstance(v, list)}
    assert set(lengths.values()) == {4}, lengths
    assert all(e > 0 for e in diagnostics["evals"]), diagnostics["evals"]
