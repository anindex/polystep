"""Trust-region ratio test activates without biased_rotation.

Regression guard: the predicted-vs-actual reduction used to be recorded only
inside the biased_rotation branch, so trust_region alone left the step-radius
multiplier frozen at 1.0. It must now activate whenever the finite-difference
model is available (use_quadratic_model + num_probe>=2 + orthoplex).
"""

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _closure(model):
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(32, 4)
    targets = torch.randn(32, 1)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


def _run(optimizer, closure, steps=12):
    for _ in range(steps):
        optimizer.step(closure)


def test_trust_region_activates_without_biased_rotation():
    """Multiplier list becomes nonempty and nonconstant with trust_region only."""
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        num_probe=2,
        polytope_type="orthoplex",
        trust_region=True,
        biased_rotation=False,
        compile=False,
        seed=0,
    )
    # trust_region auto-enables the quadratic model.
    assert opt.use_quadratic_model is True

    _run(opt, _closure(model))

    mults = opt._state.trust_region_multipliers
    assert len(mults) > 0, "trust_region recorded no ratio test; multiplier stayed frozen"
    assert len(set(mults)) > 1, "trust_region multiplier never changed"
    for m in mults:
        assert 0.1 <= m <= 3.0


def test_trust_region_inactive_without_quadratic_model():
    """num_probe=1 cannot build the FD model, so the ratio test stays off."""
    torch.manual_seed(0)
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            num_probe=1,
            polytope_type="orthoplex",
            trust_region=True,
            biased_rotation=False,
            compile=False,
            seed=0,
        )
    _run(opt, _closure(model))
    assert len(opt._state.trust_region_multipliers) == 0


def test_trust_region_warns_on_low_num_probe():
    torch.manual_seed(0)
    model = _model()
    with pytest.warns(UserWarning, match="num_probe>=2"):
        PolyStepOptimizer(
            model,
            num_probe=1,
            trust_region=True,
            compile=False,
            seed=0,
        )
