"""FactoredSubspace and the low-rank evaluator that avoids materializing weights."""

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import FactoredEvaluator, NNCostEvaluator
from polystep.factored_subspace import FactoredSubspace
from polystep.optimizer import PolyStepOptimizer
from polystep.transform import ParamLayout


def _mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))


def _setup(rank=4, seed=7):
    torch.manual_seed(0)
    model = _mlp()
    layout = ParamLayout.from_module(model)
    subspace = FactoredSubspace.from_layout(layout, rank=rank, seed=seed)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    return model, subspace, projections, base


def test_b_factors_have_orthonormal_rows():
    """``||A @ B||_F == ||A||_F``, so a coordinate step is the same length in params."""
    _, subspace, projections, _ = _setup()
    for spec in subspace.specs:
        if not spec.is_projected:
            continue
        B = projections[spec.entry_key]
        r = subspace.ranks[spec.entry_key]
        assert torch.allclose(B @ B.t(), torch.eye(r), atol=1e-5)


def test_perturbation_is_linear_in_the_coordinates():
    """Linearity is what lets absorb and the displacement history stay valid."""
    _, subspace, projections, base = _setup()
    c1 = torch.randn(subspace.subspace_dim) * 0.1
    c2 = torch.randn(subspace.subspace_dim) * 0.1

    p1 = subspace.apply_perturbation(projections, base, c1)
    p2 = subspace.apply_perturbation(projections, base, c2)
    psum = subspace.apply_perturbation(projections, base, c1 + c2)

    for key in base:
        expected = p1[key] + p2[key] - base[key]
        assert torch.allclose(psum[key], expected, atol=1e-5)


def test_factored_evaluator_matches_the_materialized_reference():
    """The whole point: same losses, without ever building a candidate weight."""
    model, subspace, projections, base = _setup()
    coords = torch.randn(5, subspace.subspace_dim) * 0.05
    inputs, targets = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
    loss_fn = nn.CrossEntropyLoss()

    materialized = subspace.reconstruct_batch(projections, base, coords)
    reference = NNCostEvaluator(model, loss_fn).evaluate(materialized, inputs, targets)

    factored = FactoredEvaluator.try_build(model, loss_fn)
    assert factored is not None
    got = factored.evaluate(subspace, projections, base, coords, inputs, targets)

    assert torch.allclose(reference, got, atol=1e-5)


def test_factored_evaluator_declines_unsupported_models():
    conv = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 26 * 26, 10))
    assert FactoredEvaluator.try_build(conv, nn.CrossEntropyLoss()) is None


def test_rotation_is_disabled_by_default_and_deterministic_when_on():
    _, subspace, projections, _ = _setup()
    assert subspace.rotate_all(projections, step=5) is projections  # interval 0

    torch.manual_seed(0)
    layout = ParamLayout.from_module(_mlp())
    rotating = FactoredSubspace.from_layout(layout, rank=4, seed=7, rotation_interval=2)
    proj = rotating.init_projections(torch.device("cpu"), torch.float32)
    once = rotating.rotate_all(proj, step=2)
    twice = rotating.rotate_all(proj, step=2)

    assert once is not proj
    for key in once:
        assert torch.equal(once[key], twice[key])  # same step -> same basis


def test_optimizer_falls_back_and_warns_for_an_unsupported_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 26 * 26, 10))
    layout = ParamLayout.from_module(model)
    subspace = FactoredSubspace.from_layout(layout, rank=2)
    inputs, targets = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, subspace=subspace, num_probe=1, seed=0)

    with pytest.warns(UserWarning, match="not a plain nn.Sequential"):
        opt.register_evaluator(evaluator, inputs, targets)

    # Still correct, just via the materializing path.
    def closure(params):
        return evaluator.evaluate(params, inputs, targets)

    assert torch.isfinite(torch.tensor(opt.step(closure)))
