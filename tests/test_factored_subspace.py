"""FactoredSubspace and the low-rank evaluator that avoids materializing weights."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import FactoredEvaluator, NNCostEvaluator, SubspaceDeltaEvaluator
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


def test_perturbation_linear_in_coordinates():
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


def test_factored_evaluator_matches_materialized_reference():
    """Same losses without building a candidate weight."""
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


@pytest.mark.parametrize("polytope", ["simplex", "orthoplex"])
def test_registered_step_matches_closure_step(polytope):
    """Contained runs score through the confined delta and runs that straddle two
    blocks through the low-rank evaluator; both must land where the closure does."""

    def spy(cls):
        return patch.object(cls, "evaluate", autospec=True, side_effect=cls.evaluate)

    def run(register):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 12), nn.Tanh(), nn.Linear(12, 5))
        subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=3)
        x, y = torch.randn(32, 16), torch.randint(0, 5, (32,))
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        opt = PolyStepOptimizer(model, subspace=subspace, polytope_type=polytope, seed=0)
        with spy(SubspaceDeltaEvaluator) as delta, spy(FactoredEvaluator) as low_rank:
            for _ in range(3):
                if register:
                    opt.register_evaluator(evaluator, x, y)
                opt.step(lambda p: evaluator.evaluate(p, x, y))
        return opt.state.X, delta.call_count, low_rank.call_count

    fast, delta_calls, low_rank_calls = run(register=True)
    closure_only, _, _ = run(register=False)
    assert delta_calls and low_rank_calls, "both paths must have run, or this proves nothing"
    torch.testing.assert_close(fast, closure_only, rtol=0, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "width, batch, chunk_size, uses_delta",
    [
        (128, 128, None, False),
        (128, 128, 1152, False),
        (128, 512, None, True),
        (128, 512, 72, False),
        (128, 512, 255, False),
        (128, 512, 256, True),
        (1024, 1, None, False),
        (2048, 1, None, True),
    ],
)
def test_cuda_delta_path_only_for_large_chunks(width, batch, chunk_size, uses_delta):
    """The delta path launches more kernels per chunk, so on CUDA it waits until one
    default chunk cannot hold every candidate and each chunk carries enough activations."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(784, width), nn.ReLU(), nn.Linear(width, 10)).cuda()
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=8)
    x, y = torch.randn(batch, 784, device="cuda"), torch.randint(0, 10, (batch,), device="cuda")
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, subspace=subspace, chunk_size=chunk_size, seed=0)
    with patch.object(
        SubspaceDeltaEvaluator, "evaluate", autospec=True, side_effect=SubspaceDeltaEvaluator.evaluate
    ) as delta:
        opt.register_evaluator(evaluator, x, y)
        opt.step(lambda p: evaluator.evaluate(p, x, y))
    assert bool(delta.call_count) is uses_delta


def test_factored_evaluator_declines_unsupported_models():
    conv = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 26 * 26, 10))
    assert FactoredEvaluator.try_build(conv, nn.CrossEntropyLoss()) is None


def test_rotation_redraws_on_interval_deterministically():
    _, subspace, projections, _ = _setup()
    assert subspace.rotate_all(projections, step=4) is projections
    assert subspace.rotate_all(projections, step=5) is not projections

    torch.manual_seed(0)
    layout = ParamLayout.from_module(_mlp())
    rotating = FactoredSubspace.from_layout(layout, rank=4, seed=7, rotation_interval=2)
    proj = rotating.init_projections(torch.device("cpu"), torch.float32)
    once = rotating.rotate_all(proj, step=2)
    twice = rotating.rotate_all(proj, step=2)

    assert once is not proj
    for key in once:
        assert torch.equal(once[key], twice[key])  # same step -> same basis


def test_factored_default_rotation_redraws_basis():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1))
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    inputs, targets = torch.randn(16, 6), torch.randn(16, 1)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(model, subspace=subspace, seed=0)
    closure = opt.register_evaluator(evaluator, inputs, targets)
    opt.step(closure)
    initial = opt.state.hybrid_projections["0.weight"].clone()
    for _ in range(5):
        opt.step(closure)

    assert not torch.equal(initial, opt.state.hybrid_projections["0.weight"])


def test_unsupported_model_warns_and_falls_back():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 26 * 26, 10))
    layout = ParamLayout.from_module(model)
    subspace = FactoredSubspace.from_layout(layout, rank=2)
    inputs, targets = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, subspace=subspace, num_probe=1, seed=0)

    with pytest.warns(UserWarning, match="not a chain of Linear and activation layers"):
        opt.register_evaluator(evaluator, inputs, targets)

    # Still correct, just via the materializing path.
    def closure(params):
        return evaluator.evaluate(params, inputs, targets)

    assert torch.isfinite(torch.tensor(opt.step(closure)))


@pytest.mark.filterwarnings("ignore:FactoredSubspace is in use:UserWarning")
def test_frozen_param_edit_survives_step():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    model[0].weight.requires_grad_(False)
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    inputs, targets = torch.randn(16, 4), torch.randn(16, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(model, subspace=subspace, seed=0)
    opt.register_evaluator(evaluator, inputs, targets)

    with torch.no_grad():
        model[0].weight.fill_(0.5)
    opt.step(lambda params: evaluator.evaluate(params, inputs, targets))

    assert torch.all(model[0].weight == 0.5)


def test_closure_after_release_skips_low_rank():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    inputs, targets = torch.randn(16, 4, dtype=torch.float64), torch.randn(16, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(model, subspace=subspace, seed=0)

    def closure(params):
        return evaluator.evaluate(params, inputs, targets)

    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(closure)
    factored = opt._factored_evaluator
    assert factored._cast_cache is not None, "the low-rank path must have run, or this proves nothing"
    opt.release_evaluator()

    assert factored._cast_cache is None
    assert torch.isfinite(torch.tensor(opt.step(closure)))
    assert opt._factored_evaluator is factored


def test_vector_only_model_rotation_noop():
    """Every parameter unprojected leaves ``init_projections()`` empty, so the
    rotation must not reach for ``next(iter(...))`` on it.
    """
    model = nn.Module()
    model.bias_only = nn.Parameter(torch.zeros(5))
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2, rotation_interval=1)
    projections = subspace.init_projections(device=torch.device("cpu"), dtype=torch.float32)

    assert subspace.rotate_all(projections, step=1) is projections


def test_factored_subspace_rejects_nonpositive_rank():
    with pytest.raises(ValueError, match="rank"):
        FactoredSubspace.from_layout(ParamLayout.from_module(nn.Linear(3, 2)), rank=0)
