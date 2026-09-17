"""Objective, cache, and numerical regression checks."""

from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from torch import nn

from polystep import PolyStepOptimizer
from polystep.ask_tell import PolyStepES
from polystep.cost_nn import NNCostEvaluator, cast_inputs_memo


@pytest.mark.parametrize("targets", [[0, -100], [-100, -100]])
def test_fast_cross_entropy_matches_ignored_target_reduction(targets):
    model = nn.Sequential(nn.Linear(2, 2)).double()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    params = {k: v.detach().unsqueeze(0).expand(3, *v.shape) for k, v in model.named_parameters()}
    inputs, targets = torch.randn(2, 2, dtype=torch.float64), torch.tensor(targets)
    torch.testing.assert_close(
        evaluator.evaluate(params, inputs, targets), evaluator._evaluate_vmap(params, inputs, targets), equal_nan=True
    )


@pytest.mark.parametrize("subspace", [False, True])
def test_delta_step_accepts_soft_cross_entropy_targets(subspace):
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    results = []
    for fast in (False, True):
        torch.manual_seed(11)
        model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).double()
        sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2) if subspace else None
        opt = PolyStepOptimizer(model, subspace=sub, solver="softmax", compile=False, seed=3)
        ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
        x = torch.randn(5, 4, dtype=torch.float64)
        y = torch.randn(5, 2, dtype=torch.float64).softmax(-1)
        if fast:
            opt.register_evaluator(ev, x, y)
        opt.step(lambda p: ev.evaluate(p, x, y))
        results.append(opt.state.X.clone())
    torch.testing.assert_close(*results, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("inference_tensor", [False, True])
def test_cast_cache_observes_input_mutation(inference_tensor):
    holder = SimpleNamespace()
    with torch.inference_mode(inference_tensor):
        inputs = torch.ones(2, dtype=torch.float64)
        first = cast_inputs_memo(holder, inputs, torch.float32)
        if not inference_tensor:
            assert cast_inputs_memo(holder, inputs, torch.float32) is first
        inputs.fill_(7)
        torch.testing.assert_close(cast_inputs_memo(holder, inputs, torch.float32), torch.full((2,), 7.0))


def test_ask_tell_never_records_nonfinite_best():
    es = PolyStepES(dim=2)
    es.ask()
    es.tell(torch.tensor([float("inf"), float("nan"), float("-inf"), float("inf")]))
    assert es.best_fitness == float("inf") and es.best_solution is None
    es.ask()
    es.tell(torch.ones(es.popsize))
    previous = es.best_solution.clone()
    es.ask()
    es.tell(torch.full((es.popsize,), float("inf")))
    assert es.best_fitness == 1.0
    torch.testing.assert_close(es.best_solution, previous)


def test_baseline_objective_never_records_nonfinite_best():
    from polystep.baselines import Objective

    obj = Objective(lambda x: torch.full((len(x),), float("inf")), dim=2, budget=4)
    obj(torch.zeros(4, 2))
    assert obj.best_loss == float("inf") and obj.best_x is None


@pytest.mark.parametrize("num_probe", [1, 2])
def test_nn_trust_prediction_sums_parameter_blocks(num_probe):
    model = nn.Linear(6, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2)
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        polytope_type="orthoplex",
        solver="softmax",
        trust_region=True,
        num_probe=num_probe,
        epsilon=1.0,
        step_radius=0.1,
        probe_radius=0.2,
        use_momentum=False,
        compile=False,
        seed=3,
    )

    def closure(params):
        return params["weight"].square().flatten(1).sum(1) / 2

    before = model.weight.square().sum().item() / 2
    opt.step(closure)
    actual = model.weight.square().sum().item() / 2 - before
    assert opt._center_loss is not None
    assert opt._prev_pre_step_loss == pytest.approx(before)
    # The stored scalar describes the entire NN, not an average parameter block.
    assert opt._prev_predicted_improvement.numel() == 1
    assert opt._prev_predicted_improvement.item() == pytest.approx(actual, abs=1e-10)
    assert opt.candidate_evals == 3 * 4 * num_probe + 1


@pytest.mark.parametrize("subspace", [False, True])
@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
def test_prefix_reuse_matches_uncached_steps_and_releases_activations(subspace, block_strategy):
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    results = []
    for reuse in (False, True):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4), nn.Sigmoid(), nn.Linear(4, 2))
        sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2) if subspace else None
        opt = PolyStepOptimizer(
            model, subspace=sub, block_strategy=block_strategy, solver="softmax", compile=False, seed=3, chunk_size=27
        )
        ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
        x, y = torch.randn(8, 8), torch.tensor([0, 1, -100, 0, 1, 0, 1, 0])
        opt.register_evaluator(ev, x, y)
        with (
            nullcontext() if reuse else patch("polystep._step_monolithic._reuse_prefixes", lambda *a: lambda f: f),
            nullcontext() if reuse else patch("polystep.cost_nn._reuse_prefixes", lambda *a: nullcontext()),
        ):
            for i in range(3):
                x.add_(0.1)  # same tensor object, different objective
                opt.step(lambda p: ev.evaluate(p, x, y), objective_token=i)
        results.append((opt.state.X.clone(), opt.state.costs))
        for name in ("_sparse_delta_evaluator", "_subspace_delta_evaluator"):
            assert getattr(getattr(opt, name, None), "_prefix_cache", None) is None
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1] == results[1][1]


def test_compiled_site_uses_current_shared_weights_and_data(monkeypatch):
    from polystep.cost_nn import SiteVmapEvaluator
    from polystep.transform import ParamLayout

    # Dynamo's eager backend checks graph guards without requiring a C++ toolchain.
    compile_fn = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compile_fn(fn, backend="eager", fullgraph=True))
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    owner = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=True)
    site_ev = SiteVmapEvaluator.try_build(owner, ParamLayout.from_module(model))
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    key = "2.weight"
    for batch in (5, 3):
        x, y = torch.randn(batch, 3), torch.randint(0, 2, (batch,))
        base["0.weight"].add_(0.2)
        candidates = base[key].unsqueeze(0) + torch.randn(3, *base[key].shape) * 0.1
        got = site_ev._vmap_over_site(key, base, candidates, x, y)
        expected = torch.stack(
            [
                nn.functional.cross_entropy(torch.func.functional_call(model, {**base, key: p}, (x,)), y)
                for p in candidates
            ]
        )
        torch.testing.assert_close(got, expected)
    assert set(site_ev._compiled_sites) == {key}
    assert not site_ev._compile_failed_sites


@pytest.mark.parametrize("heads_mask", [False, True])
def test_attention_float_padding_and_per_head_mask_match_pytorch(heads_mask):
    from polystep.layers import VmapSafeMultiHeadAttention

    reference = nn.MultiheadAttention(8, 2, batch_first=True).double().eval()
    ours = VmapSafeMultiHeadAttention(8, 2).double().eval()
    with torch.no_grad():
        for index, layer in enumerate((ours.W_q, ours.W_k, ours.W_v)):
            layer.weight.copy_(reference.in_proj_weight.chunk(3)[index])
            layer.bias.copy_(reference.in_proj_bias.chunk(3)[index])
        ours.W_o.load_state_dict(reference.out_proj.state_dict())
    q, kv = torch.randn(2, 3, 8, dtype=torch.float64), torch.randn(2, 5, 8, dtype=torch.float64)
    mask = torch.randn(4, 3, 5, dtype=torch.float64) if heads_mask else torch.zeros(3, 5, dtype=torch.float64)
    padding = torch.tensor([[0.0, 0.0, -float("inf"), -0.4, 0.0], [0.0, -0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
    expected = reference(q, kv, kv, attn_mask=mask, key_padding_mask=padding, need_weights=False)[0]
    torch.testing.assert_close(ours(q, kv, kv, attn_mask=mask, key_padding_mask=padding), expected)


def test_sparse_columns_dtype_change_invalidates_csr():
    from polystep.projection import SparseRandomProjection

    projection = SparseRandomProjection(16, 8, density=0.5, seed=2)
    projection.project(torch.ones(8))  # populate the float32 CSR cache
    projection.columns(torch.arange(8), torch.device("cpu"), torch.float64)
    actual = projection.project(torch.ones(8, dtype=torch.float64))
    expected = SparseRandomProjection(16, 8, density=0.5, seed=2).project(torch.ones(8, dtype=torch.float64))
    torch.testing.assert_close(actual, expected)


def test_mixed_dtype_site_does_not_cast_inputs_to_an_unrelated_scalar():
    from polystep.cost_nn import SiteVmapEvaluator
    from polystep.transform import ParamLayout

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
            self.weight = nn.Parameter(torch.randn(2, 3))

        def forward(self, x):
            return (x @ self.weight.t()).double() * self.scale

    model = Mixed()
    ev = NNCostEvaluator(model, nn.MSELoss())
    site_ev = SiteVmapEvaluator.try_build(ev, ParamLayout.from_module(model))
    base = dict(model.named_parameters())
    site = base["weight"][None].expand(2, -1, -1)
    x, y = torch.randn(4, 3), torch.randn(4, 2, dtype=torch.float64)
    got = site_ev._vmap_over_site("weight", base, site, x, y)
    torch.testing.assert_close(got, nn.functional.mse_loss(model(x), y).expand(2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_simplex_losses_have_exactly_zero_model_gradient(dtype):
    from polystep.geometry import get_simplex_vertices
    from polystep.quadratic_model import extract_fd_gradient

    g = extract_fd_gradient(
        torch.full((2, 9, 1), 1e6, dtype=dtype),
        torch.ones(1, dtype=dtype),
        0.01,
        8,
        get_simplex_vertices(8, dtype=dtype),
    )
    assert torch.count_nonzero(g) == 0


@pytest.mark.parametrize("centered", [False, True])
def test_old_checkpoint_migrates_only_incumbent_based_trust_predictions(centered):
    model = nn.Linear(4, 1, bias=False)
    opt = PolyStepOptimizer(model, trust_region=True, compile=False)
    checkpoint = opt.state_dict()
    checkpoint["format"] = 4
    checkpoint["control"].update(
        _prev_predicted_improvement=torch.tensor([-1.0, -2.0]), _prev_pre_step_loss=5.0, _prev_loss_from_center=centered
    )
    opt.load_state_dict(checkpoint)
    if centered:
        assert opt._prev_predicted_improvement.item() == -3.0
        assert opt._prev_pre_step_loss == 5.0
    else:
        assert opt._prev_predicted_improvement is None
        assert opt._prev_pre_step_loss is None


@pytest.mark.parametrize(
    "kwargs", [{"full_dim": 0}, {"subspace_dim": -1}, {"density": 0}, {"density": float("nan")}, {"density": 2}]
)
def test_sparse_projection_rejects_invalid_configuration(kwargs):
    from polystep.projection import SparseRandomProjection

    with pytest.raises(ValueError):
        SparseRandomProjection(**({"full_dim": 16, "subspace_dim": 8} | kwargs))


def test_factored_subspace_rejects_nonpositive_rank():
    from polystep.factored_subspace import FactoredSubspace
    from polystep.transform import ParamLayout

    with pytest.raises(ValueError, match="rank"):
        FactoredSubspace.from_layout(ParamLayout.from_module(nn.Linear(3, 2)), rank=0)


def test_register_after_reset_rebuilds_site_owner_and_buffers():
    model = nn.Sequential(nn.Linear(3, 2))
    opt = PolyStepOptimizer(model)
    ev = NNCostEvaluator(model, nn.MSELoss())
    x, y = torch.randn(4, 3), torch.randn(4, 2)
    opt.register_evaluator(ev, x, y)
    old = opt._site_vmap_evaluator
    ev.reset_vmap()
    opt.register_evaluator(ev, x, y)
    assert opt._site_vmap_evaluator is not old
    replacement = NNCostEvaluator(model, ev.loss_fn, compile_vmap=True)
    opt.register_evaluator(replacement, x, y)
    assert opt._site_vmap_evaluator._owner is replacement


def test_eager_inplace_does_not_spend_an_extra_verification_forward():
    model = nn.Linear(3, 2)
    count = []
    model.register_forward_hook(lambda *args: count.append(1))
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True, compile_forward=False)
    params = {k: v.detach()[None].expand(3, *v.shape) for k, v in model.named_parameters()}
    original = {k: v.clone() for k, v in model.state_dict().items()}
    ev.evaluate(params, torch.randn(4, 3), torch.randn(4, 2))
    assert len(count) == 3
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k])
