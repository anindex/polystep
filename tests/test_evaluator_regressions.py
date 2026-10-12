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


@pytest.mark.parametrize("behavior", ["nested_forward", "nested_hook", "loss_hook"])
def test_fast_paths_preserve_container_and_loss_behavior(behavior):
    class Offset(nn.Sequential):
        def forward(self, x):
            return super().forward(x) + 3

    inner = (Offset if behavior == "nested_forward" else nn.Sequential)(nn.Linear(3, 2))
    model, loss = nn.Sequential(inner), nn.MSELoss()
    if behavior == "nested_hook":
        inner.register_forward_hook(lambda module, args, output: output + 3)
    if behavior == "loss_hook":
        loss.register_forward_hook(lambda module, args, output: output + 3)
    ev = NNCostEvaluator(model, loss, use_inplace=False)
    params = {k: v.detach()[None] + torch.randn(4, *v.shape) * 0.1 for k, v in model.named_parameters()}
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    torch.testing.assert_close(ev.evaluate(params, x, y), ev._evaluate_loop(params, x, y))
    # Every delta evaluator shares the same compatibility check.
    from polystep.cost_nn import FactoredEvaluator, SparseDeltaEvaluator, SubspaceDeltaEvaluator
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    layout = ParamLayout.from_module(model)
    assert SparseDeltaEvaluator.try_build(model, loss, layout) is None
    assert SubspaceDeltaEvaluator.try_build(model, loss, HybridSubspace.from_layout(layout, rank=1)) is None
    assert FactoredEvaluator.try_build(model, loss) is None


def test_dense_evaluator_bounds_activation_batches_by_chunk_size():
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    ev = NNCostEvaluator(model, nn.MSELoss(), chunk_size=3, use_inplace=False)
    params = {k: v.detach()[None] + torch.randn(8, *v.shape) * 0.1 for k, v in model.named_parameters()}
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    with patch("torch.baddbmm", wraps=torch.baddbmm) as matmul:
        got = ev.evaluate(params, x, y)
    assert max(call.args[1].shape[0] for call in matmul.call_args_list) <= 3
    torch.testing.assert_close(got, ev._evaluate_loop(params, x, y))


@pytest.mark.parametrize("chunk", [0, -1, True, 2.5, "invalid"])
def test_evaluator_rejects_invalid_chunks(chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        NNCostEvaluator(nn.Linear(3, 2), nn.MSELoss(), chunk_size=chunk)


def test_reset_recomputes_automatic_chunk_size():
    ev = NNCostEvaluator(nn.Linear(3, 2), nn.MSELoss(), chunk_size="auto")
    with patch("polystep.cost_nn.auto_detect_chunk_size", side_effect=[8, 2]):
        assert ev.chunk_size == 8
        ev.reset_vmap()
        assert ev.chunk_size == 2


@pytest.mark.parametrize("site", ["0.weight", "0.bias", "2.weight", "2.bias"])
def test_dense_shared_weights_match_materialized_candidates(site):
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    params = {k: v.detach()[None].expand(8, *v.shape) for k, v in model.named_parameters()}
    params[site] = params[site] + torch.randn_like(params[site]) * 0.1
    x, y = torch.randn(5, 3).double(), torch.randn(5, 2).double()
    expected = ev._evaluate_loop(params, x, y)
    torch.testing.assert_close(ev.evaluate(params, x, y), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ev.evaluate({k: v.clone() for k, v in params.items()}, x, y), expected)


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "kind, attr",
    [("full", "_sparse_delta_evaluator"), ("hybrid", "_subspace_delta_evaluator"), ("factored", "_factored_evaluator")],
)
def test_3d_inputs_take_the_delta_paths_only_through_a_leading_flatten(kind, attr, flatten):
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    x = torch.randn(8, 5, 4, dtype=torch.float64)
    y = torch.randn(8, 2, dtype=torch.float64) if flatten else torch.randn(8, 5, 2, dtype=torch.float64)
    results = []
    for fast in (False, True):
        torch.manual_seed(5)
        head = [nn.Flatten(), nn.Linear(20, 6)] if flatten else [nn.Linear(4, 6)]
        model = nn.Sequential(*head, nn.Tanh(), nn.Linear(6, 2)).double()
        subspace = {"full": None, "hybrid": HybridSubspace, "factored": FactoredSubspace}[kind]
        subspace = subspace and subspace.from_layout(ParamLayout.from_module(model), rank=2)
        opt = PolyStepOptimizer(model, subspace=subspace, solver="softmax", compile=False, seed=3)
        ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
        if fast:
            opt.register_evaluator(ev, x, y)
            assert (getattr(opt, attr) is not None) is flatten
        opt.step(lambda p: ev.evaluate(p, x, y))
        results.append(opt.state.X.clone())
    torch.testing.assert_close(*results, rtol=1e-10, atol=1e-12)
    if not flatten:
        opt.register_evaluator(ev, x[:, 0], y[:, 0])
        assert getattr(opt, attr) is not None


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_grouped_site_casts_inputs_past_a_float32_buffer(dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2).to(dtype)
            self.register_buffer("extra", torch.zeros(()))

        def forward(self, x):
            return self.fc(x) + self.extra

    torch.manual_seed(0)
    model, loss_fn = Model(), nn.MSELoss()
    x, y = torch.randn(8, 2), torch.randn(8, 2)
    opt = PolyStepOptimizer(model, block_strategy="grouped", seed=0)
    opt.register_evaluator(NNCostEvaluator(model, loss_fn), x, y)
    with torch.no_grad():
        loss = opt.step(lambda *_: loss_fn(model(x.to(dtype)).float(), y))
    assert torch.isfinite(torch.as_tensor(loss)).all()
    assert model.fc.weight.dtype is dtype


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


@pytest.mark.parametrize("screen", [False, True])
def test_materialized_fallback_honours_chunk_size(screen):
    model = nn.Sequential(nn.Linear(2, 2))
    opt = PolyStepOptimizer(
        model, particle_dim=3, chunk_size=1, multifidelity_screen=screen, solver="softmax", compile=False, seed=0
    )
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(4, 2), torch.randn(4, 2)
    opt.register_evaluator(ev, x, y)
    rows = []

    def closure(params):
        rows.append(next(iter(params.values())).shape[0])
        return ev.evaluate(params, x, y)

    opt.step(closure, screen_closure=closure if screen else None)
    assert rows and set(rows) == {1}


@pytest.mark.parametrize(
    "layers, out, expected",
    [
        (lambda: (nn.Linear(3, 101, bias=False),), 101, 0),
        (lambda: (nn.Linear(3, 101), nn.Tanh(), nn.Linear(101, 5)), 5, 27),
    ],
    ids=["tail", "interior"],
)
@pytest.mark.parametrize("hybrid", [False, True])
def test_only_straddling_particles_fall_back(hybrid, layers, out, expected):
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    model = nn.Sequential(*layers())
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model, particle_dim=8), rank=4) if hybrid else None
    opt = PolyStepOptimizer(
        model, particle_dim=8, polytope_type="simplex", subspace=subspace, solver="softmax", compile=False, seed=0
    )
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(16, 3), torch.randn(16, out)
    opt.register_evaluator(ev, x, y)
    rows = []

    def closure(params):
        rows.append(next(iter(params.values())).shape[0])
        return ev.evaluate(params, x, y)

    opt.step(closure)
    assert sum(rows) == expected


@pytest.mark.parametrize("route", ["delta", "site", "inplace"])
@pytest.mark.parametrize(
    "kind",
    [
        "full",
        "full_bias",
        "per_layer",
        "grouped",
        "dense",
        "sparse",
        "unprojected",
        "unprojected_bias",
        "factored",
        "bf16",
    ],
)
def test_the_padded_tail_scores_like_the_materialized_step(monkeypatch, kind, route):
    from polystep import _step_blockwise, _step_monolithic
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.solvers._shared import sanitize_cost
    from polystep.transform import ParamLayout

    if (kind == "factored" and route != "delta") or (kind == "bf16" and route != "site"):
        pytest.skip("no site path for this pair")
    dtype = torch.bfloat16 if kind == "bf16" else torch.float64
    out = 6 if kind.endswith("bias") else 5

    def run(register):
        torch.manual_seed(0)
        last = nn.Linear(4, out, bias=kind.endswith("bias"))
        model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), last).to(dtype)
        layout = ParamLayout.from_module(model)
        subspace = {
            "full": lambda: None,
            "per_layer": lambda: None,
            "grouped": lambda: None,
            "dense": lambda: HybridSubspace.from_layout(layout, rank=1),
            "sparse": lambda: HybridSubspace.from_layout(layout, rank=1, sparse_threshold_bytes=0),
            "unprojected": lambda: HybridSubspace.from_layout(layout, rank=4),
            "factored": lambda: FactoredSubspace.from_layout(layout, rank=2),
            "bf16": lambda: HybridSubspace.from_layout(layout, rank=1),
        }[kind.removesuffix("_bias")]()
        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            polytope_type="simplex",
            solver="softmax",
            block_strategy=kind if kind in ("per_layer", "grouped") else "monolithic",
            compile=False,
            seed=0,
            **({"particle_dim": 8} if subspace is None else {}),
        )
        assert opt._state.X.numel() > (layout.total_params if subspace is None else subspace.subspace_dim)
        ev = NNCostEvaluator(
            model, nn.MSELoss(reduction="sum" if route == "site" else "mean"), use_inplace=route == "inplace"
        )
        x, y = torch.randn(16, 4).to(dtype), torch.randn(16, out).to(dtype)
        if register:
            opt.register_evaluator(ev, x, y)
        materialized, owners, costs = [], [], []
        for step_module in (_step_monolithic, _step_blockwise):
            monkeypatch.setattr(step_module, "sanitize_cost", lambda c: costs.append(c.clone()) or sanitize_cost(c))
        for name in (
            "_sparse_delta_evaluator",
            "_site_vmap_evaluator",
            "_subspace_delta_evaluator",
            "_factored_evaluator",
        ):
            owner = getattr(opt, name, None)
            for method in ("evaluate", "evaluate_subspace"):
                if hasattr(owner, method):
                    f = getattr(owner, method)
                    monkeypatch.setattr(owner, method, lambda *a, f=f, o=owner: owners.append(type(o)) or f(*a))
        swap = ev.evaluate_subspace_inplace
        monkeypatch.setattr(ev, "evaluate_subspace_inplace", lambda *a: materialized.append(a[3].shape[0]) or swap(*a))

        def closure(params):
            materialized.append(next(iter(params.values())).shape[0])
            return ev.evaluate(params, x, y)

        opt.step(closure)
        return materialized, owners, costs

    dense_rows, _, dense = run(register=False)
    fast_rows, owners, fast = run(register=True)
    assert sum(dense_rows) == sum(c.numel() for c in dense)
    assert not fast_rows
    assert len(owners) == (3 if kind.endswith("bias") else 2) and len(set(owners)) == 1
    tol = dict(rtol=2e-2, atol=1e-2) if kind == "bf16" else dict(rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fast, dense, **tol)


@pytest.mark.parametrize("route", ["delta", "site", "inplace"])
def test_a_chunk_of_only_the_padded_tail_scores_like_the_materialized_step(monkeypatch, route):
    from polystep import _step_monolithic
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.solvers._shared import sanitize_cost
    from polystep.transform import ParamLayout

    def run(register):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 5, bias=False)).double()
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
        opt = PolyStepOptimizer(
            model, subspace=subspace, polytope_type="simplex", solver="softmax", chunk_size=1, compile=False, seed=0
        )
        ev = NNCostEvaluator(
            model, nn.MSELoss(reduction="sum" if route == "site" else "mean"), use_inplace=route == "inplace"
        )
        x, y = torch.randn(16, 4).double(), torch.randn(16, 5).double()
        if register:
            opt.register_evaluator(ev, x, y)
        costs, calls = [], []
        monkeypatch.setattr(_step_monolithic, "sanitize_cost", lambda c: costs.append(c.clone()) or sanitize_cost(c))
        opt.step(lambda params: calls.append(1) or ev.evaluate(params, x, y))
        return costs, calls

    dense, _ = run(register=False)
    fast, calls = run(register=True)
    assert not calls
    torch.testing.assert_close(fast, dense, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("route", ["monolithic", "per_layer", "hybrid"])
def test_site_chunk_budgets_downstream_activations(route):
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64, 8)
    )
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=3) if route == "hybrid" else None
    block_strategy = "per_layer" if route == "per_layer" else "monolithic"
    opt = PolyStepOptimizer(
        model, subspace=subspace, block_strategy=block_strategy, solver="softmax", compile=False, seed=0
    )
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(64, 1, 64, 64), torch.randint(0, 8, (64,))
    opt.register_evaluator(ev, x, y)
    head = 64 * 8 / x.numel()
    assert opt._act_ratios == {None: 4.0, "0.weight": 4.0, "0.bias": 4.0, "4.weight": head, "4.bias": head}
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())

    site = opt._site_vmap_evaluator
    rows = {}
    inner = site._vmap_over_site

    def spy(key, base_sd, candidates, inputs, targets):
        rows.setdefault(key, []).append(candidates.shape[0])
        return inner(key, base_sd, candidates, inputs, targets)

    site._vmap_over_site = spy
    opt.step(lambda p: ev.evaluate(p, x, y))
    conv_chunk = max(rows["0.weight"])
    assert 2 * 4 * x.numel() * conv_chunk <= 1 << 26
    assert max(rows["4.weight"]) > conv_chunk


def test_site_budget_unexecuted_module_uses_global_max():
    from polystep._step_core import activation_elements

    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return nn.functional.linear(self.conv(x).mean((2, 3)), self.head.weight, self.head.bias)

    model = Functional()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    x = torch.randn(8, 1, 16, 16)
    opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False), x, torch.zeros(8).long())
    assert "head.weight" not in opt._act_ratios
    assert activation_elements(opt, x, "head.weight") == activation_elements(opt, x) == 4 * x.numel()


class InputNoise(nn.Module):
    def forward(self, x):
        return x + 0.01 * torch.randn_like(x)


def test_site_budget_random_forward_uses_global_max():
    model = nn.Sequential(InputNoise(), nn.Linear(4, 8), nn.Linear(8, 2))
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    x = torch.randn(8, 4)
    opt.register_evaluator(NNCostEvaluator(model, nn.MSELoss(), use_inplace=False), x, torch.zeros(8, 2))
    assert opt._act_ratios == {None: 2.0}


@pytest.mark.parametrize(
    "layer, ratios",
    [("relu", {None: 2.0, "0.weight": 2.0, "0.bias": 2.0, "2.weight": 0.5, "2.bias": 0.5}), ("linear", None)],
    ids=["relu", "linear"],
)
def test_site_budget_scripted_submodule_skips_hooks(layer, ratios):
    layers = [nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)]
    i = {"linear": 0, "relu": 1}[layer]
    layers[i] = torch.jit.script(layers[i])
    model = nn.Sequential(*layers)
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(16, 4), torch.zeros(16, 2)
    opt.register_evaluator(ev, x, y)
    assert opt._act_ratios == ratios
    assert not any(
        m._forward_hooks or m._forward_pre_hooks for m in layers if not isinstance(m, torch.jit.ScriptModule)
    )
    opt.step(lambda p: ev.evaluate(p, x, y))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("noise", [nn.Identity, InputNoise])
def test_site_chunk_peak_within_budget(noise):
    torch.manual_seed(0)
    model = nn.Sequential(
        noise(), nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128, 8)
    ).cuda()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(64, 1, 64, 64, device="cuda"), torch.randint(0, 8, (64,), device="cuda")
    opt.register_evaluator(ev, x, y)
    site = opt._site_vmap_evaluator
    inner = site._vmap_over_site
    peaks = {}

    def spy(key, base_sd, candidates, inputs, targets):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        out = inner(key, base_sd, candidates, inputs, targets)
        peaks[key] = max(peaks.get(key, 0), torch.cuda.max_memory_allocated() - start)
        return out

    site._vmap_over_site = spy
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert set(peaks) == {"1.weight", "1.bias", "5.weight", "5.bias"}
    assert max(peaks.values()) <= 1.05 * 4 * (1 << 26)


@pytest.mark.parametrize("kind", [nn.Embedding, nn.EmbeddingBag])
def test_registering_leaves_a_max_norm_embedding_untouched(kind):
    model = kind(2, 4, max_norm=1.0)
    with torch.no_grad():
        model.weight.copy_(torch.arange(2.0, 10.0).reshape(2, 4))
    before = model.weight.detach().clone()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    opt.register_evaluator(ev, torch.tensor([[0], [1]]), torch.zeros(2, 4))
    assert opt._act_ratios is not None
    assert torch.equal(model.weight, before)


class _Noisy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        self.calls += 1
        return x + 0 * torch.randn_like(x)


def test_registering_leaves_train_mode_stats_and_rng_untouched():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), _Noisy(), nn.ReLU(), nn.Linear(8, 2))
    opt = PolyStepOptimizer(model, compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss())
    model.train()
    x, y = torch.randn(32, 4) + 3, torch.randn(32, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    opt.register_evaluator(ev, x, y)
    assert opt._act_ratios is not None
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
