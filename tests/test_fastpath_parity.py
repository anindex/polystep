"""A fast evaluator path must score what the materializing path scores.

Each model here looks eligible on a shallow reading. The plan must either decline it
or reproduce the reference loss; a silently different number reorders the cost matrix.
"""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from polystep import PolyStepOptimizer
from polystep.cost_nn import BatchedLinearEvaluator, NNCostEvaluator, _batched_loss_kind
from polystep.transform import ParamLayout


def _reference_losses(model, stacked, inputs, targets, loss_fn):
    """Loss per candidate through plain functional_call, one candidate at a time."""
    buffers = dict(model.named_buffers())
    n = next(iter(stacked.values())).shape[0]
    out = []
    for i in range(n):
        params = {k: v[i] for k, v in stacked.items()}
        y = functional_call(model, {**params, **buffers}, (inputs,))
        out.append(loss_fn(y, targets))
    return torch.stack(out)


def _stack(model, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        k: (v.unsqueeze(0) + 0.1 * torch.randn(n, *v.shape, generator=g)).clone()
        for k, v in model.named_parameters(remove_duplicate=False)
    }


def test_repeated_module_declines_bmm_plan():
    """Sequential(fc, ReLU, fc) applies fc twice; named_children reports it once."""
    fc = nn.Linear(4, 4)
    model = nn.Sequential(fc, nn.ReLU(), fc)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_tied_linear_weights_decline_bmm_plan():
    a, b = nn.Linear(4, 4), nn.Linear(4, 4)
    b.weight = a.weight
    model = nn.Sequential(a, nn.ReLU(), b)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_batched_linear_applies_frozen_bias():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model[2].bias.requires_grad_(False)
    with torch.no_grad():
        model[2].bias.fill_(7.0)

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    stacked = {k: v for k, v in _stack(model, 5).items() if k != "2.bias"}

    fast = BatchedLinearEvaluator.try_build(model, loss_fn, "mse")
    assert fast is not None
    torch.testing.assert_close(
        fast.evaluate(stacked, inputs, targets),
        _reference_losses(model, stacked, inputs, targets, loss_fn),
    )


def test_forward_hook_declines_bmm_plan():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model[2].register_forward_hook(lambda m, i, o: o + 5.0)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_root_forward_hook_declines_bmm_plan():
    model = nn.Sequential(nn.Linear(4, 2))
    model.register_forward_hook(lambda m, i, o: o * 3.0)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_loss_subclass_overriding_forward_refused():
    class PlusMSE(nn.MSELoss):
        def forward(self, output, target):
            return super().forward(output, target) + 10.0

    assert _batched_loss_kind(PlusMSE()) is None
    # A subclass that only renames itself keeps the base reduction and stays eligible.
    assert _batched_loss_kind(type("Renamed", (nn.MSELoss,), {})()) == "mse"


def test_soft_label_cross_entropy_stays_batched(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 3))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert ev._batched_linear is not None
    monkeypatch.setattr(ev, "_evaluate_vmap", lambda *a: pytest.fail("soft labels left the batched path"))

    inputs = torch.randn(6, 4)
    soft = torch.softmax(torch.randn(6, 3), dim=1)
    stacked = _stack(model, 4)
    torch.testing.assert_close(
        ev.evaluate(stacked, inputs, soft),
        _reference_losses(model, stacked, inputs, soft, nn.CrossEntropyLoss()),
    )


def test_inplace_path_keeps_float64_resolution():
    """FP32 loss buffers collapse candidates that differ below FP32 resolution."""
    model = nn.Sequential(nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)

    inputs = torch.ones(1, 2, dtype=torch.float64)
    targets = torch.zeros(1, 1, dtype=torch.float64)
    bias = torch.tensor([[1.0], [1.0 + 1e-9], [1.0 + 2e-9]], dtype=torch.float64)
    stacked = {"0.weight": model[0].weight.unsqueeze(0).expand(3, 1, 2).clone(), "0.bias": bias}

    losses = ev.evaluate(stacked, inputs, targets)
    assert losses.dtype == torch.float64
    assert len(torch.unique(losses)) == 3


def test_shifted_overlapping_views_are_rejected():
    """Two windows of one buffer are not independent parameters."""
    # Parameter() copies, so assign .data to keep both views on one storage.
    shared = torch.zeros(6)
    model = nn.Module()
    model.a = nn.Parameter(torch.zeros(4))
    model.b = nn.Parameter(torch.zeros(4))
    model.a.data = shared[0:4]
    model.b.data = shared[2:6]
    with pytest.raises(ValueError, match="share storage"):
        ParamLayout.from_module(model)


def test_disjoint_buffer_slices_stay_independent():
    shared = torch.zeros(6)
    model = nn.Module()
    model.a = nn.Parameter(torch.zeros(3))
    model.b = nn.Parameter(torch.zeros(3))
    model.a.data = shared[0:3]
    model.b.data = shared[3:6]
    layout = ParamLayout.from_module(model)
    assert layout.total_params == 6
    assert layout.shared_groups == ()


def test_reset_vmap_rebuilds_after_layer_swap():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    ev = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(5, 3), torch.randn(5, 2)
    stacked = _stack(model, 4)

    model[1] = nn.Sigmoid()
    ev.reset_vmap()
    torch.testing.assert_close(
        ev.evaluate(stacked, inputs, targets),
        _reference_losses(model, stacked, inputs, targets, nn.MSELoss()),
    )


def test_reset_vmap_rebinds_buffers():
    model = nn.Sequential(nn.Linear(2, 1))
    model.register_buffer("scale", torch.zeros(()))
    ev = NNCostEvaluator(model, nn.MSELoss())
    model.scale = torch.full((), 4.0)
    ev.reset_vmap()
    assert ev._buffers["scale"].item() == 4.0


def test_batched_linear_rejects_flatten_after_linear():
    """The bmm path pre-flattens the input, which is only valid before the first Linear."""
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Flatten(), nn.Linear(6, 2))
    assert BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss()) is None

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    layout = ParamLayout.from_module(model)
    flat = layout.flatten(model).reshape(-1)
    stacked = layout.batch_unflatten(torch.stack([flat, flat * 1.01]))
    losses = evaluator.evaluate(stacked, torch.randn(5, 2, 3), torch.randint(0, 2, (5,)))
    assert losses.shape == (2,) and torch.isfinite(losses).all()


@pytest.mark.parametrize("behavior", ["nested_forward", "nested_hook", "loss_hook"])
def test_fast_paths_preserve_container_loss_behavior(behavior):
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

    layout = ParamLayout.from_module(model)
    assert SparseDeltaEvaluator.try_build(model, loss, layout) is None
    assert SubspaceDeltaEvaluator.try_build(model, loss, HybridSubspace.from_layout(layout, rank=1)) is None
    assert FactoredEvaluator.try_build(model, loss) is None


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "kind, attr",
    [("full", "_sparse_delta_evaluator"), ("hybrid", "_subspace_delta_evaluator"), ("factored", "_factored_evaluator")],
)
def test_3d_inputs_need_flatten_for_delta(kind, attr, flatten):
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace

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
def test_fast_cross_entropy_matches_ignored_targets(targets):
    model = nn.Sequential(nn.Linear(2, 2)).double()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    params = {k: v.detach().unsqueeze(0).expand(3, *v.shape) for k, v in model.named_parameters()}
    inputs, targets = torch.randn(2, 2, dtype=torch.float64), torch.tensor(targets)
    torch.testing.assert_close(
        evaluator.evaluate(params, inputs, targets), evaluator._evaluate_vmap(params, inputs, targets), equal_nan=True
    )


@pytest.mark.parametrize("targets", [[0, 1], [0, -100], [-100, -100]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_cpu_vmap_cross_entropy_matches_native_loss(targets, dtype):
    model = nn.Sequential(nn.Linear(4, 2, bias=False)).to(dtype)
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn, use_inplace=False)
    params = {k: v.to(dtype) for k, v in _stack(model, 3).items()}
    inputs, targets = torch.randn(2, 4, dtype=dtype), torch.tensor(targets)

    def reference(output, labels):
        return loss_fn(output.float() if dtype == torch.bfloat16 else output, labels)

    torch.testing.assert_close(
        evaluator._evaluate_vmap(params, inputs, targets),
        _reference_losses(model, params, inputs, targets, reference),
        equal_nan=True,
    )


@pytest.mark.parametrize("subspace", [False, True])
def test_delta_step_accepts_soft_cross_entropy(subspace):
    from polystep.hybrid_subspace import HybridSubspace

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


@pytest.mark.parametrize("subspace", [False, True])
@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
def test_prefix_reuse_matches_uncached_frees_activations(subspace, block_strategy):
    from polystep.hybrid_subspace import HybridSubspace

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
            nullcontext() if reuse else patch("polystep.optimizer._reuse_prefixes", lambda *a: nullcontext()),
        ):
            for i in range(3):
                x.add_(0.1)  # same tensor object, different objective
                opt.step(lambda p: ev.evaluate(p, x, y), objective_token=i)
        results.append((opt.state.X.clone(), opt.state.costs))
        for name in ("_sparse_delta_evaluator", "_subspace_delta_evaluator"):
            assert getattr(getattr(opt, name, None), "_prefix_cache", None) is None
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize(
    "kind, route",
    [
        (kind, route)
        for kind in ("full", "full_bias", "per_layer", "grouped", "dense", "sparse", "unprojected", "unprojected_bias")
        for route in ("delta", "site", "inplace")
    ]
    + [("factored", "delta"), ("bf16", "site")],
)
def test_padded_tail_scores_like_materialized_step(monkeypatch, kind, route):
    from polystep import _step_blockwise, _step_monolithic
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.solvers._shared import sanitize_cost

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
                    monkeypatch.setattr(
                        owner, method, lambda *a, f=f, o=owner, **k: owners.append(type(o)) or f(*a, **k)
                    )
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
def test_tail_only_chunk_scores_like_materialized(monkeypatch, route):
    from polystep import _step_monolithic
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.solvers._shared import sanitize_cost

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
