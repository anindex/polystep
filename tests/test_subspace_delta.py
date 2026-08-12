"""SubspaceDeltaEvaluator must match reconstruct_batch + closure exactly."""

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator, SubspaceDeltaEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


def _setup(model, rank=4, seed=0, sparse_threshold_bytes=None):
    layout = ParamLayout.from_module(model)
    kwargs = {} if sparse_threshold_bytes is None else {"sparse_threshold_bytes": sparse_threshold_bytes}
    sub = HybridSubspace.from_layout(layout, rank=rank, seed=seed, **kwargs)
    projections = sub.init_projections(torch.device("cpu"), torch.float32)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return layout, sub, projections, base_sd


def _reference(sub, projections, base_sd, coords, evaluator, x, y):
    params = sub.reconstruct_batch(projections, base_sd, coords)
    return evaluator.evaluate(params, x, y)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8, 64, generator=g)
    y = torch.randint(0, 10, (8,), generator=g)
    return x, y


@pytest.mark.parametrize(
    "model_fn,name",
    [
        (lambda: nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10)), "two_layer"),
        (
            lambda: nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 12), nn.Tanh(), nn.Linear(12, 10)),
            "three_layer",
        ),
        (lambda: nn.Sequential(nn.Linear(64, 10)), "one_layer"),
    ],
)
@pytest.mark.parametrize("sparse", [False, True])
def test_matches_reconstruct_batch(model_fn, name, sparse, batch):
    """Every resolvable site reproduces the materializing path's losses."""
    x, y = batch
    torch.manual_seed(0)
    model = model_fn()
    threshold = 0 if sparse else None
    layout, sub, projections, base_sd = _setup(model, sparse_threshold_bytes=threshold)
    loss_fn = nn.CrossEntropyLoss()
    ev = NNCostEvaluator(model, loss_fn, layout)
    delta_ev = SubspaceDeltaEvaluator.try_build(model, loss_fn, sub)
    assert delta_ev is not None, "plain MLP must be supported"

    pdim = 4
    g = torch.Generator().manual_seed(3)
    bary = torch.randn(sub.subspace_dim, generator=g) * 0.05
    bary_sd = sub.apply_perturbation(projections, base_sd, bary)

    n_groups, n_cand = 2, 3
    # One site per coordinate block, at the block start, so weight and bias entries
    # of every layer are covered; resolve_site rejects blocks too narrow for n_groups.
    starts = sorted({s.flat_start // pdim for s in sub.specs})
    checked, kinds = 0, set()
    for start_particle in starts:
        lo = start_particle * pdim
        hi = (start_particle + n_groups) * pdim
        spec = delta_ev.resolve_site(sub, lo, hi, sub.subspace_dim)
        if spec is None:
            continue
        checked += 1
        kinds.add(spec.is_projected)
        cols = torch.arange(lo, hi, dtype=torch.long).reshape(n_groups, pdim) - spec.flat_start
        dcoords = torch.randn(n_groups, n_cand, pdim, generator=g) * 0.1

        full = bary.unsqueeze(0).repeat(n_groups * n_cand, 1)
        rows = torch.arange(n_groups).repeat_interleave(n_cand)
        for r in range(full.shape[0]):
            gidx = int(rows[r])
            full[r, cols[gidx] + spec.flat_start] += dcoords[gidx, r % n_cand]

        expected = _reference(sub, projections, base_sd, full, ev, x, y)
        got = delta_ev.evaluate(sub, projections, bary_sd, spec, lo - spec.flat_start, dcoords, x, y)
        torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    assert checked > 0, f"{name}: no resolvable site found, test proved nothing"
    assert kinds == {True, False}, f"{name}: covered only {kinds}, want a projected and a bias site"


def test_resolve_site_rejects_a_run_straddling_two_blocks():
    """A run crossing a coordinate-block boundary has no single owning layer."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
    _, sub, _, _ = _setup(model)
    ev = SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub)
    boundary = sub.specs[0].flat_end
    assert ev.resolve_site(sub, boundary - 2, boundary + 2, sub.subspace_dim) is None


def test_resolve_site_rejects_particle_padding():
    """Coordinates past subspace_dim have no parameter behind them."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
    _, sub, _, _ = _setup(model)
    ev = SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub)
    assert ev.resolve_site(sub, sub.subspace_dim - 2, sub.subspace_dim + 2, sub.subspace_dim) is None


def test_try_build_declines_a_non_mlp():
    """Conv models fall back rather than being scored by the MLP-only forward."""
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 10))
    _, sub, _, _ = _setup(model)
    assert SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub) is None


def test_subspace_blockwise_step_scores_through_the_delta_path():
    """A subspace block-wise step must reach the delta path and agree with the dense one."""
    from polystep import HybridSubspace, ParamLayout, PolyStepOptimizer
    from polystep.cost_nn import NNCostEvaluator

    def build():
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        gen = torch.Generator().manual_seed(0)
        inputs = torch.randn(32, 64, generator=gen)
        targets = torch.randint(0, 10, (32,), generator=gen)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        calls = []

        def closure(batched_params):
            calls.append(1)
            return evaluator.evaluate(batched_params, inputs, targets)

        torch.manual_seed(7)
        opt = PolyStepOptimizer(
            model,
            epsilon=0.3,
            step_radius=0.1,
            seed=3,
            subspace=HybridSubspace.auto_from_layout(ParamLayout.from_module(model)),
            block_strategy="per_layer",
        )
        return opt, evaluator, inputs, targets, closure, calls

    opt_dense, _, _, _, dense_closure, dense_calls = build()
    dense = [opt_dense.step(dense_closure) for _ in range(4)]

    opt_fast, evaluator, inputs, targets, fast_closure, fast_calls = build()
    opt_fast.register_evaluator(evaluator, inputs, targets)
    assert opt_fast._subspace_delta_evaluator is not None, "the delta path must be available, or this proves nothing"
    fast = [opt_fast.step(fast_closure) for _ in range(4)]

    assert len(fast_calls) < len(dense_calls), "some chunks should have skipped reconstruct_batch"
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mixed,dtype,tol",
    [(False, torch.float32, 1e-6), (True, torch.bfloat16, 5e-3)],
    ids=["fp32", "bf16"],
)
def test_optimizer_step_matches_the_materializing_path(mixed, dtype, tol):
    """The step must agree with reconstruct_batch, mixed precision included."""
    from polystep import HybridSubspace, ParamLayout, PolyStepOptimizer

    def build(register):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)).to(dtype)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(32, 64, generator=g).to(dtype)
        y = torch.randint(0, 10, (32,), generator=g)
        ev = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        torch.manual_seed(7)
        opt = PolyStepOptimizer(
            model,
            epsilon=0.3,
            step_radius=0.1,
            seed=3,
            subspace=HybridSubspace.auto_from_layout(ParamLayout.from_module(model)),
            mixed_precision=mixed,
        )
        if register:
            opt.register_evaluator(ev, x, y)
        return opt, [opt.step(lambda bp: ev.evaluate(bp, x, y)) for _ in range(3)]

    _, dense = build(register=False)
    opt_fast, fast = build(register=True)
    assert opt_fast._subspace_delta_evaluator is not None, "the delta path must be built, or this proves nothing"
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=tol, atol=tol)


def test_resolve_site_declines_an_unprojected_weight():
    """An unprojected weight's coordinates index the flattened weight, not output
    units, so neither correction branch applies and it must fall back."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
    _, sub, _, _ = _setup(model)
    ev = SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub)

    for spec in sub.specs:
        site = ev._site.get(spec.entry_key)
        if site is None or site[1] == spec.is_projected:
            continue
        assert ev.resolve_site(sub, spec.flat_start, spec.flat_start + 2, sub.subspace_dim) is None


def test_basis_products_matches_the_materialized_column_block():
    """``P`` is ``(d_out*d_in, num_coords)`` row-major, so
    ``P.view(d_out, d_in, -1)[o, i, j]`` is ``M_j[o, i]``; wrong index arithmetic
    there silently scores against a transposed basis."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))
    _, sub, projections, _ = _setup(model)
    ev = SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub)
    spec = next(s for s in sub.specs if s.is_projected)
    P = projections[spec.entry_key]
    d_out, d_in = spec.original_shape
    x = torch.randn(8, d_in)

    col_start, n_cols = 2, 4
    cols = torch.arange(col_start, col_start + n_cols)
    expected = torch.matmul(P.index_select(1, cols).t().reshape(n_cols, d_out, d_in), x.t())
    got = ev._basis_products(spec, P, col_start, n_cols, x, torch.float32)
    torch.testing.assert_close(got, expected)

    # A rotation swaps the tensor, so the next call must read the new basis, not a cache.
    P2 = torch.linalg.qr(torch.randn_like(P))[0]
    projections[spec.entry_key] = P2
    expected2 = torch.matmul(P2.index_select(1, cols).t().reshape(n_cols, d_out, d_in), x.t())
    got2 = ev._basis_products(spec, P2, col_start, n_cols, x, torch.float32)
    torch.testing.assert_close(got2, expected2)
    assert not torch.allclose(got2, got), "the two bases must differ, or the test proves nothing"
