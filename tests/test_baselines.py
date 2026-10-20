"""Tests for the shared gradient-free baselines.

Each method must (a) reduce the loss on a small smooth quadratic and (b) respect
its candidate budget, and must behave identically with and without a subspace.
"""

from __future__ import annotations

import math
from importlib.util import find_spec

import pytest
import torch
import torch.nn as nn

from polystep.baselines import METHODS, BudgetExhausted, Objective, Result, centered_rank, zscore
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

METHOD_NAMES = [
    pytest.param(name, marks=pytest.mark.skipif(name == "cma_es" and find_spec("cma") is None, reason="requires cma"))
    for name in sorted(METHODS)
]

DIM = 12
BUDGET = 640

# Hyperparameters tuned only enough that each method makes progress on the
# quadratic below within BUDGET evaluations; not defaults anyone should copy.
HYPERPARAMS = {
    "openai_es": dict(sigma=0.3, lr=1.0, popsize=16),
    "spsa": dict(a=0.5, c=0.1),
    "mezo": dict(eps=1e-2, lr=1e-2),
    "random_search": dict(sigma=0.2),
    "eggroll": dict(sigma=0.3, lr=1.0, popsize=16, rank=1),
    "cma_es": dict(sigma0=0.5, popsize=16),
}

# Candidates consumed per iteration; the budget can be left with less than this.
PER_ITER = {
    "openai_es": 16,
    "spsa": 2,
    "mezo": 2,
    "random_search": 2,
    "eggroll": 16,
    "cma_es": 16,
}


def quadratic(shift: float = 1.0):
    """f(x) = ||x - shift||^2 / dim, minimum 0 at x = shift."""

    def fn(X: torch.Tensor) -> torch.Tensor:
        return ((X - shift) ** 2).mean(dim=-1)

    return fn


def run(name: str, budget: int = BUDGET, **overrides) -> tuple[Objective, Result]:
    obj = Objective(quadratic(), dim=DIM, budget=budget, shapes=[(3, 4)])
    kwargs = {**HYPERPARAMS[name], **overrides}
    return obj, METHODS[name](obj, **kwargs)


# Per-method behaviour


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_reduces_loss_and_respects_budget(name):
    obj, result = run(name)
    start = quadratic()(torch.zeros(1, DIM))[0].item()
    assert result.best_loss < start * 0.5, f"{name}: {result.best_loss} vs start {start}"
    assert torch.isfinite(result.x).all()
    assert obj.iterate is not None
    assert obj.iterate.shape == (DIM,)
    assert torch.isfinite(obj.iterate).all()

    assert result.evals == obj.evals
    assert result.evals <= BUDGET
    # It stopped because it could not afford another iteration, not early.
    # cma_es is the exception: pycma's own convergence criteria may fire first.
    if name != "cma_es":
        assert obj.remaining < PER_ITER[name], f"{name} left {obj.remaining} unspent"


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("budget", [3, 17, 33])
def test_never_overspends_odd_budgets(name, budget):
    if name == "cma_es" and budget < 16:
        pytest.skip("pycma needs at least one full population")
    obj, result = run(name, budget=budget)
    assert obj.evals <= budget
    assert result.evals == obj.evals


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_deterministic_given_seed(name):
    _, a = run(name, seed=7)
    _, b = run(name, seed=7)
    assert a.evals == b.evals
    assert a.best_loss == pytest.approx(b.best_loss)


# The counter


def test_objective_counts_candidates_not_calls():
    obj = Objective(quadratic(), dim=DIM, budget=100)
    obj(torch.zeros(8, DIM))
    assert obj.evals == 8, "one call scoring 8 candidates must cost 8"
    obj(torch.zeros(1, DIM))
    assert obj.evals == 9
    assert obj.remaining == 91


def test_objective_refuses_to_overspend():
    obj = Objective(quadratic(), dim=DIM, budget=4)
    obj(torch.zeros(4, DIM))
    with pytest.raises(BudgetExhausted):
        obj(torch.zeros(1, DIM))
    assert obj.evals == 4


def test_objective_tracks_best_and_ignores_nan():
    obj = Objective(lambda X: torch.tensor([float("nan"), -1.0, 5.0]), dim=DIM, budget=10)
    losses = obj(torch.arange(3 * DIM, dtype=torch.float32).reshape(3, DIM))
    assert losses.tolist() == [float("inf"), -1.0, 5.0]
    assert obj.best_loss == -1.0
    assert obj.best_x is not None


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_nan_candidates_keep_iterate_finite(name):
    def fn(X):
        return quadratic()(X).masked_fill(X[:, 0] > 0.2, float("nan"))

    obj = Objective(fn, dim=DIM, budget=BUDGET, shapes=[(3, 4)])
    result = METHODS[name](obj, **HYPERPARAMS[name])
    assert torch.isfinite(result.x).all()
    assert math.isfinite(result.best_loss)


def test_random_search_compares_on_same_call():
    """A per-call shift, like a fresh minibatch, must not decide acceptance."""
    gen = torch.Generator().manual_seed(0)

    def fn(X):
        return quadratic()(X) + 10.0 * torch.rand((), generator=gen)

    result = METHODS["random_search"](Objective(fn, dim=DIM, budget=BUDGET), **HYPERPARAMS["random_search"])
    assert quadratic()(result.x[None])[0] < 0.5 * quadratic()(torch.zeros(1, DIM))[0]


def test_objective_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Objective(quadratic(), dim=DIM, budget=10, shapes=[(3, 3)])
    obj = Objective(quadratic(), dim=DIM, budget=10)
    with pytest.raises(ValueError):
        obj(torch.zeros(2, DIM + 1))


def test_shaping_helpers():
    x = torch.tensor([3.0, 1.0, 2.0])
    assert centered_rank(x).tolist() == [0.5, -0.5, 0.0]
    values = torch.tensor([2.0, 1.0, 1.0, 3.0])
    assert torch.allclose(centered_rank(values), torch.tensor([1 / 6, -1 / 3, -1 / 3, 0.5]))
    permutation = torch.tensor([3, 0, 2, 1])
    assert torch.equal(centered_rank(values[permutation]), centered_rank(values)[permutation])
    assert torch.equal(centered_rank(torch.ones(32)), torch.zeros(32))
    assert torch.allclose(zscore(torch.ones(4)), torch.zeros(4))
    assert zscore(x).mean().abs() < 1e-6


def test_zscore_single_returns_zero():
    assert torch.equal(zscore(torch.tensor([3.0])), torch.zeros(1))


@pytest.mark.parametrize("shaping", ["rank", "zscore"])
def test_shaping_puts_nonfinite_losses_last(shaping):
    from polystep.baselines.methods import _shaped

    for bad in (float("nan"), float("inf")):
        u = _shaped(torch.tensor([1.0, bad, 0.0]), shaping)
        assert torch.isfinite(u).all()
        assert u.argmin() == 1


def test_zscore_rescales_overflowing_losses():
    from polystep.baselines.methods import _shaped

    u = _shaped(torch.tensor([1e38, 1e38, float("inf"), float("inf")]), "zscore")
    assert torch.isfinite(u).all()
    assert (u[:2] > u[2:]).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.float16])
@pytest.mark.parametrize(("name", "shaping"), [("openai_es", "rank"), ("eggroll", "sign"), ("eggroll", "rank")])
def test_loss_dtypes_run_es_methods(name, shaping, dtype):
    obj = Objective(lambda X: X.square().sum(1).to(dtype), dim=4, budget=8)
    result = METHODS[name](obj, popsize=4, shaping=shaping)
    assert result.x.dtype == torch.float32
    assert torch.isfinite(result.x).all()


# Subspace parity


def build_subspace(max_subspace_dim=32):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 2))
    layout = ParamLayout.from_module(model)
    hybrid = HybridSubspace.from_layout(layout, rank=2, max_subspace_dim=max_subspace_dim)
    base_sd = {e.key: p.detach().clone() for e, p in zip(layout.entries, model.parameters())}
    return model, layout, hybrid, base_sd


def subspace_objective(budget: int) -> Objective:
    """Least squares on a tiny MLP, searched in HybridSubspace coordinates."""
    model, layout, hybrid, base_sd = build_subspace()
    torch.manual_seed(1)
    X = torch.randn(16, 8)
    Y = torch.randn(16, 2)

    def loss_batch(params):
        # (N, batch, out): one shared minibatch for the whole generation.
        h = torch.tanh(torch.einsum("bi,noi->nbo", X, params["0.weight"]) + params["0.bias"].unsqueeze(1))
        out = torch.einsum("nbi,noi->nbo", h, params["2.weight"]) + params["2.bias"].unsqueeze(1)
        return ((out - Y) ** 2).mean(dim=(1, 2))

    return Objective.from_subspace(hybrid, base_sd, loss_batch, budget)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_subspace_run_keeps_counter_semantics(name):
    """Same method, same budget, same counter meaning, projected coordinates."""
    plain = Objective(quadratic(), dim=DIM, budget=BUDGET, shapes=[(3, 4)])
    projected = subspace_objective(BUDGET)
    assert projected.dim != plain.dim, "subspace should change the search dimension"

    start = projected(torch.zeros(1, projected.dim))[0].item()
    plain(torch.zeros(1, plain.dim))
    kwargs = HYPERPARAMS[name]
    r_plain = METHODS[name](plain, **kwargs)
    r_proj = METHODS[name](projected, **kwargs)
    assert r_proj.best_loss < start * 0.9, f"{name}: {r_proj.best_loss} vs start {start}"

    for obj, res in ((plain, r_plain), (projected, r_proj)):
        assert res.evals == obj.evals <= BUDGET
        assert res.best_x.numel() == obj.dim
        assert math.isfinite(res.best_loss)
    # Identical budget accounting: the counter means the same thing on both sides.
    if name != "cma_es":
        assert r_plain.evals == r_proj.evals
        assert r_plain.iters == r_proj.iters


def test_subspace_objective_reports_layer_shapes():
    _, _, hybrid, base_sd = build_subspace()
    obj = Objective.from_subspace(hybrid, base_sd, lambda p: torch.zeros(1), 10)
    assert sum(s[0] for s in obj.shapes) == obj.dim == hybrid.subspace_dim
    assert len(obj.shapes) == len(hybrid.specs)


def test_from_layout_keeps_shapes_for_eggroll():
    model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 2))
    layout = ParamLayout.from_module(model)
    obj = Objective.from_layout(layout, quadratic(), budget=64)
    assert obj.dim == layout.total_params
    assert (6, 8) in obj.shapes and (2, 6) in obj.shapes


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
        ),
    ],
)
@pytest.mark.parametrize("name", METHOD_NAMES)
def test_x0_follows_objective_dtype_device(name, device):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 2)).to(device, torch.float64)
    hybrid = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2, seed=0)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    for x0 in (None, torch.zeros(hybrid.subspace_dim)):
        obj = Objective.from_subspace(hybrid, base_sd, lambda p: p["2.bias"].sum(-1), 64, dtype=torch.float64)
        result = METHODS[name](obj, x0=x0, **HYPERPARAMS[name])
        assert result.x.dtype == result.best_x.dtype == torch.float64
        assert result.x.device.type == result.best_x.device.type == device


# EGGROLL specifics


def test_eggroll_lowrank_noise_full_rank_update():
    from polystep.baselines.methods import _lowrank_noise

    gen = torch.Generator().manual_seed(0)
    shapes = [(6, 8)]
    E = _lowrank_noise(shapes, rank=1, n=16, dim=48, generator=gen, device=torch.device("cpu"), dtype=torch.float32)
    for row in E:
        assert torch.linalg.matrix_rank(row.reshape(6, 8)) == 1
    # The population average is full rank: this is EGGROLL's whole point.
    assert torch.linalg.matrix_rank(E.sum(0).reshape(6, 8)) == 6


def test_eggroll_rank_is_clipped_per_entry():
    from polystep.baselines.methods import _lowrank_noise

    gen = torch.Generator().manual_seed(0)
    E = _lowrank_noise([(2, 3)], rank=99, n=4, dim=6, generator=gen, device=torch.device("cpu"), dtype=torch.float32)
    assert E.shape == (4, 6) and torch.isfinite(E).all()


def test_eggroll_handles_1d_entries():
    obj = Objective(quadratic(), dim=10, budget=64, shapes=[(2, 3), (4,)])
    result = METHODS["eggroll"](obj, sigma=0.1, lr=1.0, popsize=8)
    assert result.evals == 64


# MeZO specifics


def test_mezo_regenerates_same_perturbation():
    """The same seed must give the same z, or the update points nowhere."""
    seen = []

    def fn(X):
        seen.append(X.clone())
        return ((X - 1.0) ** 2).mean(dim=-1)

    obj = Objective(fn, dim=DIM, budget=2)
    result = METHODS["mezo"](obj, eps=0.5, lr=0.0)
    plus, minus = seen[0]
    # x0 is zeros, so plus = +eps*z and minus = -eps*z.
    assert torch.allclose(plus, -minus)
    assert torch.allclose(result.x, torch.zeros(DIM)), "lr=0 must not move x"


def test_mezo_nonfinite_pair_skips_weight_decay():
    obj = Objective(lambda X: torch.tensor([float("inf"), 1.0]), dim=1, budget=2)
    result = METHODS["mezo"](obj, x0=torch.ones(1), lr=0.01, weight_decay=1.0)
    assert result.x.tolist() == [1.0]
    assert (result.evals, result.iters) == (2, 1)


def test_baseline_objective_never_records_nonfinite_best():
    obj = Objective(lambda x: torch.full((len(x),), float("inf")), dim=2, budget=4)
    obj(torch.zeros(4, 2))
    assert obj.best_loss == float("inf") and obj.best_x is None
