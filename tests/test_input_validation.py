"""Tests that invalid inputs are rejected and the fused softmax path matches
SoftmaxSolver on every scale_cost mode.
"""

import math

import pytest
import torch

import torch.nn as nn

from polystep import AdaptiveSubspace, HybridSubspace, ParamLayout, PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep._compiled import _fused_softmax_project
from polystep.geometry import get_orthoplex_vertices, get_random_rotation_matrices
from polystep.solvers import KLSoftmaxSolver, SinkhornSolver, SoftmaxSolver
from polystep.solvers._shared import align_dual, align_marginal, prepare_cost, sanitize_cost


@pytest.mark.parametrize("scale_cost", ["mean", "max_cost", 2.0, None])
@pytest.mark.parametrize("variant", ["finite", "inf", "bf16"])
def test_fused_softmax_matches_solver_scale_cost(scale_cost, variant):
    """The fused path (prepare_cost -> kernel) matches SoftmaxSolver.solve for every
    scale_cost mode, including +inf hard constraints and BF16."""
    torch.manual_seed(0)
    P, dim = 6, 2
    verts = get_orthoplex_vertices(dim)  # (V, dim), V = 2*dim
    V = verts.shape[0]
    cost = torch.randn(P, V)
    if variant == "inf":
        cost[0, 0] = float("inf")
    elif variant == "bf16":
        cost = cost.to(torch.bfloat16)
    a = torch.ones(P) / P
    rot = get_random_rotation_matrices(P, dim)
    X = torch.randn(P, dim)
    eps = 0.1

    # Non-fused reference (sanitizes internally)
    ref = SoftmaxSolver(epsilon=eps).solve(cost, a=a, scale_cost=scale_cost).matrix

    scaled = prepare_cost(cost, None, scale_cost, "SoftmaxSolver")[0]
    _, fused_T = _fused_softmax_project(
        scaled,
        eps,
        verts.float(),
        rot.float(),
        0.15,
        X.float(),
    )
    assert torch.isfinite(fused_T).all()
    assert torch.allclose(ref, fused_T * a[:, None], atol=1e-5)


def test_num_probe_zero_raises():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="num_probe"):
        PolyStepOptimizer(model, num_probe=0)


@pytest.mark.parametrize("solver_class", [SinkhornSolver, KLSoftmaxSolver, SoftmaxSolver])
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("inf"), float("nan")])
def test_epsilon_revalidated_per_solve(solver_class, epsilon):
    """Schedules can mutate epsilon after construction."""
    solver = solver_class()
    solver.epsilon = epsilon
    with pytest.raises(ValueError, match="epsilon"):
        solver.solve(torch.rand(4, 6))


def test_check_every_must_be_positive():
    with pytest.raises(ValueError, match="check_every"):
        SinkhornSolver(check_every=0)


def test_sanitize_cost_promotes_and_replaces_nonfinite():
    cost = torch.tensor([[1.0, float("inf")], [float("nan"), 2.0]], dtype=torch.bfloat16)
    out = sanitize_cost(cost)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert out[0, 0].item() == pytest.approx(1.0, abs=1e-2)


def test_align_marginal_defaults_and_coerces():
    a = align_marginal(None, 5, torch.device("cpu"), torch.float32)
    assert a.shape == (5,) and torch.allclose(a.sum(), torch.tensor(1.0))
    a64 = torch.ones(5, dtype=torch.float64) / 5
    out = align_marginal(a64, 5, torch.device("cpu"), torch.float32)
    assert out.dtype == torch.float32


def test_align_dual_coerces_dtype_warns_mismatch():
    f64 = torch.zeros(5, dtype=torch.float64)
    out = align_dual(f64, 5, torch.device("cpu"), torch.float32, "init_f")
    assert out is not None and out.dtype == torch.float32
    with pytest.warns(UserWarning, match="shape mismatch"):
        bad = align_dual(torch.zeros(99), 5, torch.device("cpu"), torch.float32, "init_f")
    assert bad is None


def test_batched_linear_honors_nondefault_activations():
    """The MLP fast path must read each activation module's real config; a
    hardcoded default would compute a silently wrong loss."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(6, 8),
        nn.LeakyReLU(0.3),
        nn.Linear(8, 5),
        nn.GELU(approximate="tanh"),
        nn.Linear(5, 3),
    )
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert ev._batched_linear is not None, "fast path should activate for a pure MLP"

    N, B = 4, 10
    inputs = torch.randn(B, 6)
    targets = torch.randint(0, 3, (B,))
    base = dict(model.named_parameters())
    stacked = {k: v.detach()[None].expand(N, *v.shape) + 0.05 * torch.randn(N, *v.shape) for k, v in base.items()}
    fast = ev._batched_linear.evaluate(stacked, inputs, targets)
    ref = ev._evaluate_loop(stacked, inputs, targets)
    assert torch.allclose(fast, ref, atol=1e-5)


def test_batched_linear_skips_nondefault_flatten():
    """A non-default Flatten must fall back to vmap, not the bmm fast path."""
    model = nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(6, 3))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert ev._batched_linear is None


def test_warmstart_dual_centering_keeps_marginals():
    """Gauge-preserving re-centering must not break marginal convergence under
    overrelaxation (omega != 1), where independent centering would perturb it."""
    torch.manual_seed(0)
    cost = torch.rand(8, 8)
    solver = SinkhornSolver(epsilon=0.1, omega=1.5, threshold=1e-8, max_iterations=2000)
    res = solver.solve(cost)
    res2 = solver.solve(cost, init_f=res.f, init_g=res.g)
    P = res2.matrix
    a = torch.ones(8) / 8
    assert torch.allclose(P.sum(dim=1), a, atol=1e-4)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"block_strategy": "sequential"}, "block_strategy"),
        ({"block_strategy": "per-layer"}, "block_strategy"),
        ({"polytope_type": "hypercube"}, "polytope_type"),
    ],
)
def test_unknown_string_options_raise(kwargs, message):
    """Downstream code tests `!= "monolithic"`, so a typo would select block-wise mode."""
    with pytest.raises(ValueError, match=message):
        PolyStepOptimizer(nn.Linear(8, 4), compile=False, **kwargs)


def test_unsupported_subspace_type_raises():
    with pytest.raises(TypeError, match="subspace must be"):
        PolyStepOptimizer(nn.Linear(8, 4), subspace=object(), compile=False)


def test_newton_refinement_warns_under_block_strategy():
    """newton_refinement auto-enables the quadratic model, which block-wise ignores."""
    with pytest.warns(UserWarning, match="use_quadratic_model=True is not supported"):
        PolyStepOptimizer(
            nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4)),
            newton_refinement=True,
            use_quadratic_model=False,
            polytope_type="orthoplex",
            num_probe=2,
            block_strategy="per_layer",
            compile=False,
        )


def test_fully_frozen_model_raises_clearly():
    """A model with no trainable parameters must say so, not fail deep in the layout."""
    model = nn.Sequential(nn.Linear(4, 4))
    for param in model.parameters():
        param.requires_grad_(False)

    with pytest.raises(ValueError, match="requires_grad"):
        PolyStepOptimizer(model, epsilon=0.1, max_iterations=5)


def test_freeze_after_construction_raises():
    """The layout is fixed at construction, so a later freeze must not be ignored."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    optimizer = PolyStepOptimizer(model, seed=0)
    model[0].requires_grad_(False)
    with pytest.raises(ValueError, match="requires_grad"):
        optimizer.step(lambda params: torch.zeros(next(iter(params.values())).shape[0]))


@pytest.mark.parametrize(
    "build,match",
    [
        (
            lambda: PolyStepOptimizer(nn.Linear(4, 2), block_strategy="grouped", block_group_size=-1),
            "block_group_size",
        ),
        (lambda: PolyStepOptimizer(nn.Linear(4, 2), chunk_size=-1), "chunk_size"),
    ],
    ids=["block_group_size", "chunk_size"],
)
def test_degenerate_argument_rejected(build, match):
    """Negative block_group_size gives no blocks; negative chunk_size, no chunks."""
    with pytest.raises(ValueError, match=match):
        build()


def test_cost_batch_size_zero_is_rejected():
    """0 slices an empty batch: every candidate scores NaN, sanitization flattens
    the cost matrix to a constant, and the run reports finite losses while
    learning nothing.
    """
    with pytest.raises(ValueError, match="cost_batch_size"):
        PolyStepOptimizer(nn.Linear(4, 2), cost_batch_size=0)


@pytest.mark.parametrize("target", [0.0, -0.1, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["epsilon", "ent_epsilon"])
@pytest.mark.parametrize("scheduled", [False, True])
def test_invalid_temperature_rejected_before_fused_step(target, name, scheduled):
    """Zero freezes the run through NaN reversion and a negative value ascends."""
    from polystep.epsilon import LinearEpsilon

    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, solver="softmax")
    value = LinearEpsilon(init=1.0, target=target, decay=10.0) if scheduled else target
    setattr(opt, name, value)
    with pytest.raises(ValueError, match=f"{name} must be > 0"):
        for _ in range(4):
            opt.step(lambda p: torch.full((next(iter(p.values())).shape[0],), 1.0))


def test_subspace_particle_dim_zero_raises():
    model = nn.Linear(4, 2)
    sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    with pytest.raises(ValueError, match="subspace_particle_dim"):
        PolyStepOptimizer(model, subspace=sub, subspace_particle_dim=0, compile=False)


def test_svd_ratio_above_one_raises():
    with pytest.raises(ValueError, match="svd_ratio_final"):
        AdaptiveSubspace(full_dim=10, subspace_dim=3, svd_ratio_final=2.0)


def test_bare_adaptive_subspace_steps():
    """Without from_layout the optimizer fills the entry specs, and a size mismatch raises."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    with pytest.raises(ValueError, match="from_layout"):
        PolyStepOptimizer(model, subspace=AdaptiveSubspace(full_dim=11, subspace_dim=4), compile=False)
    sub = AdaptiveSubspace(full_dim=10, subspace_dim=4)
    opt = PolyStepOptimizer(model, subspace=sub, subspace_particle_dim=2, compile=False)
    ev = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(8, 4), torch.randn(8, 2)
    assert math.isfinite(opt.step(lambda p: ev.evaluate(p, inputs, targets)))
