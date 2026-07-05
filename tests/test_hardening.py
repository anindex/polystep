"""Regression tests for the solver-hardening pass.

Covers the shared validated prelude, fused/non-fused softmax scale_cost parity,
and the new input-validation guards (epsilon revalidation, check_every,
num_probe, numeric scale_cost).
"""

import warnings

import pytest
import torch

import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.costs import scale_cost_matrix
from polystep._compiled import _fused_softmax_project
from polystep.geometry import get_orthoplex_vertices, get_random_rotation_matrices
from polystep.solvers import SinkhornSolver, SoftmaxSolver, TemperedSoftmaxSolver
from polystep.solvers._prelude import align_dual, align_marginal, sanitize_cost


# --------------------------------------------------------------------------
# Fused vs non-fused softmax: scale_cost must be honored identically
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale_cost", ["mean", "max_cost", 2.0, None])
@pytest.mark.parametrize("variant", ["finite", "inf", "bf16"])
def test_fused_softmax_matches_solver_scale_cost(scale_cost, variant):
    """The fused path (optimizer flow: sanitize_cost -> scale -> kernel) matches
    SoftmaxSolver.solve for every scale_cost mode, including +inf hard
    constraints and BF16. The original bug: the fused path ignored
    'max_cost'/float and treated None as 'mean'. The +inf/BF16 cases guard the
    parity too, since SoftmaxSolver sanitizes the cost and the fused path must
    match it."""
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

    # Fused path as wired in the optimizer: step sanitizes, branch scales.
    scaled = scale_cost_matrix(sanitize_cost(cost), scale_cost)
    _, fused_T, _ = _fused_softmax_project(
        scaled,
        eps,
        a.float(),
        verts.float(),
        rot.float(),
        0.15,
        X.float(),
        scale_cost_mean=False,
    )
    assert torch.isfinite(fused_T).all()
    assert torch.allclose(ref, fused_T, atol=1e-5)


# --------------------------------------------------------------------------
# Input validation guards
# --------------------------------------------------------------------------


def test_num_probe_zero_raises():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="num_probe"):
        PolyStepOptimizer(model, num_probe=0)


def test_numeric_scale_cost_zero_and_inf_raise():
    cost = torch.rand(3, 4)
    with pytest.raises(ValueError, match="scale_cost"):
        scale_cost_matrix(cost, 0.0)
    with pytest.raises(ValueError, match="scale_cost"):
        scale_cost_matrix(cost, float("inf"))


def test_sinkhorn_epsilon_revalidated_per_solve():
    """Schedules mutate solver.epsilon; a bad value must be caught in solve()
    rather than dividing by zero (it is only validated in __post_init__)."""
    solver = SinkhornSolver(threshold=1e-3, max_iterations=10)
    solver.epsilon = -1.0  # simulate a misconfigured schedule
    with pytest.raises(ValueError, match="epsilon"):
        solver.solve(torch.rand(4, 6))


def test_check_every_must_be_positive():
    with pytest.raises(ValueError, match="check_every"):
        SinkhornSolver(check_every=0)


# --------------------------------------------------------------------------
# Shared prelude behavior
# --------------------------------------------------------------------------


def test_sanitize_cost_promotes_and_replaces_nonfinite():
    cost = torch.tensor([[1.0, float("inf")], [float("nan"), 2.0]], dtype=torch.bfloat16)
    out = sanitize_cost(cost)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    # finite entries preserved
    assert out[0, 0].item() == pytest.approx(1.0, abs=1e-2)


def test_align_marginal_defaults_and_coerces():
    a = align_marginal(None, 5, torch.device("cpu"), torch.float32)
    assert a.shape == (5,) and torch.allclose(a.sum(), torch.tensor(1.0))
    a64 = torch.ones(5, dtype=torch.float64) / 5
    out = align_marginal(a64, 5, torch.device("cpu"), torch.float32)
    assert out.dtype == torch.float32


def test_align_dual_coerces_dtype_and_warns_on_mismatch():
    f64 = torch.zeros(5, dtype=torch.float64)
    out = align_dual(f64, 5, torch.device("cpu"), torch.float32, "init_f")
    assert out is not None and out.dtype == torch.float32
    with pytest.warns(UserWarning, match="shape mismatch"):
        bad = align_dual(torch.zeros(99), 5, torch.device("cpu"), torch.float32, "init_f")
    assert bad is None


def test_tempered_softmax_handles_nonfinite_cost():
    """TemperedSoftmaxSolver previously lacked FP32 promotion + finite handling."""
    cost = torch.tensor([[1.0, float("inf")], [0.5, 2.0]], dtype=torch.bfloat16)
    res = TemperedSoftmaxSolver(tau=0.5).solve(cost)
    assert torch.isfinite(res.matrix).all()
    # row sums equal the (uniform) source marginal
    assert torch.allclose(res.matrix.sum(dim=1), torch.full((2,), 0.5), atol=1e-5)


def test_batched_linear_honors_nondefault_activations():
    """The MLP fast path must use the activation modules' real config; it
    previously hardcoded leaky_relu(0.01)/gelu('none') -> silently wrong loss."""
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
    ref = ev._evaluate_loop(stacked, inputs, targets)  # functional_call on real model
    assert torch.allclose(fast, ref, atol=1e-5)


def test_batched_linear_skips_nondefault_flatten():
    """A non-default Flatten must fall back to vmap, not the bmm fast path."""
    model = nn.Sequential(nn.Flatten(start_dim=2), nn.Linear(6, 3))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert ev._batched_linear is None


def test_batched_linear_rejects_inline_functional_activations():
    """A custom module applying activations inline (torch.relu in forward) has
    no activation submodule, so the bmm plan reconstructed from named_children()
    would silently omit it and compute a wrong (activation-free) loss. try_build
    must detect the non-Sequential forward and fall back to the correct vmap path."""

    class InlineMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 8)
            self.fc2 = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    ev = NNCostEvaluator(InlineMLP(), nn.CrossEntropyLoss())
    assert ev._batched_linear is None


def test_warmstart_dual_centering_keeps_marginals():
    """Gauge-preserving re-centering must not break marginal convergence under
    overrelaxation (omega != 1), where independent centering would perturb it."""
    torch.manual_seed(0)
    cost = torch.rand(8, 8)
    solver = SinkhornSolver(epsilon=0.1, omega=1.5, threshold=1e-8, max_iterations=2000)
    res = solver.solve(cost)
    # warm-start a second solve from the first solution
    res2 = solver.solve(cost, init_f=res.f, init_g=res.g, init_eps=0.1)
    P = res2.matrix
    a = torch.ones(8) / 8
    assert torch.allclose(P.sum(dim=1), a, atol=1e-4)
