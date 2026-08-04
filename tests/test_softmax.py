"""Numerical contract of the Softmax solver: overflow safety, marginals,
warnings, no caller-tensor mutation, and the runner solver pin.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch

from polystep import SoftmaxSolver

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_overflow_grid(cost_grid, dtype):
    """No NaN / Inf for any combination of cost-range x epsilon."""
    P, V = 16, 32
    torch.manual_seed(0)
    base = torch.randn(P, V, dtype=dtype)

    failures = []
    for cost_range, eps in cost_grid:
        C = base * cost_range
        solver = SoftmaxSolver(epsilon=eps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(C)

        if not torch.isfinite(result.matrix).all():
            n_bad = (~torch.isfinite(result.matrix)).sum().item()
            failures.append(f"dtype={dtype}, range={cost_range}, eps={eps}: {n_bad}/{P * V} non-finite entries")

    assert not failures, "softmax produced non-finite entries:\n" + "\n".join(failures)


def test_softmax_identical_row_returns_uniform():
    """An identical cost row must return a uniform 1/V row, not NaN."""
    P, V = 4, 8
    C = torch.full((P, V), 5.0)
    solver = SoftmaxSolver(epsilon=0.1)
    result = solver.solve(C)

    assert torch.isfinite(result.matrix).all(), "got NaN on identical rows"
    # Each row is uniform with row-sum equal to a_p = 1/P.
    expected_row_value = 1.0 / (P * V)
    assert torch.allclose(
        result.matrix,
        torch.full_like(result.matrix, expected_row_value),
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.bfloat16, 5e-3)])
def test_softmax_source_marginal_preserved(dtype, tol):
    """transport.sum(-1) must equal the source marginal ``a`` within dtype tol."""
    P, V = 8, 16
    torch.manual_seed(0)
    C = torch.randn(P, V, dtype=dtype)
    solver = SoftmaxSolver(epsilon=0.5)
    result = solver.solve(C)

    a = torch.full((P,), 1.0 / P, dtype=dtype)
    row_sums = result.matrix.sum(dim=-1)
    max_err = (row_sums - a).abs().max().item()
    assert max_err < tol, (
        f"row sums deviate from a by {max_err} (tol={tol}, dtype={dtype}). a={a.tolist()}, row_sums={row_sums.tolist()}"
    )


def test_softmax_warns_on_nonuniform_b():
    """Softmax cannot enforce a non-uniform target marginal ``b``; it must warn."""
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    solver = SoftmaxSolver(epsilon=0.5)

    b_nonuniform = torch.tensor([0.5, 0.1, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C, b=b_nonuniform)

    msgs = [str(w.message).lower() for w in caught]
    assert any("b" in m and ("ignore" in m or "softmax" in m or "marginal" in m) for m in msgs), (
        f"expected SoftmaxSolver to warn that target marginal `b` is ignored; got warnings: {msgs}"
    )


def test_softmax_warns_on_tiny_epsilon():
    """The solver must warn when eps < 1e-6 * cost_max, where -C/eps overflows."""
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V) * 10.0  # cost_max ~ 30
    solver = SoftmaxSolver(epsilon=1e-30)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C)

    msgs = [str(w.message).lower() for w in caught]
    assert any("epsilon" in m and ("underflow" in m or "small" in m or "scale" in m) for m in msgs), (
        f"expected SoftmaxSolver to warn about tiny epsilon underflow; got warnings: {msgs}"
    )


def test_softmax_does_not_mutate_caller_cost_matrix():
    """solve() must not mutate the caller's cost matrix when scale_cost is set."""
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    C_before = C.clone()

    solver = SoftmaxSolver(epsilon=0.5)
    solver.solve(C, scale_cost="mean")

    assert torch.equal(C, C_before), (
        "solver mutated caller's cost matrix via scale_cost_matrix. softmax.py:87 must clone before scaling."
    )


def test_runner_pins_softmax_solver(require_experiments):
    """Result-reporting runners must pin softmax, not inherit auto-selection.

    The gallery runners pin it through ``fairness.build_polystep``; MAX-SAT scaling
    builds its own optimizer and carries the literal.
    """
    import inspect

    from experiments.runners.fairness import build_polystep

    assert inspect.signature(build_polystep).parameters["solver"].default == "softmax"
    for relpath in ("experiments/runners/run_moe.py", "experiments/runners/run_elevation.py"):
        src = (REPO_ROOT / relpath).read_text()
        assert not re.search(r"build_polystep\([^)]*solver\s*=", src), (
            f"{relpath} overrides the pinned solver on a build_polystep call"
        )
    scaling = (REPO_ROOT / "experiments/runners/run_maxsat_softmax_scaling.py").read_text()
    assert re.search(r"solver\s*=\s*'softmax'", scaling)


@pytest.mark.parametrize("eps", [0.0, -0.1])
def test_softmax_rejects_nonpositive_epsilon(eps):
    """epsilon <= 0 must raise ValueError on solve()."""
    solver = SoftmaxSolver(epsilon=eps)
    with pytest.raises(ValueError, match="epsilon"):
        solver.solve(torch.rand(5, 8))


def test_softmax_result_invariants():
    """SolverResult fields are well-formed: f/g None, converged, n_iters=1."""
    solver = SoftmaxSolver(epsilon=0.1)
    result = solver.solve(torch.rand(5, 8))
    assert result.f is None
    assert result.g is None
    assert result.converged is True
    assert result.n_iters == 1
    assert isinstance(result.ent_reg_cost, float)


def test_softmax_init_f_and_g_are_ignored():
    """init_f / init_g are accepted for SolverProtocol parity but must not
    affect the output (softmax is closed-form, no warm start)."""
    torch.manual_seed(42)
    C = torch.rand(5, 8)
    solver = SoftmaxSolver(epsilon=0.1)
    r0 = solver.solve(C)
    r1 = solver.solve(C, init_f=torch.rand(5), init_g=torch.rand(8))
    torch.testing.assert_close(r0.matrix, r1.matrix)


def test_default_epsilon_sets_the_selectivity():
    """The 0.1 default is what an unconfigured solver picks with, so pin its weights."""
    C = torch.tensor([[0.0, 0.1, 0.2]])
    weights = SoftmaxSolver().solve(C).matrix
    expected = torch.softmax(-C / 0.1, dim=1)
    torch.testing.assert_close(weights, expected)
    assert weights[0, 0].item() == pytest.approx(0.6652409, abs=1e-6)
