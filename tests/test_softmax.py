"""Regression tests for the Softmax solver.

The Softmax solver is on the critical path for every headline number
in the paper. This file checks:

- numerical safety of ``softmax(-C/eps)`` under FP32 / BF16
- identical-row behavior (uniform output, no NaN)
- source-marginal preservation under FP32 / BF16
- a non-uniform target marginal ``b`` triggers a warning
- a tiny epsilon (relative to ``max|C|``) triggers a warning
- ``scale_cost`` does not mutate the caller's tensor in place
- every headline runner explicitly pins ``solver=softmax``
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch

from polystep import SoftmaxSolver

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Numerical safety of softmax(-C/eps) - overflow grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_overflow_grid(cost_grid, dtype):
    """No NaN / Inf for any combination of cost-range x epsilon.

    PyTorch's softmax subtracts the row-max internally; this test asserts
    the property holds even at the small-eps regime where -C/eps blows up
    before max-subtraction kicks in.
    """
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
            failures.append(
                f"dtype={dtype}, range={cost_range}, eps={eps}: "
                f"{n_bad}/{P*V} non-finite entries"
            )

    assert not failures, "softmax produced non-finite entries:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Identical rows -> uniform output, no NaN
# ---------------------------------------------------------------------------


def test_softmax_identical_row_returns_uniform():
    """When every cost in a row equals 5.0, softmax must return 1/V uniformly.

    Some implementations return NaN here when subtracting max=5 then dividing
    by exp(0)=1 produces 0/0; PyTorch handles this correctly via the
    subtract-max trick, but this test confirms it explicitly.
    """
    P, V = 4, 8
    C = torch.full((P, V), 5.0)
    solver = SoftmaxSolver(epsilon=0.1)
    result = solver.solve(C)

    assert torch.isfinite(result.matrix).all(), "got NaN on identical rows"
    # Each row is uniform with row-sum equal to a_p = 1/P.
    expected_row_value = 1.0 / (P * V)
    assert torch.allclose(
        result.matrix, torch.full_like(result.matrix, expected_row_value),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Source-marginal preservation under FP32 and BF16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.bfloat16, 5e-3)])
def test_softmax_source_marginal_preserved(dtype, tol):
    """transport.sum(-1) must equal source marginal `a` exactly within dtype tol.

    This is the property that lets callers replace Sinkhorn with softmax
    in subspace mode without rebuilding their barycentric projection step.
    """
    P, V = 8, 16
    torch.manual_seed(0)
    C = torch.randn(P, V, dtype=dtype)
    solver = SoftmaxSolver(epsilon=0.5)
    result = solver.solve(C)

    a = torch.full((P,), 1.0 / P, dtype=dtype)
    row_sums = result.matrix.sum(dim=-1)
    max_err = (row_sums - a).abs().max().item()
    assert max_err < tol, (
        f"row sums deviate from a by {max_err} (tol={tol}, dtype={dtype}). "
        f"a={a.tolist()}, row_sums={row_sums.tolist()}"
    )


# ---------------------------------------------------------------------------
# Target marginal `b` silently ignored - should warn (P3 fix)
# ---------------------------------------------------------------------------


def test_softmax_warns_on_nonuniform_b():
    """Softmax solver does not enforce target marginal b; a non-uniform b
    that the caller passes in is silently ignored. The solver warns
    so the user knows their constraint has no effect.
    """
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    solver = SoftmaxSolver(epsilon=0.5)

    # Non-uniform b that softmax cannot enforce.
    b_nonuniform = torch.tensor([0.5, 0.1, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C, b=b_nonuniform)

    msgs = [str(w.message).lower() for w in caught]
    assert any(
        "b" in m and ("ignore" in m or "softmax" in m or "marginal" in m)
        for m in msgs
    ), (
        "expected SoftmaxSolver to warn that target marginal `b` is ignored; "
        f"got warnings: {msgs}"
    )


# ---------------------------------------------------------------------------
# Tiny epsilon underflow - should warn (P2 fix)
# ---------------------------------------------------------------------------


def test_softmax_warns_on_tiny_epsilon():
    """epsilon = 1e-30 is technically positive (passes existing
    `epsilon <= 0` validation) but produces -C/eps overflow in any
    realistic cost matrix. The solver warns whenever
    `eps < 1e-6 * cost_max`.
    """
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V) * 10.0  # cost_max ~ 30
    solver = SoftmaxSolver(epsilon=1e-30)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C)

    msgs = [str(w.message).lower() for w in caught]
    assert any(
        "epsilon" in m and ("underflow" in m or "small" in m or "scale" in m)
        for m in msgs
    ), (
        "expected SoftmaxSolver to warn about tiny epsilon underflow; "
        f"got warnings: {msgs}"
    )


# ---------------------------------------------------------------------------
# Cost-matrix scaling must clone - no in-place mutation
# ---------------------------------------------------------------------------


def test_softmax_does_not_mutate_caller_cost_matrix():
    """If the caller reuses a cost matrix across multiple solve() calls
    (for example, A/B testing a few epsilons on the same probe data), the
    solver must not mutate it via scale_cost_matrix.
    """
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    C_before = C.clone()

    solver = SoftmaxSolver(epsilon=0.5)
    solver.solve(C, scale_cost="mean")

    assert torch.equal(C, C_before), (
        "solver mutated caller's cost matrix via scale_cost_matrix. "
        "softmax.py:87 must clone before scaling."
    )


# ---------------------------------------------------------------------------
# Static check: every headline runner forces softmax
# ---------------------------------------------------------------------------


HEADLINE_RUNNERS = (
    ("experiments/runners/run_moe.py", '"softmax"'),
    ("experiments/runners/run_elevation.py", '"softmax"'),
    ("experiments/runners/run_maxsat_softmax_scaling.py", "'softmax'"),
)


@pytest.mark.parametrize("relpath,literal", HEADLINE_RUNNERS)
def test_headline_runner_uses_softmax(relpath, literal):
    """Verify that headline runners hard-code solver=softmax.

    Sets a regression guard if a refactor accidentally removes the explicit pin
    and lets the auto-selection rule (subspace -> softmax, else sinkhorn)
    silently pick a different solver.
    """
    path = REPO_ROOT / relpath
    src = path.read_text()
    pattern = re.compile(rf"solver\s*=\s*{re.escape(literal)}")
    assert pattern.search(src), (
        f"{relpath} no longer pins solver={literal}. "
        f"This may indicate the headline configuration drifted off softmax."
    )


# ---------------------------------------------------------------------------
# API contract: epsilon validation, result-object invariants, init_f/g ignored
# ---------------------------------------------------------------------------


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
