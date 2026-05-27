"""Correctness tests for `polystep.solvers.kl_softmax.KLSoftmaxSolver`.

This solver implements a one-sided KL-penalized entropic OT that
interpolates between the softmax solver (`lam=0`) and the full
Sinkhorn solver (`lam=inf`). Tests cover the limit recoveries,
intermediate marginal interpolation, NaN-safety at small epsilon,
and monotonic convergence of the marginal error.
"""
from __future__ import annotations

import math

import pytest
import torch

from polystep.solvers.kl_softmax import KLSoftmaxSolver
from polystep.solvers.sinkhorn import SinkhornSolver
from polystep.solvers.softmax import SoftmaxSolver


def _make_problem(n: int = 12, m: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    C = torch.rand(n, m, generator=g) * 5.0
    a = torch.full((n,), 1.0 / n)
    b = torch.full((m,), 1.0 / m)
    return C, a, b


def test_lam_zero_recovers_softmax_row_marginals() -> None:
    C, a, b = _make_problem()
    eps = 0.1
    klsolver = KLSoftmaxSolver(epsilon=eps, lam=0.0, max_iterations=500)
    softmax = SoftmaxSolver(epsilon=eps)
    res_kl = klsolver.solve(C, a=a, b=b)
    res_sm = softmax.solve(C, a=a, b=None)
    # Row marginals must equal `a` in both
    torch.testing.assert_close(res_kl.matrix.sum(dim=1), a, atol=1e-5, rtol=0)
    torch.testing.assert_close(res_sm.matrix.sum(dim=1), a, atol=1e-5, rtol=0)
    # Transport matrices match elementwise to ~1e-4
    torch.testing.assert_close(res_kl.matrix, res_sm.matrix, atol=1e-4, rtol=1e-4)


def test_lam_huge_recovers_sinkhorn_full_marginals() -> None:
    C, a, b = _make_problem()
    eps = 0.1
    kl = KLSoftmaxSolver(epsilon=eps, lam=1e6, max_iterations=2000, threshold=1e-8)
    sink = SinkhornSolver(epsilon=eps, max_iterations=2000, threshold=1e-8)
    res_kl = kl.solve(C, a=a, b=b)
    res_sk = sink.solve(C, a=a, b=b)
    # Both should satisfy BOTH marginals
    torch.testing.assert_close(res_kl.matrix.sum(dim=1), a, atol=1e-3, rtol=0)
    torch.testing.assert_close(res_kl.matrix.sum(dim=0), b, atol=1e-3, rtol=0)
    torch.testing.assert_close(res_sk.matrix.sum(dim=0), b, atol=1e-5, rtol=0)
    # Transport matrices match to ~1e-3
    torch.testing.assert_close(res_kl.matrix, res_sk.matrix, atol=1e-3, rtol=1e-3)


def test_intermediate_lam_decreases_kl_to_target() -> None:
    """KL(P^T 1 || b) should decrease monotonically as lam increases."""
    C, a, b = _make_problem()
    eps = 0.1
    kls = []
    for lam in (0.0, 0.1, 1.0, 10.0, 1e3):
        solver = KLSoftmaxSolver(epsilon=eps, lam=lam, max_iterations=1000, threshold=1e-7)
        res = solver.solve(C, a=a, b=b)
        col_sums = res.matrix.sum(dim=0)
        # KL(col_sums || b)
        kl_val = (col_sums * (col_sums.clamp(min=1e-30).log() - b.log())).sum().item()
        kls.append(kl_val)
    # Strictly non-increasing within numerical noise
    for i in range(len(kls) - 1):
        assert kls[i + 1] <= kls[i] + 1e-6, f"KL not monotone: {kls}"


def test_intermediate_lam_softens_column_constraint() -> None:
    """At lam=1, column marginals are between softmax (free) and Sinkhorn (b)."""
    C, a, b = _make_problem()
    eps = 0.1
    kl_zero = KLSoftmaxSolver(epsilon=eps, lam=0.0, max_iterations=500).solve(C, a=a, b=b)
    kl_one = KLSoftmaxSolver(epsilon=eps, lam=1.0, max_iterations=500).solve(C, a=a, b=b)
    kl_huge = KLSoftmaxSolver(epsilon=eps, lam=1e4, max_iterations=2000, threshold=1e-8).solve(C, a=a, b=b)

    err_zero = (kl_zero.matrix.sum(dim=0) - b).abs().max().item()
    err_one = (kl_one.matrix.sum(dim=0) - b).abs().max().item()
    err_huge = (kl_huge.matrix.sum(dim=0) - b).abs().max().item()

    # Stricter constraint as lam grows: the column-marginal error must
    # be (weakly) monotone non-increasing in lam.
    assert err_huge <= err_one <= err_zero, (
        f"column-marginal error should be non-increasing in lam, "
        f"got err_zero={err_zero:.3e} err_one={err_one:.3e} "
        f"err_huge={err_huge:.3e}"
    )


def test_returns_solver_result_with_expected_fields() -> None:
    C, a, b = _make_problem()
    res = KLSoftmaxSolver(epsilon=0.1, lam=1.0).solve(C, a=a, b=b)
    assert hasattr(res, "matrix")
    assert hasattr(res, "cost")
    assert hasattr(res, "n_iters")
    assert hasattr(res, "converged")
    assert res.matrix.shape == C.shape
    # Row marginals enforced
    torch.testing.assert_close(res.matrix.sum(dim=1), a, atol=1e-3, rtol=0)


def test_nan_safe_at_small_epsilon() -> None:
    """Tiny epsilon should not produce NaN/Inf via log-domain stability."""
    C, a, b = _make_problem()
    res = KLSoftmaxSolver(epsilon=1e-3, lam=1.0, max_iterations=500).solve(C, a=a, b=b)
    assert torch.isfinite(res.matrix).all()
    assert res.matrix.min() >= 0


def test_validation_negative_lam_raises() -> None:
    with pytest.raises(ValueError):
        KLSoftmaxSolver(epsilon=0.1, lam=-1.0)


def test_validation_zero_or_negative_epsilon_raises() -> None:
    with pytest.raises(ValueError):
        KLSoftmaxSolver(epsilon=0.0, lam=1.0)
    with pytest.raises(ValueError):
        KLSoftmaxSolver(epsilon=-0.1, lam=1.0)


def test_default_uniform_marginals_when_a_b_none() -> None:
    C = torch.rand(6, 4)
    res = KLSoftmaxSolver(epsilon=0.1, lam=1.0).solve(C)
    # Default a uniform -> row sums uniform 1/6
    torch.testing.assert_close(
        res.matrix.sum(dim=1), torch.full((6,), 1.0 / 6), atol=1e-3, rtol=0,
    )


def test_inf_lam_treated_as_full_sinkhorn() -> None:
    C, a, b = _make_problem()
    res_inf = KLSoftmaxSolver(
        epsilon=0.1, lam=float("inf"), max_iterations=2000, threshold=1e-8,
    ).solve(C, a=a, b=b)
    res_huge = KLSoftmaxSolver(
        epsilon=0.1, lam=1e6, max_iterations=2000, threshold=1e-8,
    ).solve(C, a=a, b=b)
    torch.testing.assert_close(res_inf.matrix, res_huge.matrix, atol=1e-3, rtol=1e-3)
