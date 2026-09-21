"""Correctness tests for `polystep.solvers.kl_softmax.KLSoftmaxSolver`.

Covers the lam=0 / lam=inf limit recoveries, intermediate marginal
interpolation, NaN-safety at small epsilon, and monotone convergence.
"""

from __future__ import annotations


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
    torch.testing.assert_close(res_kl.matrix.sum(dim=1), a, atol=1e-5, rtol=0)
    torch.testing.assert_close(res_sm.matrix.sum(dim=1), a, atol=1e-5, rtol=0)
    torch.testing.assert_close(res_kl.matrix, res_sm.matrix, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("epsilon", [1e-3, 1e-20])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_row_mass_survives_large_row_offsets(epsilon, lam):
    costs = torch.tensor([[0.0, 1.0], [100.0, 101.0], [50.0, 51.0]])
    a = torch.tensor([0.25, 0.75, 0.0])
    result = KLSoftmaxSolver(epsilon=epsilon, lam=lam, max_iterations=20).solve(costs, a=a)
    torch.testing.assert_close(result.matrix.sum(1), a, rtol=1e-6, atol=0)
    if lam == 0:
        expected = SoftmaxSolver(epsilon=epsilon).solve(costs, a=a)
        torch.testing.assert_close(result.matrix, expected.matrix, rtol=1e-6, atol=0)
        assert result.cost == pytest.approx(expected.cost)


def test_lam_huge_recovers_sinkhorn_full_marginals() -> None:
    C, a, b = _make_problem()
    eps = 0.1
    kl = KLSoftmaxSolver(epsilon=eps, lam=1e6, max_iterations=2000, threshold=1e-8)
    sink = SinkhornSolver(epsilon=eps, max_iterations=2000, threshold=1e-8)
    res_kl = kl.solve(C, a=a, b=b)
    res_sk = sink.solve(C, a=a, b=b)
    torch.testing.assert_close(res_kl.matrix.sum(dim=1), a, atol=1e-3, rtol=0)
    torch.testing.assert_close(res_kl.matrix.sum(dim=0), b, atol=1e-3, rtol=0)
    torch.testing.assert_close(res_sk.matrix.sum(dim=0), b, atol=1e-5, rtol=0)
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

    assert err_huge <= err_one <= err_zero, (
        f"column-marginal error should be non-increasing in lam, "
        f"got err_zero={err_zero:.3e} err_one={err_one:.3e} "
        f"err_huge={err_huge:.3e}"
    )


def test_nan_safe_at_small_epsilon() -> None:
    """Tiny epsilon should not produce NaN/Inf via log-domain stability."""
    C, a, b = _make_problem()
    res = KLSoftmaxSolver(epsilon=1e-3, lam=1.0, max_iterations=500).solve(C, a=a, b=b)
    assert torch.isfinite(res.matrix).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0.1, "lam": -1.0},
        {"epsilon": 0.0, "lam": 1.0},
        {"epsilon": -0.1, "lam": 1.0},
    ],
)
def test_validation_invalid_constructor_args_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        KLSoftmaxSolver(**kwargs)


@pytest.mark.parametrize("epsilon,lam,expected", [(1e308, 1e308, 0.5), (1.0, 1e-309, 1e-309)])
def test_alpha_avoids_overflow(epsilon, lam, expected):
    assert KLSoftmaxSolver(epsilon=epsilon, lam=lam).alpha == pytest.approx(expected, abs=0.0)


def test_default_uniform_marginals_when_a_b_none() -> None:
    C = torch.rand(6, 4)
    res = KLSoftmaxSolver(epsilon=0.1, lam=1.0).solve(C)
    torch.testing.assert_close(
        res.matrix.sum(dim=1),
        torch.full((6,), 1.0 / 6),
        atol=1e-3,
        rtol=0,
    )


def test_inf_lam_treated_as_full_sinkhorn() -> None:
    C, a, b = _make_problem()
    res_inf = KLSoftmaxSolver(
        epsilon=0.1,
        lam=float("inf"),
        max_iterations=2000,
        threshold=1e-8,
    ).solve(C, a=a, b=b)
    res_huge = KLSoftmaxSolver(
        epsilon=0.1,
        lam=1e6,
        max_iterations=2000,
        threshold=1e-8,
    ).solve(C, a=a, b=b)
    torch.testing.assert_close(res_inf.matrix, res_huge.matrix, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("lam", [0.0, 1.0, 10.0, float("inf")])
@pytest.mark.parametrize("iters", [1, 2, 50])
def test_row_marginal_is_exact_at_any_iteration_count(lam, iters) -> None:
    """P1 == a is the one hard constraint this solver enforces, so it must hold at
    every iteration count rather than waiting for convergence."""
    torch.manual_seed(0)
    n, m = 6, 5
    a = torch.full((n,), 1.0 / n)
    result = KLSoftmaxSolver(epsilon=0.1, lam=lam, max_iterations=iters).solve(torch.rand(n, m))
    torch.testing.assert_close(result.matrix.sum(dim=-1), a, atol=1e-6, rtol=1e-6)


def test_optimizer_wires_lam_through() -> None:
    from polystep.optimizer import PolyStepOptimizer

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    opt = PolyStepOptimizer(model=model, particle_dim=2, seed=0, solver="kl_softmax", kl_softmax_lam=2.0)
    assert isinstance(opt.solver, KLSoftmaxSolver)
    assert opt.solver.lam == 2.0


def test_the_limits_inherit_the_guards_of_the_solver_they_become() -> None:
    """lam=inf is balanced Sinkhorn and lam=0 is the one-shot softmax, so each limit
    must inherit that solver's optimizer-side guards."""
    from polystep.epsilon import ProgressiveEpsilon
    from polystep.optimizer import PolyStepOptimizer

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    with pytest.warns(UserWarning, match="num_particles=1"):
        PolyStepOptimizer(model=model, particle_dim=14, seed=0, solver="kl_softmax", kl_softmax_lam=float("inf"))

    with pytest.raises(ValueError, match="ProgressiveEpsilon requires"):
        PolyStepOptimizer(
            model=model,
            particle_dim=2,
            seed=0,
            solver="kl_softmax",
            kl_softmax_lam=0.0,
            epsilon=ProgressiveEpsilon(init=0.5),
        )


@pytest.mark.parametrize("bad", ["f", "g", "both"])
def test_a_partly_non_finite_warm_start_is_dropped_whole(bad):
    """One poisoned dual has to discard both, or the LSE carries the NaN through."""
    C = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    solver = KLSoftmaxSolver(epsilon=0.5, lam=1.0)

    clean = solver.solve(C)
    f = torch.zeros(2)
    g = torch.zeros(3)
    if bad in ("f", "both"):
        f = f.clone()
        f[0] = float("nan")
    if bad in ("g", "both"):
        g = g.clone()
        g[1] = float("inf")

    poisoned = solver.solve(C, init_f=f, init_g=g)
    assert torch.isfinite(poisoned.matrix).all()
    torch.testing.assert_close(poisoned.matrix, clean.matrix)


def test_convergence_is_measured_on_the_dual_step_not_its_magnitude():
    """A skewed ``b`` drives the column duals away from zero, so a residual built from
    anything but the step between iterates stops reporting convergence."""
    C = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
    b = torch.tensor([0.6, 0.3, 0.1])
    result = KLSoftmaxSolver(epsilon=0.5, lam=1.0, max_iterations=2000).solve(C, b=b)
    assert result.converged
    assert result.n_iters < 500
    assert result.g.abs().max() > 0.1, "the duals must be off zero, or the test proves nothing"


def test_denormal_lam_takes_the_softmax_closed_form():
    """Below fp32's smallest normal the damped g-update underflows to zero, making the
    residual's ``/alpha`` normalization 0/0 = nan, so the solver must take the softmax
    closed form instead."""
    solver = KLSoftmaxSolver(epsilon=0.1, lam=1e-300, max_iterations=2000)
    result = solver.solve(torch.rand(16, 8))
    assert result.converged and result.n_iters < 10, (result.converged, result.n_iters)
    torch.testing.assert_close(result.g, torch.zeros_like(result.g))


def test_fp64_resolves_an_alpha_an_fp32_bound_would_flatten():
    """The softmax-limit cutoff follows the working dtype: alpha = lam/(lam+eps) ~ 1e-39
    is denormal in fp32 but ordinary in fp64, so an fp64 solve must run the damped
    iteration and return a non-zero ``g``."""
    C = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]], dtype=torch.float64)
    result = KLSoftmaxSolver(epsilon=0.1, lam=1e-40, max_iterations=2000).solve(C)
    assert result.converged
    assert result.g.abs().max() > 0, "fp64 must resolve the KL damping, not flatten it to the softmax limit"
    # fp32 takes the closed form: the same alpha is denormal there.
    fp32 = KLSoftmaxSolver(epsilon=0.1, lam=1e-40, max_iterations=2000).solve(C.float())
    assert fp32.converged and fp32.n_iters < 10
    torch.testing.assert_close(fp32.g, torch.zeros_like(fp32.g))
