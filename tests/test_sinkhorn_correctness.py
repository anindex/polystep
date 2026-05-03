"""Regression tests for the Sinkhorn solver.

Covers:

- log-sum-exp stays finite on FP32 / BF16 across wide cost ranges
- epsilon-scheduled warm starts converge fast when ``init_eps`` is
  threaded through for rescaling
- row + column marginals are enforced to ``threshold`` at convergence
- equal cost rows produce equal transport rows (ties broken by
  symmetry, not numerical noise)
- divergence detector backs omega off after sustained growth at the
  aggressive end of the omega range
- omega sweep stays finite across ``[0.5, 1.95]`` on ill-conditioned C
- dual re-centering keeps ``|f|.max`` bounded under cost shifts
- Anderson regression-check guard rejects bad combined iterates
- Anderson history is per-call (implicit restart on epsilon change)
- Anderson depth clamp: lstsq is well-defined for ``k in {1..5}``
"""
from __future__ import annotations

import warnings

import pytest
import torch

from polystep import SinkhornSolver


def _gaussian_cost(P, V, dtype=torch.float32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(P, V, generator=g, dtype=dtype) * scale


# ---------------------------------------------------------------------------
# 2B.1a Log-sum-exp safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,scale", [
    (torch.float32, 50.0),     # FP32 with C entries up to 50/eps=500
    (torch.bfloat16, 20.0),    # BF16 with C entries up to 20/eps=200
])
def test_sinkhorn_lse_safety(dtype, scale):
    """logsumexp must not overflow even when -C/eps is large.

    Pre-condition: PyTorch's logsumexp subtracts the max internally. The
    this test verifies this property holds end-to-end inside the solver.
    """
    P, V = 16, 32
    eps = 0.1
    C = _gaussian_cost(P, V, dtype=dtype) * scale
    solver = SinkhornSolver(epsilon=eps, max_iterations=200, threshold=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)

    assert torch.isfinite(result.f).all(), "f has non-finite entries"
    assert torch.isfinite(result.g).all(), "g has non-finite entries"


# ---------------------------------------------------------------------------
# 2B.1b epsilon-scheduled warm start - rescaling preserves convergence
# ---------------------------------------------------------------------------


def test_sinkhorn_warm_start_rescale_on_eps_change():
    """When epsilon changes between solves, the dual potentials must be
    rescaled by ``eps_new / eps_old`` to preserve convergence speed.

    Without rescaling the warm-started f/g are off by a factor that
    inflates iteration count; the post-fix path accepts ``init_eps`` so
    the caller can opt into rescaling.

    Pre-fix: solver accepts no `init_eps` arg -> raises TypeError.
    Post-fix: rescaled warm-start converges within 3x the iterations of
    the cold start (vs 5-10x without rescaling).
    """
    P, V = 16, 32
    C = _gaussian_cost(P, V, scale=2.0)
    eps_old, eps_new = 1.0, 0.1

    # First solve at eps_old.
    solver_old = SinkhornSolver(epsilon=eps_old, max_iterations=500, threshold=1e-4)
    res_old = solver_old.solve(C)
    assert res_old.converged

    # Cold solve at eps_new (the reference iteration count).
    solver_cold = SinkhornSolver(epsilon=eps_new, max_iterations=500, threshold=1e-4)
    res_cold = solver_cold.solve(C)
    assert res_cold.converged

    # Warm solve at eps_new with init_eps=eps_old: solver rescales internally.
    solver_warm = SinkhornSolver(epsilon=eps_new, max_iterations=500, threshold=1e-4)
    res_warm = solver_warm.solve(
        C, init_f=res_old.f, init_g=res_old.g, init_eps=eps_old,
    )
    assert res_warm.converged

    # Warm-start with rescale should be at most 3x cold-start iters
    # (typically << 1x; 3x is a generous upper bound for noise).
    assert res_warm.n_iters <= 3 * res_cold.n_iters, (
        f"warm-start with rescale used {res_warm.n_iters} iters vs cold "
        f"{res_cold.n_iters}; rescale may not be active."
    )


# ---------------------------------------------------------------------------
# 2B.1c marginal-constraint enforcement
# ---------------------------------------------------------------------------


def test_sinkhorn_marginals_satisfied_at_convergence():
    """Solver must enforce row + col marginals (L_inf, matching the
    solver's own convergence criterion) on 100 random cost matrices."""
    n_problems = 100
    P, V = 12, 18
    eps, tol = 0.1, 1e-4
    failures = []
    converged_count = 0

    for seed in range(n_problems):
        C = _gaussian_cost(P, V, seed=seed, scale=1.0)
        solver = SinkhornSolver(
            epsilon=eps, max_iterations=2000, threshold=tol, check_every=10,
        )
        result = solver.solve(C)
        if not result.converged:
            continue
        converged_count += 1
        T = result.matrix
        a = torch.full((P,), 1.0 / P)
        b = torch.full((V,), 1.0 / V)
        # Use L_inf to match the solver's convergence criterion at
        # sinkhorn.py:401-403; allow 2x slack for the gap between the last
        # in-loop check and the post-loop transport reconstruction.
        err_a = (T.sum(dim=1) - a).abs().max().item()
        err_b = (T.sum(dim=0) - b).abs().max().item()
        if err_a > tol * 2 or err_b > tol * 2:
            failures.append(
                f"seed={seed}: err_a={err_a:.2e}, err_b={err_b:.2e}"
            )

    assert converged_count > 0.9 * n_problems, (
        f"only {converged_count}/{n_problems} problems converged"
    )
    assert not failures, (
        f"marginal constraints violated on {len(failures)}/{converged_count} "
        f"converged problems:\n" + "\n".join(failures[:5])
    )


# ---------------------------------------------------------------------------
# 2B.1d ties symmetry
# ---------------------------------------------------------------------------


def test_sinkhorn_ties_yield_symmetric_transport():
    """Equal cost rows must produce equal transport rows (broken by
    symmetry, not numerical noise)."""
    # 4x4 cost where rows 0 and 1 are identical, and rows 2 and 3 are identical.
    C = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    solver = SinkhornSolver(epsilon=0.5, max_iterations=500, threshold=1e-6)
    result = solver.solve(C)

    T = result.matrix
    assert torch.allclose(T[0], T[1], atol=1e-5), (
        f"rows 0 and 1 should be equal; got diff {(T[0]-T[1]).abs().max()}"
    )
    assert torch.allclose(T[2], T[3], atol=1e-5), (
        f"rows 2 and 3 should be equal; got diff {(T[2]-T[3]).abs().max()}"
    )


# ---------------------------------------------------------------------------
# 2B.2a divergence detector
# ---------------------------------------------------------------------------


def test_sinkhorn_omega_default_is_safe():
    """The default omega=1.0 sits inside the proven-safe range
    (0, 2 - rho) for any well-conditioned cost; no divergence detector
    should ever fire on a benign MNIST-style cost matrix."""
    P, V = 32, 64
    C = _gaussian_cost(P, V, scale=1.0)
    solver = SinkhornSolver(epsilon=0.1, max_iterations=500, threshold=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C)
    msgs = [str(w.message).lower() for w in caught]
    assert not any("diverg" in m for m in msgs), (
        f"divergence detector fired on benign input: {msgs}"
    )


def test_sinkhorn_divergence_detector_backs_off_omega_on_growth():
    """When omega is set near the unstable boundary on an ill-conditioned
    cost, three consecutive growth steps must trigger an automatic
    back-off to omega=1.0 plus a UserWarning.

    Pre-fix: no detector exists; solver may diverge silently and zero the
    duals.
    Post-fix: detector backs omega off and emits a "divergence" warning.
    """
    P, V = 32, 64
    # Ill-conditioned: large dynamic range relative to eps
    C = _gaussian_cost(P, V, scale=200.0)
    eps = 0.05
    solver = SinkhornSolver(
        epsilon=eps, omega=1.95, max_iterations=200, threshold=1e-4,
        check_every=5,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.solve(C)
    msgs = [str(w.message).lower() for w in caught]
    diverg = [m for m in msgs if "diverg" in m and "omega" in m]
    assert diverg, (
        f"expected divergence-detector warning on ill-conditioned cost with "
        f"omega=1.95; got warnings: {msgs}"
    )


# ---------------------------------------------------------------------------
# 2B.2b omega sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omega", [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9, 1.95])
def test_sinkhorn_omega_sweep(omega):
    """Record iteration counts across omega values on an ill-conditioned
    cost. This test asserts the solver does not produce non-finite duals at
    any value in [0.5, 1.95]; omega=1.98 is rejected by validation."""
    P, V = 32, 64
    C = _gaussian_cost(P, V, scale=100.0)
    solver = SinkhornSolver(
        epsilon=0.1, omega=omega, max_iterations=500, threshold=1e-4,
        check_every=10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)
    assert torch.isfinite(result.f).all() and torch.isfinite(result.g).all(), (
        f"omega={omega} produced non-finite duals"
    )


# ---------------------------------------------------------------------------
# 2B.3a dual re-centering on cost shift
# ---------------------------------------------------------------------------


def test_sinkhorn_warm_start_centered_under_cost_shift():
    """Warm-started duals (and the `data_dependent_init` cold init) must
    be re-centered after warm-start validation so |f|.max stays bounded
    even when the cost matrix mean shifts between solves.

    Pre-fix: f.mean() can grow unboundedly across many cost shifts.
    Post-fix: the solver re-centers f and g (zero-mean), so |f|.max is
    bounded by O(cost_scale).
    """
    P, V = 16, 32
    eps = 0.1
    C = _gaussian_cost(P, V, scale=1.0)
    solver = SinkhornSolver(epsilon=eps, max_iterations=300, threshold=1e-4)
    result = solver.solve(C)
    f0, g0 = result.f, result.g

    # Now shift cost by +1000 and warm-start with the previous duals.
    shifted_C = C + 1000.0
    result2 = solver.solve(shifted_C, init_f=f0, init_g=g0)
    cost_scale = shifted_C.abs().max().item()
    assert result2.f.abs().max().item() < 100.0 * cost_scale, (
        f"|f|.max blew up under cost shift: "
        f"{result2.f.abs().max().item():.3e} vs cost_scale={cost_scale:.3e}"
    )


# ---------------------------------------------------------------------------
# 2B.4a Anderson regression-check guard
# ---------------------------------------------------------------------------


def test_sinkhorn_anderson_does_not_diverge_on_ill_conditioned():
    """Anderson acceleration with depth=5 on an ill-conditioned cost
    matrix must converge to similar accuracy as plain Sinkhorn (Chizat 2020:
    when Anderson would regress, fall back to the plain iterate).

    Pre-fix: no regression check; Anderson may push the iterate to a worse
    Lyapunov than plain Sinkhorn would have.
    Post-fix: regression-check guard accepts only Anderson updates that
    do not decrease the Lyapunov function below the plain iterate.
    """
    # Moderately ill-conditioned (range/eps = 10/0.1 = 100). Lower than
    # this is too easy; much higher and plain Sinkhorn does not converge
    # within max_iterations.
    P, V = 32, 64
    C = _gaussian_cost(P, V, scale=10.0)
    eps = 0.1
    plain = SinkhornSolver(
        epsilon=eps, max_iterations=3000, threshold=1e-4,
        anderson_depth=0, check_every=10,
    )
    accel = SinkhornSolver(
        epsilon=eps, max_iterations=3000, threshold=1e-4,
        anderson_depth=5, check_every=10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_plain = plain.solve(C)
        res_accel = accel.solve(C)

    assert res_plain.converged, "plain Sinkhorn failed to converge"
    assert res_accel.converged, "Anderson Sinkhorn failed to converge"

    # Anderson should reach a Lyapunov at least as high as plain (within
    # a small tolerance for stochastic-iterate effects).
    lyap_plain = res_plain.ent_reg_cost
    lyap_accel = res_accel.ent_reg_cost
    rel_gap = abs(lyap_plain - lyap_accel) / max(abs(lyap_plain), 1e-9)
    assert rel_gap < 0.05, (
        f"Anderson Lyapunov {lyap_accel:.6e} differs from plain "
        f"{lyap_plain:.6e} by {rel_gap*100:.2f}% (regression-check guard "
        f"may be missing)"
    )


# ---------------------------------------------------------------------------
# 2B.4b Anderson restart on epsilon change (per-call history)
# ---------------------------------------------------------------------------


def test_sinkhorn_anderson_history_resets_per_call():
    """Anderson history is local to each `solve()` call (created inside
    the convergence-checking branch). Verify by running two solves with
    different epsilons and asserting both converge."""
    P, V = 16, 32
    C = _gaussian_cost(P, V, scale=2.0)
    solver = SinkhornSolver(
        epsilon=1.0, max_iterations=500, threshold=1e-5,
        anderson_depth=5, check_every=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = solver.solve(C)
        # Change eps - if history leaked, the second solve might diverge.
        solver.epsilon = 0.05
        r2 = solver.solve(C)
    assert r1.converged and r2.converged


# ---------------------------------------------------------------------------
# 2B.4c Anderson depth clamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_sinkhorn_anderson_depth_well_defined(depth):
    """Anderson works for depth=1 (k=0) up to depth=5 (k=4)."""
    P, V = 16, 32
    C = _gaussian_cost(P, V, scale=2.0)
    solver = SinkhornSolver(
        epsilon=0.5, max_iterations=300, threshold=1e-5,
        anderson_depth=depth, check_every=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)
    assert result.converged, f"depth={depth} failed to converge"
    assert torch.isfinite(result.f).all(), f"depth={depth} produced NaN/Inf"
