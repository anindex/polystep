"""Unit tests for SinkhornSolver (full-rank log-domain)."""

import torch
import pytest

from polystep.costs import resolve_cost_scale
from polystep.solvers import KLSoftmaxSolver, SinkhornSolver
from polystep.solvers._shared import recenter_cost


class TestSinkhornSolver:
    def test_small_epsilon_near_deterministic(self):
        """Small epsilon pushes transport toward near one-hot rows."""
        torch.manual_seed(42)
        n = 5
        C = torch.rand(n, n)

        solver = SinkhornSolver(
            epsilon=0.001,
            max_iterations=5000,
            threshold=1e-10,
            compile=False,
        )
        result = solver.solve(C)

        P = result.matrix
        row_sums = P.sum(dim=1)
        max_vals = P.max(dim=1).values
        ratios = max_vals / row_sums
        assert (ratios > 0.9).all(), f"Row concentration ratios: {ratios.tolist()}"

    def test_known_ot_solution_diagonal(self):
        """Identity cost matrix should produce near-diagonal transport plan."""
        torch.manual_seed(42)
        n = 5
        C = 1.0 - torch.eye(n)  # 0 on diagonal, 1 off-diagonal

        solver = SinkhornSolver(
            epsilon=0.01,
            max_iterations=5000,
            threshold=1e-10,
            compile=False,
        )
        result = solver.solve(C)

        P = result.matrix
        # Diagonal dominates every off-diagonal entry.
        diag = torch.diag(P)
        off_diag_max = (P - torch.diag(diag)).max()
        assert diag.min() > off_diag_max, (
            f"Diagonal min {diag.min():.4f} should exceed off-diagonal max {off_diag_max:.4f}"
        )

    def test_scale_cost_mean(self):
        """Scale cost='mean' should still produce valid marginals for large costs."""
        torch.manual_seed(42)
        C = torch.rand(5, 5) * 100

        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=2000,
            threshold=1e-6,
            compile=False,
        )
        result = solver.solve(C, scale_cost="mean")

        P = result.matrix
        a = torch.ones(5) / 5
        assert torch.allclose(P.sum(dim=1), a, atol=1e-3), f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"


@pytest.mark.parametrize("solver_cls", [SinkhornSolver, KLSoftmaxSolver])
def test_warmstart_shape_mismatch_warns_at_caller(solver_cls):
    solver = solver_cls(threshold=1e-3, max_iterations=10)
    cost = torch.rand(5, 8)
    wrong_f = torch.zeros(99)  # wrong shape, should be (5,)
    with pytest.warns(UserWarning, match="warm-start.*shape mismatch") as caught:
        solver.solve(cost, init_f=wrong_f)
    assert {w.filename for w in caught if "shape mismatch" in str(w.message)} == {__file__}


def test_zero_max_iterations_raises():
    """max_iterations=0 would return an infeasible plan, so it is rejected."""
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        SinkhornSolver(max_iterations=0)


@pytest.mark.parametrize(
    "a",
    [
        torch.tensor([1.2, -0.2]),  # negative entry
        torch.tensor([float("nan"), 1.0]),  # non-finite
        torch.zeros(2),  # zero total mass
    ],
)
def test_invalid_marginal_raises(a):
    """A user-supplied malformed marginal fails loudly instead of being clamped."""
    solver = SinkhornSolver(max_iterations=10)
    with pytest.raises(ValueError, match="marginal a"):
        solver.solve(torch.zeros(2, 3), a=a)


class TestSharedIteration:
    """Both modes run ``_sinkhorn_iteration``, and the check reports the plan's marginal error."""

    def test_convergence_loop_runs_shared_iteration(self):
        torch.manual_seed(0)
        solver = SinkhornSolver(epsilon=0.1, threshold=1e-6, check_every=5)
        calls = []
        shared = solver._compiled.sinkhorn_iter

        def counting(*args):
            calls.append(1)
            return shared(*args)

        solver._compiled.sinkhorn_iter = counting
        result = solver.solve(torch.rand(12, 7))
        assert result.converged
        assert len(calls) == result.n_iters

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"omega": 1.0, "max_iterations": 10},
            {"omega": 1.5, "max_iterations": 10},
            {"anderson_depth": 3, "check_every": 1, "max_iterations": 6},
        ],
    )
    def test_reported_error_matches_plan_marginals(self, kwargs):
        torch.manual_seed(0)
        n, m = 12, 7
        C = torch.rand(n, m, dtype=torch.float64)
        solver = SinkhornSolver(epsilon=0.05, threshold=1e-12, **kwargs)
        result = solver.solve(C)
        P = result.matrix
        row = (P.sum(dim=1) - 1.0 / n).abs().max()
        col = (P.sum(dim=0) - 1.0 / m).abs().max()
        assert not result.converged
        assert abs(result.errors[-1] - max(row, col).item()) < 1e-12

    @pytest.mark.filterwarnings("ignore:SinkhornSolver epsilon.*very small:UserWarning")
    def test_nonfinite_duals_are_reset_while_checking(self):
        torch.manual_seed(0)
        with pytest.warns(UserWarning, match="went non-finite"):
            result = SinkhornSolver(epsilon=1e-40, threshold=1e-6).solve(torch.rand(5, 4) * 1e3)
        assert not result.converged
        assert torch.isfinite(result.f).all() and torch.isfinite(result.g).all()

    def test_duals_near_float32_limit_kept(self):
        C = torch.tensor([[1.8255e36, 1.5541e38, 6.1981e37, 2.3055e37], [1.8135e38, 8.4998e37, 2.8427e38, 3.5292e37]])
        result = SinkhornSolver(epsilon=1.7183182677327684e38, threshold=1e-6).solve(C)
        assert result.converged
        assert result.f.abs().max() + result.g.abs().max() == float("inf")

    def test_inductor_matches_eager(self):
        from polystep._compiled import _sinkhorn_iteration

        torch._dynamo.reset()
        torch.manual_seed(0)
        C = torch.rand(64, 9)
        eager = SinkhornSolver(epsilon=0.05).solve(C)
        solver = SinkhornSolver(epsilon=0.05)
        solver._compiled.sinkhorn_iter = torch.compile(_sinkhorn_iteration, fullgraph=True)
        compiled = solver.solve(C)
        torch.testing.assert_close(compiled.matrix, eager.matrix, rtol=0, atol=1e-6)
        assert compiled.converged == eager.converged
        assert abs(compiled.n_iters - eager.n_iters) <= solver.check_every


class TestOverrelaxation:
    def test_omega_default(self):
        """SinkhornSolver() with default omega=1.0 produces same result as standard Sinkhorn."""
        torch.manual_seed(42)
        C = torch.rand(10, 10)

        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=500,
            threshold=1e-6,
            compile=False,
        )
        result = solver.solve(C)
        assert result.converged, (
            f"Did not converge after {result.n_iters} iters, errors: {result.errors[-3:] if result.errors else 'none'}"
        )
        assert solver.omega == 1.0

        P = result.matrix
        a = torch.ones(10) / 10
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4)

    @pytest.mark.parametrize(
        "seed, n, cost_scale, omega",
        [
            (42, 30, 10.0, 1.5),
            (123, 25, 8.0, 1.2),
        ],
    )
    def test_omega_overrelaxed(self, seed, n, cost_scale, omega):
        """omega in (1, 1.5] converges in fewer iterations than omega=1.0."""
        torch.manual_seed(seed)
        C = torch.rand(n, n) * cost_scale  # Wider cost range makes standard Sinkhorn work harder

        solver_standard = SinkhornSolver(
            epsilon=0.5,
            max_iterations=5000,
            threshold=1e-6,
            check_every=1,
            compile=False,
            omega=1.0,
        )
        result_standard = solver_standard.solve(C)

        solver_overrelaxed = SinkhornSolver(
            epsilon=0.5,
            max_iterations=5000,
            threshold=1e-6,
            check_every=1,
            compile=False,
            omega=omega,
        )
        result_overrelaxed = solver_overrelaxed.solve(C)

        assert result_standard.converged, f"Standard did not converge in {result_standard.n_iters} iters"
        assert result_overrelaxed.converged, f"Overrelaxed did not converge in {result_overrelaxed.n_iters} iters"
        assert result_overrelaxed.n_iters < result_standard.n_iters, (
            f"Overrelaxed ({result_overrelaxed.n_iters} iters) should be faster than "
            f"standard ({result_standard.n_iters} iters)"
        )

    def test_adaptive_omega_converges_faster_than_static(self):
        """adaptive_omega derives an overrelaxation from the residual ratio and
        converges in fewer iterations than static omega=1.0."""
        torch.manual_seed(42)
        n = 30
        C = torch.rand(n, n) * 10.0

        static = SinkhornSolver(
            epsilon=0.5,
            max_iterations=5000,
            threshold=1e-6,
            check_every=1,
            compile=False,
            omega=1.0,
        )
        r_static = static.solve(C)

        adaptive = SinkhornSolver(
            epsilon=0.5,
            max_iterations=5000,
            threshold=1e-6,
            check_every=1,
            compile=False,
            omega=1.0,
            adaptive_omega=True,
        )
        r_adaptive = adaptive.solve(C)

        assert r_static.converged
        assert r_adaptive.converged
        assert r_adaptive.n_iters < r_static.n_iters, (
            f"adaptive ({r_adaptive.n_iters}) should beat static ({r_static.n_iters})"
        )

    def test_omega_invalid(self):
        """ValueError for omega=0.3 and omega=2.5."""
        with pytest.raises(ValueError, match="omega must be in"):
            SinkhornSolver(omega=0.3, compile=False)
        with pytest.raises(ValueError, match="omega must be in"):
            SinkhornSolver(omega=2.5, compile=False)


class TestAndersonAcceleration:
    """Anderson acceleration: same fixed point, finite on near-singular costs."""

    @pytest.mark.parametrize(
        "override, shape, max_iterations",
        [
            ({"omega": 1.5}, (8, 8), 500),
            ({"anderson_depth": 3}, (10, 8), 2000),
            ({"adaptive_omega": True}, (10, 8), 2000),
        ],
    )
    def test_acceleration_knobs_reach_same_fixed_point(self, override, shape, max_iterations):
        """Acceleration may change the path to the optimum, never the optimum."""
        torch.manual_seed(42)
        n, m = shape
        C = torch.rand(n, m)

        base = dict(epsilon=0.1, max_iterations=max_iterations, threshold=1e-6, compile=False)
        result_standard = SinkhornSolver(**base).solve(C)
        result_configured = SinkhornSolver(**base, **override).solve(C)

        torch.testing.assert_close(result_configured.matrix, result_standard.matrix, rtol=1e-4, atol=1e-6)
        a = torch.full((n,), 1.0 / n)
        torch.testing.assert_close(result_configured.matrix.sum(dim=1), a, rtol=1e-4, atol=1e-6)
        assert result_configured.converged

    @pytest.mark.parametrize("noise", [1e-4, 1e-6])
    def test_anderson_near_singular_no_nan(self, noise):
        """Anderson on near-singular costs must stay finite (the lstsq guard bounds alpha)."""
        torch.manual_seed(42)
        n = 12
        # Near-singular cost matrix: rank-1 outer product with tiny perturbation
        v = torch.rand(n)
        C = v.unsqueeze(1) @ v.unsqueeze(0) + torch.rand(n, n) * noise

        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=200,
            threshold=1e-6,
            compile=False,
            anderson_depth=5,
            check_every=1,
        )
        result = solver.solve(C)

        # The hardened guard must prevent NaN/Inf in dual potentials
        assert torch.isfinite(result.f).all(), (
            f"noise={noise}: f has non-finite values: "
            f"NaN={torch.isnan(result.f).sum()}, Inf={torch.isinf(result.f).sum()}"
        )
        assert torch.isfinite(result.g).all(), (
            f"noise={noise}: g has non-finite values: "
            f"NaN={torch.isnan(result.g).sum()}, Inf={torch.isinf(result.g).sum()}"
        )

        # Transport plan must be finite (no NaN propagation)
        P = result.matrix
        assert torch.isfinite(P).all(), f"noise={noise}: Transport matrix contains NaN or Inf"

    def test_anderson_acceleration_reduces_iterations(self):
        """The update subtracts dX+dR; subtracting dX alone is worse than no acceleration."""
        torch.manual_seed(0)
        cost = torch.rand(20, 30) * 5.0
        kwargs = dict(epsilon=0.05, max_iterations=400, threshold=1e-7, check_every=1)
        plain = SinkhornSolver(anderson_depth=0, **kwargs).solve(cost)
        accelerated = SinkhornSolver(anderson_depth=3, **kwargs).solve(cost)

        assert plain.converged and accelerated.converged
        assert accelerated.n_iters < plain.n_iters


def test_ill_conditioned_stays_finite():
    """On a stiff cost the divergence back-off latches, so adaptive omega
    cannot re-raise itself into a runaway; the plan stays valid."""
    torch.manual_seed(0)
    n = 20
    C = torch.full((n, n), 10.0)
    C[:, 0] = 0.0  # one vastly cheaper column stresses overrelaxation

    solver = SinkhornSolver(
        epsilon=0.01,
        max_iterations=500,
        threshold=1e-8,
        check_every=10,
        compile=False,
        adaptive_omega=True,
    )
    result = solver.solve(C)

    assert torch.isfinite(result.f).all() and torch.isfinite(result.g).all()
    a = torch.ones(n) / n
    assert torch.allclose(result.matrix.sum(dim=1), a, atol=1e-3)


class TestFixedModeLogic:
    """``fixed_mode`` is determined by ``threshold <= 0`` alone.

    ``check_every`` sets how often convergence is tested, not whether it is tested, so
    raising it above ``max_iterations`` must not divert the solver into the
    fixed-iteration compiled path, which has no Anderson acceleration and no
    adaptive omega.
    """

    def test_large_check_every_uses_eager_path(self):
        """When check_every > max_iterations but threshold > 0, solver uses eager path.

        The eager (convergence-checking) path produces valid transport plans and
        runs the overrelaxation / Anderson / adaptive-omega logic. Even though the
        actual convergence check won't fire (since check_every > max_iterations),
        the solver still runs the eager iteration with NaN divergence guards.
        """
        torch.manual_seed(42)
        n = 10
        C = torch.rand(n, n)

        # check_every=9999 >> max_iterations=500, but threshold > 0
        # This enters the eager path (not the fixed/compiled path)
        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=500,
            threshold=1e-6,
            check_every=9999,
            compile=False,
        )
        result = solver.solve(C)

        # Verify valid transport plan (eager path ran correctly)
        P = result.matrix
        a = torch.ones(n) / n
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4), f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        assert torch.allclose(P.sum(dim=0), a, atol=1e-4), f"Col marginal error: {(P.sum(dim=0) - a).abs().max():.6f}"
        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"

    @pytest.mark.parametrize(
        "override, substring",
        [
            ({"anderson_depth": 5}, "anderson_depth"),
            ({"adaptive_omega": True}, "adaptive_omega"),
        ],
    )
    def test_anderson_not_silently_disabled_by_check_every(self, override, substring):
        """``check_every > max_iterations`` must leave the acceleration enabled.

        The eager path still runs it, so warning that it has no effect would be wrong.
        """
        import warnings

        torch.manual_seed(42)
        n = 15
        C = torch.rand(n, n)

        # This should NOT produce the fixed-mode warning for the acceleration
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SinkhornSolver(
                epsilon=0.5,
                max_iterations=500,
                threshold=1e-6,
                check_every=99999,
                compile=False,
                **override,
            )
            result = solver.solve(C)

        disabled_warnings = [x for x in w if substring in str(x.message) and "fixed-iteration" in str(x.message)]
        assert len(disabled_warnings) == 0, (
            f"{substring} should NOT be disabled when threshold > 0, but got warning: {disabled_warnings[0].message}"
        )
        # Result should be finite (acceleration ran in eager path)
        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"

    def test_threshold_zero_is_fixed_mode(self):
        """threshold <= 0 correctly activates fixed_mode regardless of check_every."""
        torch.manual_seed(42)
        C = torch.rand(5, 5)

        solver = SinkhornSolver(
            epsilon=0.1,
            max_iterations=100,
            threshold=0,
            check_every=1,
            compile=False,
        )
        result = solver.solve(C)

        # Fixed mode runs every iteration (no early stop) and reports success
        # on a finite result so ProgressiveEpsilon does not inflate epsilon.
        assert result.n_iters == 100
        assert result.converged

    def test_threshold_zero_warns_about_anderson(self):
        """threshold <= 0 with anderson_depth > 0 emits a warning."""
        import warnings

        torch.manual_seed(42)
        C = torch.rand(5, 5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SinkhornSolver(
                epsilon=0.1,
                max_iterations=100,
                threshold=0,
                compile=False,
                anderson_depth=5,
            )
            solver.solve(C)

        anderson_warnings = [x for x in w if "anderson_depth" in str(x.message) and "fixed-iteration" in str(x.message)]
        assert len(anderson_warnings) == 1, f"Expected 1 Anderson fixed-mode warning, got {len(anderson_warnings)}"


@pytest.mark.parametrize("scale_cost", [None, "mean", "max_cost", 2.0])
def test_sinkhorn_ent_reg_cost_in_caller_frame(scale_cost):
    C = torch.tensor([[0.0, 3.0, 1.0, 2.0], [2.0, 0.0, 4.0, 1.0], [1.0, 2.0, 0.0, 3.0]])
    solver = SinkhornSolver(epsilon=0.5, max_iterations=5000, threshold=1e-13)
    result = solver.solve(cost_matrix=C.clone(), scale_cost=scale_cost)

    # Solving the scaled problem at eps is the raw problem at eps * divisor.
    divisor = float(resolve_cost_scale(recenter_cost(C.clone())[0], scale_cost))
    P = result.matrix
    entropy = -(P * (P.clamp(min=1e-30).log() - 1)).sum()
    expected = (C * P).sum() - 0.5 * divisor * entropy

    assert result.ent_reg_cost == pytest.approx(expected.item(), rel=1e-5)


class TestSinkhornMarginalBalance:
    @pytest.mark.parametrize("b", [torch.tensor([1.0]), None], ids=["both", "a_only"])
    def test_warns_on_unequal_total_mass(self, b):
        """Balanced OT with sum(a) != sum(b) is infeasible; warn instead of
        silently reporting converged."""
        C = torch.zeros(2, 1)
        with pytest.warns(UserWarning, match="unequal total mass"):
            SinkhornSolver(epsilon=0.5, max_iterations=50).solve(C, a=torch.tensor([1.0, 1.0]), b=b)

    def test_no_warning_on_uniform_marginals(self):
        """The hot path (a=b=None uniform) must never trip the balance warning."""
        import warnings

        C = torch.rand(4, 6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = SinkhornSolver(epsilon=0.5, max_iterations=50).solve(C)

        assert caught == [], [str(w.message) for w in caught]
        torch.testing.assert_close(result.matrix.sum(dim=1), torch.full((4,), 0.25))
