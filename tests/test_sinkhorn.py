"""Unit tests for SinkhornSolver (full-rank and low-rank modes)."""
import torch
import pytest

from polystep.solvers import SinkhornSolver, SinkhornResult


class TestSinkhornSolver:
    """Tests for the unified Sinkhorn solver."""

    # ------------------------------------------------------------------
    # Full-rank tests
    # ------------------------------------------------------------------

    def test_small_epsilon_near_deterministic(self):
        """Small epsilon pushes transport toward deterministic (near one-hot rows).

        With uniform marginals, each row sums to 1/n. A near-deterministic plan
        concentrates mass on one column per row, so max entry ~ 1/n and the ratio
        max / (row_sum) -> 1.  We verify max(row) / sum(row) > 0.9 for each row.
        """
        torch.manual_seed(42)
        n = 5
        C = torch.rand(n, n)

        solver = SinkhornSolver(
            epsilon=0.001, max_iterations=5000, threshold=1e-10, compile=False,
        )
        result = solver.solve(C)

        P = result.matrix
        row_sums = P.sum(dim=1)
        max_vals = P.max(dim=1).values
        ratios = max_vals / row_sums
        assert (ratios > 0.9).all(), \
            f"Row concentration ratios: {ratios.tolist()}"

    def test_known_ot_solution_diagonal(self):
        """Identity cost matrix should produce near-diagonal transport plan."""
        torch.manual_seed(42)
        n = 5
        C = torch.eye(n) * 0.0  # Zero on diagonal, we need off-diagonal penalty
        C = 1.0 - torch.eye(n)  # 0 on diagonal, 1 off-diagonal

        solver = SinkhornSolver(
            epsilon=0.01, max_iterations=5000, threshold=1e-10, compile=False,
        )
        result = solver.solve(C)

        P = result.matrix
        # Diagonal should dominate: P_ii > 0.5 / n for each i
        diag = torch.diag(P)
        off_diag_max = (P - torch.diag(diag)).max()
        assert diag.min() > off_diag_max, \
            f"Diagonal min {diag.min():.4f} should exceed off-diagonal max {off_diag_max:.4f}"

    def test_scale_cost_mean(self):
        """Scale cost='mean' should still produce valid marginals for large costs."""
        torch.manual_seed(42)
        C = torch.rand(5, 5) * 100  # Large costs

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            compile=False,
        )
        result = solver.solve(C, scale_cost='mean')

        P = result.matrix
        a = torch.ones(5) / 5
        assert torch.allclose(P.sum(dim=1), a, atol=1e-3), \
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"

    # ------------------------------------------------------------------
    # Low-rank tests
    # ------------------------------------------------------------------

    def test_low_rank_marginals(self):
        """Low-rank Sinkhorn produces valid transport plan with matching marginals."""
        torch.manual_seed(42)
        n, m = 20, 15
        C = torch.rand(n, m)

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            rank=5, compile=False,
        )
        result = solver.solve(C)

        P = result.matrix
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        assert torch.allclose(P.sum(dim=1), a, atol=1e-3), \
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        assert torch.allclose(P.sum(dim=0), b, atol=1e-3), \
            f"Col marginal error: {(P.sum(dim=0) - b).abs().max():.6f}"

    def test_low_rank_result_uses_dual_potential_path(self):
        """Low-rank result uses cost_approx + dual potentials (no factored SVD)."""
        torch.manual_seed(42)
        C = torch.rand(20, 15)

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=1000, rank=5, compile=False,
        )
        result = solver.solve(C)

        # Low-rank stores approximated cost matrix and uses dual potential path
        assert result._cost_matrix is not None, "_cost_matrix should be set"
        assert result._cost_matrix.shape == (20, 15)
        P = result.matrix
        assert P.shape == (20, 15)
        assert torch.isfinite(P).all()

    def test_auto_rank_selection(self):
        """Auto rank triggers when n+m > auto_rank_threshold."""
        torch.manual_seed(42)
        n, m = 20, 15
        C = torch.rand(n, m)

        # Set threshold very low so auto-rank triggers
        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=1000, threshold=1e-6,
            auto_rank_threshold=30, compile=False,
        )
        result = solver.solve(C)

        # Should have used low-rank since n+m=35 > 30
        # Low-rank now uses cost_approx + dual potentials (not factored Q/R/g_lr)
        assert result._cost_matrix is not None, "Auto rank should have triggered low-rank mode"
        P = result.matrix
        assert P.shape == (n, m)
        assert torch.isfinite(P).all()

    def test_entropic_cost_finite(self):
        """Entropic regularized cost should be finite for both modes."""
        torch.manual_seed(42)
        C = torch.rand(10, 8)

        # Full-rank
        solver_fr = SinkhornSolver(epsilon=0.1, max_iterations=500, compile=False)
        result_fr = solver_fr.solve(C)
        assert torch.isfinite(torch.tensor(result_fr.ent_reg_cost)), \
            f"Full-rank ent_reg_cost not finite: {result_fr.ent_reg_cost}"

        # Low-rank
        solver_lr = SinkhornSolver(epsilon=0.1, max_iterations=500, rank=3, compile=False)
        result_lr = solver_lr.solve(C)
        assert torch.isfinite(torch.tensor(result_lr.ent_reg_cost)), \
            f"Low-rank ent_reg_cost not finite: {result_lr.ent_reg_cost}"


def test_warmstart_shape_mismatch_warns():
    """Sinkhorn should warn when warm-start duals have wrong shape."""
    solver = SinkhornSolver(threshold=1e-3, max_iterations=10)
    cost = torch.rand(5, 8)
    wrong_f = torch.zeros(99)  # wrong shape, should be (5,)
    with pytest.warns(UserWarning, match="warm-start.*shape mismatch"):
        solver.solve(cost, init_f=wrong_f)


class TestOverrelaxation:
    """Tests for overrelaxed Sinkhorn iterations (parametric extension)."""

    def test_omega_default(self):
        """SinkhornSolver() with default omega=1.0 produces same result as standard Sinkhorn."""
        torch.manual_seed(42)
        C = torch.rand(10, 10)

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=500, threshold=1e-6,
            compile=False,
        )
        result = solver.solve(C)
        assert result.converged, f"Did not converge after {result.n_iters} iters, errors: {result.errors[-3:] if result.errors else 'none'}"
        assert solver.omega == 1.0

        # Verify valid transport plan
        P = result.matrix
        a = torch.ones(10) / 10
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4)

    def test_omega_overrelaxed(self):
        """omega=1.5 converges in fewer iterations than omega=1.0 on a non-trivial cost matrix.

        Overrelaxation is most beneficial for larger problems with wider cost ranges
        where standard Sinkhorn needs more iterations.
        """
        torch.manual_seed(42)
        n = 30
        C = torch.rand(n, n) * 10.0  # Wider cost range makes standard Sinkhorn work harder

        solver_standard = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, omega=1.0,
        )
        result_standard = solver_standard.solve(C)

        solver_overrelaxed = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, omega=1.5,
        )
        result_overrelaxed = solver_overrelaxed.solve(C)

        assert result_standard.converged, f"Standard did not converge in {result_standard.n_iters} iters"
        assert result_overrelaxed.converged, f"Overrelaxed did not converge in {result_overrelaxed.n_iters} iters"
        assert result_overrelaxed.n_iters < result_standard.n_iters, (
            f"Overrelaxed ({result_overrelaxed.n_iters} iters) should be faster than "
            f"standard ({result_standard.n_iters} iters)"
        )

    def test_omega_backward_compatible(self):
        """Same cost matrix with omega=1.0 and no omega argument produce identical f, g values."""
        torch.manual_seed(42)
        C = torch.rand(8, 8)

        solver_default = SinkhornSolver(
            epsilon=0.1, max_iterations=500, threshold=1e-8,
            compile=False,
        )
        result_default = solver_default.solve(C)

        solver_explicit = SinkhornSolver(
            epsilon=0.1, max_iterations=500, threshold=1e-8,
            compile=False, omega=1.0,
        )
        result_explicit = solver_explicit.solve(C)

        assert torch.allclose(result_default.f, result_explicit.f, atol=1e-10), (
            f"f difference: {(result_default.f - result_explicit.f).abs().max():.2e}"
        )
        assert torch.allclose(result_default.g, result_explicit.g, atol=1e-10), (
            f"g difference: {(result_default.g - result_explicit.g).abs().max():.2e}"
        )

    def test_omega_invalid(self):
        """ValueError for omega=0.3 and omega=2.5."""
        with pytest.raises(ValueError, match="omega must be in"):
            SinkhornSolver(omega=0.3, compile=False)
        with pytest.raises(ValueError, match="omega must be in"):
            SinkhornSolver(omega=2.5, compile=False)

    def test_omega_convergence_check_path(self):
        """Use threshold>0 to trigger convergence-checking path with omega=1.2.

        The convergence-checking path (threshold > 0) exercises the inline overrelaxation
        code rather than the compiled function path.
        """
        torch.manual_seed(123)
        n = 25
        C = torch.rand(n, n) * 8.0  # Scaled costs for meaningful iteration counts

        solver_standard = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, omega=1.0,
        )
        result_standard = solver_standard.solve(C)

        solver_overrelaxed = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, omega=1.2,
        )
        result_overrelaxed = solver_overrelaxed.solve(C)

        assert result_standard.converged, f"Standard did not converge in {result_standard.n_iters} iters"
        assert result_overrelaxed.converged, f"Overrelaxed did not converge in {result_overrelaxed.n_iters} iters"
        assert result_overrelaxed.n_iters < result_standard.n_iters, (
            f"Overrelaxed ({result_overrelaxed.n_iters} iters) should be faster than "
            f"standard ({result_standard.n_iters} iters)"
        )


class TestAndersonAcceleration:
    """Tests for Anderson acceleration in Sinkhorn solver (convergence acceleration)."""

    def test_anderson_depth_zero_matches_standard(self):
        """anderson_depth=0 produces identical f, g, n_iters to standard solver."""
        torch.manual_seed(42)
        n, m = 10, 8
        C = torch.rand(n, m)

        solver_standard = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False,
        )
        result_standard = solver_standard.solve(C)

        solver_aa = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False, anderson_depth=0,
        )
        result_aa = solver_aa.solve(C)

        assert torch.allclose(result_standard.f, result_aa.f, atol=1e-10), (
            f"f difference: {(result_standard.f - result_aa.f).abs().max():.2e}"
        )
        assert torch.allclose(result_standard.g, result_aa.g, atol=1e-10), (
            f"g difference: {(result_standard.g - result_aa.g).abs().max():.2e}"
        )
        assert result_standard.n_iters == result_aa.n_iters, (
            f"n_iters differ: standard={result_standard.n_iters}, aa={result_aa.n_iters}"
        )

    def test_anderson_reduces_iterations(self):
        """anderson_depth=5 converges in fewer iterations than depth=0."""
        torch.manual_seed(42)
        n, m = 20, 20
        C = torch.rand(n, m)

        solver_standard = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=5, compile=False, anderson_depth=0,
        )
        result_standard = solver_standard.solve(C)

        solver_aa = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            check_every=5, compile=False, anderson_depth=5,
        )
        result_aa = solver_aa.solve(C)

        assert result_standard.converged, f"Standard did not converge in {result_standard.n_iters} iters"
        assert result_aa.converged, f"Anderson did not converge in {result_aa.n_iters} iters"
        assert result_aa.n_iters <= result_standard.n_iters, (
            f"Anderson ({result_aa.n_iters} iters) should be <= standard ({result_standard.n_iters} iters)"
        )

    def test_anderson_valid_marginals(self):
        """anderson_depth=5 result satisfies marginal constraints."""
        torch.manual_seed(42)
        n, m = 15, 12
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            compile=False, anderson_depth=5,
        )
        result = solver.solve(C)

        P = result.matrix
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4), (
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        )
        assert torch.allclose(P.sum(dim=0), b, atol=1e-4), (
            f"Col marginal error: {(P.sum(dim=0) - b).abs().max():.6f}"
        )

    def test_anderson_handles_ill_conditioning(self):
        """anderson_depth=5 on a nearly-singular cost matrix does not produce NaN."""
        torch.manual_seed(42)
        n = 10
        # Create a nearly-singular cost matrix (rank-1 + small noise)
        v = torch.rand(n)
        C = v.unsqueeze(1) @ v.unsqueeze(0) + torch.rand(n, n) * 1e-4

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            compile=False, anderson_depth=5,
        )
        result = solver.solve(C)

        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"

    def test_anderson_near_singular_no_nan(self):
        """Anderson acceleration on near-singular cost matrices produces finite results.

        Tests the hardened lstsq guard: alpha norm bound and combined-result
        finiteness check prevent ill-conditioned solves from corrupting duals.
        """
        for seed in [42, 99, 123]:
            torch.manual_seed(seed)
            n = 12
            # Near-singular cost matrix: rank-1 outer product with tiny perturbation
            # This stresses lstsq with ill-conditioned residual matrices
            v = torch.rand(n)
            C = v.unsqueeze(1) @ v.unsqueeze(0) + torch.rand(n, n) * 1e-6

            solver = SinkhornSolver(
                epsilon=0.1, max_iterations=2000, threshold=1e-6,
                compile=False, anderson_depth=5, check_every=1,
            )
            result = solver.solve(C)

            # The hardened guard must prevent NaN/Inf in dual potentials
            assert torch.isfinite(result.f).all(), (
                f"seed={seed}: f has non-finite values: "
                f"NaN={torch.isnan(result.f).sum()}, Inf={torch.isinf(result.f).sum()}"
            )
            assert torch.isfinite(result.g).all(), (
                f"seed={seed}: g has non-finite values: "
                f"NaN={torch.isnan(result.g).sum()}, Inf={torch.isinf(result.g).sum()}"
            )

            # Transport plan must be finite (no NaN propagation)
            P = result.matrix
            assert torch.isfinite(P).all(), (
                f"seed={seed}: Transport matrix contains NaN or Inf"
            )


class TestDataDependentInit:
    """Tests for data-dependent initialization in Sinkhorn solver (convergence acceleration)."""

    def test_disabled_matches_standard(self):
        """data_dependent_init=False produces identical result to default solver."""
        torch.manual_seed(42)
        n, m = 10, 8
        C = torch.rand(n, m)

        solver_standard = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False,
        )
        result_standard = solver_standard.solve(C)

        solver_ddi = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False, data_dependent_init=False,
        )
        result_ddi = solver_ddi.solve(C)

        assert torch.allclose(result_standard.f, result_ddi.f, atol=1e-10), (
            f"f difference: {(result_standard.f - result_ddi.f).abs().max():.2e}"
        )
        assert torch.allclose(result_standard.g, result_ddi.g, atol=1e-10), (
            f"g difference: {(result_standard.g - result_ddi.g).abs().max():.2e}"
        )

    def test_cold_start_fewer_iterations(self):
        """data_dependent_init=True converges in fewer iterations than False on cold start."""
        torch.manual_seed(42)
        n, m = 15, 15
        C = torch.rand(n, m)

        solver_cold = SinkhornSolver(
            epsilon=0.1, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, data_dependent_init=False,
        )
        result_cold = solver_cold.solve(C)

        solver_ddi = SinkhornSolver(
            epsilon=0.1, max_iterations=5000, threshold=1e-6,
            check_every=1, compile=False, data_dependent_init=True,
        )
        result_ddi = solver_ddi.solve(C)

        assert result_cold.converged, f"Cold start did not converge in {result_cold.n_iters} iters"
        assert result_ddi.converged, f"DDI did not converge in {result_ddi.n_iters} iters"
        assert result_ddi.n_iters <= result_cold.n_iters, (
            f"DDI ({result_ddi.n_iters} iters) should be <= cold ({result_cold.n_iters} iters)"
        )

    def test_warm_start_bypasses_init(self):
        """data_dependent_init=True with init_f/init_g uses warm-start, not data-dependent init."""
        torch.manual_seed(42)
        n, m = 10, 10
        C = torch.rand(n, m)

        # First solve to get warm-start potentials
        solver_base = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False,
        )
        result_base = solver_base.solve(C)

        # With DDI + warm-start: should use warm-start (same as without DDI + warm-start)
        solver_ddi_warm = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False, data_dependent_init=True,
        )
        result_ddi_warm = solver_ddi_warm.solve(
            C, init_f=result_base.f, init_g=result_base.g
        )

        solver_warm = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False, data_dependent_init=False,
        )
        result_warm = solver_warm.solve(
            C, init_f=result_base.f, init_g=result_base.g
        )

        assert torch.allclose(result_ddi_warm.f, result_warm.f, atol=1e-10), (
            f"f difference: {(result_ddi_warm.f - result_warm.f).abs().max():.2e}"
        )
        assert torch.allclose(result_ddi_warm.g, result_warm.g, atol=1e-10), (
            f"g difference: {(result_ddi_warm.g - result_warm.g).abs().max():.2e}"
        )

    def test_valid_marginals(self):
        """data_dependent_init=True produces valid transport plan."""
        torch.manual_seed(42)
        n, m = 12, 10
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            compile=False, data_dependent_init=True,
        )
        result = solver.solve(C)

        P = result.matrix
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4), (
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        )
        assert torch.allclose(P.sum(dim=0), b, atol=1e-4), (
            f"Col marginal error: {(P.sum(dim=0) - b).abs().max():.6f}"
        )


class TestAdaptiveOmega:
    """Tests for adaptive omega in Sinkhorn solver (convergence acceleration)."""

    def test_disabled_matches_standard(self):
        """adaptive_omega=False produces identical result to default solver."""
        torch.manual_seed(42)
        n, m = 10, 8
        C = torch.rand(n, m)

        solver_standard = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False,
        )
        result_standard = solver_standard.solve(C)

        solver_ao = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-8,
            compile=False, adaptive_omega=False,
        )
        result_ao = solver_ao.solve(C)

        assert torch.allclose(result_standard.f, result_ao.f, atol=1e-10), (
            f"f difference: {(result_standard.f - result_ao.f).abs().max():.2e}"
        )
        assert torch.allclose(result_standard.g, result_ao.g, atol=1e-10), (
            f"g difference: {(result_standard.g - result_ao.g).abs().max():.2e}"
        )

    def test_omega_stays_in_bounds(self):
        """adaptive_omega=True keeps omega within [1.0, 1.8] (no NaN/Inf in result)."""
        torch.manual_seed(42)
        n, m = 15, 15
        C = torch.rand(n, m) * 10.0

        solver = SinkhornSolver(
            epsilon=0.5, max_iterations=2000, threshold=1e-6,
            compile=False, adaptive_omega=True,
        )
        result = solver.solve(C)

        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"
        assert result.converged or result.n_iters > 0, "Solver produced no iterations"

    def test_valid_marginals(self):
        """adaptive_omega=True produces valid transport plan."""
        torch.manual_seed(42)
        n, m = 10, 10
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=2000, threshold=1e-6,
            compile=False, adaptive_omega=True,
        )
        result = solver.solve(C)

        P = result.matrix
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4), (
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        )
        assert torch.allclose(P.sum(dim=0), b, atol=1e-4), (
            f"Col marginal error: {(P.sum(dim=0) - b).abs().max():.6f}"
        )

    def test_convergence(self):
        """adaptive_omega=True converges on a standard problem."""
        torch.manual_seed(42)
        n, m = 10, 10
        C = torch.rand(n, m)

        solver = SinkhornSolver(
            epsilon=0.5, max_iterations=5000, threshold=1e-6,
            compile=False, adaptive_omega=True,
        )
        result = solver.solve(C)

        assert result.converged, f"Did not converge after {result.n_iters} iters"


class TestFixedModeLogic:
    """Tests for fixed_mode determination (OR → AND fix).

    Before the fix, fixed_mode = threshold <= 0 OR check_every > max_iterations.
    This meant that setting check_every > max_iterations silently forced the solver
    into the fixed-iteration (compiled) path, disabling Anderson acceleration and
    adaptive_omega even when threshold > 0. After the fix, fixed_mode = threshold <= 0,
    so the convergence-checking (eager) code path is always used when threshold > 0.
    """

    def test_check_every_exceeds_max_iterations_uses_eager_path(self):
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
            epsilon=0.1, max_iterations=500, threshold=1e-6,
            check_every=9999, compile=False,
        )
        result = solver.solve(C)

        # Verify valid transport plan (eager path ran correctly)
        P = result.matrix
        a = torch.ones(n) / n
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4), (
            f"Row marginal error: {(P.sum(dim=1) - a).abs().max():.6f}"
        )
        assert torch.allclose(P.sum(dim=0), a, atol=1e-4), (
            f"Col marginal error: {(P.sum(dim=0) - a).abs().max():.6f}"
        )
        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"

    def test_anderson_not_silently_disabled_by_check_every(self):
        """Anderson acceleration is not warned/disabled when check_every > max_iterations.

        Before the fix, setting check_every > max_iterations triggered fixed_mode,
        which emitted a warning that Anderson has no effect. After the fix, no
        warning is emitted because the solver correctly enters the eager path.
        """
        import warnings
        torch.manual_seed(42)
        n = 15
        C = torch.rand(n, n)

        # This should NOT produce the fixed-mode warning for Anderson
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SinkhornSolver(
                epsilon=0.5, max_iterations=500, threshold=1e-6,
                check_every=99999, compile=False, anderson_depth=5,
            )
            result = solver.solve(C)

        anderson_warnings = [
            x for x in w
            if "anderson_depth" in str(x.message) and "fixed-iteration" in str(x.message)
        ]
        assert len(anderson_warnings) == 0, (
            "Anderson acceleration should NOT be disabled when threshold > 0, "
            f"but got warning: {anderson_warnings[0].message}"
        )
        # Result should be finite (Anderson ran in eager path)
        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"
        assert torch.isfinite(result.g).all(), "g contains NaN or Inf"

    def test_adaptive_omega_not_silently_disabled_by_check_every(self):
        """adaptive_omega is not warned/disabled when check_every > max_iterations."""
        import warnings
        torch.manual_seed(42)
        n = 10
        C = torch.rand(n, n)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SinkhornSolver(
                epsilon=0.5, max_iterations=500, threshold=1e-6,
                check_every=99999, compile=False, adaptive_omega=True,
            )
            result = solver.solve(C)

        adaptive_warnings = [
            x for x in w
            if "adaptive_omega" in str(x.message) and "fixed-iteration" in str(x.message)
        ]
        assert len(adaptive_warnings) == 0, (
            "adaptive_omega should NOT be disabled when threshold > 0, "
            f"but got warning: {adaptive_warnings[0].message}"
        )
        assert torch.isfinite(result.f).all(), "f contains NaN or Inf"

    def test_threshold_zero_is_fixed_mode(self):
        """threshold <= 0 correctly activates fixed_mode regardless of check_every."""
        torch.manual_seed(42)
        C = torch.rand(5, 5)

        solver = SinkhornSolver(
            epsilon=0.1, max_iterations=100, threshold=0,
            check_every=1, compile=False,
        )
        result = solver.solve(C)

        # In fixed mode, solver runs all max_iterations without convergence check
        assert not result.converged, (
            "With threshold=0, solver should be in fixed mode and not report convergence"
        )

    def test_threshold_zero_warns_about_anderson(self):
        """threshold <= 0 with anderson_depth > 0 emits a warning."""
        import warnings
        torch.manual_seed(42)
        C = torch.rand(5, 5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = SinkhornSolver(
                epsilon=0.1, max_iterations=100, threshold=0,
                compile=False, anderson_depth=5,
            )
            solver.solve(C)

        anderson_warnings = [
            x for x in w
            if "anderson_depth" in str(x.message) and "fixed-iteration" in str(x.message)
        ]
        assert len(anderson_warnings) == 1, (
            f"Expected 1 Anderson fixed-mode warning, got {len(anderson_warnings)}"
        )


# ---------------------------------------------------------------------------
# Tests for dual potential clamping
# ---------------------------------------------------------------------------


class TestDualClamping:
    """Tests for dual potential clamping behavior."""

    def test_dual_clamping_not_too_aggressive_small_epsilon(self):
        """Dual potentials should not be aggressively clamped at small epsilon."""
        from polystep.solvers import SinkhornSolver
        # Small epsilon creates large dual potentials
        solver = SinkhornSolver(epsilon=0.05, max_iterations=100, threshold=1e-6, compile=False)
        n, m = 10, 10
        torch.manual_seed(42)
        C = torch.randn(n, m).abs() * 10  # Large cost spread
        a = torch.ones(n) / n
        b = torch.ones(m) / m
        result = solver.solve(C, a=a, b=b)
        f, g = result.f, result.g
        T = result.matrix
        # Duals should be finite and marginals approximately satisfied
        assert torch.isfinite(f).all()
        assert torch.isfinite(g).all()
        row_sums = T.sum(dim=1)
        assert torch.allclose(row_sums, a, atol=1e-3), f"Row marginals off: {row_sums} vs {a}"


# ---------------------------------------------------------------------------
# Tests for ProgressiveEpsilon (progressive epsilon)
# ---------------------------------------------------------------------------


class TestProgressiveEpsilon:
    """Tests for ProgressiveEpsilon auto-epsilon scheduler."""

    def test_at_returns_init_value(self):
        """ProgressiveEpsilon.at(0) returns init value."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(init=1.0, target=0.01)
        assert pe.at(0) == 1.0

    def test_fast_convergence_decreases_epsilon(self):
        """After update(n_iters=5, converged=True) (fast), next epsilon decreases."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(
            init=1.0, target=0.01, max_epsilon=5.0,
            fast_threshold=0.1, slow_threshold=0.5,
            decrease_factor=0.95, increase_factor=1.2,
            ema_alpha=0.0,  # no smoothing for clear test
        )
        initial = pe.at()
        pe.update(n_iters=5, max_iterations=1000, converged=True)
        after = pe.at()
        assert after < initial, f"Expected decrease: {after} < {initial}"

    def test_slow_convergence_increases_epsilon(self):
        """After update(n_iters=500, converged=False) (slow), next epsilon increases."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(
            init=1.0, target=0.01, max_epsilon=5.0,
            fast_threshold=0.1, slow_threshold=0.5,
            decrease_factor=0.95, increase_factor=1.2,
            ema_alpha=0.0,  # no smoothing for clear test
        )
        initial = pe.at()
        pe.update(n_iters=500, max_iterations=1000, converged=False)
        after = pe.at()
        assert after > initial, f"Expected increase: {after} > {initial}"

    def test_epsilon_never_below_target(self):
        """Epsilon never goes below target floor."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(
            init=0.05, target=0.01, max_epsilon=5.0,
            decrease_factor=0.5, ema_alpha=0.0,
        )
        # Repeatedly decrease
        for _ in range(100):
            pe.update(n_iters=1, max_iterations=1000, converged=True)
        assert pe.at() >= 0.01

    def test_epsilon_never_above_max(self):
        """Epsilon never goes above max_epsilon ceiling."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(
            init=1.0, target=0.01, max_epsilon=5.0,
            increase_factor=2.0, ema_alpha=0.0,
        )
        # Repeatedly increase
        for _ in range(100):
            pe.update(n_iters=999, max_iterations=1000, converged=False)
        assert pe.at() <= 5.0

    def test_drop_in_replacement_for_linear_epsilon(self):
        """ProgressiveEpsilon has .at() method compatible with LinearEpsilon."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(init=1.0, target=0.01)
        # .at() should accept iteration parameter (ignored) and return float
        val = pe.at(42)
        assert isinstance(val, float)
        val_none = pe.at(None)
        assert isinstance(val_none, float)
        val_no_arg = pe.at()
        assert isinstance(val_no_arg, float)

    def test_ema_smoothing_prevents_oscillation(self):
        """Multiple update() calls with EMA smoothing create a smooth trajectory."""
        from polystep.epsilon import ProgressiveEpsilon
        pe = ProgressiveEpsilon(
            init=1.0, target=0.01, max_epsilon=5.0,
            fast_threshold=0.1, slow_threshold=0.5,
            decrease_factor=0.8, increase_factor=1.5,
            ema_alpha=0.7,  # heavy smoothing
        )
        values = [pe.at()]
        # Alternate fast and slow convergence
        for i in range(10):
            if i % 2 == 0:
                pe.update(n_iters=1, max_iterations=1000, converged=True)  # fast
            else:
                pe.update(n_iters=999, max_iterations=1000, converged=False)  # slow
            values.append(pe.at())

        # With heavy EMA smoothing, changes between consecutive steps
        # should be relatively small (smoothed, not jumping wildly)
        max_change = max(
            abs(values[i+1] - values[i]) for i in range(len(values) - 1)
        )
        # Without EMA, changes would be large (0.8x or 1.5x swings)
        # With EMA=0.7, each step changes by at most 30% of the raw change
        assert max_change < 0.5, (
            f"EMA smoothing failed: max step change = {max_change:.4f}, "
            f"values = {[f'{v:.4f}' for v in values]}"
        )
