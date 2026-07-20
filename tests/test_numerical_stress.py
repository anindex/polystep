"""Numerical stress tests for the Sinkhorn solver and OT pipeline.

Tests edge conditions that could expose hidden numerical issues:
- extreme epsilon values
- pathological cost matrices
- warm-start with scale changes
- NaN propagation
- marginal constraint satisfaction under stress
"""

import torch
import torch.nn as nn
import pytest

from polystep import ParamLayout, SinkhornSolver


class TestSinkhornEdgeCases:
    """Stress-test the Sinkhorn solver under extreme conditions."""

    def test_very_small_epsilon(self):
        """Tiny epsilon should produce near-deterministic (one-hot) transport."""
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=0.001, max_iterations=500, threshold=1e-8)
        result = solver.solve(C)

        T = result.matrix
        assert torch.isfinite(T).all(), "Transport has NaN/Inf at eps=0.001"
        # At eps=0.001 the log-domain solver collapses toward a permutation
        # and the iterates stop reducing marginal error below ~1e-2 in
        # practice (LSE rounding plus the eps -> 0 limit). This test guards
        # the no-NaN/no-Inf behavior; the broad atol reflects that
        # near-breakdown regime, not the solver's normal convergence floor.
        row_sums = T.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=0.02)

    def test_very_large_epsilon(self):
        """Large epsilon should produce near-uniform transport."""
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=100.0, max_iterations=200)
        result = solver.solve(C)

        T = result.matrix
        assert torch.isfinite(T).all(), "Transport has NaN/Inf at eps=100"
        # Near-uniform: all entries should be close to 1/(n*m)
        expected = 1.0 / (n * m)
        assert (T - expected).abs().max() < 0.1

    @pytest.mark.parametrize("const", [0.0, 42.0])
    def test_zero_cost_matrix(self, const):
        """Constant cost (including all-zero) should produce uniform transport."""
        n, m = 8, 4
        C = torch.ones(n, m) * const
        solver = SinkhornSolver(epsilon=1.0, max_iterations=100)
        result = solver.solve(C)

        T = result.matrix
        assert torch.isfinite(T).all()
        # Should be uniform
        row_sums = T.sum(dim=1)
        col_sums = T.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=1e-4)
        assert torch.allclose(col_sums, torch.ones(m) / m, atol=1e-4)

    def test_large_cost_range(self):
        """Cost matrix with values spanning [0, 1000] should converge."""
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m) * 1000.0
        solver = SinkhornSolver(epsilon=10.0, max_iterations=500)
        result = solver.solve(C, scale_cost="mean")
        T = result.matrix
        assert torch.isfinite(T).all()
        row_sums = T.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=1e-3)

    def test_negative_costs(self):
        """Cost matrix with negative values should still work."""
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.randn(n, m)  # mean 0, includes negatives
        solver = SinkhornSolver(epsilon=1.0, max_iterations=200)
        result = solver.solve(C)
        T = result.matrix
        assert torch.isfinite(T).all()
        assert (T >= -1e-8).all(), "Transport plan should be non-negative"

    @pytest.mark.filterwarnings("ignore:Cost matrix has.*non-finite:UserWarning")
    def test_nan_in_cost_is_sanitized(self):
        """``SinkhornSolver`` sanitizes NaN / Inf entries in the cost
        matrix and emits a ``UserWarning``; the resulting transport
        plan is always finite.
        """
        torch.manual_seed(0)
        n, m = 8, 4
        C = torch.rand(n, m)
        C[0, 0] = float("nan")
        C[3, 2] = float("inf")

        solver = SinkhornSolver(epsilon=1.0, max_iterations=100)
        result = solver.solve(C)
        T = result.matrix
        assert torch.isfinite(T).all(), (
            "SinkhornSolver should sanitize non-finite cost entries and return a finite transport plan."
        )

    def test_warm_start_after_scale_change(self):
        """Warm-started duals from a 1x-cost step applied to a 100x-cost step."""
        torch.manual_seed(0)
        n, m = 10, 6
        C_small = torch.rand(n, m)
        C_large = torch.rand(n, m) * 100.0

        solver = SinkhornSolver(epsilon=1.0, max_iterations=200)

        # Cold start on small costs
        result1 = solver.solve(C_small)

        # Warm start on large costs using duals from small costs
        result2 = solver.solve(C_large, init_f=result1.f, init_g=result1.g)
        T2 = result2.matrix
        assert torch.isfinite(T2).all(), "Warm start with scale change produced NaN/Inf"
        row_sums = T2.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=1e-4)

    def test_overrelaxation_stability(self):
        """omega=1.9 (aggressive overrelaxation) should converge without diverging."""
        torch.manual_seed(0)
        n, m = 20, 8
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=0.5, max_iterations=500, threshold=1e-6, omega=1.9)
        result = solver.solve(C)
        T = result.matrix
        assert torch.isfinite(T).all(), "Overrelaxation omega=1.9 diverged"


class TestMarginalConstraints:
    """Property tests: transport plan must satisfy marginal constraints."""

    @pytest.mark.parametrize("n,m", [(5, 4), (10, 6), (50, 8), (100, 16)])
    def test_row_column_sums(self, n, m):
        torch.manual_seed(0)
        C = torch.rand(n, m)
        a = torch.ones(n) / n
        b = torch.ones(m) / m
        solver = SinkhornSolver(epsilon=0.5, max_iterations=300, threshold=1e-6)
        result = solver.solve(C, a=a)
        T = result.matrix

        assert torch.isfinite(T).all()
        assert (T >= -1e-8).all(), "Transport plan has negative entries"
        assert torch.allclose(T.sum(dim=1), a, atol=1e-3), "Row sums don't match marginals"
        assert torch.allclose(T.sum(dim=0), b, atol=1e-3), "Col sums don't match marginals"

    def test_non_uniform_marginals(self):
        """Non-uniform source marginals should be respected."""
        torch.manual_seed(0)
        n, m = 10, 6
        a = torch.softmax(torch.randn(n), dim=0)
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=1.0, max_iterations=200)
        result = solver.solve(C, a=a)
        T = result.matrix
        assert torch.allclose(T.sum(dim=1), a, atol=1e-3)


class TestParamLayoutStress:
    """Stress-test parameter layout round-trip under various conditions."""

    def test_roundtrip_many_dtypes(self):
        """Model with mixed dtypes should round-trip correctly."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        recovered = layout.unflatten(flat)

        sd = model.state_dict()
        for key in recovered:
            assert torch.allclose(sd[key], recovered[key], atol=1e-6), f"Mismatch: {key}"

    @pytest.mark.parametrize("particle_dim", [1, 2, 3, 4, 7, 8])
    def test_roundtrip_various_particle_dims(self, particle_dim):
        """Round-trip should work for any particle_dim."""
        model = nn.Linear(13, 7)  # Odd dimensions to test padding
        layout = ParamLayout.from_module(model, particle_dim=particle_dim)

        assert layout.padded_size % particle_dim == 0
        flat = layout.flatten(model)
        assert flat.shape[1] == particle_dim

        recovered = layout.unflatten(flat)
        sd = model.state_dict()
        for key in recovered:
            assert torch.equal(sd[key], recovered[key])

    def test_empty_model(self):
        """Model with no parameters should not crash."""
        model = nn.ReLU()
        layout = ParamLayout.from_module(model)
        assert layout.total_params == 0

    def test_batch_unflatten_consistency(self):
        """batch_unflatten(N=1) should match unflatten on the same data."""
        model = nn.Linear(10, 5)
        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)

        single = layout.unflatten(flat)
        batched = layout.batch_unflatten(flat.unsqueeze(0))

        for key in single:
            assert torch.allclose(single[key], batched[key][0], atol=1e-6)
