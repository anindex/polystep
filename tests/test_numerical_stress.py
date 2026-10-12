"""Numerical stress tests for the Sinkhorn solver and OT pipeline."""

import torch
import pytest

from polystep import SinkhornSolver


class TestSinkhornEdgeCases:
    def test_smaller_epsilon_concentrates_plan(self):
        """Lower eps means a sharper plan: each row's mass piles onto fewer columns.

        Rows that must split across two columns cap the eps -> 0 row share at 0.5.
        """
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m)

        def mean_row_share(eps):
            T = SinkhornSolver(epsilon=eps, max_iterations=500, threshold=1e-8).solve(C).matrix
            assert torch.isfinite(T).all(), f"Transport has NaN/Inf at eps={eps}"
            return (T.max(dim=1).values / T.sum(dim=1)).mean().item()

        sharp, soft = mean_row_share(0.001), mean_row_share(0.05)
        assert sharp > soft + 0.1, f"eps=0.001 share {sharp:.3f} not sharper than eps=0.05 share {soft:.3f}"

    def test_very_large_epsilon_is_near_uniform(self):
        """As eps -> inf the plan approaches the independent coupling ``a b^T``.

        The bound is relative: an absolute tolerance could not fail at 1/(n*m) scale.
        """
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=100.0, max_iterations=200)
        T = solver.solve(C).matrix

        assert torch.isfinite(T).all(), "Transport has NaN/Inf at eps=100"
        expected = 1.0 / (n * m)
        assert (T - expected).abs().max() < expected * 0.05

    @pytest.mark.parametrize("const", [0.0, 42.0, 1_000_000.0])
    def test_zero_cost_matrix(self, const):
        """Constant cost (including all-zero) should produce uniform transport."""
        n, m = 8, 4
        C = torch.ones(n, m) * const
        solver = SinkhornSolver(epsilon=1.0, max_iterations=100)
        result = solver.solve(C)

        T = result.matrix
        assert torch.isfinite(T).all()
        assert torch.allclose(T, torch.full((n, m), 1.0 / (n * m)), atol=1e-6)
        row_sums = T.sum(dim=1)
        col_sums = T.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=1e-4)
        assert torch.allclose(col_sums, torch.ones(m) / m, atol=1e-4)

    def test_warm_start_after_scale_change(self):
        """Warm-started duals from a 1x-cost step applied to a 100x-cost step."""
        torch.manual_seed(0)
        n, m = 10, 6
        C_small = torch.rand(n, m)
        C_large = torch.rand(n, m) * 100.0

        solver = SinkhornSolver(epsilon=1.0, max_iterations=200)

        result1 = solver.solve(C_small)

        result2 = solver.solve(C_large, init_f=result1.f, init_g=result1.g)
        T2 = result2.matrix
        assert torch.isfinite(T2).all(), "Warm start with scale change produced NaN/Inf"
        row_sums = T2.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n) / n, atol=1e-4)


def test_non_uniform_marginals():
    """Non-uniform source marginals should be respected."""
    torch.manual_seed(0)
    n, m = 10, 6
    a = torch.softmax(torch.randn(n), dim=0)
    C = torch.rand(n, m)
    solver = SinkhornSolver(epsilon=1.0, max_iterations=200)
    result = solver.solve(C, a=a)
    T = result.matrix
    assert torch.allclose(T.sum(dim=1), a, atol=1e-3)
