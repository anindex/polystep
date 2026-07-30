"""Numerical stress tests for the Sinkhorn solver and OT pipeline.

Edge conditions that could expose hidden numerical issues: extreme epsilon at both
limits, constant and negative costs, warm start across a 100x cost-scale change,
non-uniform marginals, and ParamLayout round-trip under padding.

Non-finite cost sanitisation lives in test_input_validation.py (unit) and
test_regressions.py (the ordering contract); marginal satisfaction over
many random problems lives in test_sinkhorn_numerics.py.
"""

import torch
import torch.nn as nn
import pytest

from polystep import ParamLayout, SinkhornSolver


class TestSinkhornEdgeCases:
    def test_smaller_epsilon_concentrates_the_plan(self):
        """Lower eps means a sharper plan: each row's mass piles onto fewer columns.

        The bound is relative, not absolute. With 10 rows of mass 1/10 feeding 6 columns
        of capacity 1/6, some rows must split across two columns no matter how small eps
        gets, so the eps -> 0 limit here has a min row share of 0.5, not 1.
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

        The bound has to be relative: with uniform marginals every entry is already in
        ``[0, 1/n]``, so an absolute tolerance of 0.1 against ``1/60`` cannot fail.
        """
        torch.manual_seed(0)
        n, m = 10, 6
        C = torch.rand(n, m)
        solver = SinkhornSolver(epsilon=100.0, max_iterations=200)
        T = solver.solve(C).matrix

        assert torch.isfinite(T).all(), "Transport has NaN/Inf at eps=100"
        expected = 1.0 / (n * m)
        assert (T - expected).abs().max() < expected * 0.05

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


class TestParamLayoutStress:
    # nn.Linear(13, 7) is 98 params: pdim 1 needs no padding, 3 and 8 do.
    @pytest.mark.parametrize("particle_dim", [1, 3, 8])
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
