"""Numerical stress tests for the Sinkhorn solver and OT pipeline."""

import torch
import pytest

from polystep import SinkhornSolver
from polystep.solvers import KLSoftmaxSolver, SoftmaxSolver


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
        ),
    ]
)
def device(request):
    return request.param


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


@pytest.mark.parametrize(
    "solver_type,kwargs", [(SinkhornSolver, {}), (KLSoftmaxSolver, {}), (KLSoftmaxSolver, {"lam": 0.4})]
)
@pytest.mark.parametrize("mass", [1.0, 1e-60])
@pytest.mark.parametrize("zero_support", [False, True])
def test_tiny_marginals_and_zero_support(solver_type, kwargs, mass, zero_support, device):
    C = torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.4], [0.1, 0.5, 0.3]], dtype=torch.float64, device=device)
    a = C.new_tensor([0.0 if zero_support else 1e-60, 0.3, 0.7]) * mass
    b = C.new_tensor([0.4, 0.0 if zero_support else 1e-60, 0.6]) * mass
    solver = solver_type(epsilon=0.3, threshold=1e-10, **kwargs)
    result = solver.solve(C, a=a, b=b)
    resumed = solver.solve(C, a=a, b=b, init_f=result.f, init_g=result.g)
    for out in (result, resumed):
        assert out.converged
        torch.testing.assert_close(out.matrix.sum(1), a, atol=0, rtol=1e-9)
        if not kwargs:
            torch.testing.assert_close(out.matrix.sum(0), b, atol=0, rtol=1e-9)
        else:
            active = b > 0
            torch.testing.assert_close(out.g[active], -kwargs["lam"] * (out.matrix.sum(0)[active] / b[active]).log())
        assert torch.isfinite(torch.tensor(out.ent_reg_cost, dtype=torch.float64))
    if zero_support:
        reference = solver_type(epsilon=0.3, threshold=1e-10, **kwargs).solve(
            C[a > 0][:, b > 0], a=a[a > 0], b=b[b > 0]
        )
        torch.testing.assert_close(result.matrix[a > 0][:, b > 0], reference.matrix, atol=0, rtol=1e-9)
        assert result.ent_reg_cost == pytest.approx(reference.ent_reg_cost, rel=1e-9, abs=0)


@pytest.mark.parametrize("kwargs", [{"omega": 1.5}, {"anderson_depth": 3}, {"adaptive_omega": True}, {"threshold": 0}])
def test_sinkhorn_zero_support_acceleration(kwargs, device):
    C = torch.rand(4, 3, dtype=torch.float64, device=device)
    a, b = C.new_tensor([0.0, 0.2, 0.3, 0.5]), C.new_tensor([0.4, 0.6, 0.0])
    result = SinkhornSolver(epsilon=0.5, max_iterations=200, **kwargs).solve(C, a=a, b=b)
    assert result.converged
    torch.testing.assert_close(result.matrix.sum(1), a, atol=0, rtol=1e-4)
    torch.testing.assert_close(result.matrix.sum(0), b, atol=0, rtol=1e-4)


@pytest.mark.parametrize("solver_type", [SinkhornSolver, KLSoftmaxSolver, SoftmaxSolver])
@pytest.mark.parametrize("scale", ["mean", "max_cost"])
def test_scaled_extreme_costs_match_double(solver_type, scale, device):
    C = torch.tensor([[-3e38, 3e38, 1e38], [-3e38, 0.0, 2e38]], device=device)
    solver = solver_type(epsilon=0.5)
    result = solver.solve(C, scale_cost=scale)
    expected = solver.solve(C.double(), scale_cost=scale)
    assert result.matrix.dtype == C.dtype
    torch.testing.assert_close(result.matrix.double(), expected.matrix, atol=1e-5, rtol=1e-4)
    assert result.ent_reg_cost == pytest.approx(expected.ent_reg_cost, rel=1e-4)
