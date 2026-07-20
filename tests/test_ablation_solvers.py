"""Tests for ablation solver classes: MinCostGreedy, TopKMean, TemperedSoftmax.

Validates transport matrix properties, shape consistency, and integration
with PolyStepOptimizer's solver selection.
"""

import pytest
import torch
import torch.nn as nn

from polystep.solvers.base import SolverResult
from polystep.solvers.greedy import MinCostGreedySolver, TopKMeanSolver
from polystep.solvers.tempered_softmax import TemperedSoftmaxSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cost_matrix():
    """Small (4, 6) cost matrix with known structure."""
    torch.manual_seed(42)
    return torch.rand(4, 6)


@pytest.fixture
def source_marginal():
    """Uniform source marginal for P=4."""
    return torch.ones(4) / 4


# ---------------------------------------------------------------------------
# MinCostGreedySolver
# ---------------------------------------------------------------------------


class TestMinCostGreedySolver:
    def test_basic_assignment(self, cost_matrix, source_marginal):
        solver = MinCostGreedySolver()
        result = solver.solve(cost_matrix, a=source_marginal)

        assert isinstance(result, SolverResult)
        assert result.matrix.shape == (4, 6)
        assert result.converged is True
        assert result.n_iters == 1
        assert result.f is None
        assert result.g is None

    def test_argmin_correctness(self, cost_matrix, source_marginal):
        """Each row should have mass only at the argmin column."""
        solver = MinCostGreedySolver()
        result = solver.solve(cost_matrix, a=source_marginal)
        T = result.matrix

        for i in range(4):
            min_col = cost_matrix[i].argmin().item()
            # Only the argmin column should be non-zero
            assert T[i, min_col].item() == pytest.approx(source_marginal[i].item(), abs=1e-7)
            # All other columns should be zero
            mask = torch.ones(6, dtype=torch.bool)
            mask[min_col] = False
            assert T[i, mask].sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_deterministic(self, cost_matrix, source_marginal):
        solver = MinCostGreedySolver()
        r1 = solver.solve(cost_matrix, a=source_marginal)
        r2 = solver.solve(cost_matrix, a=source_marginal)
        assert torch.equal(r1.matrix, r2.matrix)

    def test_default_uniform_marginal(self, cost_matrix):
        solver = MinCostGreedySolver()
        result = solver.solve(cost_matrix)
        row_sums = result.matrix.sum(dim=1)
        expected = torch.ones(4) / 4
        assert torch.allclose(row_sums, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# TopKMeanSolver
# ---------------------------------------------------------------------------


class TestTopKMeanSolver:
    def test_basic_assignment(self, cost_matrix, source_marginal):
        solver = TopKMeanSolver(k=3)
        result = solver.solve(cost_matrix, a=source_marginal)

        assert isinstance(result, SolverResult)
        assert result.matrix.shape == (4, 6)
        assert result.converged is True
        assert result.n_iters == 1

    def test_top_k_correctness(self, cost_matrix, source_marginal):
        """Each row should have exactly k non-zero entries at the top-k positions."""
        solver = TopKMeanSolver(k=3)
        result = solver.solve(cost_matrix, a=source_marginal)
        T = result.matrix

        for i in range(4):
            _, topk_idx = cost_matrix[i].topk(3, largest=False)
            nonzero_mask = T[i] > 0
            assert nonzero_mask.sum().item() == 3
            for idx in topk_idx:
                assert T[i, idx].item() > 0

    def test_uniform_weights_within_top_k(self, cost_matrix, source_marginal):
        """Each selected vertex should get a[i]/k mass."""
        solver = TopKMeanSolver(k=3)
        result = solver.solve(cost_matrix, a=source_marginal)
        T = result.matrix

        for i in range(4):
            nonzero = T[i][T[i] > 0]
            expected = source_marginal[i] / 3
            assert torch.allclose(nonzero, expected.expand_as(nonzero), atol=1e-7)

    def test_v_less_than_k(self):
        """When V < k, should use all V vertices gracefully."""
        C = torch.rand(3, 2)  # Only 2 vertices, k=3
        a = torch.ones(3) / 3
        solver = TopKMeanSolver(k=3)
        result = solver.solve(C, a=a)

        # Should use all 2 vertices
        T = result.matrix
        for i in range(3):
            assert (T[i] > 0).sum().item() == 2
        assert torch.allclose(T.sum(dim=1), a, atol=1e-7)

    def test_k_equals_1_matches_greedy(self, cost_matrix, source_marginal):
        """TopKMean with k=1 should match greedy exactly."""
        greedy = MinCostGreedySolver()
        topk1 = TopKMeanSolver(k=1)
        r_greedy = greedy.solve(cost_matrix, a=source_marginal)
        r_topk1 = topk1.solve(cost_matrix, a=source_marginal)
        assert torch.allclose(r_greedy.matrix, r_topk1.matrix, atol=1e-7)


# ---------------------------------------------------------------------------
# TemperedSoftmaxSolver
# ---------------------------------------------------------------------------


class TestTemperedSoftmaxSolver:
    def test_basic(self, cost_matrix, source_marginal):
        solver = TemperedSoftmaxSolver(tau=1.0)
        result = solver.solve(cost_matrix, a=source_marginal)

        assert isinstance(result, SolverResult)
        assert result.matrix.shape == (4, 6)
        assert result.converged is True

    def test_uses_tau_not_epsilon(self, cost_matrix, source_marginal):
        """Output should depend on tau, not epsilon."""
        solver1 = TemperedSoftmaxSolver(epsilon=0.1, tau=1.0)
        solver2 = TemperedSoftmaxSolver(epsilon=99.0, tau=1.0)
        r1 = solver1.solve(cost_matrix, a=source_marginal)
        r2 = solver2.solve(cost_matrix, a=source_marginal)
        # Same tau -> same result regardless of epsilon
        assert torch.allclose(r1.matrix, r2.matrix, atol=1e-7)

    def test_different_tau_different_result(self, cost_matrix, source_marginal):
        solver1 = TemperedSoftmaxSolver(tau=0.1)
        solver2 = TemperedSoftmaxSolver(tau=10.0)
        r1 = solver1.solve(cost_matrix, a=source_marginal)
        r2 = solver2.solve(cost_matrix, a=source_marginal)
        assert not torch.allclose(r1.matrix, r2.matrix, atol=1e-3)

    def test_tau_validation(self):
        solver = TemperedSoftmaxSolver(tau=-1.0)
        C = torch.rand(3, 4)
        with pytest.raises(ValueError, match="tau must be > 0"):
            solver.solve(C)

    def test_low_tau_approaches_greedy(self):
        """Very low tau should approximate greedy (argmin) on unscaled costs."""
        # Use a well-separated cost matrix so there are no near-ties
        C = torch.tensor(
            [
                [10.0, 1.0, 5.0, 8.0],
                [3.0, 7.0, 2.0, 9.0],
                [6.0, 4.0, 8.0, 1.0],
            ]
        )
        a = torch.ones(3) / 3
        solver = TemperedSoftmaxSolver(tau=0.001)
        result = solver.solve(C, a=a)
        T = result.matrix

        greedy = MinCostGreedySolver()
        r_greedy = greedy.solve(C, a=a)

        # tau=1e-3 with well-separated costs (gaps >= 1) makes exp(-gap/tau)
        # underflow to ~0, so the tempered softmax collapses to the greedy
        # assignment up to float32 rounding.
        assert torch.allclose(T, r_greedy.matrix, atol=1e-5)


# ---------------------------------------------------------------------------
# Shape Consistency: all solvers return (P, V) matrix
# ---------------------------------------------------------------------------


class TestShapeConsistency:
    @pytest.mark.parametrize("P,V", [(1, 4), (10, 2), (50, 16), (100, 8)])
    def test_various_sizes(self, P, V):
        C = torch.rand(P, V)
        a = torch.ones(P) / P
        for solver in [MinCostGreedySolver(), TopKMeanSolver(k=3), TemperedSoftmaxSolver(tau=1.0)]:
            result = solver.solve(C, a=a)
            assert result.matrix.shape == (P, V)
            assert torch.allclose(result.matrix.sum(dim=1), a, atol=1e-6)


# ---------------------------------------------------------------------------
# PolyStepOptimizer solver selection integration
# ---------------------------------------------------------------------------


class TestSolverSelection:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    @pytest.mark.parametrize(
        "solver_name",
        [
            "softmax",
            "sinkhorn",
            "min_cost_greedy",
            "top_k_mean",
            "tempered_softmax",
        ],
    )
    def test_optimizer_accepts_solver(self, solver_name, simple_model):
        """PolyStepOptimizer should accept all solver strings without error."""
        from polystep.optimizer import PolyStepOptimizer
        from polystep.hybrid_subspace import HybridSubspace
        from polystep.transform import ParamLayout

        layout = ParamLayout.from_module(simple_model)
        subspace = HybridSubspace.from_layout(layout, rank=2)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            solver=solver_name,
            epsilon=1.0,
        )
        # Verify correct solver type
        from polystep.solvers import (
            SoftmaxSolver,
            SinkhornSolver,
            MinCostGreedySolver,
            TopKMeanSolver,
            TemperedSoftmaxSolver,
        )

        expected = {
            "softmax": SoftmaxSolver,
            "sinkhorn": SinkhornSolver,
            "min_cost_greedy": MinCostGreedySolver,
            "top_k_mean": TopKMeanSolver,
            "tempered_softmax": TemperedSoftmaxSolver,
        }
        assert isinstance(opt.solver, expected[solver_name])

    def test_invalid_solver_raises(self, simple_model):
        from polystep.optimizer import PolyStepOptimizer

        with pytest.raises(ValueError, match="Unknown solver"):
            PolyStepOptimizer(simple_model, solver="nonexistent")
