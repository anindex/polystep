"""Unit tests for SolverResult dataclass schema."""

import torch


class TestSolverResult:
    """SolverResult dataclass field defaults and overrides."""

    def test_solver_result_fields(self):
        """SolverResult has all required fields with correct defaults."""
        from polystep.solvers import SolverResult

        matrix = torch.rand(5, 8)
        result = SolverResult(matrix=matrix, cost=1.23)
        assert torch.equal(result.matrix, matrix)
        assert result.cost == 1.23
        assert result.f is None
        assert result.g is None
        assert result.converged is True
        assert result.n_iters == 1
        assert result.ent_reg_cost == 0.0
