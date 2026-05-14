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

    def test_solver_result_custom_fields(self):
        """SolverResult accepts all field overrides."""
        from polystep.solvers import SolverResult
        f = torch.rand(5)
        g = torch.rand(8)
        matrix = torch.rand(5, 8)
        result = SolverResult(
            matrix=matrix, cost=2.5, f=f, g=g,
            converged=False, n_iters=42, ent_reg_cost=3.14,
        )
        assert torch.equal(result.f, f)
        assert torch.equal(result.g, g)
        assert result.converged is False
        assert result.n_iters == 42
        assert result.ent_reg_cost == 3.14
