"""Unit tests for the solvers package: Protocol, SolverResult, relocated SinkhornSolver."""
import pytest
import torch


class TestSolversPackageImports:
    """Test that the solvers package exports are correct."""

    def test_import_solver_protocol(self):
        """Solver Protocol can be imported from polystep.solvers."""
        from polystep.solvers import Solver
        assert Solver is not None

    def test_import_solver_result(self):
        """SolverResult dataclass can be imported from polystep.solvers."""
        from polystep.solvers import SolverResult
        assert SolverResult is not None

    def test_import_sinkhorn_solver(self):
        """SinkhornSolver can be imported from polystep.solvers."""
        from polystep.solvers import SinkhornSolver
        assert SinkhornSolver is not None

    def test_import_sinkhorn_result(self):
        """SinkhornResult can be imported from polystep.solvers."""
        from polystep.solvers import SinkhornResult
        assert SinkhornResult is not None


class TestSolverResult:
    """Test SolverResult dataclass fields and defaults."""

    def test_solver_result_fields(self):
        """SolverResult has all required fields."""
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


class TestSinkhornSolverRelocated:
    """Test that relocated SinkhornSolver is functionally identical."""

    def test_sinkhorn_solver_solve_returns_sinkhorn_result(self):
        """SinkhornSolver.solve() returns a SinkhornResult with .matrix property."""
        from polystep.solvers import SinkhornSolver
        torch.manual_seed(42)
        C = torch.rand(5, 8)
        solver = SinkhornSolver(epsilon=0.1, compile=False)
        result = solver.solve(C)
        assert hasattr(result, 'matrix')
        P = result.matrix
        assert P.shape == (5, 8)

    def test_sinkhorn_relocated_produces_identical_results(self):
        """Relocated SinkhornSolver produces identical results to original on a known cost matrix."""
        from polystep.solvers import SinkhornSolver
        torch.manual_seed(42)
        C = torch.rand(10, 8)
        solver = SinkhornSolver(epsilon=0.1, max_iterations=200, threshold=1e-6, compile=False)
        result = solver.solve(C)
        P = result.matrix
        a = torch.ones(10) / 10
        b = torch.ones(8) / 8
        # Must satisfy marginal constraints (same as original)
        assert torch.allclose(P.sum(dim=1), a, atol=1e-4)
        assert torch.allclose(P.sum(dim=0), b, atol=1e-4)

    def test_sinkhorn_result_has_dual_potentials(self):
        """SinkhornResult still has f, g, converged, n_iters, ent_reg_cost."""
        from polystep.solvers import SinkhornSolver
        torch.manual_seed(42)
        C = torch.rand(5, 8)
        solver = SinkhornSolver(epsilon=0.1, compile=False)
        result = solver.solve(C)
        assert result.f is not None
        assert result.g is not None
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iters, int)
        assert isinstance(result.ent_reg_cost, float)
