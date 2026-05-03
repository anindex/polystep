"""Unit tests for SoftmaxSolver."""
import pytest
import torch


class TestSoftmaxSolverImports:
    """Test that SoftmaxSolver can be imported."""

    def test_import_softmax_solver(self):
        """SoftmaxSolver can be imported from polystep.solvers."""
        from polystep.solvers import SoftmaxSolver
        assert SoftmaxSolver is not None

    def test_import_softmax_result(self):
        """SoftmaxResult can be imported from polystep.solvers."""
        from polystep.solvers import SoftmaxResult
        assert SoftmaxResult is not None


class TestSoftmaxSolverBasic:
    """Test basic SoftmaxSolver.solve() behavior."""

    def test_solve_returns_solver_result_with_correct_shape(self):
        """SoftmaxSolver.solve() returns SolverResult with .matrix shape (P, V)."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 10, 8
        C = torch.rand(P, V)
        a = torch.ones(P) / P
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C, a=a)
        assert result.matrix.shape == (P, V)

    def test_row_sums_equal_source_marginal(self):
        """Transport matrix row sums equal source marginal a (within tolerance 1e-6)."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 10, 8
        C = torch.rand(P, V)
        a = torch.tensor([0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05])
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C, a=a)
        row_sums = result.matrix.sum(dim=1)
        torch.testing.assert_close(row_sums, a, atol=1e-6, rtol=0)

    def test_uniform_marginal_row_sums(self):
        """With uniform a=1/P, transport matrix row sums are 1/P."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 50, 8
        C = torch.rand(P, V)
        a = torch.ones(P) / P
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C, a=a)
        expected_row_sums = torch.ones(P) / P
        torch.testing.assert_close(result.matrix.sum(dim=1), expected_row_sums, atol=1e-6, rtol=0)

    def test_default_marginal_is_uniform(self):
        """Default a (None) produces uniform 1/P marginal."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 10, 8
        C = torch.rand(P, V)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        expected_row_sums = torch.ones(P) / P
        torch.testing.assert_close(result.matrix.sum(dim=1), expected_row_sums, atol=1e-6, rtol=0)


class TestSoftmaxSolverResultFields:
    """Test SoftmaxResult field values."""

    def test_f_is_none(self):
        """SoftmaxResult.f is None."""
        from polystep.solvers import SoftmaxSolver
        C = torch.rand(5, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert result.f is None

    def test_g_is_none(self):
        """SoftmaxResult.g is None."""
        from polystep.solvers import SoftmaxSolver
        C = torch.rand(5, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert result.g is None

    def test_converged_is_true(self):
        """SoftmaxResult.converged is True."""
        from polystep.solvers import SoftmaxSolver
        C = torch.rand(5, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert result.converged is True

    def test_n_iters_is_one(self):
        """SoftmaxResult.n_iters is 1."""
        from polystep.solvers import SoftmaxSolver
        C = torch.rand(5, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert result.n_iters == 1

    def test_ent_reg_cost_is_float(self):
        """SoftmaxResult.ent_reg_cost is a float."""
        from polystep.solvers import SoftmaxSolver
        C = torch.rand(5, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert isinstance(result.ent_reg_cost, float)


class TestSoftmaxSolverScaling:
    """Test cost scaling behavior."""

    def test_scale_cost_mean(self):
        """scale_cost='mean' normalizes cost matrix before softmax."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 10, 8
        C = torch.rand(P, V) * 100  # Large costs
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C, scale_cost='mean')
        # Row sums should still equal 1/P with uniform marginal
        expected_row_sums = torch.ones(P) / P
        torch.testing.assert_close(result.matrix.sum(dim=1), expected_row_sums, atol=1e-6, rtol=0)


class TestSoftmaxSolverNumericalStability:
    """Test numerical stability."""

    @pytest.mark.filterwarnings("ignore:SoftmaxSolver epsilon=.*is very small:UserWarning")
    def test_large_cost_values_no_nan(self):
        """Large cost values (1e6) do not produce NaN (numerical stability)."""
        from polystep.solvers import SoftmaxSolver
        P, V = 5, 8
        C = torch.ones(P, V) * 1e6
        C[0, 0] = 0.0  # One cheap option
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert not torch.isnan(result.matrix).any(), "NaN in transport matrix"
        assert not torch.isinf(result.matrix).any(), "Inf in transport matrix"

    def test_identical_cost_rows_produce_uniform_weights(self):
        """Identical cost rows produce uniform weights per row."""
        from polystep.solvers import SoftmaxSolver
        P, V = 5, 8
        C = torch.ones(P, V) * 3.0  # All costs equal
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        # Each row should have uniform weights: a_i / V for each element
        expected_per_element = 1.0 / (P * V)
        torch.testing.assert_close(
            result.matrix,
            torch.full((P, V), expected_per_element),
            atol=1e-6, rtol=0,
        )

    def test_epsilon_zero_raises_error(self):
        """Epsilon <= 0 raises ValueError."""
        from polystep.solvers import SoftmaxSolver
        solver = SoftmaxSolver(epsilon=0.0)
        C = torch.rand(5, 8)
        with pytest.raises(ValueError, match="epsilon"):
            solver.solve(C)

    def test_epsilon_negative_raises_error(self):
        """Negative epsilon raises ValueError."""
        from polystep.solvers import SoftmaxSolver
        solver = SoftmaxSolver(epsilon=-0.1)
        C = torch.rand(5, 8)
        with pytest.raises(ValueError, match="epsilon"):
            solver.solve(C)


class TestSoftmaxSolverEdgeCases:
    """Test edge cases."""

    def test_init_f_and_init_g_accepted_but_ignored(self):
        """init_f and init_g parameters are accepted but ignored."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        P, V = 5, 8
        C = torch.rand(P, V)
        solver = SoftmaxSolver(epsilon=0.1)
        result_no_init = solver.solve(C)
        result_with_init = solver.solve(
            C, init_f=torch.rand(P), init_g=torch.rand(V)
        )
        torch.testing.assert_close(result_no_init.matrix, result_with_init.matrix)

    def test_single_particle(self):
        """Single particle (P=1) works correctly."""
        from polystep.solvers import SoftmaxSolver
        C = torch.tensor([[1.0, 2.0, 3.0]])
        solver = SoftmaxSolver(epsilon=0.5)
        result = solver.solve(C)
        assert result.matrix.shape == (1, 3)
        # Row sum should be 1/1 = 1.0
        torch.testing.assert_close(result.matrix.sum(dim=1), torch.tensor([1.0]), atol=1e-6, rtol=0)

    def test_typical_size(self):
        """Typical size (P=50, V=8) runs without error."""
        from polystep.solvers import SoftmaxSolver
        torch.manual_seed(42)
        C = torch.rand(50, 8)
        solver = SoftmaxSolver(epsilon=0.1)
        result = solver.solve(C)
        assert result.matrix.shape == (50, 8)
        expected_row_sums = torch.ones(50) / 50
        torch.testing.assert_close(result.matrix.sum(dim=1), expected_row_sums, atol=1e-6, rtol=0)
