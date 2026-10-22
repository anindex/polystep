"""Tests for the ablation solvers MinCostGreedy and TopKMean."""

import pytest
import torch
import torch.nn as nn

from polystep.solvers.greedy import MinCostGreedySolver, TopKMeanSolver
from polystep.solvers.kl_softmax import KLSoftmaxSolver
from polystep.solvers.sinkhorn import SinkhornSolver
from polystep.solvers.softmax import SoftmaxSolver


@pytest.fixture
def cost_matrix():
    """Small (4, 6) cost matrix with known structure."""
    torch.manual_seed(42)
    return torch.rand(4, 6)


@pytest.fixture
def source_marginal():
    """Uniform source marginal for P=4."""
    return torch.ones(4) / 4


class TestMinCostGreedySolver:
    def test_argmin_correctness(self, cost_matrix, source_marginal):
        """Each row should have mass only at the argmin column."""
        solver = MinCostGreedySolver()
        result = solver.solve(cost_matrix, a=source_marginal)
        T = result.matrix

        for i in range(4):
            min_col = cost_matrix[i].argmin().item()
            assert T[i, min_col].item() == pytest.approx(source_marginal[i].item(), abs=1e-7)
            mask = torch.ones(6, dtype=torch.bool)
            mask[min_col] = False
            assert T[i, mask].sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_greedy_scaled_cost_in_caller_frame(self):
        C = torch.tensor([[0.0, 4.0], [6.0, 2.0]])
        plain = MinCostGreedySolver().solve(cost_matrix=C.clone())
        scaled = MinCostGreedySolver().solve(cost_matrix=C.clone(), scale_cost="max_cost")

        assert torch.equal(plain.matrix, scaled.matrix)  # argmin is scale-invariant
        assert scaled.ent_reg_cost == pytest.approx(plain.ent_reg_cost)


class TestTopKMeanSolver:
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
                assert T[i, idx].item() == pytest.approx(source_marginal[i].item() / 3)

    def test_v_less_than_k(self):
        """When V < k, should use all V vertices gracefully."""
        C = torch.rand(3, 2)
        a = torch.ones(3) / 3
        solver = TopKMeanSolver(k=3)
        result = solver.solve(C, a=a)

        T = result.matrix
        for i in range(3):
            assert (T[i] > 0).sum().item() == 2
        assert torch.allclose(T.sum(dim=1), a, atol=1e-7)

    def test_masked_vertices_get_no_mass(self):
        """Masked (inf) vertices must get no mass even with fewer than k finite entries."""
        inf = float("inf")
        C = torch.tensor([[1.0, inf, inf, inf], [1.0, 2.0, inf, 3.0]])
        a = torch.tensor([0.5, 0.5])
        T = TopKMeanSolver(k=3).solve(C, a=a).matrix

        assert T[0].tolist() == [0.5, 0.0, 0.0, 0.0]
        assert T[1, 2] == 0.0
        torch.testing.assert_close(T.sum(dim=1), a)

    def test_k_equals_1_matches_greedy(self, cost_matrix, source_marginal):
        greedy = MinCostGreedySolver()
        topk1 = TopKMeanSolver(k=1)
        r_greedy = greedy.solve(cost_matrix, a=source_marginal)
        r_topk1 = topk1.solve(cost_matrix, a=source_marginal)
        assert torch.allclose(r_greedy.matrix, r_topk1.matrix, atol=1e-7)


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
            "kl_softmax",
        ],
    )
    def test_optimizer_accepts_solver(self, solver_name, simple_model):
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
        expected = {
            "softmax": SoftmaxSolver,
            "sinkhorn": SinkhornSolver,
            "min_cost_greedy": MinCostGreedySolver,
            "top_k_mean": TopKMeanSolver,
            "kl_softmax": KLSoftmaxSolver,
        }
        assert isinstance(opt.solver, expected[solver_name])

    def test_invalid_solver_raises(self, simple_model):
        from polystep.optimizer import PolyStepOptimizer

        with pytest.raises(ValueError, match="Unknown solver"):
            PolyStepOptimizer(simple_model, solver="nonexistent")
        with pytest.raises(ValueError, match="solver='softmax' with ent_epsilon"):
            PolyStepOptimizer(simple_model, solver="tempered_softmax")


@pytest.mark.parametrize(
    "solver",
    [
        MinCostGreedySolver(),
        TopKMeanSolver(k=2),
        SoftmaxSolver(epsilon=0.1),
        SinkhornSolver(epsilon=0.1, threshold=1e-6),
    ],
    ids=["min_cost", "top_k", "softmax", "sinkhorn"],
)
def test_half_cost_gives_fp32_plan(solver):
    """sanitize_cost promotes the cost to fp32 first."""
    C = torch.randn(5, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    result = solver.solve(C)
    assert result.matrix.dtype == torch.float32
    assert torch.allclose(result.matrix.sum(dim=1), torch.full((5,), 0.2), atol=1e-5)


@pytest.mark.parametrize(
    "solver",
    [MinCostGreedySolver(), TopKMeanSolver(k=1), SoftmaxSolver(), KLSoftmaxSolver(), SinkhornSolver()],
    ids=["min_cost", "top_k", "softmax", "kl_softmax", "sinkhorn"],
)
def test_integer_cost_matches_float_plan(solver):
    C = torch.tensor([[1, 2], [3, 0]])
    assert torch.equal(solver.solve(C).matrix, solver.solve(C.float()).matrix)


@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
@pytest.mark.parametrize("num_probe", [1, 3])
def test_optimizer_topk_ignores_infeasible_vertices(block_strategy, num_probe):
    """Only the candidate with the largest first weight scores; every other one is NaN.

    TopK must not receive the sanitize penalty as a finite cost, or it averages
    infeasible vertices into the move. A vertex with any infeasible probe is
    infeasible, so with three probes none is left and the particle stays put.
    """
    from polystep.optimizer import PolyStepOptimizer

    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)
    opt = PolyStepOptimizer(
        model,
        solver="top_k_mean",
        polytope_type="orthoplex",
        num_probe=num_probe,
        use_momentum=False,
        block_strategy=block_strategy,
        seed=0,
    )
    start = model.weight.detach().reshape(-1).clone()
    scored = []

    def closure(params):
        w = params["weight"].reshape(-1, 2)
        best = w[:, 0].argmax()
        scored.append(w[best].clone())
        return torch.full((w.shape[0],), float("nan")).index_fill(0, best, 0.0)

    opt.step(closure)

    assert len(scored) == 1
    expected = scored[0] if num_probe == 1 else start
    torch.testing.assert_close(model.weight.detach().reshape(-1), expected)


def test_topk_excludes_nonfinite_freezes_uninformative_rows():
    costs = torch.tensor([[1.0, float("-inf"), float("nan"), float("inf")], [float("nan")] * 4, [2.0] * 4])
    plan = TopKMeanSolver(k=3).solve(costs).matrix
    expected = torch.tensor([[1 / 3, 0.0, 0.0, 0.0], [1 / 12] * 4, [1 / 12] * 4])
    torch.testing.assert_close(plan, expected)


@pytest.mark.parametrize("solver", [MinCostGreedySolver(), TopKMeanSolver(k=1)])
def test_dtype_max_cost_excludes_infeasible(solver):
    """The penalty clamps to the dtype max; ranking on it would make the row look flat."""
    top = torch.finfo(torch.float32).max
    costs = torch.tensor([[top, float("inf")], [float("nan"), top]])
    plan = solver.solve(costs).matrix
    torch.testing.assert_close(plan, torch.tensor([[0.5, 0.0], [0.0, 0.5]]))
