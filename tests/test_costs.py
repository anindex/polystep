"""Unit tests for cost matrix computation and scaling."""

import pytest
import torch
from polystep.costs import compute_cost_matrix, resolve_cost_scale, scale_cost_matrix


@pytest.mark.parametrize("divisor", [-1.0, 0.0, float("inf")])
def test_scale_cost_matrix_rejects_invalid_divisor(divisor):
    """A negative divisor reverses the objective, zero undefines it and inf erases it."""
    with pytest.raises(ValueError, match="positive"):
        scale_cost_matrix(torch.tensor([[0.0, 1.0]]), divisor)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.float64])
def test_compute_cost_matrix_averages_probes(dtype):
    values = torch.arange(12, dtype=dtype).reshape(2, 3, 2, 1)
    actual = compute_cost_matrix(lambda x: x[:, 0], values, chunk_size=5)
    expected = values.squeeze(-1).to(torch.float64 if dtype == torch.float64 else torch.float32).mean(-1)
    torch.testing.assert_close(actual, expected)


def test_compute_cost_matrix_chunked_matches_unchunked():
    batch, verts, probes, dim = 3, 5, 2, 4
    X = torch.randn(batch, verts, probes, dim)

    def fn(x):
        return x.norm(dim=-1)

    C_full = compute_cost_matrix(fn, X)
    C_chunked = compute_cost_matrix(fn, X, chunk_size=4)
    assert torch.allclose(C_full, C_chunked, atol=1e-6)


def test_scale_cost_matrix_none():
    """None scaling should return unchanged matrix."""
    C = torch.randn(3, 5)
    assert torch.equal(scale_cost_matrix(C, None), C)


@pytest.mark.parametrize("mode, divisor", [("mean", 5.0), ("max_cost", 8.0)])
def test_scale_cost_matrix_named_modes(mode, divisor):
    C = torch.tensor([[2.0, 4.0], [6.0, 8.0]])  # mean 5, max 8
    assert torch.allclose(scale_cost_matrix(C, mode), C / divisor)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")],
        ),
    ],
)
@pytest.mark.parametrize("mode", ["mean", "max_cost"])
def test_all_finite_mask_keeps_divisor(mode, device):
    """243 * fl(1/243) is not 1 in fp32, so a ratio of counts would move a clean divisor."""
    for dtype in (torch.float32, torch.float64):
        C = torch.rand(81, 3, dtype=dtype, device=device)
        assert torch.equal(resolve_cost_scale(C, mode, torch.isfinite(C)), resolve_cost_scale(C, mode))


def test_scale_cost_matrix_float():
    """Float scaling divides by that float."""
    C = torch.tensor([[2.0, 4.0]])
    scaled = scale_cost_matrix(C, 5.0)
    assert torch.allclose(scaled, C / 5.0)


def test_scale_cost_matrix_zero_safe():
    """Scaling should not produce inf on zero-mean matrix."""
    C = torch.zeros(3, 5)
    scaled = scale_cost_matrix(C, "mean")
    assert torch.isfinite(scaled).all()


def test_infinite_cost_keeps_plan_contrast():
    """An absolute 1e6 penalty enters the 'mean' reduction and flattens every finite
    vertex to the same weight; the penalty must scale with the finite costs.

    Suppression strength depends on epsilon, which the substitution cannot see, so the
    contract is ordering plus preserved contrast, not an absolute weight.
    """
    from polystep.solvers import SoftmaxSolver

    cost = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reference = SoftmaxSolver(epsilon=0.5).solve(cost, scale_cost="mean").matrix

    masked = torch.tensor([[1.0, 2.0, 3.0, float("inf")]])
    plan = SoftmaxSolver(epsilon=0.5).solve(masked, scale_cost="mean").matrix

    # The masked vertex ranks below every finite one.
    assert plan[0, 3] < plan[0, :3].min()
    # The finite vertices keep their ordering and stay far from uniform.
    assert plan[0, 0] > plan[0, 1] > plan[0, 2]
    assert plan[0, 0] / plan[0, 2] > 1.5
    assert reference[0, 0] > reference[0, 1] > reference[0, 2]


@pytest.mark.parametrize("mode", ["mean", "max_cost"])
def test_nan_cost_excluded_from_divisor(mode):
    """Otherwise the NaN penalty flattens the finite plan."""
    from polystep.solvers import SoftmaxSolver

    clean = SoftmaxSolver(epsilon=0.5).solve(torch.tensor([[0.0, 1.0, 2.0]]), scale_cost=mode).matrix[0]
    masked = torch.tensor([[0.0, 1.0, 2.0, float("nan")]])
    finite = SoftmaxSolver(epsilon=0.5).solve(masked, scale_cost=mode).matrix[0, :3]

    torch.testing.assert_close(finite / finite.sum(), clean)


@pytest.mark.parametrize("mode", ["mean", "max_cost"])
@pytest.mark.parametrize("name", ["SoftmaxSolver", "SinkhornSolver"])
def test_nan_vertex_loses_to_flat_row(mode, name):
    """Equal finite costs recenter to 0, so only the +1 keeps the penalty above them."""
    import polystep.solvers as solvers

    cost = torch.tensor([[1.0, 1.0, float("nan")], [1.0, 1.0, 1.0]])
    row = getattr(solvers, name)(epsilon=0.1).solve(cost, scale_cost=mode).matrix[0]

    assert row[2] < 1e-3 * row[0]


def test_entropic_plan_invariant_to_cost_shift():
    from polystep.solvers import SinkhornSolver, SoftmaxSolver

    cost = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]])
    for solver in (SinkhornSolver(epsilon=0.5, max_iterations=500), SoftmaxSolver(epsilon=0.5)):
        for mode in ("mean", "max_cost"):
            base = solver.solve(cost, scale_cost=mode).matrix
            shifted = solver.solve(cost + 1000.0, scale_cost=mode).matrix
            assert torch.allclose(base, shifted, atol=1e-6), (mode, (base - shifted).abs().max())
