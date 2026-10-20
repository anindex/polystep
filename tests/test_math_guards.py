"""Numerical and update-rule guards."""

import math

import pytest
import torch
import torch.nn as nn

from polystep._compiled import _barycentric_projection, _rotate_and_translate
from polystep.geometry import get_orthoplex_vertices
from polystep.solver import PolyStep
from polystep.solvers import MinCostGreedySolver, SinkhornSolver, SoftmaxSolver
from polystep.solvers._shared import exp_plan, sanitize_cost, solver_health
from polystep import PolyStepOptimizer


def test_exp_plan_saturates_instead_of_overflowing():
    f = torch.tensor([1e4, 1e4])
    g = torch.tensor([1e4, 1e4])
    C = torch.zeros(2, 2)
    P = exp_plan(f, g, C, eps=1e-3)
    assert torch.isfinite(P).all()
    assert (P > 0).all()


def test_exp_plan_exact_where_plain_exp_finite():
    torch.manual_seed(0)
    f, g = torch.randn(4), torch.randn(5)
    C = torch.rand(4, 5)
    ref = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / 0.1)
    torch.testing.assert_close(exp_plan(f, g, C, 0.1), ref)


def test_diverged_sinkhorn_duals_keep_plan_finite():
    solver = SinkhornSolver(epsilon=1e-4, max_iterations=5, threshold=0.0)
    res = solver.solve(cost_matrix=torch.randn(4, 6) * 1e3)
    assert torch.isfinite(res.matrix).all()


def test_greedy_all_infeasible_row_stays_put():
    C = torch.tensor([[float("inf"), float("nan")], [0.0, 1.0]])
    T = MinCostGreedySolver(epsilon=0.1).solve(cost_matrix=C).matrix
    # Row 0 got no information, so its mass is spread evenly.
    torch.testing.assert_close(T[0], torch.full((2,), 0.25))
    # Row 1 still picks its argmin.
    torch.testing.assert_close(T[1], torch.tensor([0.5, 0.0]))


def test_zero_transport_row_keeps_particle():
    """An underflowed row must not move the particle to the coordinate origin."""
    torch.manual_seed(0)
    X = torch.randn(2, 8)
    rot = torch.linalg.qr(torch.randn(2, 8, 8))[0]
    X_vertices, _ = _rotate_and_translate(rot, get_orthoplex_vertices(8), X, 0.5)

    torch.testing.assert_close(_barycentric_projection(torch.zeros(2, 16), X_vertices), X)

    # Over-relaxed Sinkhorn reaches this from the public API.
    T = (
        SinkhornSolver(epsilon=1e-3, max_iterations=1, threshold=0.0, omega=1.9)
        .solve(torch.rand(2, 16) * 10, init_f=torch.tensor([-1e4, 0.0]), init_g=torch.zeros(16))
        .matrix
    )
    dead = T.sum(dim=1) == 0
    assert dead.any(), "expected an underflowed transport row"
    torch.testing.assert_close(_barycentric_projection(T, X_vertices)[dead], X[dead])


def test_infinite_transport_row_keeps_particle():
    torch.manual_seed(0)
    X = torch.randn(2, 8)
    rot = torch.linalg.qr(torch.randn(2, 8, 8))[0]
    X_vertices, _ = _rotate_and_translate(rot, get_orthoplex_vertices(8), X, 0.5)
    T = torch.rand(2, 16)
    T[0, 3] = torch.inf
    torch.testing.assert_close(_barycentric_projection(T, X_vertices)[0], X[0])


def test_tiny_row_projects_to_normalized_barycentre():
    torch.manual_seed(0)
    T = torch.rand(2, 6, dtype=torch.float64)
    X_vertices = torch.randn(2, 6, 3, dtype=torch.float64)
    tiny = T * torch.tensor([[1.0], [1e-13]], dtype=torch.float64)
    torch.testing.assert_close(_barycentric_projection(tiny, X_vertices), _barycentric_projection(T, X_vertices))


def test_ess_ignores_row_scale():
    torch.manual_seed(0)
    T = torch.rand(3, 5, dtype=torch.float64)
    displacement = torch.zeros(3, 4, dtype=torch.float64)
    ess, _ = solver_health(T, displacement, 1.0)
    tiny_ess, _ = solver_health(T * 1e-13, displacement, 1.0)
    torch.testing.assert_close(tiny_ess, ess)


def test_nonfinite_costs_rank_below_finite():
    """-inf is a diverged evaluation, not a free optimum."""
    C = torch.tensor([[0.0, float("-inf"), 1.0]])
    sanitized = sanitize_cost(C)
    assert sanitized[0, 1] > sanitized[0, 0], "-inf tied with a finite cost"
    assert MinCostGreedySolver(epsilon=0.1).solve(cost_matrix=C).matrix.argmax().item() == 0

    # Nothing finite means no ranking, which is what the step's warning promises.
    allbad = torch.full((1, 3), float("nan"))
    allbad[0, 0] = float("-inf")
    assert sanitize_cost(allbad).unique().numel() == 1


def test_negative_radii_rejected():
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="probe_radius must be > 0"):
        PolyStepOptimizer(model, probe_radius=-0.1, compile=False)
    with pytest.raises(ValueError, match="step_radius must be >= 0"):
        PolyStepOptimizer(model, step_radius=-0.1, compile=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(momentum_final=1.2),
        dict(momentum_final=1.0),
        dict(momentum_init=1.0),
        dict(amortize_ema=1.5),
        dict(velocity_lr=-1.0),
        dict(velocity_lr=0.0),
        dict(radius_min=2.0, radius_max=0.5),
        dict(radius_min=0.0),
        dict(radius_increase=0.5),
        dict(radius_decrease=0.0),
    ],
)
def test_unstable_dynamics_values_raise(kwargs):
    with pytest.raises(ValueError, match="must"):
        PolyStepOptimizer(nn.Sequential(nn.Linear(2, 2)), compile=False, **kwargs)


def test_single_particle_polystep_uses_softmax():
    """The substitution follows the state, so states of different sizes can coexist."""
    solver = PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2)
    with pytest.warns(UserWarning, match="Single particle"):
        one = solver.init_state(torch.zeros(2))
    assert isinstance(solver._solver_for(one), SoftmaxSolver)

    many = solver.init_state(torch.zeros(8, 2))
    assert isinstance(solver._solver_for(many), SinkhornSolver)
    # The one-particle state still gets the softmax after the second init_state.
    assert isinstance(solver._solver_for(one), SoftmaxSolver)


def test_single_particle_polystep_descends():
    torch.manual_seed(0)
    solver = PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2, max_iterations=30)
    with pytest.warns(UserWarning):
        state = solver.init_state(torch.tensor([[2.0, 2.0]]))
    for _ in range(30):
        state = solver.step(state)
    assert state.X.norm().item() < 2.0 * math.sqrt(2) * 0.9


def test_constant_speed_is_not_convergence():
    solver = PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2, threshold=1e-3)
    state = solver.init_state(torch.zeros(1, 2))
    state.iteration_count = 5
    state.displacement_sqnorms = [1.0, 1.0, 1.0]
    assert not solver._converged(state)
    state.displacement_sqnorms = [1.0, 1e-9, 1e-9]
    assert solver._converged(state)


def test_polystep_runs_through_infeasible_region():
    target = torch.tensor([3.0, 0.0])

    def objective(x):
        return (x - target).pow(2).sum(-1).masked_fill(x.norm(dim=-1) > 3, torch.inf)

    torch.manual_seed(0)
    solver = PolyStep(objective_fn=objective, dim=2, max_iterations=10, threshold=1e-12)
    state = solver.run(torch.randn(8, 2) * 0.05 + torch.tensor([2.9, 0.0]))
    assert len(state.costs) == 10
    assert all(math.isfinite(c) for c in state.costs)


def test_polystep_all_nonfinite_candidates_report_inf():
    solver = PolyStep(objective_fn=lambda x: torch.full(x.shape[:-1], torch.inf), dim=2, threshold=1e-12)
    X = torch.randn(8, 2, generator=torch.Generator().manual_seed(0))
    state = solver.run(X)
    assert state.costs[-1] == math.inf
    assert len(state.costs) == solver.min_iterations
    torch.testing.assert_close(state.X, X)


def test_all_nonfinite_step_flagged_keeps_duals():
    from polystep.epsilon import ProgressiveEpsilon

    objective = {"fn": lambda x: x.pow(2).sum(-1)}
    eps = ProgressiveEpsilon(init=1.0)
    solver = PolyStep(objective_fn=lambda x: objective["fn"](x), dim=2, epsilon=eps, scale_cost=None)
    state = solver.step(solver.init_state(torch.tensor([[1.0, 2.0], [3.0, 4.0]])), torch.Generator().manual_seed(0))
    f, g, before = state.f.clone(), state.g.clone(), eps._current

    objective["fn"] = lambda x: torch.full(x.shape[:-1], torch.inf)
    state = solver.step(state, torch.Generator().manual_seed(1))
    assert state.costs[-1] == math.inf
    assert state.linear_convergence[-1] is False
    assert eps._current > before
    torch.testing.assert_close(state.f, f)
    torch.testing.assert_close(state.g, g)


def test_polystep_nonfinite_particle_reports_diverged():
    solver = PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2, threshold=1e-12)
    X = torch.randn(8, 2, generator=torch.Generator().manual_seed(0))
    X[3, 0] = torch.nan
    state = solver.run(X)
    assert solver._diverged(state)
    assert len(state.costs) == solver.min_iterations
    assert math.isnan(state.costs[-1]) and math.isnan(state.displacement_sqnorms[-1])
    torch.testing.assert_close(state.X, X, equal_nan=True)
