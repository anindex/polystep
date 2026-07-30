"""Numerical and update-rule guards."""

import math

import pytest
import torch
import torch.nn as nn

from polystep.cma import compute_cma_hyperparameters, update_covariance_diagonal, update_evolution_path_sigma
from polystep.solver import PolyStep
from polystep.solvers import MinCostGreedySolver, SinkhornSolver, SoftmaxSolver
from polystep.solvers._shared import exp_plan
from polystep import PolyStepOptimizer


def test_exp_plan_saturates_instead_of_overflowing():
    f = torch.tensor([1e4, 1e4])
    g = torch.tensor([1e4, 1e4])
    C = torch.zeros(2, 2)
    P = exp_plan(f, g, C, eps=1e-3)
    assert torch.isfinite(P).all()
    assert (P > 0).all()


def test_exp_plan_is_exact_where_the_plain_exp_was_finite():
    torch.manual_seed(0)
    f, g = torch.randn(4), torch.randn(5)
    C = torch.rand(4, 5)
    ref = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / 0.1)
    torch.testing.assert_close(exp_plan(f, g, C, 0.1), ref)


def test_diverged_sinkhorn_duals_do_not_yield_a_nan_plan():
    solver = SinkhornSolver(epsilon=1e-4, max_iterations=5, threshold=0.0)
    res = solver.solve(cost_matrix=torch.randn(4, 6) * 1e3)
    assert torch.isfinite(res.matrix).all()


def test_p_sigma_ignores_c_diag_on_a_whitened_step():
    """C_diag=None must equal passing y = sqrt(C) z with the same C."""
    n = 6
    torch.manual_seed(0)
    C_diag = torch.rand(n) + 0.5
    z = torch.randn(n)
    z = z / z.norm()
    p0 = torch.zeros(n)

    whitened = update_evolution_path_sigma(p0, z, None, c_sigma=0.3, mu_eff=1.0)
    round_trip = update_evolution_path_sigma(p0, torch.sqrt(C_diag) * z, C_diag, c_sigma=0.3, mu_eff=1.0)
    torch.testing.assert_close(whitened, round_trip)


def test_covariance_normalizes_then_clamps():
    n = 8
    hp = compute_cma_hyperparameters(n, 1.0)
    C = torch.ones(n)
    p_c = torch.zeros(n)
    p_c[0] = 10.0
    out = update_covariance_diagonal(
        C_diag=C,
        p_c=p_c,
        rank_mu=torch.zeros(n),
        c_1=hp["c_1"],
        c_mu=0.0,
        h_sigma=True,
        c_c=hp["c_c"],
        trace_scale=float(n),
        cov_min=1e-6,
        cov_max=1e6,
        trace=float(n),
    )
    assert (out >= 1e-6).all() and (out <= 1e6).all()
    torch.testing.assert_close(out.sum(), torch.tensor(float(n)))


def test_greedy_leaves_an_all_infeasible_row_where_it_is():
    C = torch.tensor([[float("inf"), float("nan")], [0.0, 1.0]])
    T = MinCostGreedySolver(epsilon=0.1).solve(cost_matrix=C).matrix
    # Row 0 got no information, so its mass is spread evenly.
    torch.testing.assert_close(T[0], torch.full((2,), 0.25))
    # Row 1 still picks its argmin.
    torch.testing.assert_close(T[1], torch.tensor([0.5, 0.0]))


def test_a_negative_probe_radius_is_rejected():
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="probe_radius must be > 0"):
        PolyStepOptimizer(model, probe_radius=-0.1, compile=False)
    with pytest.raises(ValueError, match="step_radius must be >= 0"):
        PolyStepOptimizer(model, step_radius=-0.1, compile=False)


def test_single_particle_polystep_swaps_in_the_softmax_solver():
    solver = PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2)
    assert isinstance(solver.sinkhorn_solver, SinkhornSolver)
    with pytest.warns(UserWarning, match="Single particle"):
        solver.init_state(torch.zeros(2))
    assert isinstance(solver.sinkhorn_solver, SoftmaxSolver)


def test_single_particle_polystep_actually_descends():
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
