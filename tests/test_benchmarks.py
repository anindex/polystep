"""Convergence tests for PolyStep on synthetic objectives (Ackley, Rosenbrock, Rastrigin, Sphere)."""

import math

import pytest
import torch

from polystep import LinearEpsilon
from polystep.solver import PolyStep
from polystep.objectives import Ackley, Rosenbrock, Rastrigin, Sphere


def _run_benchmark(
    objective,
    dim,
    num_particles=50,
    max_iters=100,
    epsilon=0.5,
    step_radius=1.0,
    probe_radius=2.0,
    num_probe=5,
    init_scale=3.0,
):
    """Run PolyStep on a synthetic objective and return (final_state, X_init)."""
    torch.manual_seed(42)
    solver = PolyStep(
        objective_fn=objective,
        dim=dim,
        epsilon=epsilon,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        max_iterations=max_iters,
        min_iterations=10,
        compile=False,
    )
    X_init = torch.randn(num_particles, dim) * init_scale
    gen = torch.Generator().manual_seed(42)
    state = solver.run(X_init, generator=gen)
    return state, X_init


class TestSyntheticBenchmarks:
    """Convergence benchmarks for PolyStep on synthetic objectives."""

    @pytest.mark.parametrize(
        "objective,dim,run_kwargs,optimum,dist_factor",
        [
            (Ackley(dim=2), 2, {}, torch.zeros(2), 1.0),
            (Rastrigin(dim=2), 2, {}, torch.zeros(2), 1.0),
            (Sphere(dim=2), 2, {}, torch.zeros(2), 0.5),
            (Rosenbrock(dim=2), 2, {"epsilon": 0.5, "step_radius": 0.5, "probe_radius": 1.0}, torch.ones(2), 1.0),
            (
                Ackley(dim=10),
                10,
                {"num_particles": 20, "max_iters": 25, "epsilon": 1.0, "step_radius": 0.5, "probe_radius": 1.0},
                torch.zeros(10),
                1.0,
            ),
        ],
        ids=["ackley-2d", "rastrigin-2d", "sphere-2d", "rosenbrock-2d", "ackley-10d"],
    )
    def test_convergence(self, objective, dim, run_kwargs, optimum, dist_factor):
        """Cost decreases, particles move toward the optimum, and no state goes non-finite."""
        state, X_init = _run_benchmark(objective, dim=dim, **run_kwargs)

        assert state.costs[-1] < state.costs[0], f"cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"

        assert torch.isfinite(state.X).all(), "NaN/Inf in final particles"
        assert all(math.isfinite(c) for c in state.costs), f"non-finite cost in {state.costs}"
        assert all(math.isfinite(d) for d in state.displacement_sqnorms), "non-finite displacement"
        if state.f is not None:
            assert torch.isfinite(state.f).all(), "NaN/Inf in dual potential f"
        if state.g is not None:
            assert torch.isfinite(state.g).all(), "NaN/Inf in dual potential g"

        init_dist = torch.norm(X_init - optimum, dim=-1).mean().item()
        final_dist = torch.norm(state.X - optimum, dim=-1).mean().item()
        assert final_dist < init_dist * dist_factor, (
            f"particles did not converge: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

    def test_health_series_stay_aligned_with_costs(self):
        """``record_solver_health`` promises ess/rho/evals aligned with ``costs`` in length and bounds."""
        state, _ = _run_benchmark(Sphere(dim=2), dim=2, num_particles=8, max_iters=6)

        n = len(state.costs)
        assert n > 0
        assert len(state.ess) == len(state.rho) == len(state.evals) == n
        # ESS/V is a fraction of the vertex count and rho is a fraction of the probe
        # radius, so both live in a bounded range whatever the objective.
        assert all(0.0 < e <= 1.0 for e in state.ess), state.ess
        assert all(r >= 0.0 for r in state.rho), state.rho
        assert all(v > 0 for v in state.evals), state.evals

    def test_run_stops_early_once_the_displacement_settles(self):
        """A huge ``threshold`` must trip the convergence break right after ``min_iterations``."""
        torch.manual_seed(0)
        solver = PolyStep(
            objective_fn=Sphere(dim=2),
            dim=2,
            epsilon=0.5,
            step_radius=1.0,
            probe_radius=2.0,
            num_probe=5,
            max_iterations=200,
            min_iterations=3,
            threshold=1e9,  # any relative change counts as settled
            compile=False,
        )
        state = solver.run(torch.randn(8, 2), generator=torch.Generator().manual_seed(0))
        assert state.iteration_count == 3, state.iteration_count

    def test_epsilon_schedule_synthetic(self):
        """LinearEpsilon schedule with Sphere: solver completes and converges."""
        obj = Sphere(dim=2)
        eps_schedule = LinearEpsilon(init=1.0, target=0.05, decay=0.01)

        torch.manual_seed(42)
        solver = PolyStep(
            objective_fn=obj,
            dim=2,
            epsilon=eps_schedule,
            step_radius=1.0,
            probe_radius=2.0,
            num_probe=5,
            max_iterations=80,
            min_iterations=10,
            compile=False,
        )
        X_init = torch.randn(50, 2) * 3.0
        gen = torch.Generator().manual_seed(42)
        state = solver.run(X_init, generator=gen)

        assert state.costs[-1] < state.costs[0], (
            f"Sphere+LinearEpsilon cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        assert state.epsilon < 1.0, f"Epsilon did not decay: final epsilon={state.epsilon}"

        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Sphere+LinearEpsilon did not converge: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )
