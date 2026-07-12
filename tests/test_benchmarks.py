"""Synthetic optimization benchmark tests for PolyStep solver.

Verifies convergence on standard test functions (Ackley, Rosenbrock,
Rastrigin, Sphere) in low dimensions.
"""

import math

import pytest
import torch

from polystep import PolyStep, LinearEpsilon
from polystep.objectives import Ackley, Rosenbrock, Rastrigin, ObjectiveFn


class Sphere(ObjectiveFn):
    """Sphere function: f(x) = sum(x_i^2). Global minimum at origin."""

    def __init__(self, dim: int = 2):
        bounds = torch.tensor([[-5.0, 5.0]] * dim)
        optimizers = torch.zeros(1, dim)
        super().__init__(
            dim=dim,
            bounds=bounds,
            optimizers=optimizers,
            optimal_value=0.0,
        )

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(X**2, dim=-1)


ALL_OBJECTIVES = [
    ("ackley", Ackley(dim=2)),
    ("rosenbrock", Rosenbrock(dim=2)),
    ("rastrigin", Rastrigin(dim=2)),
    ("sphere", Sphere(dim=2)),
]


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


# --- Test class ---


class TestSyntheticBenchmarks:
    """Convergence benchmarks for PolyStep on synthetic objectives."""

    def test_ackley_convergence(self):
        """Particles move toward origin and cost decreases on Ackley."""
        obj = Ackley(dim=2)
        state, X_init = _run_benchmark(obj, dim=2)

        # Cost should decrease over the run
        assert state.costs[-1] < state.costs[0], (
            f"Ackley cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Particles should be closer to origin than initial
        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Ackley particles did not move toward origin: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

    def test_rosenbrock_convergence(self):
        """Particles move toward (1,1) and cost decreases on Rosenbrock."""
        obj = Rosenbrock(dim=2)
        state, X_init = _run_benchmark(
            obj,
            dim=2,
            epsilon=0.5,
            step_radius=0.5,
            probe_radius=1.0,
        )

        # Cost should decrease
        assert state.costs[-1] < state.costs[0], (
            f"Rosenbrock cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Particles should be closer to (1,1) than initial
        optimum = torch.ones(2)
        init_dist = torch.norm(X_init - optimum, dim=-1).mean().item()
        final_dist = torch.norm(state.X - optimum, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Rosenbrock particles did not move toward (1,1): init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

    def test_rastrigin_convergence(self):
        """Particles move toward origin and cost decreases on Rastrigin."""
        obj = Rastrigin(dim=2)
        state, X_init = _run_benchmark(obj, dim=2)

        # Cost should decrease
        assert state.costs[-1] < state.costs[0], (
            f"Rastrigin cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Particles should be closer to origin
        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Rastrigin particles did not move toward origin: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

    def test_sphere_convergence(self):
        """Strong convergence to origin on convex Sphere function."""
        obj = Sphere(dim=2)
        state, X_init = _run_benchmark(obj, dim=2)

        # Cost should decrease
        assert state.costs[-1] < state.costs[0], (
            f"Sphere cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Particles should be much closer to origin (convex, easier)
        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist * 0.5, (
            f"Sphere did not show strong convergence: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

    @pytest.mark.parametrize("name,objective", ALL_OBJECTIVES, ids=[n for n, _ in ALL_OBJECTIVES])
    def test_all_objectives_no_nan(self, name, objective):
        """No NaN or Inf in solver state across all benchmark runs."""
        state, _ = _run_benchmark(objective, dim=2, max_iters=30)

        # Check particles
        assert torch.isfinite(state.X).all(), f"{name}: NaN/Inf in final particles"

        # Check costs
        for i, c in enumerate(state.costs):
            assert math.isfinite(c), f"{name}: non-finite cost at iteration {i}: {c}"

        # Check displacement norms
        for i, d in enumerate(state.displacement_sqnorms):
            assert math.isfinite(d), f"{name}: non-finite displacement at iteration {i}: {d}"

        # Check dual potentials if present
        if state.f is not None:
            assert torch.isfinite(state.f).all(), f"{name}: NaN/Inf in dual potential f"
        if state.g is not None:
            assert torch.isfinite(state.g).all(), f"{name}: NaN/Inf in dual potential g"

    def test_higher_dim_ackley(self):
        """Ackley in 10D: solver runs without error and cost decreases."""
        obj = Ackley(dim=10)
        state, X_init = _run_benchmark(
            obj,
            dim=10,
            num_particles=20,
            max_iters=25,
            epsilon=1.0,
            step_radius=0.5,
            probe_radius=1.0,
        )

        # Cost should decrease
        assert state.costs[-1] < state.costs[0], (
            f"Ackley 10D cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Particles should move toward origin
        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Ackley 10D particles did not move toward origin: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )

        # No NaN
        assert torch.isfinite(state.X).all(), "NaN/Inf in 10D Ackley particles"

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

        # Cost should decrease
        assert state.costs[-1] < state.costs[0], (
            f"Sphere+LinearEpsilon cost did not decrease: {state.costs[0]:.4f} -> {state.costs[-1]:.4f}"
        )

        # Epsilon should have decayed (final epsilon < initial)
        assert state.epsilon < 1.0, f"Epsilon did not decay: final epsilon={state.epsilon}"

        # Particles should converge toward origin
        init_dist = torch.norm(X_init, dim=-1).mean().item()
        final_dist = torch.norm(state.X, dim=-1).mean().item()
        assert final_dist < init_dist, (
            f"Sphere+LinearEpsilon did not converge: init_dist={init_dist:.4f}, final_dist={final_dist:.4f}"
        )
