"""Tests for the ask/tell adapter (PolyStepES)."""

import pytest
import torch

from polystep import PolyStepES, PolyStepOptimizer, minimize
from polystep.solver import PolyStep
from polystep.solvers import SinkhornSolver, SoftmaxSolver


def _sphere(X):
    return (X**2).sum(dim=-1)


def test_ask_returns_population_shape():
    dim, P = 5, 3
    es = PolyStepES(dim, num_particles=P, seed=0)
    cand = es.ask()
    assert cand.shape == (P * 2 * dim, dim)
    assert es.popsize == P * 2 * dim


def test_tell_requires_ask_first():
    es = PolyStepES(4, seed=0)
    with pytest.raises(RuntimeError, match="before ask"):
        es.tell(torch.zeros(es.popsize))


def test_tell_keeps_best_shown_candidate():
    """Across all rounds, not only the latest."""
    es = PolyStepES(3, num_particles=2, seed=0)
    candidates = es.ask()
    fitness = _sphere(candidates)
    es.tell(fitness)

    best = int(torch.argmin(fitness))
    assert es.best_fitness == pytest.approx(fitness[best].item())
    torch.testing.assert_close(es.best_solution, candidates[best])

    # A worse round must not overwrite it.
    es.tell(_sphere(es.ask()) + 1e6)
    assert es.best_fitness == pytest.approx(fitness[best].item())


def test_tell_ignores_nonfinite_update():
    es = PolyStepES(3, num_particles=2, seed=0)
    es.ask()
    before = es.X.clone()
    es.tell(torch.full((es.popsize,), float("nan")))
    torch.testing.assert_close(es.X, before)


def test_unconverged_sinkhorn_row_keeps_particle():
    """A near-empty row averages its own vertices, not the origin."""
    x0 = torch.tensor([[0.0], [10.0]])
    es = PolyStepES(
        1,
        num_particles=2,
        x0=x0,
        scale_cost=None,
        solver=SinkhornSolver(epsilon=1.0, omega=0.5, max_iterations=1),
    )
    es.ask()
    es.tell(torch.tensor([0.0, 0.0, 100.0, 100.0]))
    torch.testing.assert_close(es.X, x0)


def test_unseeded_population_follows_global_rng():
    def first_population(seed):
        torch.manual_seed(seed)
        return PolyStepES(4, num_particles=2).ask()

    torch.testing.assert_close(first_population(0), first_population(0))
    assert not torch.equal(first_population(0), first_population(1))


def test_ask_twice_before_tell_raises():
    es = PolyStepES(4, seed=0)
    es.ask()
    with pytest.raises(RuntimeError):
        es.ask()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 4, "num_particles": 0},
        {"dim": 4, "epsilon": 0.0},
        {"dim": 4, "epsilon": -1.0},
        {"dim": 4, "step_radius": float("inf")},
        {"dim": 4, "step_radius": -0.1},
        {"dim": 4, "scale_cost": -1.0},
        {"dim": 4, "num_particles": 2, "x0": torch.zeros(3, 4)},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        PolyStepES(**kwargs)


def test_scale_cost_typo_raises_at_init():
    with pytest.raises(ValueError, match="Unknown scale_cost"):
        PolyStepES(dim=3, scale_cost="maen")
    with pytest.raises(ValueError, match="Unknown scale_cost"):
        PolyStepOptimizer(torch.nn.Linear(3, 1), scale_cost="maen")
    with pytest.raises(ValueError, match="Unknown scale_cost"):
        PolyStep(objective_fn=lambda x: x.pow(2).sum(-1), dim=2, scale_cost="maen")


def test_sinkhorn_single_particle_warns():
    with pytest.warns(UserWarning):
        PolyStepES(4, num_particles=1, solver=SinkhornSolver(epsilon=0.1))


def test_minimize_reduces_sphere():
    es = minimize(_sphere, dim=8, steps=150, step_radius=0.3, epsilon=0.1, x0=torch.full((8,), 2.0), seed=0)
    # Started at ||x||^2 = 8 * 4 = 32; a working optimizer gets far below that.
    assert es.best_fitness < 1.0
    assert _sphere(es.mean.unsqueeze(0)).item() < 4.0


def test_sinkhorn_solver_variant_runs():
    es = PolyStepES(
        6,
        num_particles=2,
        solver=SinkhornSolver(epsilon=0.1, max_iterations=50, threshold=1e-4),
        x0=torch.full((6,), 1.5),
        seed=0,
    )
    for _ in range(40):
        es.tell(_sphere(es.ask()))
    assert torch.isfinite(es.X).all()
    assert es.best_fitness < _sphere(torch.full((1, 6), 1.5)).item()


def test_es_best_skips_nan_fitness():
    """A single NaN makes ``torch.min`` return NaN, so ``fmin < best_fitness`` would be
    False and no finite candidate recorded, even with valid evaluations in the same
    population.
    """
    es = PolyStepES(dim=3, num_particles=2, seed=0)
    asked = es.ask()
    fitness = torch.arange(asked.shape[0], dtype=torch.float32)
    fitness[3] = float("nan")

    es.tell(fitness)

    assert es.best_solution is not None, "no finite best recorded despite valid candidates"
    assert es.best_fitness == 0.0


def test_user_solver_keeps_own_temperature():
    """tell() must not overwrite it with PolyStepES.epsilon."""
    es = PolyStepES(dim=3, num_particles=2, solver=SoftmaxSolver(epsilon=0.01), seed=0)
    es.ask()
    es.tell(torch.rand(es.popsize))
    assert es.solver.epsilon == 0.01


def test_neginf_fitness_never_captures_best():
    """-inf survives nan_to_num and wins torch.min, so nothing could ever beat it."""
    es = PolyStepES(dim=3, num_particles=2, seed=0)
    f = torch.full((es.popsize,), float("-inf"))
    f[1], f[2] = 2.0, 3.0
    candidates = es.ask()
    es.tell(f)
    assert es.best_fitness == 2.0
    assert torch.equal(es.best_solution, candidates[1])


def test_ask_tell_never_records_nonfinite_best():
    es = PolyStepES(dim=2)
    es.ask()
    es.tell(torch.tensor([float("inf"), float("nan"), float("-inf"), float("inf")]))
    assert es.best_fitness == float("inf") and es.best_solution is None
    es.ask()
    es.tell(torch.ones(es.popsize))
    previous = es.best_solution.clone()
    es.ask()
    es.tell(torch.full((es.popsize,), float("inf")))
    assert es.best_fitness == 1.0
    torch.testing.assert_close(es.best_solution, previous)
