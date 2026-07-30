"""Tests for the ask/tell adapter (PolyStepES)."""

import pytest
import torch

from polystep import PolyStepES, minimize
from polystep.solvers import SinkhornSolver


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


def test_tell_keeps_the_best_candidate_it_was_shown():
    """``best_solution`` is what the adapter exists to return, and only its type was checked."""
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


def test_tell_ignores_a_non_finite_update():
    """A NaN fitness must leave the particles where they were, not poison them."""
    es = PolyStepES(3, num_particles=2, seed=0)
    es.ask()
    before = es.X.clone()
    es.tell(torch.full((es.popsize,), float("nan")))
    torch.testing.assert_close(es.X, before)


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
        {"dim": 4, "num_particles": 2, "x0": torch.zeros(3, 4)},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        PolyStepES(**kwargs)


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
