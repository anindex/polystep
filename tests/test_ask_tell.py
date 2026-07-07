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
    try:
        es.tell(torch.zeros(es.popsize))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_ask_twice_before_tell_raises():
    es = PolyStepES(4, seed=0)
    es.ask()
    with pytest.raises(RuntimeError):
        es.ask()


def test_x0_particle_count_mismatch_raises():
    with pytest.raises(ValueError):
        PolyStepES(4, num_particles=2, x0=torch.zeros(3, 4))


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
