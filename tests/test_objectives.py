"""Contract tests: each objective's declared optima and bounds match its implementation."""

import pytest
import torch

from polystep.objectives import (
    Ackley,
    Beale,
    Branin,
    Griewank,
    Levy,
    Rastrigin,
    Rosenbrock,
    Sphere,
    StyblinskiTang,
)

# Tolerance at the declared optimizer; Branin and StyblinskiTang use rounded
# literals, so they get a looser tolerance than the exactly-zero optima.
ALL = [
    (lambda: Ackley(dim=2), 1e-6),
    (lambda: Ackley(dim=5), 1e-6),
    (lambda: Rosenbrock(dim=2), 1e-6),
    (lambda: Rastrigin(dim=3), 1e-6),
    (lambda: StyblinskiTang(dim=3), 1e-4),
    (lambda: Levy(dim=3), 1e-6),
    (lambda: Griewank(dim=3), 1e-6),
    (lambda: Beale(), 1e-6),
    (lambda: Branin(), 1e-5),
    (lambda: Sphere(dim=4), 1e-6),
]
IDS = ["ackley2", "ackley5", "rosenbrock", "rastrigin", "styblinski", "levy", "griewank", "beale", "branin", "sphere"]


@pytest.mark.parametrize("factory,tol", ALL, ids=IDS)
def test_objective_contract(factory, tol):
    obj = factory()
    assert obj.bounds.shape == (obj.dim, 2)
    assert obj.optimizers.shape[1] == obj.dim
    assert (obj.optimizers >= obj.bounds[:, 0] - tol).all()
    assert (obj.optimizers <= obj.bounds[:, 1] + tol).all()
    values = obj.evaluate(obj.optimizers)
    assert values.shape == (obj.optimizers.shape[0],)
    torch.testing.assert_close(values, torch.full_like(values, obj.optimal_value), atol=tol, rtol=0)
    lo, hi = obj.bounds[:, 0], obj.bounds[:, 1]
    points = lo + (hi - lo) * torch.rand(3, 5, obj.dim, generator=torch.Generator().manual_seed(0))
    values = obj.evaluate(points)
    assert values.shape == (3, 5)
    assert values.min() >= obj.optimal_value - tol


def test_negate_flips_sign():
    """``negate=True`` turns the minimization into a maximization."""
    plain, flipped = Sphere(dim=3), Sphere(dim=3, negate=True)
    X = torch.randn(8, 3, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(flipped(X), -plain(X))


def test_noise_additive_zero_mean_seeded():
    """``noise_std`` perturbs the cost reproducibly for a given generator."""
    clean, noisy = Sphere(dim=3), Sphere(dim=3, noise_std=0.5)
    X = torch.randn(4096, 3, generator=torch.Generator().manual_seed(3))

    a = noisy(X, generator=torch.Generator().manual_seed(7))
    b = noisy(X, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)

    assert not torch.equal(a, clean(X))
    residual = a - clean(X)
    assert abs(residual.mean().item()) < 0.05
    assert abs(residual.std().item() - 0.5) < 0.05


def test_zero_noise_std_is_noop():
    """``noise_std=0`` must take the clean path, not draw a zero-width normal."""
    X = torch.randn(16, 3, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(Sphere(dim=3, noise_std=0.0)(X), Sphere(dim=3)(X))


def test_rosenbrock_dim1_raises():
    """dim=1 makes Rosenbrock the zero function."""
    with pytest.raises(ValueError, match="dim >= 2"):
        Rosenbrock(dim=1)
