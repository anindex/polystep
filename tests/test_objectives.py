"""Contract tests: each objective's declared optima and bounds match its implementation."""

import pytest
import torch

from polystep.objectives import (
    Ackley,
    Beale,
    Branin,
    Griewank,
    Levy,
    ObjectiveFn,
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
def test_declared_optimum_is_actual_optimum(factory, tol):
    """``evaluate(optimizers)`` returns ``optimal_value``, at every declared optimizer."""
    obj = factory()
    values = obj.evaluate(obj.optimizers)
    assert values.shape == (obj.optimizers.shape[0],)
    torch.testing.assert_close(
        values,
        torch.full_like(values, obj.optimal_value),
        atol=tol,
        rtol=0,
    )


@pytest.mark.parametrize("factory,tol", ALL, ids=IDS)
def test_optimum_beats_random_points_in_bounds(factory, tol):
    """The declared optimum is a minimum, not just a point that happens to match a constant."""
    obj = factory()
    gen = torch.Generator().manual_seed(0)
    lo, hi = obj.bounds[:, 0], obj.bounds[:, 1]
    X = lo + (hi - lo) * torch.rand(64, obj.dim, generator=gen)
    assert obj.evaluate(X).min() >= obj.optimal_value - tol


@pytest.mark.parametrize("factory,tol", ALL, ids=IDS)
def test_declared_bounds_contain_optimizers(factory, tol):
    obj = factory()
    assert obj.bounds.shape == (obj.dim, 2)
    assert obj.optimizers.shape[1] == obj.dim
    assert (obj.optimizers >= obj.bounds[:, 0] - tol).all()
    assert (obj.optimizers <= obj.bounds[:, 1] + tol).all()


@pytest.mark.parametrize("factory,tol", ALL, ids=IDS)
def test_evaluate_is_batch_shape_preserving(factory, tol):
    """Leading dimensions pass through, which the solver relies on for ``(P, V, dim)`` probes."""
    obj = factory()
    gen = torch.Generator().manual_seed(1)
    X = torch.randn(3, 5, obj.dim, generator=gen)
    assert obj.evaluate(X).shape == (3, 5)


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


def test_objective_fn_is_abstract():
    """``evaluate`` is the required hook; the base class must not be instantiable."""
    with pytest.raises(TypeError):
        ObjectiveFn(dim=2)


def test_rosenbrock_dim1_raises():
    """dim=1 makes Rosenbrock the zero function."""
    with pytest.raises(ValueError, match="dim >= 2"):
        Rosenbrock(dim=1)


def test_cpu_generator_drives_noise_on_any_device():
    """normal_ needs the generator on the tensor's device; the docs show a CPU one."""
    fn = Sphere(dim=3, noise_std=0.5)
    gen = torch.Generator().manual_seed(7)
    assert torch.isfinite(fn(torch.zeros(4, 3), generator=gen)).all()


def test_negated_objective_negates_optimum():
    """An unnegated optimum puts regret of cost - optimal_value off by 2*optimal_value."""
    plain, flipped = Rastrigin(dim=2), Rastrigin(dim=2, negate=True)
    assert flipped.optimal_value == -plain.optimal_value
