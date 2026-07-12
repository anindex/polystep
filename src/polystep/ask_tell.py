"""PolyStep as a standard ask/tell optimizer.

Exposes the Sinkhorn Step update behind the ``ask``/``tell`` interface used by
evolution-strategy libraries (evosax, NeuroEvoBench, CMA-ES). ``ask`` returns
candidate points (rotated polytope vertices around each particle) for the caller
to evaluate; ``tell`` takes their fitness (lower is better), builds the cost
matrix, and applies one softmax-weighted barycentric step. PolyStep is then a
drop-in gradient-free optimizer comparable to any ES.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Union

import torch

from ._compiled import _barycentric_projection, _rotate_and_translate
from .geometry import get_orthoplex_vertices, get_random_rotation_matrices
from .solvers import SinkhornSolver, SoftmaxSolver

__all__ = ["PolyStepES", "minimize"]


class PolyStepES:
    """Ask/tell wrapper around the Sinkhorn Step update.

    Args:
        dim: Dimensionality of a solution vector.
        num_particles: Number of independent particles (each proposes a
            polytope of candidates). Population size is ``num_particles * 2 * dim``.
        epsilon: Entropic OT temperature for the softmax weighting.
        step_radius: Geometric step size along the polytope directions.
        solver: OT solver instance. Defaults to :class:`SoftmaxSolver`; pass a
            :class:`SinkhornSolver` for the full entropic-OT plan.
        scale_cost: Cost-matrix scaling passed to the solver ("mean", "max_cost",
            a float divisor, or None).
        x0: Initial position(s), shape ``(dim,)`` or ``(num_particles, dim)``.
        seed: Seed for the rotation generator.
        device, dtype: Tensor device and dtype.
    """

    def __init__(
        self,
        dim: int,
        num_particles: int = 1,
        epsilon: float = 0.5,
        step_radius: float = 0.5,
        solver: Optional[Union[SoftmaxSolver, SinkhornSolver]] = None,
        scale_cost: Optional[Union[str, float]] = "mean",
        x0: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}.")
        if num_particles < 1:
            raise ValueError(f"num_particles must be >= 1, got {num_particles}.")
        if not epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")
        if not (math.isfinite(step_radius) and step_radius >= 0):
            raise ValueError(f"step_radius must be finite and >= 0, got {step_radius}.")
        self.dim = dim
        self.num_particles = num_particles
        self.epsilon = epsilon
        self.step_radius = step_radius
        self.scale_cost = scale_cost
        self.device = torch.device(device)
        self.dtype = dtype
        self.solver = solver if solver is not None else SoftmaxSolver(epsilon=epsilon)
        if isinstance(self.solver, SinkhornSolver) and num_particles == 1:
            warnings.warn(
                "SinkhornSolver with num_particles=1 yields a uniform transport plan "
                "(the column marginal forces it), so steps ignore fitness. Use the "
                "default SoftmaxSolver, or num_particles > 1.",
                stacklevel=2,
            )

        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self.vertices = get_orthoplex_vertices(dim, device=self.device, dtype=dtype)  # (2d, d)
        self.num_vertices = self.vertices.shape[0]

        if x0 is None:
            X = torch.zeros(num_particles, dim, device=self.device, dtype=dtype)
        else:
            X = torch.as_tensor(x0, device=self.device, dtype=dtype).reshape(-1, dim)
            if X.shape[0] == 1 and num_particles > 1:
                X = X.expand(num_particles, dim).clone()
        if X.shape[0] != num_particles:
            raise ValueError(f"x0 has {X.shape[0]} particle rows but num_particles={num_particles}.")
        self.X = X.clone()
        self.a = torch.full((num_particles,), 1.0 / num_particles, device=self.device, dtype=dtype)

        self._pending: Optional[torch.Tensor] = None
        self.best_solution: Optional[torch.Tensor] = None
        self.best_fitness: float = float("inf")

    @property
    def popsize(self) -> int:
        return self.num_particles * self.num_vertices

    @property
    def mean(self) -> torch.Tensor:
        """Mean particle position (the current solution estimate)."""
        return self.X.mean(dim=0)

    @torch.inference_mode()
    def ask(self) -> torch.Tensor:
        """Return candidate points of shape ``(popsize, dim)`` to evaluate."""
        if self._pending is not None:
            raise RuntimeError("ask() called twice before tell(); tell() the previous population first.")
        rot = get_random_rotation_matrices(
            self.num_particles, self.dim, device=self.device, dtype=self.dtype, generator=self.generator
        )
        X_vertices, _ = _rotate_and_translate(rot, self.vertices, self.X, self.step_radius)  # (P, V, d)
        self._pending = X_vertices
        return X_vertices.reshape(self.popsize, self.dim)

    @torch.inference_mode()
    def tell(self, fitness: torch.Tensor) -> None:
        """Update particles from the fitness of the last ``ask`` (lower is better)."""
        if self._pending is None:
            raise RuntimeError("tell() called before ask()")
        cost = torch.as_tensor(fitness, device=self.device, dtype=self.dtype).reshape(
            self.num_particles, self.num_vertices
        )
        self.solver.epsilon = self.epsilon
        # Source marginal defaults to uniform 1/P inside the solver, which is
        # exactly ``self.a`` -- pass None so the solver skips the (host-syncing)
        # user-marginal validation on this per-step path.
        transport = self.solver.solve(cost, scale_cost=self.scale_cost).matrix
        X_new = _barycentric_projection(transport, self.a, self._pending)
        if torch.isfinite(X_new).all():
            self.X = X_new

        flat_cost = cost.reshape(-1)
        fmin, idx = torch.min(flat_cost, dim=0)
        if fmin.item() < self.best_fitness:
            self.best_fitness = fmin.item()
            self.best_solution = self._pending.reshape(self.popsize, self.dim)[idx].clone()
        self._pending = None


def minimize(
    fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    steps: int = 200,
    **kwargs,
) -> PolyStepES:
    """Minimize a batched black-box ``fn: (popsize, dim) -> (popsize,)``.

    Runs ``steps`` ask/tell rounds and returns the optimizer, whose
    ``best_solution`` / ``best_fitness`` / ``mean`` hold the result.
    """
    es = PolyStepES(dim, **kwargs)
    for _ in range(steps):
        candidates = es.ask()
        es.tell(fn(candidates))
    return es
