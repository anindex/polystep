"""PolyStep behind the ask/tell interface used by evolution-strategy libraries."""

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
        num_particles: Number of independent particles. Population size is
            ``num_particles * 2 * dim``.
        epsilon: Entropic OT temperature (drives the default solver only).
        step_radius: Geometric step size along the polytope directions.
        solver: OT solver. Defaults to :class:`SoftmaxSolver`; pass a
            :class:`SinkhornSolver` for the full entropic-OT plan. A caller-supplied
            solver keeps its own ``epsilon``.
        scale_cost: Cost-matrix scaling passed to the solver ("mean", "max_cost",
            a float divisor, or None).
        x0: Initial position(s), shape ``(dim,)`` or ``(num_particles, dim)``.
        seed: Seed for the rotation generator.
        device: Tensor device. ``None`` follows ``x0``, else CPU.
        dtype: Tensor dtype.
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
        device: Optional[Union[str, torch.device]] = None,
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
        # A CUDA x0 on CPU internals would fail later with a device mismatch.
        if device is None:
            device = x0.device if isinstance(x0, torch.Tensor) else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        # Only the default solver is driven by self.epsilon.
        self._own_solver = solver is None
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

        self.vertices = get_orthoplex_vertices(dim, device=self.device, dtype=dtype)
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
        if self._own_solver:
            self.solver.epsilon = self.epsilon
        # None skips the solver's host-syncing marginal validation on this per-step path.
        transport = self.solver.solve(cost, scale_cost=self.scale_cost).matrix
        X_new = _barycentric_projection(transport, self._pending)
        if torch.isfinite(X_new).all():
            self.X = X_new

        # NaN and -inf would poison torch.min, so map both to +inf first.
        flat_cost = torch.nan_to_num(cost.reshape(-1), nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
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
    """Minimize a batched black-box ``fn: (popsize, dim) -> (popsize,)`` for ``steps`` ask/tell rounds."""
    es = PolyStepES(dim, **kwargs)
    for _ in range(steps):
        candidates = es.ask()
        es.tell(fn(candidates))
    return es
