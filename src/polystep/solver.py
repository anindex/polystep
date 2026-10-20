"""PolyStep solver: gradient-free optimization via entropic OT."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from ._compiled import CompiledFunctions
from .costs import compute_cost_matrix, resolve_cost_scale
from .epsilon import feed_solver_stats, resolve_epsilon, resolve_radius, LinearEpsilon, ProgressiveEpsilon
from .geometry import get_random_rotation_matrices, POLYTOPE_MAP
from .solvers import SinkhornSolver, SoftmaxSolver
from .solvers._shared import sanitize_cost, solver_health


@dataclass
class SolverState:
    """State of the PolyStep solver. ``costs`` is the mean over all ``P * V`` probe vertices after ``sanitize_cost``, ``inf`` when every vertex cost is nonfinite and ``nan`` when ``X`` is, not a best-seen value."""

    X: torch.Tensor
    costs: List[float] = field(default_factory=list)
    ess: List[float] = field(default_factory=list)
    rho: List[float] = field(default_factory=list)
    evals: List[int] = field(default_factory=list)
    linear_convergence: List[bool] = field(default_factory=list)
    displacement_sqnorms: List[float] = field(default_factory=list)
    iteration_count: int = 0
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    epsilon: Optional[float] = None
    base_params: Optional[dict] = None
    subspace: Optional[object] = None
    block_duals: Optional[list] = None
    velocity: Optional[torch.Tensor] = None
    stagnation_count: int = 0
    radius_multiplier: float = 1.0
    prev_loss: float = float("inf")
    # Adaptive subspace state
    projection: Optional[torch.Tensor] = None
    displacement_history: Optional[torch.Tensor] = None
    displacement_history_full: Optional[torch.Tensor] = None
    displacement_history_idx: int = 0
    displacement_history_count: int = 0
    absorb_count: int = 0
    # Dual potential momentum
    prev_prev_f: Optional[torch.Tensor] = None
    prev_prev_g: Optional[torch.Tensor] = None
    # Epsilon of the last solve; a >2x jump drops the dual-momentum history.
    last_solve_eps: Optional[float] = None
    # Trust region diagnostics
    trust_region_multipliers: List[float] = field(default_factory=list)
    # HybridSubspace state
    hybrid_projections: Optional[dict] = None

    def record_solver_health(self, ess=None, rho=None, evals=0) -> None:
        """Append one entry each to ess/rho/evals, aligned with ``costs``."""
        self.ess.append(ess if ess is not None else (self.ess[-1] if self.ess else 0.0))
        self.rho.append(rho if rho is not None else (self.rho[-1] if self.rho else 0.0))
        # Rounded, not truncated, so fractional counts don't lose an eval per step.
        self.evals.append(round(evals))


@dataclass
class PolyStep:
    """Batch gradient-free PolyStep solver for non-convex objectives; use ``PolyStepOptimizer`` for neural network training."""

    objective_fn: Callable
    dim: int
    polytope_type: str = "orthoplex"
    epsilon: Union[float, LinearEpsilon] = 0.1
    ent_epsilon: Optional[Union[float, LinearEpsilon]] = None
    scale_cost: Optional[Union[str, float]] = 1.0
    step_radius: float = 1.0
    probe_radius: float = 2.0
    # K=1 matches PolyStepOptimizer's default: multi-probe averaging duplicates entropic variance reduction.
    num_probe: int = 1
    max_iterations: int = 50
    min_iterations: int = 5
    threshold: float = 1e-3
    sinkhorn_max_iters: int = 2000
    chunk_size: Optional[int] = None
    compile: bool = False

    def __post_init__(self):
        """Initialize derived state: polytope template, probes, solver, compiled fns."""
        if self.num_probe < 1:
            raise ValueError(
                f"num_probe must be >= 1, got {self.num_probe}. num_probe=0 yields an "
                "empty probe tensor, so the cost matrix is a mean over nothing (NaN) "
                "and every step becomes a no-op with a fabricated finite cost."
            )
        if isinstance(self.epsilon, ProgressiveEpsilon) and self.ent_epsilon is not None:
            raise ValueError("ProgressiveEpsilon adapts the solve epsilon; it cannot be combined with ent_epsilon.")
        resolve_cost_scale(torch.ones((), dtype=torch.float64, device="cpu"), self.scale_cost)

        self.polytope_vertices = POLYTOPE_MAP[self.polytope_type](self.dim, radius=1.0)
        self.probes = torch.linspace(0, 1, self.num_probe + 2)[1 : self.num_probe + 1]
        self._templates = {}
        self.sinkhorn_solver = SinkhornSolver(
            max_iterations=self.sinkhorn_max_iters,
            compile=self.compile,
        )
        self._compiled = CompiledFunctions(compile=self.compile)

    @classmethod
    def create(
        cls,
        objective_fn: Callable,
        dim: Optional[int] = None,
        **kwargs,
    ) -> "PolyStep":
        """Factory for a PolyStep solver; ``dim`` falls back to ``objective_fn.dim``."""
        if dim is None:
            dim = objective_fn.dim

        return cls(objective_fn=objective_fn, dim=dim, **kwargs)

    def init_state(self, X_init: torch.Tensor) -> SolverState:
        """Initialize solver state from ``X_init``."""
        # Normalize a 1-D point to one particle so the particle count is not read from dim.
        if X_init.dim() == 1:
            X_init = X_init.unsqueeze(0)

        num_points = X_init.shape[0]

        # One particle forces a uniform plan, so the step ignores the cost and never
        # moves. Only warn here; ``_solver_for`` makes the substitution per step, so a
        # second state with a different particle count is not answered by this one's.
        if num_points == 1 and isinstance(self.sinkhorn_solver, SinkhornSolver):
            warnings.warn(
                "Single particle: balanced Sinkhorn yields a uniform transport plan, so "
                "steps would ignore the cost. Using the one-sided SoftmaxSolver instead.",
                stacklevel=2,
            )

        return SolverState(X=X_init.clone())

    def _solver_for(self, state: SolverState):
        """The solver this state's particle count needs, without mutating ``self``.

        One particle makes balanced Sinkhorn return a uniform plan, so it falls back
        to the one-sided softmax. States of different sizes can share one ``PolyStep``.
        """
        solver = self.sinkhorn_solver
        if state.X.shape[0] != 1 or not isinstance(solver, SinkhornSolver):
            return solver
        cached = getattr(self, "_softmax_fallback", None)
        if cached is None:
            cached = self._softmax_fallback = SoftmaxSolver(epsilon=solver.epsilon)
        return cached

    @torch.inference_mode()
    def step(
        self,
        state: SolverState,
        generator: Optional[torch.Generator] = None,
    ) -> SolverState:
        """Run one PolyStep iteration."""
        iteration = state.iteration_count
        X = state.X
        device = X.device

        # Shared with PolyStepOptimizer so a scheduled radius means the same distance via either entry point.
        current_eps = resolve_epsilon(self.epsilon, iteration, "epsilon")
        step_radius = resolve_radius(self.step_radius, iteration, current_eps)
        probe_radius = resolve_radius(self.probe_radius, iteration, current_eps)

        templates = self._templates.get((device, X.dtype))
        if templates is None:
            templates = self._templates[(device, X.dtype)] = (
                self.polytope_vertices.to(device=device, dtype=X.dtype),
                self.probes.to(device=device, dtype=X.dtype),
            )
        polytope_verts, probes = templates

        # Pre-normalize to avoid a shape branch in the compiled fn.
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch, dim = X.shape

        # Uses torch.Generator, so not compilable.
        rot_mats = get_random_rotation_matrices(
            batch,
            dim,
            device=device,
            dtype=X.dtype,
            generator=generator,
        )

        X_vertices, rotated = self._compiled.rotate_and_translate(
            rot_mats,
            polytope_verts,
            X,
            step_radius,
        )

        X_probe = self._compiled.compute_probe_points(
            X,
            rotated,
            probes,
            probe_radius,
        )

        cost_matrix = compute_cost_matrix(
            self.objective_fn,
            X_probe,
            chunk_size=self.chunk_size,
        )

        ent_eps = resolve_epsilon(self.ent_epsilon, iteration, "ent_epsilon")
        ot_epsilon = ent_eps if ent_eps is not None else current_eps

        ot_solver = self._solver_for(state)
        ot_solver.epsilon = ot_epsilon
        ot_result = ot_solver.solve(
            cost_matrix=cost_matrix,
            # a=None lets the solver build the uniform marginal itself; passing a only buys host-sync validation.
            a=None,
            init_f=state.f,
            init_g=state.g,
            scale_cost=self.scale_cost,
        )

        transport_matrix = ot_result.matrix  # (batch, num_vertices)
        X_new = self._compiled.barycentric_projection(
            transport_matrix,
            X_vertices,
        )
        ess, rho = solver_health(transport_matrix, X_new - X, step_radius)
        feasible = torch.isfinite(cost_matrix).any()
        cost_mean = torch.where(feasible, sanitize_cost(cost_matrix).mean(), torch.inf)
        finite, feasible, disp_sqnorm, cost_mean, ess, rho = torch.stack(
            [
                torch.isfinite(X_new).all().to(X.dtype),
                feasible.to(X.dtype),
                ((X_new - X) ** 2).sum(-1).mean(),
                cost_mean,
                ess,
                rho,
            ]
        ).tolist()

        # NaN-safe: revert if X_new has NaN.
        _nan_reverted = not finite
        if _nan_reverted:
            X_new = X.clone()
            disp_sqnorm = 0.0
            if not torch.isfinite(X).all():
                cost_mean = disp_sqnorm = math.nan

        state.X = X_new
        state.costs.append(cost_mean)
        state.record_solver_health(ess, rho, X_probe[..., 0].numel())
        converged = ot_result.converged and bool(feasible)
        state.linear_convergence.append(converged)
        feed_solver_stats(self.epsilon, ot_solver, ot_result.n_iters, converged)
        state.displacement_sqnorms.append(disp_sqnorm)
        state.iteration_count += 1
        if _nan_reverted:
            state.f = None
            state.g = None
        elif feasible:
            # None on the one-particle SoftmaxSolver path, which has no duals.
            state.f = ot_result.f.detach() if ot_result.f is not None else None
            state.g = ot_result.g.detach() if ot_result.g is not None else None
        state.epsilon = current_eps

        return state

    @torch.inference_mode()
    def _converged(self, state: SolverState) -> bool:
        """Converged when the displacement has settled and gone small."""
        if state.iteration_count < 3:
            return False
        d = state.displacement_sqnorms
        settled = abs(d[-1] - d[-2]) / (abs(d[-2]) + 1e-10) < self.threshold
        return settled and d[-1] < self.threshold * max(d[0], 1e-30)

    def _diverged(self, state: SolverState) -> bool:
        """Check if the solver has diverged (non-finite cost)."""
        return bool(state.costs) and not math.isfinite(state.costs[-1])

    def run(
        self,
        X_init: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> SolverState:
        """Run the full outer loop."""
        state = self.init_state(X_init)

        for i in range(self.max_iterations):
            state = self.step(state, generator=generator)

            # i is 0-based, so i + 1 steps have run.
            if i + 1 >= self.min_iterations:
                if self._converged(state) or self._diverged(state):
                    break

        return state
