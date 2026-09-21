"""PolyStep solver: gradient-free optimization via entropic OT."""

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from ._compiled import CompiledFunctions
from .costs import compute_cost_matrix
from .epsilon import feed_solver_stats, resolve_radius, LinearEpsilon
from .geometry import get_random_rotation_matrices, POLYTOPE_MAP
from .solvers import SinkhornSolver, SoftmaxSolver
from .solvers._shared import solver_health, validate_positive


@dataclass
class SolverState:
    """State of the Sinkhorn Step solver. ``costs`` is the mean over all ``P * V`` probe vertices, not a best-seen value."""

    X: torch.Tensor
    costs: List[float] = field(default_factory=list)
    ess: List[float] = field(default_factory=list)
    rho: List[float] = field(default_factory=list)
    evals: List[int] = field(default_factory=list)
    linear_convergence: List[bool] = field(default_factory=list)
    displacement_sqnorms: List[float] = field(default_factory=list)
    a: Optional[torch.Tensor] = None
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
    # CMA-ES state
    p_c: Optional[torch.Tensor] = None
    p_sigma: Optional[torch.Tensor] = None
    C_diag: Optional[torch.Tensor] = None
    generation: int = 0
    # Dual potential momentum
    prev_prev_f: Optional[torch.Tensor] = None
    prev_prev_g: Optional[torch.Tensor] = None
    # Epsilon under which ``f`` and ``g`` were computed by the previous
    # solve. Lets the next ``SinkhornSolver.solve`` rescale the warm-
    # started duals if the schedule moved epsilon between calls.
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
    """Batch gradient-free solver for non-convex objectives using Sinkhorn Step; use ``PolyStepOptimizer`` for neural network training."""

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
    subspace: Optional[object] = None
    nn_evaluator: Optional[object] = None
    train_inputs: Optional[object] = None
    train_targets: Optional[object] = None

    def __post_init__(self):
        """Initialize derived state: polytope template, probes, solver, compiled fns."""
        if self.num_probe < 1:
            raise ValueError(
                f"num_probe must be >= 1, got {self.num_probe}. num_probe=0 yields an "
                "empty probe tensor, so the cost matrix is a mean over nothing (NaN) "
                "and every step becomes a no-op with a fabricated finite cost."
            )

        self.polytope_vertices = POLYTOPE_MAP[self.polytope_type](self.dim, radius=1.0)
        self.probes = torch.linspace(0, 1, self.num_probe + 2)[1 : self.num_probe + 1]
        self.sinkhorn_solver = SinkhornSolver(
            max_iterations=self.sinkhorn_max_iters,
            compile=self.compile,
        )
        self._compiled = CompiledFunctions(compile=self.compile and torch.cuda.is_available())

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

    def init_state(
        self,
        X_init: torch.Tensor,
        base_params: Optional[dict] = None,
    ) -> SolverState:
        """Initialize solver state from ``X_init`` and optional ``base_params``."""
        # Normalize a 1-D point to one particle so the marginal ``a`` is sized by particle count, not dim.
        if X_init.dim() == 1:
            X_init = X_init.unsqueeze(0)

        num_points = X_init.shape[0]
        a = torch.ones(num_points, device=X_init.device, dtype=X_init.dtype) / num_points

        # One particle forces a uniform plan, so the step ignores the cost and never
        # moves. Only warn here; ``_solver_for`` makes the substitution per step, so a
        # second state with a different particle count is not answered by this one's.
        if num_points == 1 and isinstance(self.sinkhorn_solver, SinkhornSolver):
            warnings.warn(
                "Single particle: balanced Sinkhorn yields a uniform transport plan, so "
                "steps would ignore the cost. Using the one-sided SoftmaxSolver instead.",
                stacklevel=2,
            )

        state = SolverState(X=X_init.clone(), a=a)

        if self.subspace is not None and base_params is not None:
            state.base_params = base_params
            state.subspace = self.subspace

        return state

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

    def _get_epsilon(self, iteration: int) -> float:
        """Resolve epsilon at ``iteration``; duck-types on ``.at()`` so schedules and plain floats both work."""
        if hasattr(self.epsilon, "at"):
            return validate_positive(self.epsilon.at(iteration), "epsilon")
        return validate_positive(self.epsilon, "epsilon")

    def _get_ent_epsilon(self, iteration: int) -> Optional[float]:
        """Resolve ent_epsilon at current iteration (supports schedule objects)."""
        if self.ent_epsilon is None:
            return None
        if hasattr(self.ent_epsilon, "at"):
            return validate_positive(self.ent_epsilon.at(iteration), "ent_epsilon")
        return validate_positive(self.ent_epsilon, "ent_epsilon")

    @torch.inference_mode()
    def step(
        self,
        state: SolverState,
        generator: Optional[torch.Generator] = None,
    ) -> SolverState:
        """Run one Sinkhorn Step iteration."""
        iteration = state.iteration_count
        X = state.X
        device = X.device

        # Shared with PolyStepOptimizer so a scheduled radius means the same distance via either entry point.
        current_eps = self._get_epsilon(iteration)
        step_radius = resolve_radius(self.step_radius, iteration, current_eps)
        probe_radius = resolve_radius(self.probe_radius, iteration, current_eps)

        polytope_verts = self.polytope_vertices.to(device=device, dtype=X.dtype)
        probes = self.probes.to(device=device, dtype=X.dtype)

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

        if state.subspace is not None and self.nn_evaluator is not None:
            # Subspace mode: reconstruct full params from subspace probes.
            P, V, K, D = X_probe.shape
            flat_probes = X_probe.reshape(P * V * K, D)
            stacked_params = state.subspace.reconstruct_batch(
                state.base_params,
                flat_probes,
            )
            losses = self.nn_evaluator.evaluate(
                stacked_params,
                self.train_inputs,
                self.train_targets,
            )
            cost_matrix = losses.reshape(P, V, K).mean(dim=-1)
        else:
            cost_matrix = compute_cost_matrix(
                self.objective_fn,
                X_probe,
                chunk_size=self.chunk_size,
            )

        # No sanitize here: prepare_cost does it; a second pass would double-penalize.

        ent_eps = self._get_ent_epsilon(iteration)
        ot_epsilon = ent_eps if ent_eps is not None else current_eps

        # Forward the previous solve's epsilon so warm-started duals rescale when the schedule moved.
        ot_solver = self._solver_for(state)
        ot_solver.epsilon = ot_epsilon
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            # a=None lets the solver build the uniform marginal itself; passing a only buys host-sync validation.
            a=None,
            init_f=state.f,
            init_g=state.g,
            scale_cost=self.scale_cost,
        )
        ot_result = ot_solver.solve(**solve_kwargs)
        state.last_solve_eps = ot_epsilon

        transport_matrix = ot_result.matrix  # (batch, num_vertices)
        X_new = self._compiled.barycentric_projection(
            transport_matrix,
            X_vertices,
        )
        ess, rho = solver_health(transport_matrix, X_new - X, step_radius)

        # NaN-safe: revert if X_new has NaN.
        _nan_reverted = not torch.isfinite(X_new).all()
        if _nan_reverted:
            X_new = X.clone()

        disp_sqnorm = torch.mean(torch.sum((X_new - X) ** 2, dim=-1)).item()

        state.X = X_new
        # Raw mean objective, not the regularized dual, so min(costs) is the best objective.
        state.costs.append(cost_matrix.mean().item())
        state.record_solver_health(ess.item(), rho.item(), X_probe[..., 0].numel())
        state.linear_convergence.append(ot_result.converged)
        feed_solver_stats(self.epsilon, ot_solver, ot_result.n_iters, ot_result.converged)
        state.displacement_sqnorms.append(disp_sqnorm)
        state.iteration_count += 1
        if _nan_reverted:
            state.f = None
            state.g = None
        else:
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
        """Run the full outer loop; subspace mode raises (it needs ``init_state``'s ``base_params``)."""
        if self.subspace is not None:
            raise ValueError(
                "PolyStep.run() cannot drive subspace mode: the subspace needs "
                "base_params, which only init_state() accepts. Call "
                "init_state(X_init, base_params=...) and step() in your own loop."
            )
        state = self.init_state(X_init)

        for i in range(self.max_iterations):
            state = self.step(state, generator=generator)

            # i is 0-based, so i + 1 steps have run.
            if i + 1 >= self.min_iterations:
                if self._converged(state) or self._diverged(state):
                    break

        return state
