"""PolyStep solver: gradient-free optimization via entropic OT.

Implements the core Sinkhorn Step algorithm that samples polytope vertices
around particles, solves entropic OT, and updates via barycentric projection.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from ._compiled import CompiledFunctions
from .costs import compute_cost_matrix
from .epsilon import feed_solver_stats, LinearEpsilon
from .geometry import get_random_rotation_matrices, POLYTOPE_MAP
from .solvers import SinkhornSolver, SoftmaxSolver
from .solvers._shared import sanitize_cost, solver_health


@dataclass
class SolverState:
    """State of the Sinkhorn Step solver.

    Most fields are named for what they hold. The ones below carry a convention that
    is not readable off the name:

    ``costs`` is the mean over all ``P * V`` probe vertices, not the entropic dual and
    not the objective at any particle, so ``min(costs)`` is the best step-mean, not a
    best-seen value. ``ess``, ``rho`` and ``evals`` are index-aligned with it.

    ``ess`` is the effective sample size of the transport weights over the vertex
    count: 1.0 means uniform weights, so the barycenter is a plain mean and the
    entropic machinery is doing nothing at that temperature.

    ``rho`` is ``||Delta|| / step_radius``, how far the barycenter moves as a
    fraction of the polytope it came from. Solvers differ in contraction by ``1/rho``,
    so comparing two of them without re-tuning the step size for each is invalid.

    ``evals`` counts candidate evaluations actually spent, net of probe reuse and of
    whatever the screen dropped; multiply by the batch size for sample-forwards.

    ``displacement_history`` is in subspace coordinates and
    ``displacement_history_full`` in parameter space, each entry keeping the basis it
    was measured in: the basis rotates between steps, so re-projecting stored
    coordinates through the current one would mix frames.

    ``p_c``, ``p_sigma``, ``C_diag`` and ``generation`` are the sep-CMA-ES state,
    ``(subspace_dim,)`` each and None until covariance adaptation is enabled.

    ``prev_prev_f`` / ``prev_prev_g`` feed the dual momentum extrapolation
    ``f_init = f + beta * (f - prev_prev_f)``. None until two OT solves have
    completed, and reset on absorb, rotation and epsilon change.
    """

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
        """Append one entry each to ess/rho/evals, keeping them aligned with ``costs``.

        A step with no OT solve passes ``ess=rho=None``, carrying the last value forward.
        """
        self.ess.append(ess if ess is not None else (self.ess[-1] if self.ess else 0.0))
        self.rho.append(rho if rho is not None else (self.rho[-1] if self.rho else 0.0))
        # Rounded, not truncated: a screened step spends a fractional count and
        # truncation loses up to one evaluation per step over a long run.
        self.evals.append(round(evals))


@dataclass
class PolyStep:
    """Batch gradient-free solver for non-convex objectives using entropic OT.

    The solver implements the Sinkhorn Step algorithm: for each iteration, it
    samples polytope vertices (candidate directions) around current particle
    positions, evaluates the objective at each vertex to build a cost matrix,
    solves an entropic optimal transport problem to find an optimal assignment
    between particles and vertices, and moves particles via barycentric
    projection (weighted average of vertices using transport plan weights).

    This is the low-level solver for direct optimization of scalar objectives
    (e.g., Ackley, Rastrigin). For neural network training, use
    ``PolyStepOptimizer`` which wraps this with parameter management and
    closure handling.

    Example::

        from polystep.solver import PolyStep
        from polystep import Ackley

        objective = Ackley(dim=10)
        solver = PolyStep.create(objective, epsilon=0.5, max_iterations=100)
        state = solver.run(torch.randn(50, 10))
        print(f"Best cost: {min(state.costs):.4f}")

    See Also:
        ``PolyStepOptimizer`` for neural network training with automatic
        closure and parameter management.

    Attributes:
        objective_fn: Callable evaluating the objective at points.
        dim: Problem dimensionality.
        polytope_type: Type of polytope ('orthoplex', 'simplex', 'cube').
        epsilon: Entropic regularization (float or LinearEpsilon).
        ent_epsilon: Optional separate entropy for OT cost geometry.
        scale_cost: Cost scaling strategy ('mean', 'max_cost', float, or None).
        step_radius: Step distance multiplier (scaled by epsilon).
        probe_radius: Probe distance multiplier (scaled by epsilon).
        num_probe: Number of probe points per direction.
        max_iterations: Maximum outer iterations.
        min_iterations: Minimum outer iterations before convergence checks.
        threshold: Convergence threshold on relative displacement change.
        sinkhorn_max_iters: Max inner Sinkhorn iterations.
        chunk_size: Chunk size for cost evaluation memory control.
    """

    objective_fn: Callable
    dim: int
    polytope_type: str = "orthoplex"
    epsilon: Union[float, LinearEpsilon] = 0.1
    ent_epsilon: Optional[Union[float, LinearEpsilon]] = None
    scale_cost: Optional[Union[str, float]] = 1.0
    step_radius: float = 1.0
    probe_radius: float = 2.0
    # K=1 matches PolyStepOptimizer's default: multi-probe averaging duplicates the
    # variance reduction entropic regularization already gives.
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
        """Factory method for creating a PolyStep solver.

        Args:
            objective_fn: Objective function (should have .dim attribute if dim not provided).
            dim: Problem dimensionality. If None, inferred from objective_fn.dim.
            **kwargs: Additional PolyStep arguments.

        Returns:
            PolyStep instance.
        """
        if dim is None:
            dim = objective_fn.dim

        return cls(objective_fn=objective_fn, dim=dim, **kwargs)

    def warm_start(self) -> None:
        """Pre-compile all hot paths by running dummy inputs.

        Triggers JIT compilation warmup for both the PolyStep geometry
        functions and the inner Sinkhorn solver iteration. Call before
        benchmarking to exclude compilation time from measurements.
        """
        device = self.polytope_vertices.device
        self._compiled.warm_start(dim=self.dim, device=device)
        self.sinkhorn_solver._compiled.warm_start(dim=self.dim, device=device)

    def init_state(
        self,
        X_init: torch.Tensor,
        base_params: Optional[dict] = None,
    ) -> SolverState:
        """Initialize solver state.

        Args:
            X_init: Initial particle positions of shape (num_particles, dim).
            base_params: Optional base state_dict for subspace mode.
                Required when ``self.subspace`` is set.

        Returns:
            Initial SolverState.
        """
        # Normalize a 1-D point ``(dim,)`` to a single particle ``(1, dim)`` so
        # the source marginal ``a`` is sized by particle count, not by ``dim``
        # (``step()`` unsqueezes the cost matrix to ``(1, V)``, so a length-dim
        # ``a`` would mismatch the Sinkhorn marginal).
        if X_init.dim() == 1:
            X_init = X_init.unsqueeze(0)

        num_points = X_init.shape[0]
        a = torch.ones(num_points, device=X_init.device, dtype=X_init.dtype) / num_points

        # One particle: the column marginal forces a uniform plan, so the step ignores
        # the cost and the point never moves. PolyStepOptimizer swaps solvers here too.
        if num_points == 1 and isinstance(self.sinkhorn_solver, SinkhornSolver):
            warnings.warn(
                "Single particle: balanced Sinkhorn yields a uniform transport plan, so "
                "steps would ignore the cost. Using the one-sided SoftmaxSolver instead.",
                stacklevel=2,
            )
            self.sinkhorn_solver = SoftmaxSolver(epsilon=self.sinkhorn_solver.epsilon)

        state = SolverState(X=X_init.clone(), a=a)

        if self.subspace is not None and base_params is not None:
            state.base_params = base_params
            state.subspace = self.subspace

        return state

    def _get_epsilon(self, iteration: int) -> float:
        """Resolve epsilon at current iteration.

        Duck-types on ``.at()`` so any scheduler (``LinearEpsilon``,
        ``CosineEpsilon``, ``ProgressiveEpsilon``) resolves to a float; a
        plain float passes through.
        """
        if hasattr(self.epsilon, "at"):
            return self.epsilon.at(iteration)
        return self.epsilon

    def _get_ent_epsilon(self, iteration: int) -> Optional[float]:
        """Resolve ent_epsilon at current iteration (supports schedule objects)."""
        if self.ent_epsilon is None:
            return None
        if hasattr(self.ent_epsilon, "at"):
            return self.ent_epsilon.at(iteration)
        return self.ent_epsilon

    @torch.inference_mode()
    def step(
        self,
        state: SolverState,
        generator: Optional[torch.Generator] = None,
    ) -> SolverState:
        """Run one iteration of the Sinkhorn Step algorithm.

        1. Resolve epsilon and scale radii
        2. Sample rotated polytope vertices and probe points
        3. Compute cost matrix from objective evaluations at probes
        4. Solve entropic OT with warm-started duals
        5. Barycentric projection: X_new = sum(vertices * transport_weights)

        Args:
            state: Current solver state.
            generator: Optional random generator for reproducibility.

        Returns:
            Updated SolverState.
        """
        iteration = state.iteration_count
        X = state.X
        device = X.device

        # Resolve epsilon and radii
        current_eps = self._get_epsilon(iteration)
        step_radius = self.step_radius * current_eps
        probe_radius = self.probe_radius * current_eps

        # Move templates to device if needed
        polytope_verts = self.polytope_vertices.to(device=device, dtype=X.dtype)
        probes = self.probes.to(device=device, dtype=X.dtype)

        # Sample polytope and probes
        # Pre-normalize for compiled path (avoid shape branch inside compiled fn)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch, dim = X.shape

        # Generate rotation matrices eagerly (uses torch.Generator, not compilable)
        rot_mats = get_random_rotation_matrices(
            batch,
            dim,
            device=device,
            dtype=X.dtype,
            generator=generator,
        )

        # Compiled rotation + translation
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
            # Subspace mode: reconstruct full params from subspace probes
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

        # Sanitize cost matrix before OT solve. Use the shared branch-free
        # helper (no host sync, FP32 promotion, clamped penalty) rather than an
        # inline copy that .item()-synced and could overflow the penalty to inf.
        cost_matrix = sanitize_cost(cost_matrix)

        ent_eps = self._get_ent_epsilon(iteration)
        ot_epsilon = ent_eps if ent_eps is not None else current_eps

        # Solve entropic OT. Forward the previous solve's epsilon so the
        # solver can rescale the warm-started duals when the schedule moved
        # epsilon: consistent with PolyStepOptimizer's monolithic step,
        self.sinkhorn_solver.epsilon = ot_epsilon
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            # state.a is the uniform marginal the solver builds itself from a=None.
            # Passing it explicitly only buys align_marginal's validation, three host
            # syncs per step, which the other call sites already skip.
            a=None,
            init_f=state.f,
            init_g=state.g,
            scale_cost=self.scale_cost,
        )
        if state.last_solve_eps is not None and isinstance(self.sinkhorn_solver, SinkhornSolver):
            solve_kwargs["init_eps"] = state.last_solve_eps
        ot_result = self.sinkhorn_solver.solve(**solve_kwargs)
        state.last_solve_eps = ot_epsilon

        transport_matrix = ot_result.matrix  # (batch, num_vertices)
        X_new = self._compiled.barycentric_projection(
            transport_matrix,
            X_vertices,
        )
        ess, rho = solver_health(transport_matrix, X_new - X, step_radius)

        # NaN-safe state update - revert if X_new has NaN
        _nan_reverted = not torch.isfinite(X_new).all()
        if _nan_reverted:
            X_new = X.clone()

        disp_sqnorm = torch.mean(torch.sum((X_new - X) ** 2, dim=-1)).item()

        state.X = X_new
        # The raw mean objective, not the OT-regularized dual, so ``min(state.costs)``
        # is the best objective value. The dual stays available as ``ent_reg_cost``.
        state.costs.append(cost_matrix.mean().item())
        state.record_solver_health(ess.item(), rho.item(), X_probe[..., 0].numel())
        state.linear_convergence.append(ot_result.converged)
        feed_solver_stats(self.epsilon, self.sinkhorn_solver, ot_result.n_iters, ot_result.converged)
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
        """Converged when the displacement has both settled and gone small.

        The relative test alone accepts a constant-speed trajectory: ``[1, 1, 1]``
        looks settled while the particle is still crossing one step per iteration.
        """
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
        """Run the full Sinkhorn Step outer loop.

        Args:
            X_init: Initial particle positions of shape (num_particles, dim).
            generator: Optional random generator for reproducibility.

        Returns:
            Final SolverState with converged particles.

        Raises:
            ValueError: If a subspace is configured. Subspace mode needs the base
                state_dict, which only ``init_state(X_init, base_params=...)``
                accepts; running here would silently optimize the plain objective.
        """
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
