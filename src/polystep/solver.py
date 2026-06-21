"""PolyStep solver: gradient-free optimization via entropic OT.

Implements the core Sinkhorn Step algorithm that samples polytope vertices
around particles, solves entropic OT, and updates via barycentric projection.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from ._compiled import CompiledFunctions
from .costs import compute_cost_matrix
from .epsilon import LinearEpsilon
from .geometry import get_random_rotation_matrices, POLYTOPE_MAP
from .solvers import SinkhornSolver


@dataclass
class SolverState:
    """State of the Sinkhorn Step solver.

    Attributes:
        X: Current particle positions of shape (num_particles, dim).
        costs: Per-iteration entropic OT costs.
        linear_convergence: Per-iteration convergence flags.
        displacement_sqnorms: Per-iteration mean squared displacements.
        a: Source marginal weights.
        iteration_count: Number of completed iterations.
        f: Dual potential for warm-start.
        g: Dual potential for warm-start.
        epsilon: Current epsilon value (for diagnostics).
        base_params: Base state_dict for subspace mode (None when unused).
        subspace: LowRankSubspace instance for subspace mode (None when unused).
        block_duals: Per-block dual potentials for warm-start in block-wise mode.
            List of (f_block, g_block) tuples, one per block. None when unused.
        velocity: Momentum velocity tensor, same shape as X. None until first momentum step.
        stagnation_count: Consecutive iterations with small relative loss change.
        radius_multiplier: Adaptive radius scaling factor applied to step_radius.
        prev_loss: Previous iteration loss for stagnation detection.
        projection: Orthonormal projection matrix for adaptive subspace mode.
            Shape (full_dim, subspace_dim). None when not using AdaptiveSubspace.
        displacement_history: Rolling buffer of recent subspace displacement vectors.
            Shape (history_size, subspace_dim). None when not using AdaptiveSubspace.
        displacement_history_idx: Current write position in the rolling buffer.
        displacement_history_count: Number of entries filled (0..history_size).
        absorb_count: Total number of absorb events performed.
        p_c: Evolution path for covariance adaptation.
            Shape (subspace_dim,). Tracks cumulative displacement direction over
            generations, used to update the covariance matrix. None until CMA enabled.
        p_sigma: Evolution path for step-size adaptation.
            Shape (subspace_dim,). Tracks cumulative step length for CSA
            (Cumulative Step-size Adaptation). None until CMA enabled.
        C_diag: Diagonal covariance for sep-CMA-ES.
            Shape (subspace_dim,). Scales per-dimension search variance.
            Full covariance would be C = diag(C_diag) in the diagonal case.
            None until CMA enabled.
        sigma: Global step-size controlled by CSA.
            Replaces radius_multiplier when use_csa=True.
        generation: CMA generation counter for hyperparameter scheduling.
            Used in the Heaviside function computation for evolution path updates.
        use_csa: Flag indicating CSA mode is active.
            When True, sigma replaces the heuristic radius_multiplier.
        prev_prev_f: Previous-previous dual potential f for dual momentum extrapolation.
            Used with dual_momentum_beta > 0 to compute f_init = f + beta*(f - prev_prev_f).
            None until at least 2 OT solves have completed. Reset on absorb/rotation/epsilon change.
        prev_prev_g: Previous-previous dual potential g (matching prev_prev_f).
    """

    X: torch.Tensor
    costs: List[float] = field(default_factory=list)
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
    prev_loss: float = float('inf')
    # Adaptive subspace state
    projection: Optional[torch.Tensor] = None
    displacement_history: Optional[torch.Tensor] = None
    displacement_history_idx: int = 0
    displacement_history_count: int = 0
    absorb_count: int = 0
    # CMA-ES state
    p_c: Optional[torch.Tensor] = None
    p_sigma: Optional[torch.Tensor] = None
    C_diag: Optional[torch.Tensor] = None
    sigma: float = 1.0
    generation: int = 0
    use_csa: bool = False
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
        from polystep.costs import Ackley

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
    polytope_type: str = 'orthoplex'
    epsilon: Union[float, LinearEpsilon] = 0.1
    ent_epsilon: Optional[Union[float, LinearEpsilon]] = None
    scale_cost: Optional[Union[str, float]] = 1.0
    step_radius: float = 1.0
    probe_radius: float = 2.0
    # K=1 is empirically optimal for the softmax solver and is what
    # every headline runner uses; matches PolyStepOptimizer's default.
    # Multi-probe averaging adds variance reduction that the entropic
    # regularization already provides.
    num_probe: int = 1
    max_iterations: int = 50
    min_iterations: int = 5
    threshold: float = 1e-3
    sinkhorn_max_iters: int = 2000
    chunk_size: Optional[int] = None
    compile: bool = True
    subspace: Optional[object] = None
    nn_evaluator: Optional[object] = None
    layout: Optional[object] = None
    train_inputs: Optional[object] = None
    train_targets: Optional[object] = None
    block_strategy: str = 'monolithic'
    block_group_size: int = 2

    def __post_init__(self):
        """Initialize derived state: polytope template, probes, solver, compiled fns."""
        # Validate: subspace + block-wise not yet supported together
        if self.subspace is not None and self.block_strategy != 'monolithic':
            raise NotImplementedError(
                "Combined subspace + block-wise mode is not yet supported. "
                "Use subspace or block_strategy independently."
            )

        self.polytope_vertices = POLYTOPE_MAP[self.polytope_type](self.dim, radius=1.0)
        self.probes = torch.linspace(0, 1, self.num_probe + 2)[1:self.num_probe + 1]
        self.sinkhorn_solver = SinkhornSolver(
            max_iterations=self.sinkhorn_max_iters,
            compile=self.compile,
        )
        self._compiled = CompiledFunctions(
            compile=self.compile and torch.cuda.is_available()
        )

        # Create blocks if block-wise mode
        self._blocks = None
        if self.block_strategy != 'monolithic' and self.layout is not None:
            from .blockwise import create_per_layer_blocks, create_grouped_blocks
            if self.block_strategy == 'per_layer':
                self._blocks = create_per_layer_blocks(
                    self.layout, particle_dim=self.layout.particle_dim,
                )
            elif self.block_strategy == 'grouped':
                self._blocks = create_grouped_blocks(
                    self.layout,
                    group_size=self.block_group_size,
                    particle_dim=self.layout.particle_dim,
                )
            else:
                raise ValueError(
                    f"Unknown block_strategy: {self.block_strategy!r}. "
                    f"Use 'monolithic', 'per_layer', or 'grouped'."
                )

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
        num_points = X_init.shape[0]
        a = torch.ones(num_points, device=X_init.device, dtype=X_init.dtype) / num_points

        state = SolverState(X=X_init.clone(), a=a)

        if self.subspace is not None and base_params is not None:
            state.base_params = base_params
            state.subspace = self.subspace

        # Initialize per-block dual potentials if block-wise mode
        if self._blocks is not None:
            state.block_duals = [(None, None) for _ in self._blocks]

        return state

    def _get_epsilon(self, iteration: int) -> float:
        """Resolve epsilon at current iteration."""
        if isinstance(self.epsilon, LinearEpsilon):
            return self.epsilon.at(iteration)
        return self.epsilon

    def _get_ent_epsilon(self, iteration: int) -> Optional[float]:
        """Resolve ent_epsilon at current iteration."""
        if self.ent_epsilon is None:
            return None
        if isinstance(self.ent_epsilon, LinearEpsilon):
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

        # 1. Resolve epsilon and radii
        current_eps = self._get_epsilon(iteration)
        step_radius = self.step_radius * current_eps
        probe_radius = self.probe_radius * current_eps

        # Block-wise mode: independent OT solve per block
        if self._blocks is not None and self.nn_evaluator is not None:
            return self._step_blockwise(state, generator, current_eps,
                                         step_radius, probe_radius)

        # --- Monolithic mode (original path) ---

        # Move templates to device if needed
        polytope_verts = self.polytope_vertices.to(device=device, dtype=X.dtype)
        probes = self.probes.to(device=device, dtype=X.dtype)

        # 2. Sample polytope and probes
        # Pre-normalize for compiled path (avoid shape branch inside compiled fn)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        batch, dim = X.shape

        # Generate rotation matrices eagerly (uses torch.Generator, not compilable)
        rot_mats = get_random_rotation_matrices(
            batch, dim, device=device, dtype=X.dtype, generator=generator,
        )

        # Compiled rotation + translation
        X_vertices, rotated = self._compiled.rotate_and_translate(
            rot_mats, polytope_verts, X, step_radius,
        )

        # Compiled probe generation
        X_probe = self._compiled.compute_probe_points(
            X, rotated, probes, probe_radius,
        )

        # 3. Compute cost matrix
        if state.subspace is not None and self.nn_evaluator is not None:
            # Subspace mode: reconstruct full params from subspace probes
            P, V, K, D = X_probe.shape
            flat_probes = X_probe.reshape(P * V * K, D)
            stacked_params = state.subspace.reconstruct_batch(
                state.base_params, flat_probes,
            )
            losses = self.nn_evaluator.evaluate(
                stacked_params, self.train_inputs, self.train_targets,
            )
            cost_matrix = losses.reshape(P, V, K).mean(dim=-1)
        else:
            cost_matrix = compute_cost_matrix(
                self.objective_fn, X_probe, chunk_size=self.chunk_size,
            )

        # Sanitize cost matrix before OT solve
        if not torch.isfinite(cost_matrix).all():
            max_finite = cost_matrix[torch.isfinite(cost_matrix)]
            penalty = max_finite.abs().max().item() * 2.0 + 1.0 if max_finite.numel() > 0 else 1e6
            cost_matrix = torch.where(
                torch.isfinite(cost_matrix), cost_matrix,
                torch.full_like(cost_matrix, penalty),
            )

        # 4. Resolve OT epsilon
        ent_eps = self._get_ent_epsilon(iteration)
        ot_epsilon = ent_eps if ent_eps is not None else current_eps

        # 5. Solve entropic OT. Forward the previous solve's epsilon so the
        # solver can rescale the warm-started duals when the schedule moved
        # epsilon -- consistent with PolyStepOptimizer's monolithic step,
        # which previously was the only path that passed init_eps.
        self.sinkhorn_solver.epsilon = ot_epsilon
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            a=state.a,
            init_f=state.f,
            init_g=state.g,
            scale_cost=self.scale_cost,
        )
        if state.last_solve_eps is not None:
            solve_kwargs["init_eps"] = state.last_solve_eps
        ot_result = self.sinkhorn_solver.solve(**solve_kwargs)
        state.last_solve_eps = ot_epsilon

        # 6. Barycentric projection (compiled)
        transport_matrix = ot_result.matrix  # (batch, num_vertices)
        X_new = self._compiled.barycentric_projection(
            transport_matrix, state.a, X_vertices,
        )

        # NaN-safe state update - revert if X_new has NaN
        _nan_reverted = not torch.isfinite(X_new).all()
        if _nan_reverted:
            X_new = X.clone()

        # 7. Update state
        disp_sqnorm = torch.mean(torch.sum((X_new - X) ** 2, dim=-1)).item()

        state.X = X_new
        state.costs.append(ot_result.ent_reg_cost)
        state.linear_convergence.append(ot_result.converged)
        state.displacement_sqnorms.append(disp_sqnorm)
        state.iteration_count += 1
        if _nan_reverted:
            state.f = None
            state.g = None
        else:
            state.f = ot_result.f.detach()
            state.g = ot_result.g.detach()
        state.epsilon = current_eps

        return state

    @torch.inference_mode()
    def _step_blockwise(
        self,
        state: SolverState,
        generator: Optional[torch.Generator],
        current_eps: float,
        step_radius: float,
        probe_radius: float,
    ) -> SolverState:
        """Run one block-wise Sinkhorn Step iteration.

        Each block gets its own polytope sampling, cost evaluation (full model
        forward with only the current block perturbed), independent OT solve
        with per-block warm-started duals, and barycentric projection.

        Args:
            state: Current solver state.
            generator: Optional random generator for reproducibility.
            current_eps: Current epsilon value.
            step_radius: Step radius (already scaled by epsilon).
            probe_radius: Probe radius (already scaled by epsilon).

        Returns:
            Updated SolverState.
        """
        from .blockwise import split_particles, reassemble_blocks, compute_block_cost_matrix
        from .blockwise import layout_flat_to_block_flat, blocks_to_layout_flat

        X = state.X
        device = X.device
        blocks = self._blocks

        # Convert layout-indexed flat to block-indexed before splitting.
        total_flat_size = sum(b.flat_end - b.flat_start for b in blocks)
        block_flat = layout_flat_to_block_flat(
            X.reshape(-1), blocks, self.layout,
        )
        block_X_2d = block_flat.reshape(-1, X.shape[-1]) if X.dim() > 1 else block_flat
        all_block_particles = split_particles(block_X_2d, blocks)

        # Resolve OT epsilon
        ent_eps = self._get_ent_epsilon(state.iteration_count)
        ot_epsilon = ent_eps if ent_eps is not None else current_eps

        updated_block_particles = []
        new_block_duals = []
        total_ent_cost = 0.0
        all_converged = True
        total_disp = 0.0
        total_particles = 0

        # Python loop over blocks (sequential per-block OT solves)
        for block_idx, block in enumerate(blocks):
            block_X = all_block_particles[block_idx]
            block_dim = block.particle_dim

            # Per-block polytope template and probes
            block_polytope_verts = POLYTOPE_MAP[self.polytope_type](
                block_dim, device=device, dtype=X.dtype, radius=1.0,
            )
            block_probes = self.probes.to(device=device, dtype=X.dtype)

            P_block = block.num_particles
            if block_X.dim() == 1:
                block_X = block_X.unsqueeze(0)

            # Rotation matrices for this block
            rot_mats = get_random_rotation_matrices(
                P_block, block_dim, device=device, dtype=X.dtype,
                generator=generator,
            )

            # Rotate and translate
            X_vertices, rotated = self._compiled.rotate_and_translate(
                rot_mats, block_polytope_verts, block_X, step_radius,
            )

            # Probe generation
            X_probe = self._compiled.compute_probe_points(
                block_X, rotated, block_probes, probe_radius,
            )

            # Block cost: full forward, perturb only this block
            cost_matrix = compute_block_cost_matrix(
                block_idx=block_idx,
                X_probe_block=X_probe,
                all_block_particles=all_block_particles,
                blocks=blocks,
                layout=self.layout,
                evaluator=self.nn_evaluator,
                inputs=self.train_inputs,
                targets=self.train_targets,
            )

            # Per-block OT solve with warm-started duals
            self.sinkhorn_solver.epsilon = ot_epsilon
            init_f, init_g = state.block_duals[block_idx]
            ot_result = self.sinkhorn_solver.solve(
                cost_matrix=cost_matrix,
                a=torch.ones(P_block, device=device, dtype=X.dtype) / P_block,
                init_f=init_f,
                init_g=init_g,
                scale_cost=self.scale_cost,
            )

            # Barycentric projection for this block
            transport_matrix = ot_result.matrix
            block_a = torch.ones(P_block, device=device, dtype=X.dtype) / P_block
            X_new_block = self._compiled.barycentric_projection(
                transport_matrix, block_a, X_vertices,
            )

            # Track per-block displacement
            block_disp = torch.sum((X_new_block - block_X) ** 2, dim=-1).sum().item()
            total_disp += block_disp
            total_particles += P_block

            updated_block_particles.append(X_new_block)
            new_block_duals.append((ot_result.f.detach(), ot_result.g.detach()))
            total_ent_cost += ot_result.ent_reg_cost
            all_converged = all_converged and ot_result.converged

        # Reassemble and convert back to layout-indexed format
        full_flat = reassemble_blocks(updated_block_particles, blocks, total_flat_size)
        layout_flat_new = blocks_to_layout_flat(full_flat, blocks, self.layout)
        X_new = layout_flat_new.reshape(X.shape)

        # Update state
        disp_sqnorm = total_disp / total_particles if total_particles > 0 else 0.0

        state.X = X_new
        state.costs.append(total_ent_cost)
        state.linear_convergence.append(all_converged)
        state.displacement_sqnorms.append(disp_sqnorm)
        state.iteration_count += 1
        state.block_duals = new_block_duals
        state.epsilon = current_eps

        return state

    def _converged(self, state: SolverState) -> bool:
        """Check if the solver has converged based on displacement change."""
        it = state.iteration_count
        if it < 3:
            return False
        d = state.displacement_sqnorms
        return abs(d[-1] - d[-2]) / (abs(d[-2]) + 1e-10) < self.threshold

    def _diverged(self, state: SolverState) -> bool:
        """Check if the solver has diverged (non-finite cost)."""
        if not state.costs:
            return False
        import math
        return not math.isfinite(state.costs[-1])

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
        """
        state = self.init_state(X_init)

        for i in range(self.max_iterations):
            state = self.step(state, generator=generator)

            if i >= self.min_iterations:
                if self._converged(state) or self._diverged(state):
                    break

        return state
