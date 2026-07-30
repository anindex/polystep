"""PolyStepOptimizer: user-facing wrapper with closure-based step interface.

Composes low-level geometry, compiled functions, Sinkhorn solver, and dynamics
(momentum + adaptive radius) into a clean step(closure) API. The optimizer
manages model parameter synchronization, state tracking, and optional
subspace/block-wise decomposition.

Multi-particle architecture: Model parameters are reshaped into
(num_particles, particle_dim) where particle_dim is typically 2. Each particle
is an independent unit in the OT problem. The polytope operates in
particle_dim space (e.g., 4 orthoplex vertices in 2D), giving a tractable
OT problem of shape (num_particles, num_vertices).
"""

from __future__ import annotations

import collections
import logging
import os
import warnings
from dataclasses import dataclass, fields as dataclass_fields
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .cost_nn import NNCostEvaluator

logger = logging.getLogger(__name__)

INF = float("inf")

# Minimum params for sparse projection (below this, dense is more efficient)
_MIN_PARAMS_FOR_SPARSE = 10_000

# Threshold for auto-selecting sparse vs dense (device-dependent)
_AUTO_SPARSE_THRESHOLD_CPU = 1_000_000  # 1M params for CPU
_AUTO_SPARSE_THRESHOLD_GPU = 2_000_000  # 2M params for GPU


def _select_projection_type(
    num_params: int,
    device: torch.device,
    projection_type: str,
) -> str:
    """Determine whether to use dense or sparse projection.

    Dense projection wins on small models, where setup dominates. Sparse wins on
    large ones, where the dense matrix would be O(num_params * num_coords).

    Args:
        num_params: Total model parameters.
        device: Model device (CPU or CUDA).
        projection_type: User preference ('dense', 'sparse', 'auto').

    Returns:
        'dense' or 'sparse' (resolved projection type).
    """
    # Explicit user choice
    if projection_type in ("dense", "sparse"):
        return projection_type

    # Auto-selection based on device and model size
    if device.type == "cuda":
        threshold = _AUTO_SPARSE_THRESHOLD_GPU
    else:
        threshold = _AUTO_SPARSE_THRESHOLD_CPU

    if num_params >= threshold:
        return "sparse"
    else:
        return "dense"


from ._compiled import CompiledFunctions
from .blockwise import (
    BlockConfig,
    create_grouped_blocks,
    create_per_layer_blocks,
    create_subspace_blocks,
)
from .epsilon import LinearEpsilon
from .geometry import POLYTOPE_MAP
from .solvers import (
    KLSoftmaxSolver,
    MinCostGreedySolver,
    SinkhornSolver,
    SoftmaxSolver,
    TemperedSoftmaxSolver,
    TopKMeanSolver,
)
from .solver import SolverState
from .adaptive_subspace import AdaptiveSubspace
from .cma import compute_cma_hyperparameters
from .cma_subspace import CMAAdaptiveSubspace
from .subspace import LowRankSubspace, LinearSubspace
from .factored_subspace import FactoredSubspace
from .hybrid_subspace import HybridSubspace
from ._serialization import SerializationMixin
from .transform import ParamLayout, create_generator

from ._step_monolithic import step_monolithic as _step_monolithic_fn
from ._step_blockwise import step_blockwise as _step_blockwise_fn
from ._step_blockwise import step_subspace_blockwise as _step_subspace_blockwise_fn
from ._step_momentum import step_momentum as _step_momentum_fn

_THREAD_WARNING_ISSUED = False


def _warn_thread_oversubscription() -> None:
    """Warn once when the intra-op pool is sized to saturate every core.

    A step is thousands of small forwards, so the OpenMP barrier is hit constantly. With
    the pool at or one below the core count, the pool threads and the main thread
    oversubscribe and the spin-wait takes over. The cost depends on core count and on
    what else the machine is running; see docs/performance.md for a measured table.
    """
    global _THREAD_WARNING_ISSUED
    if _THREAD_WARNING_ISSUED:
        return
    ncpu = os.cpu_count() or 1
    threads = torch.get_num_threads()
    if ncpu >= 4 and threads >= ncpu - 1:
        _THREAD_WARNING_ISSUED = True
        warnings.warn(
            f"torch.get_num_threads()={threads} on {ncpu} cores saturates the intra-op "
            f"pool, where OpenMP spin-wait dominates PolyStep's many small forwards. "
            f"Call torch.set_num_threads({max(1, ncpu - 8)}) "
            f"before building the optimizer, or set OMP_WAIT_POLICY=PASSIVE. "
            f"See docs/performance.md.",
            stacklevel=3,
        )


@dataclass
class RankSchedule:
    """Progressive rank expansion schedule for subspace optimization.

    Maps step number to target rank. Rank increases at specified steps,
    triggering absorb + subspace reconstruction.

    Example::

        schedule = RankSchedule(stages=[(0, 2), (100, 4), (300, 8)])
        # rank=2 for steps 0-99, rank=4 for 100-299, rank=8 for 300+
    """

    stages: List[Tuple[int, int]]  # (start_step, rank) pairs

    def __post_init__(self):
        # Sort by start_step
        self.stages = sorted(self.stages, key=lambda x: x[0])
        if not self.stages:
            raise ValueError("RankSchedule requires at least one stage")
        if self.stages[0][0] != 0:
            raise ValueError("First stage must start at step 0")
        for _, rank in self.stages:
            if rank < 1:
                raise ValueError(f"Rank must be >= 1, got {rank}")

    def at(self, step: int) -> int:
        """Return rank at given step."""
        current_rank = self.stages[0][1]
        for start_step, rank in self.stages:
            if step >= start_step:
                current_rank = rank
        return current_rank

    def transitions(self) -> List[int]:
        """Return step numbers where rank changes (excluding step 0)."""
        return [s for s, _ in self.stages if s > 0]


class PolyStepOptimizer(SerializationMixin):
    """Gradient-free optimizer via entropic optimal transport.

    Wraps polytope sampling, OT solve, barycentric projection, momentum,
    and adaptive radius into a single ``step(closure)`` interface. Updates
    the model weights in-place after each step.

    Each step solves an entropically regularized transport problem between the
    particles and the polytope vertices around them, then moves every particle to the
    transport-weighted mean of its vertices. Higher ``epsilon`` smooths the plan.
    Dual potentials are warm-started across iterations.

    This is a standalone class (NOT a ``torch.optim.Optimizer`` subclass)
    because Sinkhorn Step is gradient-free and uses neither parameter groups nor
    ``zero_grad()``.

    **Multi-particle architecture:** Model parameters are laid out as
    ``(num_particles, particle_dim)`` where ``particle_dim`` defaults to 2.
    The polytope operates in ``particle_dim`` space (e.g., orthoplex in 2D
    has 4 vertices), producing a tractable OT problem. For each probe
    evaluation, the full model parameters are reconstructed from the
    particle array with one row replaced by the probe position.

    Example::

        import torch
        import torch.nn as nn
        from polystep import PolyStepOptimizer

        model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        optimizer = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.15)

        # Define a closure that evaluates loss at batched parameter configs
        def closure(batched_params):
            # batched_params: {key: (N, *shape)}, N candidate param sets
            # Return losses tensor of shape (N,)
            ...

        cost = optimizer.step(closure)

    See Also:
        ``train()`` for a high-level training loop that builds closures
        automatically from a DataLoader and loss function.
        ``TrainConfig`` for training loop configuration.

    Args:
        model: The ``nn.Module`` to optimize. Weights are updated in-place.
        polytope_type: Polytope template ('simplex', 'orthoplex', 'cube').
            Default ``'simplex'``: ``k+1`` vertices, the minimum positive spanning set of
            ``R^k`` (Davis 1954), so the smallest probe set that can still guarantee a
            descent direction. The ``2k``-vertex orthoplex spends its extra ``k-1``
            vertices cancelling the curvature term the simplex leaks.

            Take the orthoplex when ``cur/|lin|`` at your probe radius is ``O(1)`` or
            larger, or for a feature that reads its antithetic vertex ordering:
            ``use_quadratic_model``, ``newton_refinement``, ``trust_region``, and the
            contrast-ranked ``multifidelity_screen``. The constructor warns if one of
            those is enabled on another polytope. Otherwise the simplex is the cheaper
            step at equal accuracy. See ``docs/performance.md``.
        epsilon: Entropic regularization: a float, or any scheduler exposing ``.at()``
            (``LinearEpsilon``, ``CosineEpsilon``, ``ProgressiveEpsilon``). A
            ``ProgressiveEpsilon`` here is fed solver stats exactly as ``auto_epsilon``
            does, so it needs an iterative solver.

            ``epsilon`` is overloaded. With float ``step_radius``/``probe_radius`` it
            multiplies both (the geometry scale), and it is also the softmax/Sinkhorn
            temperature when ``ent_epsilon is None``. Polytopes are mean-centred, so
            ``Delta/r = sum_j (W_j - 1/V) u_j ~= -(1/(V*eps)) sum_j (C_j - Cbar) u_j``.
            With ``sum_j u_j u_j^T = (V/d) I`` over the particle dimension ``d``, that
            gives ``Delta = -(step_radius * probe_radius * sbar * eps / d) * g`` to first
            order, where ``sbar`` is the mean probe scale (0.5 at ``num_probe=1``).
            ``step_radius * probe_radius * epsilon`` is therefore the learning rate.

            So with float radii, sweeping ``epsilon`` alone mostly moves step size, not
            weight concentration. With ``rho = ||Delta||/r`` and
            ``sigma_C ~ probe_radius^alpha``, ``d log rho / d log eps = alpha - 1`` coupled
            versus ``-1`` decoupled. ``alpha = 1`` for smooth objectives, so there
            ``epsilon`` is inert on concentration; for piecewise-constant objectives
            (sign/quantized/spiking nets) threshold crossings add incoherently and
            ``alpha ~ 1/2``, so it is weak. Set ``ent_epsilon`` to control temperature
            independently, or use scheduled radii, which epsilon does not multiply.
        ent_epsilon: Separate OT solver epsilon. If None, uses epsilon, which couples the
            entropic temperature to the probe/step geometry. See ``epsilon``.
        scale_cost: Cost scaling strategy ('mean', 'max_cost', float, or None).
        step_radius: Base step radius multiplied by epsilon (see ``epsilon``).
            Measured in subspace coordinates, and the subspace classes do not share a
            normalization. ``LinearSubspace`` amplifies by
            ``sqrt(num_params / num_coords)`` (4.38 on a 784-256-10 MLP at rank 8),
            while ``HybridSubspace`` and ``FactoredSubspace`` are unit gain. The same
            ``step_radius`` is therefore a ~4.4x different weight-space step across
            classes, and switching class needs it retuned.
        probe_radius: Base probe radius multiplied by epsilon (see ``epsilon``).
        num_probe: Number of probe points per direction (default 1).
            K=1 is optimal: multi-probe averaging is redundant when entropic
            regularization is active, and K forward passes per direction drop to one
            at no accuracy cost.
        adaptive_probes: Reuse the previous step's cost row for stagnant particles
            (small displacement) instead of recomputing it, saving ``V * K`` forward
            passes each. ``None`` (default) enables it wherever it is implemented,
            which is ``block_strategy='monolithic'``. Reuse requires the step and probe
            radii to be unchanged since the cached row was measured, so it does not
            fire under ``use_adaptive_radius`` or ``probe_radius_jitter > 0``.
        adaptive_probes_threshold: Displacement squared norm below which a
            particle is considered stagnant (default ``1e-6``).
        max_iterations: Maximum outer iterations (for momentum warmup schedule).
        sinkhorn_max_iters: Maximum inner Sinkhorn iterations.
        chunk_size: Chunk size for cost evaluation memory control.
        cost_batch_size: Optional mini-batch size for cost matrix evaluation.
            stored for the training loop closure to read; not used
            internally by the optimizer.
        compile: Whether to compile hot-path tensor functions. Defaults to
            False because ablation experiments show no measurable end-to-end
            speedup while incurring JIT warm-up overhead.
        solver: OT/weighting solver strategy. 'softmax' for direct softmax
            weighting (fast, no dual potentials), 'sinkhorn' for entropic OT
            solver (iterative, with warm-started duals), 'kl_softmax' for the
            one-sided KL-penalized interpolation between the two (``kl_softmax_lam=0``
            is softmax, ``inf`` is Sinkhorn), 'min_cost_greedy' and 'top_k_mean' for
            selection rather than weighting, 'tempered_softmax' for a softmax at a
            fixed temperature the epsilon schedule does not touch. None (default) auto-selects: softmax for subspace modes,
            sinkhorn for full-space.
            ProgressiveEpsilon (auto_epsilon=True) is incompatible with softmax.

            Solvers differ in contraction by ``1/rho``, where ``rho = ||Delta||/r`` is
            the fraction of the step radius the step travels: a selection solver has
            ``rho = 1`` by construction, a hot softmax far less. Swapping solvers without
            re-sweeping ``step_radius`` measures the step size, not the solver.
            ``get_diagnostics`` reports ``rho`` and ``ess`` per step.
        tempered_softmax_tau: Fixed temperature for ``solver='tempered_softmax'``, a
            divisor on the cost. Lower is sharper; at the default 1.0 against an
            epsilon of 0.1 it is flatter than the plain softmax, not sharper.
        kl_softmax_lam: Column-marginal KL penalty for ``solver='kl_softmax'``.
            ``0`` reduces it to softmax, ``inf`` (default) to Sinkhorn.
        subspace: Subspace object for subspace mode. ``HybridSubspace`` is the
            recommended choice; ``AdaptiveSubspace``, ``CMAAdaptiveSubspace``,
            ``LinearSubspace`` and ``LowRankSubspace`` are also accepted. None runs in
            full parameter space.
        subspace_particle_dim: Particle dimension for subspace mode (default 8).
            In subspace mode, this overrides ``particle_dim`` for the OT
            polytope dimension. Use this parameter (not ``particle_dim``)
            to control polytope geometry in subspace experiments.
            Higher values give more vertices (2*dim for orthoplex) and stronger
            per-step signal. Only used when subspace is not None.
        absorb_every: Periodic absorb interval (default 0 = disabled). When > 0, folds
            the current perturbation into the base weights every N steps, zeroing the
            subspace vector to explore new regions.

            Applies only to ``LowRankSubspace`` and ``LinearSubspace``, and only with
            ``block_strategy='monolithic'``. ``AdaptiveSubspace``, ``HybridSubspace`` and
            ``CMAAdaptiveSubspace`` run their own absorb schedule, so set ``absorb_mode``
            and ``absorb_interval`` on the subspace object instead; ``absorb_every`` has
            no effect for them.
        block_strategy: 'monolithic', 'per_layer', or 'grouped'.
        block_group_size: Number of consecutive entries per block group.
        biased_rotation: Replace the chart's first axis with the previous step's descent
            direction instead of leaving the rotation fully Haar (default False). Two
            mechanisms sit behind this flag and ``num_probe`` picks which. With
            ``use_quadratic_model=True``, ``num_probe >= 2`` and
            ``polytope_type='orthoplex'`` the direction is the finite-difference gradient
            from the ``(P, V, K)`` loss tensor. Otherwise, including at the default
            ``num_probe=1`` where that model cannot be built, it is the OT barycenter's
            displacement; the step warns once when it takes that path. Either way this is
            rank-1 subspace alignment, not preconditioning.
        use_momentum: Enable momentum velocity accumulation (default False). The velocity
            lives in the subspace basis, so absorb zeroes it and the averaging window
            cannot exceed ``absorb_interval``. It also multiplies the effective step by
            ``1/(1-beta)`` while improving the direction only by ``sqrt(1/(1-beta))``, so
            ``step_radius`` has to be re-tuned per ``momentum_final`` or the comparison
            measures the step size instead.
        momentum_init: Starting momentum coefficient.
        momentum_final: Final momentum coefficient, reached linearly over ``max_iterations``.
        velocity_lr: Learning rate for velocity update.
        use_adaptive_radius: Enable stagnation-based radius adaptation. The stagnation
            counter is tracked either way, since ``absorb_mode="stagnation"`` reads it;
            this flag only controls whether the radius reacts. A radius boost consumes the
            counter, so with ``stagnation_patience < absorb_patience`` the boost fires
            first and stagnation absorb never happens. The constructor warns.
        stagnation_threshold: Relative change below which is stagnation.
        stagnation_patience: Stagnation iterations before radius boost.
        multifidelity_screen: Two-stage probe evaluation (default False, orthoplex
            only). Stage 1 evaluates every direction on a cheap fidelity (a
            ``screen_fidelity`` slice of the batch) and ranks them by
            ``|L(+e_i) - L(-e_i)|``. Stage 2 spends the full fidelity only on the top
            ``screen_keep_ratio`` directions, keeping both signs of each so the
            orthoplex stays antithetic. Dropped vertices keep their cheap value plus a
            per-particle offset calibrated on the kept ones.

            Skipped, with a warning, unless
            ``screen_fidelity/num_probe + screen_keep_ratio < 1``; above that it buys
            work rather than saving it. Also skipped while ``use_quadratic_model``,
            ``newton_refinement`` or ``trust_region`` is on, since those need a full
            single-fidelity ``(P, V, K)`` loss tensor. Requires a cheap closure: pass
            ``screen_closure`` to :meth:`step` (see :meth:`screen_closure_from`), or
            use ``api.train``, which builds one. ``docs/performance.md`` has the
            measured wall-clock crossover.
        screen_keep_ratio: Fraction of directions promoted to full fidelity by
            ``multifidelity_screen`` (default 0.5).
        screen_fidelity: Fraction of the batch used for the screening pass
            (default 0.25).
        radius_increase: Multiplicative factor for radius boost.
        radius_decrease: Multiplicative factor for radius decay.
        radius_min: Minimum allowed radius multiplier.
        radius_max: Maximum allowed radius multiplier.
        use_covariance_adaptation: Enable diagonal CMA-ES covariance learning.
            Requires CMAAdaptiveSubspace. Learns per-dimension scaling of the search
            distribution. The diagonal is renormalized to mean 1 each step, so it
            redistributes step size across coordinates without changing the overall
            scale, which stays owned by the radii.
        seed: Optional seed for reproducible random rotations.
        mixed_precision: Cast the model, particles and projections to BF16, keeping
            the Sinkhorn internals in FP32. Halves weight memory. Because the
            parameters themselves are BF16, a candidate perturbation below their
            resolution rounds away; use ``candidate_autocast`` to keep FP32 masters.
            Default False.
        candidate_autocast: Run the candidate forward's arithmetic in BF16 under
            ``torch.amp.autocast`` while the parameters stay at their own dtype. The
            delta evaluators are excluded: their correction is a small offset on a
            full-scale output, which half precision erases. Default False.
        projection_type: Type of projection for AdaptiveSubspace mode.
            'dense' uses QR-orthogonalized dense matrices (default).
            'sparse' uses SparseRandomProjection for memory efficiency.
            'auto' will auto-select based on model size.

    See ``docs/performance.md`` for starting hyperparameters per subspace.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        polytope_type: str = "simplex",
        particle_dim: int = 2,
        epsilon: Union[float, LinearEpsilon] = 0.1,
        ent_epsilon: Optional[Union[float, LinearEpsilon]] = None,
        # Auto-epsilon: ProgOT-inspired feedback-driven epsilon
        auto_epsilon: bool = False,
        auto_epsilon_config: Optional[dict] = None,
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        # Theorem 4.2 condition (iv): eta_t ~ U[-eta_max, eta_max] on the probe radius
        # makes the joint (rotation, jitter) distribution absolutely continuous on a
        # tube around the sphere. The proof needs it > 0; the default 0.0 keeps existing
        # experiments reproducible. Try 0.05.
        probe_radius_jitter: float = 0.0,
        num_probe: int = 1,
        # Adaptive probe count: reduce K during exploitation.
        # None means "on wherever implemented"; see the class docstring.
        adaptive_num_probe: Optional[bool] = None,
        adaptive_probe_warmup: int = 20,
        # Adaptive probes: reduce evaluations for stagnant particles
        adaptive_probes: Optional[bool] = None,
        adaptive_probes_threshold: float = 1e-6,
        max_iterations: int = 50,
        sinkhorn_max_iters: int = 2000,
        chunk_size: Optional[int] = None,
        # Micro-batch cost evaluation: subsample training batch for cost matrix
        cost_batch_size: Optional[int] = None,
        # Amortized OT: alternate between full OT steps and cheap momentum steps
        amortize_steps: int = 1,
        amortize_ema: float = 0.7,
        compile: bool = False,
        solver: Optional[str] = None,
        tempered_softmax_tau: float = 1.0,
        kl_softmax_lam: float = float("inf"),
        subspace: Optional[
            Union[LowRankSubspace, LinearSubspace, AdaptiveSubspace, HybridSubspace, CMAAdaptiveSubspace]
        ] = None,
        subspace_particle_dim: int = 8,
        absorb_every: int = 0,
        rank_schedule: Optional[RankSchedule] = None,
        block_strategy: str = "monolithic",
        block_group_size: int = 2,
        use_momentum: bool = False,
        momentum_init: float = 0.5,
        momentum_final: float = 0.95,
        velocity_lr: float = 1.0,
        use_adaptive_radius: bool = False,
        stagnation_threshold: float = 1e-4,
        stagnation_patience: int = 10,
        radius_increase: float = 1.5,
        radius_decrease: float = 0.9,
        radius_min: float = 0.5,
        radius_max: float = 3.0,
        # Dual potential momentum: extrapolate warm-start duals
        dual_momentum_beta: float = 0.0,
        # Sinkhorn solver improvements: wire through to SinkhornSolver
        anderson_depth: int = 0,
        adaptive_omega: bool = False,
        data_dependent_init: bool = False,
        # Transport-biased rotation: seed first polytope direction from previous OT descent
        biased_rotation: bool = False,
        # Quadratic model: extract FD gradient/Hessian from cost evaluations
        use_quadratic_model: bool = False,
        # Newton refinement: post-OT correction using quadratic model
        newton_refinement: bool = False,
        newton_refinement_alpha: float = 0.3,
        # Trust region: adapt step_radius from predicted vs actual improvement
        trust_region: bool = False,
        # Multi-fidelity screening: rank directions on a cheap fidelity, then spend
        # the full fidelity only on the informative ones.
        multifidelity_screen: bool = False,
        screen_keep_ratio: float = 0.5,
        screen_fidelity: float = 0.25,
        # CMA-ES configuration
        use_covariance_adaptation: bool = False,
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        candidate_autocast: bool = False,
        projection_type: str = "dense",
        # Compile vmap in NNCostEvaluator (fusion only, no CUDA graphs).
        compile_evaluator: bool = False,
        # CUDA-graph-compile the in-place forward+loss closure (reduce-overhead).
        # Only bites on the in-place path (>500K-param GPU models); the
        # launch-bound win for recurrent nets. See NNCostEvaluator.compile_forward.
        compile_forward: Optional[bool] = None,
    ) -> None:
        if projection_type not in ("dense", "sparse", "auto"):
            raise ValueError(f"Invalid projection_type: {projection_type!r}. Use 'dense', 'sparse', or 'auto'.")
        # Every downstream site tests `!= "monolithic"`, so an unrecognised value would
        # silently select block-wise mode rather than raise.
        if block_strategy not in ("monolithic", "per_layer", "grouped"):
            raise ValueError(
                f"Invalid block_strategy: {block_strategy!r}. Use 'monolithic', 'per_layer', or 'grouped'."
            )
        if polytope_type not in POLYTOPE_MAP:
            raise ValueError(f"Invalid polytope_type: {polytope_type!r}. Use one of {sorted(POLYTOPE_MAP)}.")
        self._requested_projection_type = projection_type
        self._compile_evaluator = compile_evaluator
        self._compile_forward = compile_forward

        # Particle dimension validation
        if particle_dim < 2:
            raise ValueError(
                f"particle_dim must be >= 2, got {particle_dim}. The OT polytope requires at least 2 dimensions."
            )
        if particle_dim > 4 and polytope_type == "cube":
            warnings.warn(
                f"particle_dim={particle_dim} with polytope_type='cube' produces "
                f"2^{particle_dim}={2**particle_dim} vertices (exponential). "
                f"Consider polytope_type='orthoplex' (2*{particle_dim}={2 * particle_dim} vertices) "
                f"or 'simplex' ({particle_dim + 1} vertices) for better scaling.",
                stacklevel=2,
            )
        self._full_space_particle_dim = particle_dim

        _warn_thread_oversubscription()

        # Features that read the orthoplex's antithetic vertex ordering. The default
        # polytope is the simplex, so enabling one of these without also asking for the
        # orthoplex leaves it inert: the finite-difference extractors check
        # polytope_type, and the contrast-ranked screen needs vertex j and j+k to be a
        # +/- pair.
        _needs_orthoplex = [
            name
            for name, on in (
                ("use_quadratic_model", use_quadratic_model),
                ("newton_refinement", newton_refinement),
                ("trust_region", trust_region),
                # A selection solver ranks vertices directly, so its screen needs no pairing.
                ("multifidelity_screen", multifidelity_screen and solver not in ("min_cost_greedy", "top_k_mean")),
            )
            if on
        ]
        if polytope_type != "orthoplex" and _needs_orthoplex:
            warnings.warn(
                f"{', '.join(_needs_orthoplex)} depend on the orthoplex's antithetic vertex "
                f"ordering, but polytope_type={polytope_type!r}. They will not take effect. Pass "
                f"polytope_type='orthoplex' to use them, or drop them to keep the simplex, which "
                f"is the minimal positive spanning set and the cheaper step.",
                stacklevel=2,
            )

        # Warn if particle_dim is set but will be overridden by subspace_particle_dim
        if subspace is not None and particle_dim != 2:
            warnings.warn(
                f"particle_dim={particle_dim} is ignored in subspace mode. "
                f"The OT polytope uses subspace_particle_dim={subspace_particle_dim} instead. "
                f"Pass subspace_particle_dim={particle_dim} to control polytope geometry.",
                stacklevel=2,
            )

        # In combined mode the blocks slice subspace coordinates, not parameter entries,
        # so the entry-grouping strategy has nothing to group and both settings produce
        # the same blocks. Say so rather than accepting a knob that does nothing.
        if subspace is not None and block_strategy == "grouped":
            warnings.warn(
                "block_strategy='grouped' has no effect in subspace mode: blocks divide the "
                "subspace coordinates evenly, not the parameter entries, so it produces exactly "
                "like 'per_layer'. Use block_strategy='per_layer', or drop the subspace to group "
                "parameter entries.",
                stacklevel=2,
            )

        # Combined mode: the global projection compresses full params to subspace
        # coords, then per-block OT decomposes the coordinate optimization.
        self._subspace_blockwise = subspace is not None and block_strategy != "monolithic"

        # Mixed precision config. Default to the model's own dtype rather than a
        # hardcoded fp32: a float64 model would otherwise get fp32 projections and
        # CMA state, and the reconstruct matmul fails on the dtype mismatch.
        self._mixed_precision = mixed_precision
        # Distinct from mixed_precision, which casts the parameters: autocast leaves
        # them alone and runs only the candidate forward's arithmetic in bf16, so a
        # small perturbation still reaches the loss.
        self._candidate_autocast_dtype = torch.bfloat16 if candidate_autocast else None
        try:
            self._model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self._model_dtype = torch.float32
        if not self._model_dtype.is_floating_point:
            self._model_dtype = torch.float32

        self.model = model
        self.polytope_type = polytope_type
        self.epsilon = epsilon
        self.ent_epsilon = ent_epsilon

        # ProgressiveEpsilon needs solver feedback, which only ``_progressive_epsilon``
        # receives. A scheduler passed as ``epsilon`` is adopted here so it advances the
        # same way ``auto_epsilon=True`` does; otherwise it would sit frozen at ``init``.
        from .epsilon import ProgressiveEpsilon

        if isinstance(epsilon, ProgressiveEpsilon):
            self._progressive_epsilon = epsilon
        elif auto_epsilon:
            if isinstance(epsilon, LinearEpsilon):
                prog_init = epsilon.init
                prog_target = epsilon.target
            elif isinstance(epsilon, (int, float)):
                prog_init = float(epsilon)
                prog_target = max(0.01, prog_init * 0.1)
            else:
                prog_init = 1.0
                prog_target = 0.01
            config = auto_epsilon_config or {}
            self._progressive_epsilon = ProgressiveEpsilon(
                init=config.get("init", prog_init),
                target=config.get("target", prog_target),
                max_epsilon=config.get("max_epsilon", prog_init * 5.0),
                increase_factor=config.get("increase_factor", 1.2),
                decrease_factor=config.get("decrease_factor", 0.95),
                fast_threshold=config.get("fast_threshold", 0.1),
                slow_threshold=config.get("slow_threshold", 0.5),
                ema_alpha=config.get("ema_alpha", 0.7),
            )
        else:
            self._progressive_epsilon = None
        self.scale_cost = scale_cost
        # A negative radius flips the finite-difference denominator, which
        # clamp(min=1e-10) then turns positive and scales the gradient by 1e10.
        # step_radius=0 is allowed: it is the no-movement control.
        if isinstance(step_radius, (int, float)) and step_radius < 0:
            raise ValueError(f"step_radius must be >= 0, got {step_radius}.")
        if isinstance(probe_radius, (int, float)) and probe_radius <= 0:
            raise ValueError(f"probe_radius must be > 0, got {probe_radius}.")
        self.step_radius = step_radius
        self.probe_radius = probe_radius
        if not (0.0 <= probe_radius_jitter < 1.0):
            raise ValueError(
                f"probe_radius_jitter must be in [0, 1), got {probe_radius_jitter}. "
                f"Values >= 1 risk negative effective probe radius."
            )
        self.probe_radius_jitter = probe_radius_jitter
        if num_probe < 1:
            raise ValueError(
                f"num_probe must be >= 1, got {num_probe}. "
                f"At least one probe point per direction is required; "
                f"num_probe=0 yields an empty probe tensor and NaN costs."
            )
        self.num_probe = num_probe
        # The fused softmax path divides by epsilon directly and never reaches a solver's
        # own validation. A zero gives NaN, a negative silently inverts the plan into ascent.
        for _name, _value in (("epsilon", epsilon), ("ent_epsilon", ent_epsilon)):
            if isinstance(_value, (int, float)) and _value <= 0:
                raise ValueError(f"{_name} must be > 0, got {_value}.")
        # Both features are wired into the monolithic step only. Defaulting them to
        # None rather than True keeps the "you asked for this and it was ignored"
        # warning below honest: it fires on an explicit True, not on the default.
        _savings_default = block_strategy == "monolithic"
        # Whether the caller asked for reuse, as opposed to inheriting the default.
        # Diagnostics about reuse never firing are addressed to the caller who asked.
        self._adaptive_probes_explicit = adaptive_probes is not None
        # Reducing K to 1 needs a K above 1 to reduce, so the default stays off at the
        # default num_probe=1 rather than arming machinery with nothing to do.
        if adaptive_num_probe is None:
            adaptive_num_probe = _savings_default and num_probe > 1
        adaptive_probes = _savings_default if adaptive_probes is None else adaptive_probes
        self.adaptive_num_probe = adaptive_num_probe
        self._adaptive_probe_warmup = adaptive_probe_warmup
        self._loss_decreasing_count = 0
        # OT-step-only costs for adaptive_num_probe check (avoids mixing with momentum costs)
        self._ot_step_costs: collections.deque = collections.deque(maxlen=3)
        self._adaptive_probes = adaptive_probes
        self._adaptive_probes_threshold = adaptive_probes_threshold
        # Configuration the cached cost matrix was measured at. Reuse needs X to be
        # unchanged: a candidate replaces one row of X, so every row of the matrix
        # depends on all the others.
        self._prev_X: Optional[torch.Tensor] = None
        # Previous cost matrix, reused whole when X has not moved
        self._prev_cost_matrix: Optional[torch.Tensor] = None
        self._param_write_cache = None
        # Previous rotation matrices: they come back with the matrix so the rows
        # still describe the vertices they were evaluated at.
        self._prev_rot_mats: Optional[torch.Tensor] = None
        # Track K_eff and both radii to invalidate _prev_cost_matrix on change.
        # probe_r matters: it is the radius the cached costs were measured at.
        self._prev_k_eff: Optional[int] = None
        self._prev_step_r: Optional[float] = None
        self._prev_probe_r: Optional[float] = None
        # Identity of the objective the cached rows were measured against. Reuse across
        # two different minibatches would put costs from different data in one cost
        # matrix, and the OT plan would rank vertices partly by which batch they came
        # from. None means the caller asserts a stationary objective.
        self._prev_objective_token: object = None
        self._objective_token_warned = False
        self.max_iterations = max_iterations
        self.chunk_size = chunk_size
        if cost_batch_size is not None and cost_batch_size <= 0:
            # 0 slices an empty batch, so every candidate scores NaN, sanitize flattens
            # the matrix, and the run trains nothing while diagnostics stay finite.
            raise ValueError(f"cost_batch_size must be > 0 or None, got {cost_batch_size}.")
        self.cost_batch_size = cost_batch_size
        self.amortize_steps = max(1, amortize_steps)
        self.amortize_ema = amortize_ema

        # SNN-like models lose accuracy badly under a shrinking step_radius: the
        # discrete spike landscape is too chaotic. Warn on that combination.
        # Heuristic: matches substrings of module class names.
        if hasattr(step_radius, "at"):
            module_classes = {type(m).__name__ for m in model.modules()}
            snn_markers = ("lif", "leaky", "spik", "spiking", "alif")
            if any(marker in cls.lower() for cls in module_classes for marker in snn_markers):
                warnings.warn(
                    "Detected SNN-like module (LIF/Leaky/Spiking) with a "
                    "scheduled step_radius (CosineEpsilon or similar). "
                    "Per the paper experiments scheduling step_radius on "
                    "SNN models collapses accuracy from ~93% to 10-47%. "
                    "Pass a flat float for step_radius on SNN tasks.",
                    stacklevel=2,
                )
        self._amortize_counter = 0
        self._transport_direction_ema = None
        # Transport-biased rotation: store previous OT descent direction
        self.biased_rotation = biased_rotation
        self._prev_descent_direction: Optional[torch.Tensor] = None
        self._prev_descent_direction_finite: bool = False
        # Quadratic model: FD gradient/Hessian extraction from cost evaluations
        self.use_quadratic_model = use_quadratic_model
        self._losses_3d = None  # (P, V, K) retained for quadratic model
        # Newton refinement: post-OT correction using quadratic model
        self._newton_refinement = newton_refinement
        self._newton_refinement_alpha = newton_refinement_alpha
        if newton_refinement and not use_quadratic_model:
            self.use_quadratic_model = True
            logger.info(
                "newton_refinement=True auto-enables use_quadratic_model=True "
                "(needed to retain probe losses for Newton correction)"
            )
        if newton_refinement and num_probe < 2:
            warnings.warn(
                "newton_refinement needs num_probe>=2 to build the finite-difference "
                "model; it stays inactive until num_probe>=2.",
                stacklevel=2,
            )

        self._newton_direction = None  # (P, pdim) Newton step in original space
        self._center_loss = None  # (P,) f(X) when the K=1 quadratic model asked for it
        # Trust region: adapt step_radius via multiplier based on quadratic model
        self.trust_region = trust_region
        self._trust_region_multiplier = 1.0  # Multiplier on step_radius, range [0.1, 3.0]
        self._prev_predicted_improvement = None
        self._prev_pre_step_loss = None  # f(X), or the min-cost proxy when no centre ran
        self._prev_loss_from_center = False
        # Trust region needs the finite-difference model to form predicted-vs-actual
        # ratios; auto-enable it (mirrors newton_refinement) and warn on the two
        # prerequisites that would otherwise leave the multiplier frozen at 1.0.
        if trust_region and not self.use_quadratic_model:
            self.use_quadratic_model = True
            logger.info(
                "trust_region=True auto-enables use_quadratic_model=True "
                "(needed for the predicted-vs-actual ratio test)"
            )
        # num_probe=1 is fine for trust_region: the step evaluates one f(X) per particle
        # and reads curvature from L(+s) + L(-s) - 2 L(0). newton_refinement still needs
        # the regression, hence the separate warning above.
        if trust_region and block_strategy != "monolithic":
            warnings.warn(
                f"trust_region is only applied with block_strategy='monolithic'; "
                f"ignored for block_strategy='{block_strategy}'.",
                stacklevel=2,
            )
        # Vertex screening: rank cheaply, then pay full fidelity only for the survivors.
        if not 0.0 < screen_keep_ratio <= 1.0:
            raise ValueError(f"screen_keep_ratio must be in (0, 1], got {screen_keep_ratio}")
        if not 0.0 < screen_fidelity <= 1.0:
            raise ValueError(f"screen_fidelity must be in (0, 1], got {screen_fidelity}")
        self.multifidelity_screen = multifidelity_screen
        self.screen_keep_ratio = screen_keep_ratio
        self.screen_fidelity = screen_fidelity
        self._last_screen_savings = 0.0
        self.subspace = subspace
        self._subspace_particle_dim = subspace_particle_dim
        self.absorb_every = absorb_every
        self._rank_schedule = rank_schedule
        # Validate: rank_schedule requires a subspace
        if rank_schedule is not None and subspace is None:
            raise ValueError("rank_schedule requires a subspace")
        # rank_schedule transitions only run in the monolithic step; disable it
        # elsewhere so it is a clear no-op instead of a silent one (or a crash).
        if rank_schedule is not None and block_strategy != "monolithic":
            warnings.warn(
                f"rank_schedule is only applied with block_strategy='monolithic'; "
                f"ignored for block_strategy='{block_strategy}'.",
                stacklevel=2,
            )
            self._rank_schedule = None
        # Rank the subspace was last rebuilt at. None until the first step, because a
        # subspace does not record the rank it was built from and may not be at stage 0.
        self._applied_rank = None
        self.block_strategy = block_strategy
        self.block_group_size = block_group_size

        # Reads self., not the argument: newton_refinement and trust_region auto-enable
        # the quadratic model above, and the argument alone would miss both.
        if self.use_quadratic_model and block_strategy != "monolithic":
            warnings.warn(
                f"use_quadratic_model=True is not supported with "
                f"block_strategy='{block_strategy}'. Quadratic model will be "
                f"silently disabled for block-wise steps.",
                stacklevel=2,
            )

        # Momentum config
        self.use_momentum = use_momentum
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.velocity_lr = velocity_lr

        # Adaptive radius config
        self.use_adaptive_radius = use_adaptive_radius
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_patience = stagnation_patience
        self.radius_increase = radius_increase
        self.radius_decrease = radius_decrease
        self.radius_min = radius_min
        self.radius_max = radius_max

        self._dual_momentum_beta = dual_momentum_beta

        # Detect CMA subspace mode early for validation
        self._cma_subspace = isinstance(subspace, CMAAdaptiveSubspace)

        # Validate: CMA features require CMAAdaptiveSubspace
        if use_covariance_adaptation and not self._cma_subspace:
            warnings.warn("use_covariance_adaptation requires CMAAdaptiveSubspace. It will be disabled.")
            use_covariance_adaptation = False

        # Validate: CMA covariance scaling and the CMA state update are wired
        # only into the monolithic step; blockwise samples through per-block
        # projections, so the paths/covariance never adapt there.
        if use_covariance_adaptation and self.block_strategy != "monolithic":
            warnings.warn(
                "use_covariance_adaptation is only supported with block_strategy='monolithic'; "
                f"disabled for block_strategy='{self.block_strategy}'."
            )
            use_covariance_adaptation = False

        # Blockwise re-evaluates the full closure per block, never calls screen_closure
        # and never populates the reuse cache, so these flags save nothing there. Only
        # an explicit True reaches here; the default resolved to False above.
        if self.block_strategy != "monolithic":
            _ignored = [
                name
                for name, on in (
                    ("adaptive_probes", adaptive_probes),
                    ("adaptive_num_probe", adaptive_num_probe),
                    ("multifidelity_screen", multifidelity_screen),
                )
                if on
            ]
            if _ignored:
                warnings.warn(
                    f"{', '.join(_ignored)} {'is' if len(_ignored) == 1 else 'are'} only "
                    f"implemented for block_strategy='monolithic' and will be ignored for "
                    f"block_strategy='{self.block_strategy}'. No forward evaluations are saved.",
                    stacklevel=2,
                )

        # A radius boost consumes the stagnation counter, so if it fires first or at the
        # same time, a stagnation absorb never reaches its own patience.
        _absorb_mode = getattr(subspace, "absorb_mode", None)
        _absorb_patience = getattr(subspace, "absorb_patience", None)
        if (
            use_adaptive_radius
            and _absorb_mode == "stagnation"
            and _absorb_patience is not None
            and stagnation_patience <= _absorb_patience
        ):
            warnings.warn(
                f"use_adaptive_radius=True with stagnation_patience={stagnation_patience} <= "
                f"subspace.absorb_patience={_absorb_patience}: the radius boost resets the "
                f"stagnation counter before absorb_mode='stagnation' can trigger, so absorb "
                f"will never fire. Raise stagnation_patience above absorb_patience, use "
                f"absorb_mode='periodic' with absorb_interval > 0, or set "
                f"use_adaptive_radius=False.",
                stacklevel=2,
            )

        self.use_covariance_adaptation = use_covariance_adaptation
        # Coord-to-param projection for the current step, covariance-scaled for
        # CMA. Cached so a step's probes and its sync share one metric.
        self._sampling_projection = None
        # Shape-keyed scratch buffers for the monolithic chunk loop, reused across steps.
        # Fully overwritten on use, so they hold no state worth serializing.
        self._step_buffers = None

        # Detect model device for tensor creation
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            raise ValueError("Model has no trainable parameters. PolyStepOptimizer requires at least one parameter.")

        # Auto-selection of projection type based on model size
        num_params = sum(p.numel() for p in model.parameters())
        self._actual_projection_type = _select_projection_type(
            num_params, model_device, self._requested_projection_type
        )

        # Log auto-selection choice
        if self._requested_projection_type == "auto":
            logger.info(
                f"Auto-selected {self._actual_projection_type} projection for "
                f"{num_params / 1e6:.1f}M params on {model_device}"
            )

        # Fallback for tiny models: sparse has overhead that isn't worth it
        if self._actual_projection_type == "sparse" and num_params < _MIN_PARAMS_FOR_SPARSE:
            logger.info(
                f"Model has {num_params:,} params (<{_MIN_PARAMS_FOR_SPARSE:,}). "
                f"Using dense projection instead of sparse."
            )
            self._actual_projection_type = "dense"

        # Mixed precision: cast model to BF16 for memory savings
        if mixed_precision:
            if not self._bf16_supported():
                warnings.warn(
                    "BF16 not supported on this device. Falling back to FP32. "
                    "For GPU: requires compute capability >= 7.0 (Volta+)."
                )
            else:
                model.bfloat16()
                self._model_dtype = torch.bfloat16

        # Build the layout after any mixed-precision cast so it captures the
        # BF16 param dtype; building it first left full-space candidates FP32
        # while the model was BF16. Thread particle_dim for full-space mode;
        # subspace mode ignores it (uses subspace_particle_dim).
        self.layout = ParamLayout.from_module(model, particle_dim=self._full_space_particle_dim)
        # The layout only covers requires_grad parameters, so a fully frozen model
        # yields an empty layout and fails later with an opaque range() error.
        if self.layout.total_params == 0:
            raise ValueError(
                "Model has no parameters with requires_grad=True. PolyStepOptimizer "
                "optimizes the requires_grad parameters, so at least one must be trainable."
            )

        # Detect adaptive subspace mode
        self._adaptive = isinstance(subspace, AdaptiveSubspace)

        # Detect hybrid subspace mode
        self._hybrid = isinstance(subspace, HybridSubspace)
        self._hybrid_subspace = subspace if self._hybrid else None

        # FactoredSubspace presents the same per-layer-projections protocol as
        # HybridSubspace (init_projections / apply_perturbation(projections, base, flat)),
        # so it shares the state setup and the model sync below.
        self._factored = isinstance(subspace, FactoredSubspace)
        self._per_layer_projections = self._hybrid or self._factored

        # A per-entry projection keeps each parameter's own dtype. Full space and a single
        # global projection instead hold every parameter in one vector at the layout's
        # dominant dtype, so a minority-dtype weight would be optimized at the majority's
        # precision. Say so rather than downgrade it silently.
        _param_dtypes = {e.dtype for e in self.layout.entries if e.dtype.is_floating_point}
        if len(_param_dtypes) > 1 and (subspace is None or self._adaptive):
            _names = ", ".join(sorted(str(d) for d in _param_dtypes))
            raise ValueError(
                f"Model mixes parameter dtypes ({_names}). "
                f"{'AdaptiveSubspace' if self._adaptive else 'Full-space mode'} stores every parameter in one "
                f"vector at {self.layout.dominant_dtype}, so the others would be optimized at that precision. "
                "Cast the model to a single dtype, or use HybridSubspace or LinearSubspace, "
                "which give each parameter its own projection and keep its dtype."
            )

        if subspace is not None:
            # Subspace mode: subspace coords reshaped to multi-particle format.
            # Use the user-specified particle_dim if it was explicitly set;
            # otherwise fall back to subspace_particle_dim (default 8) for a
            # stronger per-step OT signal (orthoplex in 8D -> 16 vertices).
            self._base_params = {k: v.clone() for k, v in model.state_dict().items()}
            sub_dim = subspace.subspace_dim
            pdim = subspace_particle_dim
            # Pad subspace_dim to be divisible by particle_dim
            padded_sub = sub_dim + (-sub_dim % pdim)
            num_sub_particles = padded_sub // pdim
            # Start at base params => delta=0 => subspace coords = zeros
            # Use model_dtype for mixed precision compatibility
            x_dtype = self._model_dtype if self._mixed_precision else self.layout.dominant_dtype
            X_init = torch.zeros(
                num_sub_particles,
                pdim,
                dtype=x_dtype,
                device=model_device,
            )
            particle_dim = pdim
        else:
            # Full parameter mode: (num_particles, particle_dim) from layout
            flat_2d = self.layout.flatten(model)  # (rows, particle_dim)
            X_init = flat_2d.to(model_device)
            particle_dim = self.layout.particle_dim

        self._particle_dim = particle_dim
        device = X_init.device

        # Polytope template in particle_dim space (NOT full parameter space).
        # This gives a small number of vertices (e.g., 4 for orthoplex in 2D).
        # Skip when using block-wise mode (per-block polytopes used instead).
        if block_strategy == "monolithic":
            self._polytope_vertices = POLYTOPE_MAP[polytope_type](
                particle_dim,
                device=device,
                dtype=X_init.dtype,
                radius=1.0,
            )
        else:
            self._polytope_vertices = None  # per-block polytopes used instead

        # Probe linspace (exclude endpoints)
        self._probes = torch.linspace(0, 1, num_probe + 2)[1 : num_probe + 1]

        # One particle leaves balanced OT one feasible plan: the column constraint pins
        # T_0j = 1/V, so the plan carries no cost information and the mean-centered
        # polytope returns X unchanged. The optimizer would freeze with no error.
        _single_particle = X_init.shape[0] == 1 and block_strategy == "monolithic"
        if _single_particle and solver is None:
            # Never let the *default* produce a frozen optimizer.
            solver = "softmax"
            warnings.warn(
                "num_particles=1: balanced Sinkhorn yields a uniform transport plan "
                "(the column marginal forces it), so steps would ignore the cost and "
                "the parameters would never move. Selecting the one-sided "
                "SoftmaxSolver instead. Pass solver='sinkhorn' explicitly to override, "
                "or reduce particle_dim so the layout gives more than one particle.",
                stacklevel=2,
            )
        elif _single_particle and (solver == "sinkhorn" or (solver == "kl_softmax" and kl_softmax_lam == INF)):
            warnings.warn(
                f"num_particles=1 with solver={solver!r}: the column marginal forces a "
                "uniform transport plan, so every step ignores the cost and the "
                "parameters will not move. Use solver='softmax' (one-sided), or reduce "
                "particle_dim so the layout gives more than one particle.",
                stacklevel=2,
            )

        if solver is None:
            solver = "softmax" if subspace is not None else "sinkhorn"

        if solver != "sinkhorn":
            inert = [
                name
                for name, value in (
                    ("anderson_depth", anderson_depth),
                    ("adaptive_omega", adaptive_omega),
                    ("data_dependent_init", data_dependent_init),
                )
                if value
            ]
            if inert:
                warnings.warn(
                    f"{', '.join(inert)} accelerate the Sinkhorn fixed-point iteration and are "
                    f"ignored by solver={solver!r}.",
                    stacklevel=2,
                )

        if solver == "softmax":
            self.solver = SoftmaxSolver(compile=compile)
        elif solver == "sinkhorn":
            self.solver = SinkhornSolver(
                max_iterations=sinkhorn_max_iters,
                compile=compile,
                anderson_depth=anderson_depth,
                adaptive_omega=adaptive_omega,
                data_dependent_init=data_dependent_init,
            )
        elif solver == "min_cost_greedy":
            self.solver = MinCostGreedySolver(compile=compile)
        elif solver == "top_k_mean":
            self.solver = TopKMeanSolver(compile=compile)
        elif solver == "tempered_softmax":
            self.solver = TemperedSoftmaxSolver(
                tau=tempered_softmax_tau,
                compile=compile,
            )
        elif solver == "kl_softmax":
            self.solver = KLSoftmaxSolver(
                lam=kl_softmax_lam,
                max_iterations=sinkhorn_max_iters,
            )
        else:
            raise ValueError(
                f"Unknown solver: {solver!r}. Expected 'softmax', 'sinkhorn', "
                f"'min_cost_greedy', 'top_k_mean', 'tempered_softmax', 'kl_softmax', or None."
            )

        # KLSoftmaxSolver interpolates between the two limits: lam=0 is the one-shot
        # softmax and lam=inf is balanced Sinkhorn, so each limit inherits that
        # solver's guards.
        _one_shot = isinstance(
            self.solver, (SoftmaxSolver, MinCostGreedySolver, TopKMeanSolver, TemperedSoftmaxSolver)
        ) or (isinstance(self.solver, KLSoftmaxSolver) and self.solver.lam == 0)
        self._balanced_ot = isinstance(self.solver, SinkhornSolver) or (
            isinstance(self.solver, KLSoftmaxSolver) and self.solver.lam == INF
        )
        if self._progressive_epsilon is not None and _one_shot:
            raise ValueError(
                "ProgressiveEpsilon requires Sinkhorn convergence feedback. "
                "Use LinearEpsilon or CosineEpsilon with non-iterative solvers."
            )

        # Fused softmax fast path flag. Cost scaling for the fused path is
        # applied via scale_cost_matrix(...) in the step functions (honoring
        # every scale_cost mode), so the compiled kernel does no scaling.
        self._use_fused_softmax = isinstance(self.solver, SoftmaxSolver)

        # Compiled functions
        self._compiled = CompiledFunctions(compile=compile and torch.cuda.is_available())

        # Random generator (created early so AdaptiveSubspace init can use it)
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = create_generator(seed, device)

        num_points = X_init.shape[0]
        a = torch.ones(num_points, device=device, dtype=X_init.dtype) / num_points
        self._state = SolverState(X=X_init.clone(), a=a)

        if subspace is not None:
            self._state.base_params = self._base_params
            self._state.subspace = subspace

        self._init_subspace(subspace, seed, model_device)

        # Initialize CMA-ES state (evolution paths, covariance, step-size)
        self._init_cma(subspace, seed, model_device)

        if use_momentum:
            self._state.velocity = torch.zeros_like(X_init)

        # Initialize block-wise decomposition (polytopes, duals)
        self._init_blocks(
            subspace,
            subspace_particle_dim,
            block_strategy,
            block_group_size,
            polytope_type,
            X_init,
            device,
        )

        # Stage 0 before the first step, so the first sweep runs at the scheduled rank
        # rather than whatever the supplied subspace carried. X is still zero, so the
        # absorb inside the transition is a no-op.
        if self._rank_schedule is not None and self.subspace is not None:
            self._transition_rank(self._rank_schedule.at(0))

    def _init_subspace(self, subspace, seed, model_device) -> None:
        """Initialize subspace projections and displacement history."""
        if subspace is None:
            return

        # AdaptiveSubspace: initialize projection and displacement history
        if self._adaptive:
            if self._actual_projection_type == "sparse":
                from .projection import SparseRandomProjection

                self._state.projection = SparseRandomProjection(
                    full_dim=subspace.full_dim,
                    subspace_dim=subspace.subspace_dim,
                    seed=seed if seed is not None else 0,
                )
            else:
                self._state.projection = subspace.init_projection(
                    generator=self._generator,
                    device=model_device,
                    dtype=self._model_dtype,
                )
            self._state.displacement_history = torch.zeros(
                subspace.displacement_history_size,
                subspace.subspace_dim,
                device=model_device,
                dtype=self._model_dtype,
            )

        # Per-layer projections (Hybrid / Factored): build them and the history buffer.
        if self._per_layer_projections:
            self._state.hybrid_projections = subspace.init_projections(
                model_device,
                self._model_dtype,
            )
            if hasattr(subspace, "build_fused_projection"):
                subspace.build_fused_projection(self._state.hybrid_projections)
            self._state.displacement_history = torch.zeros(
                subspace.displacement_history_size,
                subspace.subspace_dim,
                device=model_device,
                dtype=self._model_dtype,
            )

    def _init_cma(self, subspace, seed, model_device) -> None:
        """Initialize CMA-ES state: projection, displacement history, evolution paths."""
        self._cma_params = None
        if not self._cma_subspace:
            return

        # CMAAdaptiveSubspace wraps AdaptiveSubspace, so it also needs projection init
        if self._actual_projection_type == "sparse":
            from .projection import SparseRandomProjection

            self._state.projection = SparseRandomProjection(
                full_dim=subspace.full_dim,
                subspace_dim=subspace.subspace_dim,
                seed=seed if seed is not None else 0,
            )
        else:
            self._state.projection = subspace.init_projection(
                generator=self._generator,
                device=model_device,
                dtype=self._model_dtype,
            )
        self._state.displacement_history = torch.zeros(
            subspace.base.displacement_history_size,
            subspace.subspace_dim,
            device=model_device,
            dtype=self._model_dtype,
        )

        if self.use_covariance_adaptation:
            # The accumulators run at fp32 even under mixed_precision. c_1 is
            # ~2/(n+1.3)^2 (8e-6 at n=512) and bf16 has 8 mantissa bits, so
            # 1 + c_1*x rounds straight back to 1 and C_diag never moves, leaving
            # adaptation silently inert. The solvers promote for the same reason.
            cma_state = subspace.init_cma_state(
                device=model_device,
                dtype=torch.float32,
            )
            self._state.p_c = cma_state["p_c"]
            self._state.p_sigma = cma_state["p_sigma"]
            self._state.C_diag = cma_state["C_diag"]
            self._state.generation = 0
            # The step feeds the paths a unit-norm direction, not a weighted
            # recombination of mu offspring, so the rates must be derived at
            # mu_eff = 1 too. Deriving them from the vertex count instead gave
            # c_sigma ~ 2/3, a path memory the updates never used.
            n = subspace.subspace_dim
            mu_eff = float(subspace.mu_eff) if getattr(subspace, "_mu_eff_explicit", False) else 1.0
            hyperparams = compute_cma_hyperparameters(n, mu_eff)
            self._cma_params = {
                **hyperparams,
                # A rate the caller set explicitly wins over the derived one; without
                # this the constructor argument is accepted and then discarded.
                **getattr(subspace, "_explicit_rates", {}),
                "mu_eff": mu_eff,
                "cov_min": subspace.cov_min,
                "cov_max": subspace.cov_max,
            }
            # __post_init__ fills mu_eff from default_mu_eff(n) when the caller left it
            # unset, which is not the value resolved above. Write back so the attribute
            # reports what the step actually runs at.
            subspace.mu_eff = mu_eff

    def _init_blocks(
        self,
        subspace,
        subspace_particle_dim,
        block_strategy,
        block_group_size,
        polytope_type,
        X_init,
        device,
    ) -> None:
        """Initialize block-wise decomposition: blocks, polytopes, duals."""
        self._blocks: Optional[List[BlockConfig]] = None
        self._block_polytopes: Optional[List[torch.Tensor]] = None
        self._subspace_blocks: Optional[List[BlockConfig]] = None
        self._subspace_block_polytopes: Optional[List[torch.Tensor]] = None

        if self._subspace_blockwise:
            sub_dim = subspace.subspace_dim
            num_blocks = min(len(self.layout.entries), 8)
            num_blocks = max(2, num_blocks)

            self._subspace_blocks = create_subspace_blocks(
                subspace_dim=sub_dim,
                num_blocks=num_blocks,
                subspace_particle_dim=subspace_particle_dim,
            )

            self._subspace_block_polytopes = []
            for block in self._subspace_blocks:
                self._subspace_block_polytopes.append(
                    POLYTOPE_MAP[polytope_type](
                        block.particle_dim,
                        device=device,
                        dtype=X_init.dtype,
                        radius=1.0,
                    )
                )

            self._state.block_duals = [(None, None) for _ in self._subspace_blocks]

        elif block_strategy != "monolithic":
            if block_strategy == "per_layer":
                self._blocks = create_per_layer_blocks(
                    self.layout,
                    particle_dim=self.layout.particle_dim,
                )
            elif block_strategy == "grouped":
                self._blocks = create_grouped_blocks(
                    self.layout,
                    group_size=block_group_size,
                    particle_dim=self.layout.particle_dim,
                )
            else:
                raise ValueError(
                    f"Unknown block_strategy: {block_strategy!r}. Use 'monolithic', 'per_layer', or 'grouped'."
                )
            self._block_polytopes = []
            for block in self._blocks:
                self._block_polytopes.append(
                    POLYTOPE_MAP[polytope_type](
                        block.particle_dim,
                        device=device,
                        dtype=X_init.dtype,
                        radius=1.0,
                    )
                )
            self._state.block_duals = [(None, None) for _ in self._blocks]

            # Same single-particle degeneracy as the monolithic path, but per block:
            # a block holding one particle solves a balanced OT with one row, whose
            # plan is forced uniform, so that block freezes while the rest train and
            # nothing shows in the loss. Small layers hit this under 'per_layer'.
            if self._balanced_ot:
                frozen = [b.name for b in self._blocks if b.num_particles == 1]
                if frozen:
                    warnings.warn(
                        f"{len(frozen)} block(s) hold a single particle "
                        f"({', '.join(frozen[:5])}{', ...' if len(frozen) > 5 else ''}); with a "
                        "balanced solver the column marginal forces a uniform plan there, "
                        "so those blocks will never move. Use solver='softmax', "
                        "block_strategy='grouped', or a smaller particle_dim.",
                        stacklevel=2,
                    )

    @property
    def state(self) -> SolverState:
        """Read-only access to the current solver state."""
        return self._state

    @property
    def mixed_precision(self) -> bool:
        """Whether mixed precision (BF16) is enabled."""
        return self._mixed_precision

    @property
    def model_dtype(self) -> torch.dtype:
        """Current dtype of the model parameters."""
        return self._model_dtype

    @property
    def projection_type(self) -> str:
        """Actual projection type being used ('dense' or 'sparse').

        When 'auto' is requested, this returns the resolved type based
        on model size (sparse for >1M params on CPU, >2M on GPU).
        """
        return self._actual_projection_type

    def _get_epsilon(self, iteration: int) -> float:
        """Resolve epsilon at current iteration."""
        if self._progressive_epsilon is not None:
            return self._progressive_epsilon.at(iteration)
        if hasattr(self.epsilon, "at"):
            return self.epsilon.at(iteration)
        return self.epsilon

    def _get_step_radius(self, iteration: int) -> float:
        """Resolve step_radius at current iteration (supports schedule objects)."""
        if hasattr(self.step_radius, "at"):
            return self.step_radius.at(iteration)
        return self.step_radius

    def _get_probe_radius(self, iteration: int) -> float:
        """Resolve probe_radius at current iteration (supports schedule objects)."""
        if hasattr(self.probe_radius, "at"):
            return self.probe_radius.at(iteration)
        return self.probe_radius

    def _apply_probe_radius_jitter(self, probe_r: float) -> float:
        """Apply per-step uniform multiplicative jitter to the probe radius.

        Samples ``eta ~ Uniform[-eta_max, +eta_max]`` and returns
        ``probe_r * (1 + eta)``. When ``probe_radius_jitter == 0`` (default)
        this is a no-op and consumes no random state.
        """
        eta_max = float(self.probe_radius_jitter)
        if eta_max <= 0.0:
            return probe_r
        if self._generator is not None:
            device = self._generator.device
            eta = torch.empty((1,), device=device).uniform_(-eta_max, eta_max, generator=self._generator).item()
        else:
            eta = float(torch.empty((1,)).uniform_(-eta_max, eta_max).item())
        return float(probe_r) * (1.0 + eta)

    def _get_ent_epsilon(self, iteration: int) -> Optional[float]:
        """Resolve ent_epsilon at current iteration."""
        if self.ent_epsilon is None:
            return None
        if hasattr(self.ent_epsilon, "at"):
            return self.ent_epsilon.at(iteration)
        return self.ent_epsilon

    def _bf16_supported(self) -> bool:
        """Check if BF16 is supported on the model's device.

        Returns:
            True if BF16 is supported and should be used.
        """
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return False
        if device.type == "cuda":
            # torch's own check, which also covers emulated bf16 on older cards.
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif device.type == "cpu":
            return True  # CPU BF16 works, may be slower without AMX
        else:
            return True  # MPS, XLA, etc. - let PyTorch error if unsupported

    def _invalidate_reuse_cache(self) -> None:
        """Drop the cached cost rows, rotations, and probe state used by
        adaptive-probe reuse. Call whenever the cost geometry changes (subspace
        rotation or absorb, rank transition), since a reused row would then
        describe stale directions.
        """
        self._prev_cost_matrix = None
        self._prev_rot_mats = None
        self._losses_3d = None
        self._prev_X = None
        self._prev_k_eff = None
        self._prev_step_r = None
        self._prev_probe_r = None
        self._prev_objective_token = None

    @property
    def compile_evaluator(self) -> bool:
        """Whether evaluators should fusion-compile the vmapped forward.

        Public, read-only view of the constructor flag. Consumed by
        ``api.train()`` and propagated to a registered evaluator's
        ``compile_vmap`` in ``register_evaluator``.
        """
        return self._compile_evaluator

    @property
    def compile_forward(self) -> Optional[bool]:
        """Whether evaluators should CUDA-graph-compile the in-place forward.

        Read-only view of the constructor flag, propagated to a registered evaluator in
        ``register_evaluator``. ``None`` leaves the evaluator's own default, which
        enables it on the in-place path; ``False`` opts out.
        """
        return self._compile_forward

    def register_evaluator(
        self,
        evaluator: "NNCostEvaluator",
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> None:
        """Register evaluator + data for fused inplace evaluation.

        When registered, the optimizer's chunk loop can bypass
        ``reconstruct_batch`` + ``closure()`` and instead call
        ``evaluator.evaluate_subspace_inplace()`` directly, which
        reconstructs weights one-at-a-time via in-place swap.

        This reduces memory from O(N x model_params + N x activation) to
        O(model_params + 1 x activation), enabling training of larger models.

        Call this before each ``step()`` with the current mini-batch data.
        The fused path is only used when the evaluator has ``_use_inplace=True``
        (auto-detected for GPU models >500K params).

        The evaluator's ``loss_fn`` becomes the objective on every site-aware chunk,
        while chunks that fall back score ``closure()``. Give both the same objective:
        a closure that adds regularization the evaluator does not know about makes one
        step rank part of its candidates on a different function.

        The optimizer's ``compile_evaluator`` / ``compile_forward`` flags are
        propagated to the evaluator here, so those public knobs work on the
        fused/runner path too (not only via ``api.train()``). torch.compile is
        lazy inside the evaluator, so setting the flags before the first
        ``evaluate*`` call is sufficient.

        Args:
            evaluator: NNCostEvaluator instance.
            inputs: Current mini-batch inputs.
            targets: Current mini-batch targets (optional).
        """
        self._cost_evaluator = evaluator
        self._fused_inputs = inputs
        self._fused_targets = targets
        if self._candidate_autocast_dtype is not None:
            evaluator.autocast_dtype = self._candidate_autocast_dtype
        if self._compile_evaluator:
            evaluator._compile_vmap = True
        # None leaves the evaluator's own default, which pairs CUDA graphs with the
        # in-place path they exist for. Only an explicit True/False overrides it.
        if self._compile_forward is not None:
            evaluator._compile_forward = self._compile_forward
        # Rebuild the fast paths when the objective changes: they cache the model and
        # loss_fn they were built from, and would otherwise keep scoring the previous
        # evaluator's objective.
        fastpath_key = (id(evaluator.model), id(evaluator.loss_fn))
        if getattr(self, "_fastpath_key", None) != fastpath_key:
            self._fastpath_key = fastpath_key
            for attr in (
                "_factored_evaluator",
                "_sparse_delta_evaluator",
                "_site_vmap_evaluator",
                "_subspace_delta_evaluator",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
        # A FactoredSubspace can be scored through the low-rank identity, which never
        # builds a candidate weight. Built once; None when the model is outside the
        # supported module set, in which case the step falls back to reconstruct_batch.
        if self._factored and not hasattr(self, "_factored_evaluator"):
            from .cost_nn import FactoredEvaluator

            self._factored_evaluator = FactoredEvaluator.try_build(evaluator.model, evaluator.loss_fn)
            if self._factored_evaluator is None:
                warnings.warn(
                    f"FactoredSubspace is in use but {type(evaluator.model).__name__} is not a "
                    "plain nn.Sequential of Linear/activation layers, so candidates must be "
                    "materialized through reconstruct_batch and the low-rank speedup does not apply.",
                    stacklevel=2,
                )
        # Full-space candidates perturb one contiguous run, so every other layer keeps
        # the shared base weight and the per-candidate bmm is avoidable. Built once;
        # None when the model is outside the supported module set.
        if self.subspace is None and not hasattr(self, "_sparse_delta_evaluator"):
            from .cost_nn import SiteVmapEvaluator, SparseDeltaEvaluator

            self._sparse_delta_evaluator = SparseDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.layout
            )
            # Same locality, no assumption about the module set: batches only the
            # perturbed tensor so the graph ahead of it is shared. Picks up the chunks
            # the sparse-delta path declines, and the models it cannot be built for.
            self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)
        # Subspace candidates perturb one particle's coordinate run, so when that run
        # sits inside one layer's coordinate block only that layer's weight moves.
        # Built once; None when the model is outside the supported module set.
        if self._hybrid and not hasattr(self, "_subspace_delta_evaluator"):
            from .cost_nn import SiteVmapEvaluator, SubspaceDeltaEvaluator

            self._subspace_delta_evaluator = SubspaceDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.subspace
            )
            # A per-layer block maps to one parameter, so the same site argument holds
            # in coordinate space. Covers the models the delta path declines.
            self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)

    def release_evaluator(self) -> None:
        """Drop the registered evaluator and its batch data.

        ``register_evaluator`` keeps a reference to the mini-batch it was handed, which
        otherwise outlives the training loop and holds that memory on the device.
        """
        self._cost_evaluator = None
        self._fused_inputs = None
        self._fused_targets = None

    def _screen_data(self, use_fused: bool):
        """Low-fidelity ``(inputs, targets)`` for the fused in-place screen pass.

        A leading slice of the registered batch, not a random subsample: every
        candidate in a step must see the *same* data, or the cost matrix compares
        vertices across different objectives (common random numbers).
        """
        inputs = getattr(self, "_fused_inputs", None)
        targets = getattr(self, "_fused_targets", None)
        if not use_fused or inputs is None or not self.multifidelity_screen:
            return inputs, targets
        n = inputs.shape[0]
        m = max(1, int(round(n * self.screen_fidelity)))
        if m >= n:
            return inputs, targets
        return inputs[:m], (targets[:m] if targets is not None else None)

    def screen_closure_from(self, closure: Callable, inputs: torch.Tensor, targets=None) -> Callable:
        """Wrap a full-fidelity data batch into a cheap screening closure.

        Returns ``None`` when screening is off or the batch is too small to split,
        so callers can pass the result straight to :meth:`step`.
        """
        if not self.multifidelity_screen:
            return None
        n = inputs.shape[0]
        m = max(1, int(round(n * self.screen_fidelity)))
        if m >= n:
            return None
        sub_in = inputs[:m]
        sub_tgt = targets[:m] if targets is not None else None

        def screen(batched_params, _in=sub_in, _tgt=sub_tgt):
            return closure(batched_params, _in, _tgt)

        return screen

    @torch.inference_mode()
    def step(
        self,
        closure: Callable,
        screen_closure: Optional[Callable] = None,
        objective_token: object = None,
    ) -> float:
        """Run one optimization step.

        Samples polytope vertices around current particles, calls the user
        closure to evaluate costs at probe points, solves entropic OT, and
        updates particles via barycentric projection. Optionally applies
        momentum and adaptive radius.

        The closure is called once per chunk of probe positions (possibly
        multiple times if chunking is enabled). It should return a scalar
        loss per candidate parameter configuration.

        Args:
            closure: ``closure(batched_params) -> losses`` where
                ``batched_params`` is ``{key: (N, *shape)}`` and ``losses``
                is a 1D tensor of shape ``(N,)``.
            screen_closure: Optional cheap-fidelity closure with the same
                signature, used by ``multifidelity_screen`` to rank directions
                before spending full-fidelity forwards on the survivors. Build one
                with :meth:`screen_closure_from`. Ignored unless
                ``multifidelity_screen=True``.
            objective_token: Identity of the objective ``closure`` measures. When it
                differs from the previous step's, ``adaptive_probes`` will not reuse
                cached cost rows, since those were measured against a different
                objective. Pass the batch index (or any per-batch value) when the
                closure changes between steps; :func:`~polystep.api.train` does this
                automatically. Leave as ``None`` for a stationary objective, where
                reuse is sound.

        Returns:
            Mean raw model cost for this step (a diagnostic scalar, the mean of
            the cost matrix), not the OT entropic-regularized dual.
        """
        if objective_token != self._prev_objective_token:
            if (
                self._adaptive_probes_explicit
                and self._adaptive_probes
                and self._prev_cost_matrix is not None
                and not self._objective_token_warned
            ):
                self._objective_token_warned = True
                warnings.warn(
                    "adaptive_probes=True but the objective changed between steps "
                    "(objective_token differs), so no cached cost rows can be reused and no "
                    "forward evaluations are saved. Reuse is only sound for a stationary "
                    "objective, e.g. full-batch training.",
                    stacklevel=2,
                )
            self._invalidate_reuse_cache()
            # A prediction made on the previous minibatch cannot be scored against
            # this one's loss.
            self._prev_predicted_improvement = None
            self._prev_pre_step_loss = None
            self._prev_objective_token = objective_token
        # Amortized OT: cheap momentum steps between full OT solves
        if (
            self.amortize_steps > 1
            and self._amortize_counter % self.amortize_steps != 0
            and self._transport_direction_ema is not None
        ):
            # Coasting moves the parameters with nothing predicting the move, so the
            # deferred trust-region comparison would charge them to the last OT step.
            self._prev_predicted_improvement = None
            self._prev_pre_step_loss = None
            result = self._step_momentum(closure)
            self._amortize_counter += 1
            return result

        # Full OT step
        # HybridSubspace runs the monolithic step with per-layer projections.
        if self._subspace_blockwise:
            # Combined subspace + block-wise mode
            result = self._step_subspace_blockwise(closure)
        elif self._blocks is not None:
            result = self._step_blockwise(closure)
        else:
            result = self._step_monolithic(closure, screen_closure)

        self._amortize_counter += 1
        return result

    def _step_monolithic(self, closure: Callable, screen_closure: Optional[Callable] = None) -> float:
        """Monolithic step: single OT solve over all particles.

        Delegates to ``_step_monolithic.step_monolithic()``.
        """
        return _step_monolithic_fn(self, closure, screen_closure)

    def _step_blockwise(self, closure: Callable) -> float:
        """Block-wise step: per-block OT solve with full-model closure calls.

        Delegates to ``_step_blockwise.step_blockwise()``.
        """
        return _step_blockwise_fn(self, closure)

    def _step_subspace_blockwise(self, closure: Callable) -> float:
        """Combined subspace + block-wise step: per-block OT in subspace coords.

        Delegates to ``_step_blockwise.step_subspace_blockwise()``.
        """
        return _step_subspace_blockwise_fn(self, closure)

    def _step_momentum(self, closure: Callable) -> float:
        """Cheap step: reapply the last direction with decay, no forward passes.

        Delegates to ``_step_momentum.step_momentum()``.
        """
        return _step_momentum_fn(self, closure)

    def _transition_rank(self, new_rank: int) -> None:
        """Transition subspace to new rank, preserving accumulated progress via absorb.

        Absorbs the current perturbation into base weights, then reconstructs
        the subspace at the new rank. Resets particles, duals, and displacement
        history to match the new subspace dimension.

        Args:
            new_rank: Target rank for the new subspace.
        """
        state = self._state
        self._applied_rank = new_rank

        # Absorb current perturbation into base weights
        old_subspace = self.subspace
        if isinstance(old_subspace, HybridSubspace):
            flat_sub = state.X.reshape(-1)[: old_subspace.subspace_dim]
            new_base, _ = old_subspace.absorb(
                state.hybrid_projections,
                state.base_params,
                flat_sub,
            )
            state.base_params = new_base
        elif isinstance(old_subspace, LinearSubspace):
            flat_sub = state.X.reshape(-1)[: old_subspace.subspace_dim]
            new_base, _ = old_subspace.absorb(state.base_params, flat_sub)
            state.base_params = new_base
        else:
            # Other subspace types: skip transition
            import warnings

            warnings.warn(f"Rank transition not supported for {type(old_subspace).__name__}, skipping")
            return

        # Reconstruct subspace at new rank
        if isinstance(old_subspace, HybridSubspace):
            # Carry every config field across the transition, so a field added later
            # is not silently dropped at the rank change.
            # Rebuilt by from_layout from the layout itself, so carrying them over would
            # collide with the explicit keyword and pin the old rank's values.
            _structural = {
                "specs",
                "subspace_dim",
                "compression_ratio",
                "_total_params",
                "_max_subspace_dim",
                "_entry_dtypes",
                "seed",
            }
            carried = {
                f.name: getattr(old_subspace, f.name)
                for f in dataclass_fields(old_subspace)
                if f.name not in _structural
            }
            self.subspace = HybridSubspace.from_layout(
                self.layout,
                rank=new_rank,
                seed=old_subspace.seed,
                max_subspace_dim=getattr(old_subspace, "_max_subspace_dim", None),
                **carried,
            )
        elif isinstance(old_subspace, LinearSubspace):
            self.subspace = LinearSubspace.from_layout(
                self.layout,
                rank=new_rank,
                seed=old_subspace.seed,
                max_subspace_dim=getattr(old_subspace, "_max_subspace_dim", None),
            )

        state.subspace = self.subspace

        # Resize particle array for new subspace dimension
        sub_dim = self.subspace.subspace_dim
        pdim = self._subspace_particle_dim
        padded_sub_dim = ((sub_dim + pdim - 1) // pdim) * pdim
        new_X = torch.zeros(
            padded_sub_dim // pdim,
            pdim,
            dtype=state.X.dtype,
            device=state.X.device,
        )
        state.X = new_X

        state.f = None
        state.g = None

        # Per-particle state carries the old particle count, so momentum either raises a
        # shape error next step or, when the old count was 1, broadcasts one stale row.
        if state.velocity is not None:
            state.velocity = torch.zeros_like(new_X)
        if state.p_c is not None:
            state.p_c = torch.zeros(sub_dim, dtype=new_X.dtype, device=new_X.device)
        if state.p_sigma is not None:
            state.p_sigma = torch.zeros(sub_dim, dtype=new_X.dtype, device=new_X.device)
        if state.C_diag is not None:
            state.C_diag = torch.ones(sub_dim, dtype=new_X.dtype, device=new_X.device)

        # Re-initialize displacement history at new subspace dimension
        if isinstance(self.subspace, HybridSubspace):
            state.displacement_history = torch.zeros(
                self.subspace.displacement_history_size,
                self.subspace.subspace_dim,
                dtype=state.X.dtype,
                device=state.X.device,
            )
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            # Regenerate per-layer projections for new subspace
            state.hybrid_projections = self.subspace.init_projections(
                state.X.device,
                state.X.dtype,
            )
            # The fused matrix is per basis, and the step only rebuilds it when the basis
            # object changes. Nothing changes it again after a transition, so skipping
            # this leaves _fused_P at the old rank's shape, or None, for the whole run.
            if hasattr(self.subspace, "build_fused_projection"):
                self.subspace.build_fused_projection(state.hybrid_projections)
            self._hybrid_subspace = self.subspace

        # Update uniform distribution to match new particle count
        num_points = state.X.shape[0]
        state.a = torch.ones(num_points, device=state.X.device, dtype=state.X.dtype) / num_points

        # Clear stale adaptive probe state (shape changed with new rank)
        self._invalidate_reuse_cache()
        self._transport_direction_ema = None
        self._newton_direction = None

        logger.info(
            f"Rank transition: rank={new_rank}, subspace_dim={self.subspace.subspace_dim}, particles={num_points}"
        )

    def _update_sampling_projection(self) -> None:
        """Cache the coord-to-param projection for this step.

        Scaled by sqrt(C_diag) when CMA covariance adaptation is on, else the
        plain projection. Cached at step start so a step's probes and its
        end-of-step sync share one metric (C_diag only changes at step end).

        Absorbs first when the scaling moved. ``X`` is read through this
        projection, so rescaling it under a nonzero ``X`` moves the represented
        parameters without any candidate having scored the move, and the drift
        compounds every step.
        """
        state = self._state
        proj = state.projection
        if not (
            self._cma_subspace
            and self.use_covariance_adaptation
            and isinstance(proj, torch.Tensor)
            and state.C_diag is not None
        ):
            self._sampling_projection = proj
            return

        # Identity, not value: C_diag is always rebound, never written in place, so this
        # skips a clone, a torch.equal sync and the full projection multiply on every
        # step that did not change it. Not the projection itself, which
        # apply_covariance_scaling returns from a cached buffer.
        previous_C = getattr(self, "_sampling_C_diag", None)
        if previous_C is state.C_diag and getattr(self, "_sampling_proj_src", None) is proj:
            return
        if previous_C is not None and previous_C.shape == state.C_diag.shape and self._sampling_projection is not None:
            flat = state.X.reshape(-1)[: state.subspace.subspace_dim]
            if bool(flat.any()):
                # Absorb under the projection X was measured in, before the buffer is
                # overwritten with the new scaling.
                state.base_params, _ = state.subspace.absorb(self._sampling_projection, state.base_params, flat)
                self._base_params = state.base_params
                state.X = torch.zeros_like(state.X)
                self._invalidate_reuse_cache()

        self._sampling_C_diag = state.C_diag
        self._sampling_proj_src = proj
        self._sampling_projection = self.subspace.apply_covariance_scaling(proj, state.C_diag)

    def resync_from_model(self) -> None:
        """Re-read the optimizer's particle state from the model's current weights.

        Call this after writing weights externally (loading a snapshot, clipping,
        an external scheduler). Without it the next ``step`` ends in ``_sync_model``
        and overwrites those weights with the state the optimizer still holds.
        """
        if self.subspace is not None:
            # Subspace coords are relative to base_params, so re-anchor the base and
            # zero the coordinates: the represented point is the model as it stands.
            self._base_params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self._state.base_params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self._state.X = torch.zeros_like(self._state.X)
        else:
            self._state.X = self.layout.flatten(self.model).to(self._state.X.device, self._state.X.dtype)
        self._state.f = None
        self._state.g = None
        self._state.prev_prev_f = None
        self._state.prev_prev_g = None
        self._invalidate_reuse_cache()
        # Every direction and counter below was measured at the old anchor. Keeping them
        # would let the next step() coast along a stale EMA direction without evaluating
        # anything, moving the weights the caller just wrote.
        self._amortize_counter = 0
        self._transport_direction_ema = None
        self._newton_direction = None
        self._prev_descent_direction = None
        self._prev_descent_direction_finite = False
        self._prev_block_descent_directions = None
        self._prev_predicted_improvement = None
        self._prev_pre_step_loss = None
        self._prev_loss_from_center = False
        if self._state.velocity is not None:
            self._state.velocity = torch.zeros_like(self._state.velocity)
        self._state._prev_prev_block_duals = None

    def _write_params(self, sd: dict) -> None:
        """Copy the parameter entries of ``sd`` into the live parameters, once per step.

        ``load_state_dict`` re-walks the module tree, re-validates every shape and
        rebuilds its own bookkeeping on each call. The keys come from the layout built
        from this model, so the mapping is fixed: cache it and issue one grouped copy.

        Non-parameter keys are skipped. In subspace mode ``sd`` is reconstructed from
        ``base_params``, a ``state_dict()`` snapshot taken at construction, so it carries
        buffers as well; writing those back would both bypass this copy for the whole
        dict and revert live buffers (BatchNorm running stats, say) to the snapshot.
        """
        cache = self._param_write_cache
        if cache is None or cache[0] is not self.model:
            cache = self._param_write_cache = (self.model, dict(self.model.named_parameters()))
        params = cache[1]
        keys = [k for k in sd if k in params]
        torch._foreach_copy_([params[k].data for k in keys], [sd[k] for k in keys])

    def _sync_model(self) -> None:
        """Write current particles back to the model."""
        state = self._state

        if self.subspace is not None:
            # Subspace: flatten multi-particle X back to subspace coords,
            # then reconstruct full params from base + perturbation.
            X = state.X  # (num_sub_particles, particle_dim)
            flat_sub = X.reshape(-1)[: state.subspace.subspace_dim]
            if self._per_layer_projections:
                # Hybrid / Factored take the per-layer projections dict
                full_sd = state.subspace.apply_perturbation(
                    state.hybrid_projections,
                    state.base_params,
                    flat_sub,
                )
            elif self._adaptive or self._cma_subspace:
                # AdaptiveSubspace and CMAAdaptiveSubspace require projection argument.
                # Use the step's cached (possibly covariance-scaled) projection so
                # the sync matches the metric the probes were evaluated in.
                proj = self._sampling_projection if self._sampling_projection is not None else state.projection
                full_sd = state.subspace.apply_perturbation(
                    proj,
                    state.base_params,
                    flat_sub,
                )
            else:
                full_sd = state.subspace.apply_perturbation(
                    state.base_params,
                    flat_sub,
                )
            self._write_params(full_sd)
        else:
            # Multi-particle mode: X is already (num_particles, particle_dim)
            particles_2d = state.X
            if particles_2d.dim() == 1:
                particles_2d = particles_2d.reshape(-1, self._particle_dim)

            sd = self.layout.unflatten(particles_2d)
            self._write_params(sd)
