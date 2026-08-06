"""Gradient-free optimizer with a closure-based step interface."""

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

# Below this many params, dense projection is cheaper.
_MIN_PARAMS_FOR_SPARSE = 10_000

_AUTO_SPARSE_THRESHOLD_CPU = 1_000_000
_AUTO_SPARSE_THRESHOLD_GPU = 2_000_000


def _select_projection_type(
    num_params: int,
    device: torch.device,
    projection_type: str,
) -> str:
    """Resolve 'dense' vs 'sparse' projection from device and model size."""
    if projection_type in ("dense", "sparse"):
        return projection_type

    if device.type == "cuda":
        threshold = _AUTO_SPARSE_THRESHOLD_GPU
    else:
        threshold = _AUTO_SPARSE_THRESHOLD_CPU

    if num_params >= threshold:
        return "sparse"
    else:
        return "dense"


from ._compiled import CompiledFunctions
from ._step_core import invalidate_for_basis_change
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
    """Warn once when the intra-op thread pool saturates every core."""
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
    """Progressive rank expansion schedule for subspace optimization."""

    stages: List[Tuple[int, int]]  # (start_step, rank) pairs

    def __post_init__(self):
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
    """Gradient-free optimizer built on entropic optimal transport.

    Updates model weights in-place each ``step(closure)``. Not a
    ``torch.optim.Optimizer`` subclass.

    Args:
        model: Module to optimize; weights are updated in-place.
        polytope_type: 'simplex', 'orthoplex', or 'cube'. Only ``newton_refinement``
            needs the orthoplex's antithetic vertex pairs.
        epsilon: Entropic regularization; a float or a scheduler with ``.at()``.
            With float radii it multiplies both radii, so it is the learning rate.
        ent_epsilon: Separate solver temperature; if None, uses ``epsilon``.
        scale_cost: Cost scaling ('mean', 'max_cost', float, or None).
        step_radius: Step size. A scalar multiplies ``epsilon``; a schedule with
            ``.at()`` is the physical distance.
        probe_radius: Probe radius, same two parameterizations as ``step_radius``. Probe
            ``k`` sits at ``k/(num_probe+1)`` of it, so the default single probe is at half.
        num_probe: Probe points per direction; 1 is optimal.
        adaptive_probes: Reuse the previous cost matrix when nothing moved. Default
            on for monolithic blocks.
        adaptive_probes_threshold: Squared displacement below which the matrix is reused.
        max_iterations: Maximum outer iterations.
        sinkhorn_max_iters: Maximum inner Sinkhorn iterations.
        chunk_size: Chunk size for cost-evaluation memory control.
        cost_batch_size: Mini-batch size for cost evaluation, read by the training loop.
        compile: Compile hot-path tensor functions (default False).
        solver: OT solver. None auto-selects softmax for subspace modes, sinkhorn
            otherwise.
        tempered_softmax_tau: Fixed temperature for 'tempered_softmax'.
        kl_softmax_lam: KL penalty for 'kl_softmax'; 0 is softmax, inf is Sinkhorn.
        subspace: Subspace object, or None for full parameter space.
        subspace_particle_dim: OT polytope dimension in subspace mode (default 8).
        block_strategy: 'monolithic', 'per_layer', or 'grouped'.
        block_group_size: Consecutive entries per block group.
        biased_rotation: Align the chart's first axis with the previous descent direction.
        use_momentum: Accumulate a velocity (default False). Heavy-ball, so at steady
            state the move is ``velocity_lr/(1-beta)`` times the barycentric displacement,
            20x at the default ``momentum_final``. ``step_radius`` does not bound it.
        momentum_init: Starting momentum coefficient.
        momentum_final: Final momentum coefficient, reached linearly.
        velocity_lr: Velocity update learning rate.
        use_adaptive_radius: Stagnation-based radius adaptation.
        stagnation_threshold: Relative change below which is stagnation.
        stagnation_patience: Stagnation iterations before a radius boost.
        use_quadratic_model: Fit gradient and curvature from the vertex losses. Costs one
            shared ``f(X)`` per step and needs ``num_probe=1`` off the orthoplex.
        newton_refinement: Correct the barycentre with a Newton step. Orthoplex only,
            ``num_probe>=2``.
        newton_refinement_alpha: Step size for that correction.
        trust_region: Scale ``step_radius`` by predicted against actual improvement. Both
            must come from the same objective, so it engages only on a stationary one and
            at ``amortize_steps=1``. On a minibatch stream pass ``objective_token`` per
            batch, or the ratio is measured across two different batches and the radius
            collapses to ``radius_min``.
        multifidelity_screen: Screen directions on a cheap fidelity first.
        screen_keep_ratio: Fraction of directions promoted to full fidelity.
        screen_fidelity: Fraction of the batch used for screening.
        radius_increase: Multiplicative factor for radius boost.
        radius_decrease: Multiplicative factor for radius decay.
        radius_min: Minimum radius multiplier.
        radius_max: Maximum radius multiplier.
        use_covariance_adaptation: Diagonal CMA-ES covariance learning (needs
            CMAAdaptiveSubspace).
        seed: Seed for reproducible random rotations.
        mixed_precision: Cast model, particles, and projections to BF16.
        candidate_autocast: Run the candidate forward in BF16 under autocast.
        projection_type: 'dense', 'sparse', or 'auto' for AdaptiveSubspace mode.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        polytope_type: str = "simplex",
        particle_dim: int = 2,
        epsilon: Union[float, LinearEpsilon] = 0.1,
        ent_epsilon: Optional[Union[float, LinearEpsilon]] = None,
        # Feedback-driven epsilon.
        auto_epsilon: bool = False,
        auto_epsilon_config: Optional[dict] = None,
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        # Jitter keeps probes off the discontinuity set; the convergence proof needs it > 0.
        # Default 0.0: it also makes the recorded cost noisy, which the reuse heuristics
        # misread as progress.
        probe_radius_jitter: float = 0.0,
        probe_radius_jitter_dist: str = "smooth",
        # Makes the one-step law of the iterate continuous within its plane.
        step_radius_jitter: float = 0.0,
        num_probe: int = 1,
        # None means "on wherever implemented".
        adaptive_num_probe: Optional[bool] = None,
        adaptive_probe_warmup: int = 20,
        adaptive_probes: Optional[bool] = None,
        adaptive_probes_threshold: float = 1e-6,
        max_iterations: int = 50,
        sinkhorn_max_iters: int = 2000,
        chunk_size: Optional[int] = None,
        cost_batch_size: Optional[int] = None,
        # Alternate full OT steps with cheap momentum steps.
        amortize_steps: int = 1,
        amortize_ema: float = 0.7,
        compile: bool = False,
        solver: Optional[str] = None,
        tempered_softmax_tau: float = 1.0,
        kl_softmax_lam: float = float("inf"),
        subspace: Optional[Union[AdaptiveSubspace, HybridSubspace, CMAAdaptiveSubspace, FactoredSubspace]] = None,
        subspace_particle_dim: int = 8,
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
        # Extrapolate warm-start duals.
        dual_momentum_beta: float = 0.0,
        anderson_depth: int = 0,
        adaptive_omega: bool = False,
        data_dependent_init: bool = False,
        biased_rotation: bool = False,
        # Extract finite-difference gradient/Hessian from cost evaluations.
        use_quadratic_model: bool = False,
        # Post-OT correction using the quadratic model.
        newton_refinement: bool = False,
        newton_refinement_alpha: float = 0.3,
        # Adapt step_radius from predicted vs actual improvement.
        trust_region: bool = False,
        # Rank directions on a cheap fidelity, then pay full fidelity only for the survivors.
        multifidelity_screen: bool = False,
        screen_keep_ratio: float = 0.5,
        screen_fidelity: float = 0.25,
        use_covariance_adaptation: bool = False,
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        candidate_autocast: bool = False,
        projection_type: str = "dense",
        # Fusion-compile the vmapped forward (no CUDA graphs).
        compile_evaluator: bool = False,
        # CUDA-graph the in-place forward. Reached whenever the evaluator selects the
        # in-place path, which it auto-enables for large GPU models, not only under an
        # explicit use_inplace=True.
        compile_forward: Optional[bool] = None,
    ) -> None:
        if projection_type not in ("dense", "sparse", "auto"):
            raise ValueError(f"Invalid projection_type: {projection_type!r}. Use 'dense', 'sparse', or 'auto'.")
        # Downstream code tests `!= "monolithic"`, so an unknown value would silently
        # pick block-wise mode rather than raise.
        if block_strategy not in ("monolithic", "per_layer", "grouped"):
            raise ValueError(
                f"Invalid block_strategy: {block_strategy!r}. Use 'monolithic', 'per_layer', or 'grouped'."
            )
        # Below 1 gives no blocks, and reassembly then zeros every trainable parameter.
        if block_strategy == "grouped" and block_group_size < 1:
            raise ValueError(f"block_group_size must be >= 1, got {block_group_size}.")
        if polytope_type not in POLYTOPE_MAP:
            raise ValueError(f"Invalid polytope_type: {polytope_type!r}. Use one of {sorted(POLYTOPE_MAP)}.")
        self._requested_projection_type = projection_type
        self._compile_evaluator = compile_evaluator
        self._compile_forward = compile_forward

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

        # newton_refinement reads the antithetic vertex order directly. The gradient and
        # the curvature are closed forms on any centred tight frame, but off the orthoplex
        # the curvature comes from a shared f(X), which only num_probe=1 evaluates.
        if polytope_type != "orthoplex":
            if newton_refinement:
                warnings.warn(
                    f"newton_refinement reads the orthoplex's antithetic vertex ordering, but "
                    f"polytope_type={polytope_type!r}. It will not take effect. Pass "
                    f"polytope_type='orthoplex' to use it.",
                    stacklevel=2,
                )
            if (use_quadratic_model or trust_region) and num_probe != 1:
                warnings.warn(
                    f"use_quadratic_model on polytope_type={polytope_type!r} takes its curvature "
                    f"from one shared f(X), which the step only evaluates at num_probe=1; got "
                    f"num_probe={num_probe}, so the model will not take effect. Pass num_probe=1, "
                    f"or polytope_type='orthoplex' to regress across probe scales instead.",
                    stacklevel=2,
                )

        if subspace is not None and particle_dim != 2:
            warnings.warn(
                f"particle_dim={particle_dim} is ignored in subspace mode. "
                f"The OT polytope uses subspace_particle_dim={subspace_particle_dim} instead. "
                f"Pass subspace_particle_dim={particle_dim} to control polytope geometry.",
                stacklevel=2,
            )

        # In combined mode blocks slice subspace coordinates, so entry grouping has
        # nothing to group.
        if subspace is not None and block_strategy == "grouped":
            warnings.warn(
                "block_strategy='grouped' has no effect in subspace mode: blocks divide the "
                "subspace coordinates evenly, not the parameter entries, so it produces exactly "
                "like 'per_layer'. Use block_strategy='per_layer', or drop the subspace to group "
                "parameter entries.",
                stacklevel=2,
            )

        # Global projection compresses to subspace coords, then per-block OT.
        self._subspace_blockwise = subspace is not None and block_strategy != "monolithic"

        # Match the model's dtype, not hardcoded fp32, or a float64 model's
        # reconstruct matmul fails on the mismatch.
        self._mixed_precision = mixed_precision
        # Unlike mixed_precision, autocast leaves the params and casts only the
        # forward, so small perturbations survive.
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

        # Adopt a scheduler passed as ``epsilon`` so it advances like ``auto_epsilon``
        # does, rather than sitting frozen at ``init``.
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
        # A negative radius flips the finite-difference denominator, which the
        # clamp then turns positive and scales the gradient by 1e10. 0 is the
        # no-movement control.
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
        if probe_radius_jitter_dist not in ("smooth", "uniform"):
            raise ValueError(
                f"probe_radius_jitter_dist must be 'smooth' or 'uniform', got {probe_radius_jitter_dist!r}"
            )
        self.probe_radius_jitter = probe_radius_jitter
        self.probe_radius_jitter_dist = probe_radius_jitter_dist
        if not (0.0 <= step_radius_jitter < 1.0):
            raise ValueError(
                f"step_radius_jitter must be in [0, 1), got {step_radius_jitter}. "
                f"Values >= 1 risk a negative effective step radius."
            )
        self.step_radius_jitter = step_radius_jitter
        if num_probe < 1:
            raise ValueError(
                f"num_probe must be >= 1, got {num_probe}. "
                f"At least one probe point per direction is required; "
                f"num_probe=0 yields an empty probe tensor and NaN costs."
            )
        self.num_probe = num_probe
        # The fused softmax path divides by epsilon without the solvers' validation;
        # zero gives NaN, negative inverts the plan into ascent.
        for _name, _value in (("epsilon", epsilon), ("ent_epsilon", ent_epsilon)):
            if isinstance(_value, (int, float)) and _value <= 0:
                raise ValueError(f"{_name} must be > 0, got {_value}.")
        # These run only in the monolithic step; default to None so the ignore warning
        # fires on an explicit True, not the default.
        _savings_default = block_strategy == "monolithic"
        # Whether the caller asked for reuse explicitly.
        self._adaptive_probes_explicit = adaptive_probes is not None
        # Reducing K needs a K above 1, so stay off at the default num_probe=1.
        if adaptive_num_probe is None:
            adaptive_num_probe = _savings_default and num_probe > 1
        adaptive_probes = _savings_default if adaptive_probes is None else adaptive_probes
        self.adaptive_num_probe = adaptive_num_probe
        self._adaptive_probe_warmup = adaptive_probe_warmup
        self._loss_decreasing_count = 0
        # OT-step costs only, to avoid mixing in momentum costs.
        self._ot_step_costs: collections.deque = collections.deque(maxlen=3)
        self._adaptive_probes = adaptive_probes
        self._adaptive_probes_threshold = adaptive_probes_threshold
        # A candidate replaces one row of X, so reuse needs every row unchanged.
        self._prev_X: Optional[torch.Tensor] = None
        self._prev_cost_matrix: Optional[torch.Tensor] = None
        self._param_write_cache = None
        # Cached rotations, restored with the matrix so rows match their vertices.
        self._prev_rot_mats: Optional[torch.Tensor] = None
        # Track K_eff and both radii to invalidate the cached matrix on change.
        self._prev_k_eff: Optional[int] = None
        self._prev_step_r: Optional[float] = None
        self._prev_probe_r: Optional[float] = None
        # Reusing across minibatches would rank vertices partly by which batch they came
        # from. None asserts a stationary objective.
        self._prev_objective_token: object = None
        self._objective_token_warned = False
        self.max_iterations = max_iterations
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 or None, got {chunk_size}.")
        self.chunk_size = chunk_size
        if cost_batch_size is not None and cost_batch_size <= 0:
            # 0 slices an empty batch, so every candidate scores NaN and nothing trains.
            raise ValueError(f"cost_batch_size must be > 0 or None, got {cost_batch_size}.")
        self.cost_batch_size = cost_batch_size
        self.amortize_steps = max(1, amortize_steps)
        self.amortize_ema = amortize_ema

        # Jitter makes the per-step cost noisy, which the reuse heuristics read as
        # progress, so disable them here for every caller.
        if self.probe_radius_jitter > 0.0:
            disabled = []
            if self._adaptive_probes:
                self._adaptive_probes = False
                if self._adaptive_probes_explicit:
                    disabled.append("adaptive_probes")
            if self.amortize_steps > 1:
                self.amortize_steps = 1
                disabled.append("amortize_steps")
            if disabled:
                warnings.warn(
                    f"probe_radius_jitter={self.probe_radius_jitter} disables "
                    f"{', '.join(disabled)}: jitter makes the per-step cost a noisy "
                    f"estimate, which those heuristics read as progress. Set "
                    f"probe_radius_jitter=0.0 to keep them.",
                    stacklevel=2,
                )

        # SNN-like models lose accuracy under a shrinking step_radius. Match on
        # module class-name substrings.
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
        self.biased_rotation = biased_rotation
        self._prev_descent_direction: Optional[torch.Tensor] = None
        self._prev_descent_direction_finite: bool = False
        self.use_quadratic_model = use_quadratic_model
        self._losses_3d = None  # (P, V, K) loss tensor
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
        self._center_loss = None  # (P,) f(X) from the K=1 quadratic model
        self.trust_region = trust_region
        self._trust_region_multiplier = 1.0  # Multiplier on step_radius in [0.1, 3.0]
        self._prev_predicted_improvement = None
        self._prev_pre_step_loss = None  # f(X), or the min-cost proxy when no centre ran
        self._prev_loss_from_center = False
        # Trust region needs the finite-difference model for its predicted-vs-actual ratio.
        if trust_region and not self.use_quadratic_model:
            self.use_quadratic_model = True
            logger.info(
                "trust_region=True auto-enables use_quadratic_model=True "
                "(needed for the predicted-vs-actual ratio test)"
            )
        # They scale the same step_r in opposite directions.
        if trust_region and use_adaptive_radius:
            warnings.warn(
                "trust_region and use_adaptive_radius both scale step_radius, and in "
                "opposite directions: the trust region expands on an accurate step while "
                "the stagnation controller contracts on an improving one. Enable one.",
                stacklevel=2,
            )
        if trust_region and block_strategy != "monolithic":
            warnings.warn(
                f"trust_region is only applied with block_strategy='monolithic'; "
                f"ignored for block_strategy='{block_strategy}'.",
                stacklevel=2,
            )
        # A momentum step predicts nothing, so it drops the pending comparison and the
        # next OT step has none to score.
        if trust_region and amortize_steps > 1:
            warnings.warn(
                f"trust_region never updates at amortize_steps={amortize_steps}: the "
                "momentum steps in between discard the prediction it would score. Pass "
                "amortize_steps=1, or drop trust_region and keep the amortization.",
                stacklevel=2,
            )
        if not 0.0 < screen_keep_ratio <= 1.0:
            raise ValueError(f"screen_keep_ratio must be in (0, 1], got {screen_keep_ratio}")
        if not 0.0 < screen_fidelity <= 1.0:
            raise ValueError(f"screen_fidelity must be in (0, 1], got {screen_fidelity}")
        self.multifidelity_screen = multifidelity_screen
        self.screen_keep_ratio = screen_keep_ratio
        self.screen_fidelity = screen_fidelity
        self._last_screen_savings = 0.0
        #: Candidate evaluations charged by the step itself, so fast paths stay counted.
        self.candidate_evals = 0
        self.subspace = subspace
        self._subspace_particle_dim = subspace_particle_dim
        self._rank_schedule = rank_schedule
        if rank_schedule is not None and subspace is None:
            raise ValueError("rank_schedule requires a subspace")
        # Rank transitions run only in the monolithic step; make it a clear no-op elsewhere.
        if rank_schedule is not None and block_strategy != "monolithic":
            warnings.warn(
                f"rank_schedule is only applied with block_strategy='monolithic'; "
                f"ignored for block_strategy='{block_strategy}'.",
                stacklevel=2,
            )
            self._rank_schedule = None
        # Rank the subspace was last rebuilt at; None until the first step.
        self._applied_rank = None
        self.block_strategy = block_strategy
        self.block_group_size = block_group_size

        # Read self., not the argument: newton_refinement and trust_region auto-enable
        # the quadratic model above.
        if self.use_quadratic_model and block_strategy != "monolithic":
            warnings.warn(
                f"use_quadratic_model=True is not supported with "
                f"block_strategy='{block_strategy}'. Quadratic model will be "
                f"silently disabled for block-wise steps.",
                stacklevel=2,
            )

        self.use_momentum = use_momentum
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.velocity_lr = velocity_lr

        self.use_adaptive_radius = use_adaptive_radius
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_patience = stagnation_patience
        self.radius_increase = radius_increase
        self.radius_decrease = radius_decrease
        self.radius_min = radius_min
        self.radius_max = radius_max

        self._dual_momentum_beta = dual_momentum_beta

        self._cma_subspace = isinstance(subspace, CMAAdaptiveSubspace)

        if use_covariance_adaptation and not self._cma_subspace:
            warnings.warn("use_covariance_adaptation requires CMAAdaptiveSubspace. It will be disabled.")
            use_covariance_adaptation = False

        # CMA covariance scaling and state update run only in the monolithic step.
        if use_covariance_adaptation and self.block_strategy != "monolithic":
            warnings.warn(
                "use_covariance_adaptation is only supported with block_strategy='monolithic'; "
                f"disabled for block_strategy='{self.block_strategy}'."
            )
            use_covariance_adaptation = False

        # Blockwise re-evaluates per block, so these save nothing there.
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

        # A radius boost consumes the stagnation counter, so a stagnation absorb may
        # never reach its own patience.
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
        # Coord-to-param projection for this step, covariance-scaled for CMA, so a
        # step's probes and its sync share one metric.
        self._sampling_projection = None
        # Shape-keyed scratch buffers, reused across steps and overwritten on use.
        self._step_buffers = None

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            raise ValueError("Model has no trainable parameters. PolyStepOptimizer requires at least one parameter.")

        num_params = sum(p.numel() for p in model.parameters())
        self._actual_projection_type = _select_projection_type(
            num_params, model_device, self._requested_projection_type
        )

        if self._requested_projection_type == "auto":
            logger.info(
                f"Auto-selected {self._actual_projection_type} projection for "
                f"{num_params / 1e6:.1f}M params on {model_device}"
            )

        if self._actual_projection_type == "sparse" and num_params < _MIN_PARAMS_FOR_SPARSE:
            logger.info(
                f"Model has {num_params:,} params (<{_MIN_PARAMS_FOR_SPARSE:,}). "
                f"Using dense projection instead of sparse."
            )
            self._actual_projection_type = "dense"

        if mixed_precision:
            if not self._bf16_supported():
                warnings.warn(
                    "BF16 not supported on this device. Falling back to FP32. "
                    "For GPU: requires compute capability >= 7.0 (Volta+)."
                )
                # So the property reports the cast that never happened.
                self._mixed_precision = False
            else:
                model.bfloat16()
                self._model_dtype = torch.bfloat16

        # Build after the cast so the layout captures the BF16 dtype.
        self.layout = ParamLayout.from_module(model, particle_dim=self._full_space_particle_dim)
        # The layout only covers requires_grad params, so a frozen model fails
        # later with an opaque range() error.
        if self.layout.total_params == 0:
            raise ValueError(
                "Model has no parameters with requires_grad=True. PolyStepOptimizer "
                "optimizes the requires_grad parameters, so at least one must be trainable."
            )

        self._adaptive = isinstance(subspace, AdaptiveSubspace)

        self._hybrid = isinstance(subspace, HybridSubspace)
        self._hybrid_subspace = subspace if self._hybrid else None

        # FactoredSubspace shares HybridSubspace's per-layer projection protocol.
        self._factored = isinstance(subspace, FactoredSubspace)
        self._per_layer_projections = self._hybrid or self._factored

        # Full space and one global projection hold every parameter at the layout's
        # dominant dtype, so a minority-dtype weight would be optimized at that precision.
        _param_dtypes = {e.dtype for e in self.layout.entries if e.dtype.is_floating_point}
        if len(_param_dtypes) > 1 and (subspace is None or self._adaptive):
            _names = ", ".join(sorted(str(d) for d in _param_dtypes))
            raise ValueError(
                f"Model mixes parameter dtypes ({_names}). "
                f"{'AdaptiveSubspace' if self._adaptive else 'Full-space mode'} stores every parameter in one "
                f"vector at {self.layout.dominant_dtype}, so the others would be optimized at that precision. "
                "Cast the model to a single dtype, or use HybridSubspace, which gives each "
                "parameter its own projection and keeps its dtype."
            )

        if subspace is not None:
            # Subspace coords reshaped to (num_particles, subspace_particle_dim).
            self._base_params = {k: v.clone() for k, v in model.state_dict().items()}
            sub_dim = subspace.subspace_dim
            pdim = subspace_particle_dim
            # Pad to a multiple of particle_dim.
            padded_sub = sub_dim + (-sub_dim % pdim)
            num_sub_particles = padded_sub // pdim
            # Start at base params, so coordinates are zero.
            x_dtype = self._model_dtype if self._mixed_precision else self.layout.dominant_dtype
            X_init = torch.zeros(
                num_sub_particles,
                pdim,
                dtype=x_dtype,
                device=model_device,
            )
            particle_dim = pdim
        else:
            flat_2d = self.layout.flatten(model)  # (rows, particle_dim)
            X_init = flat_2d.to(model_device)
            particle_dim = self.layout.particle_dim

        self._particle_dim = particle_dim
        device = X_init.device

        # Polytope in particle_dim space (not full parameter space); skipped in
        # block-wise mode, where per-block polytopes are used.
        if block_strategy == "monolithic":
            self._polytope_vertices = POLYTOPE_MAP[polytope_type](
                particle_dim,
                device=device,
                dtype=X_init.dtype,
                radius=1.0,
            )
        else:
            self._polytope_vertices = None  # per-block polytopes used instead

        # Interior probe positions.
        self._probes = torch.linspace(0, 1, num_probe + 2)[1 : num_probe + 1]

        # With one particle, balanced OT pins a uniform plan and the step freezes.
        _single_particle = X_init.shape[0] == 1 and block_strategy == "monolithic"
        if _single_particle and solver is None:
            # Never let the default produce a frozen optimizer.
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

        # KLSoftmax interpolates softmax (lam=0) to Sinkhorn (lam=inf), inheriting
        # each limit's guards.
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

        # Fused softmax path: cost scaling is applied in the step functions, not
        # the compiled kernel.
        self._use_fused_softmax = isinstance(self.solver, SoftmaxSolver)

        self._compiled = CompiledFunctions(compile=compile and torch.cuda.is_available())

        # Created early so AdaptiveSubspace init can use it.
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

        self._init_cma(subspace, seed, model_device)

        if use_momentum:
            self._state.velocity = torch.zeros_like(X_init)

        self._init_blocks(
            subspace,
            subspace_particle_dim,
            block_strategy,
            block_group_size,
            polytope_type,
            X_init,
            device,
        )

        # Transition before the first step so the first sweep runs at the scheduled rank.
        if self._rank_schedule is not None and self.subspace is not None:
            self._transition_rank(self._rank_schedule.at(0))

    def _init_subspace(self, subspace, seed, model_device) -> None:
        """Initialize subspace projections and displacement history."""
        if subspace is None:
            return

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
        """CMA evolution paths; _init_subspace already built the projection and history."""
        self._cma_params = None
        if not self._cma_subspace:
            return

        if self.use_covariance_adaptation and self._actual_projection_type == "sparse":
            # _update_sampling_projection only scales a dense Tensor.
            raise ValueError(
                "use_covariance_adaptation needs a dense projection; "
                f"projection_type resolved to 'sparse' for full_dim={subspace.full_dim}. "
                "Pass projection_type='dense', or drop use_covariance_adaptation."
            )

        if self.use_covariance_adaptation:
            # fp32 accumulators: c_1 ~ 2/(n+1.3)^2 rounds to 1 in bf16, so C_diag
            # would never move.
            cma_state = subspace.init_cma_state(
                device=model_device,
                dtype=torch.float32,
            )
            self._state.p_c = cma_state["p_c"]
            self._state.p_sigma = cma_state["p_sigma"]
            self._state.C_diag = cma_state["C_diag"]
            self._state.generation = 0
            # The step feeds the paths a unit-norm direction, so derive rates at
            # mu_eff = 1 too.
            n = subspace.subspace_dim
            mu_eff = float(subspace.mu_eff) if getattr(subspace, "_mu_eff_explicit", False) else 1.0
            hyperparams = compute_cma_hyperparameters(n, mu_eff)
            self._cma_params = {
                **hyperparams,
                # A rate the caller set explicitly wins over the derived one.
                **getattr(subspace, "_explicit_rates", {}),
                "mu_eff": mu_eff,
                "cov_min": subspace.cov_min,
                "cov_max": subspace.cov_max,
            }
            # Write back so the attribute reports what the step actually runs at.
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

            if self._balanced_ot:
                frozen = sum(1 for b in self._subspace_blocks if b.num_particles == 1)
                if frozen:
                    warnings.warn(
                        f"{frozen} of {len(self._subspace_blocks)} subspace block(s) hold a "
                        "single particle; with a balanced solver the column marginal forces "
                        "a uniform plan there, so those blocks will never move. Use "
                        "solver='softmax', a larger subspace_dim, or a smaller "
                        "subspace_particle_dim.",
                        stacklevel=2,
                    )

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

            # A one-particle block's plan is forced uniform, so that block freezes.
            # Small layers hit this under 'per_layer'.
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
        """Actual projection type in use ('dense' or 'sparse')."""
        return self._actual_projection_type

    @staticmethod
    def _check_temperature(value: float, name: str) -> float:
        """Reject a non-positive temperature the fused kernel would divide by unguarded."""
        if not value > 0:
            raise ValueError(f"{name} must resolve to > 0, got {value}.")
        return value

    def _get_epsilon(self, iteration: int) -> float:
        """Resolve epsilon at current iteration."""
        if self._progressive_epsilon is not None:
            return self._check_temperature(self._progressive_epsilon.at(iteration), "epsilon")
        if hasattr(self.epsilon, "at"):
            return self._check_temperature(self.epsilon.at(iteration), "epsilon")
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
        """Return ``probe_r * (1 + eta)``; a no-op when jitter is 0."""
        eta_max = float(self.probe_radius_jitter)
        if eta_max <= 0.0:
            return probe_r
        return float(probe_r) * (1.0 + self._sample_jitter_pooled(eta_max))

    #: Inverse-CDF knot count for the mollifier sampler.
    _JITTER_ICDF_KNOTS = 4097
    #: Scalar draws are pooled to amortise the one device-to-host sync.
    _JITTER_POOL = 1024

    def _jitter_icdf(self, device) -> "torch.Tensor":
        """Cached inverse CDF of the standard mollifier on (-1, 1)."""
        cache = getattr(self, "_jitter_icdf_cache", None)
        if cache is None:
            cache = self._jitter_icdf_cache = {}
        key = str(device)
        table = cache.get(key)
        if table is not None:
            return table

        # Integrate then invert the density; float64 on CPU once.
        m = 20001
        t = torch.linspace(-1.0, 1.0, m, dtype=torch.float64)
        inner = 1.0 - t * t
        dens = torch.where(inner > 0, torch.exp(-1.0 / inner.clamp_min(1e-300)), torch.zeros_like(t))
        cdf = torch.cumulative_trapezoid(dens, t)
        cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])
        cdf = cdf / cdf[-1]
        # Make the cdf strictly increasing so searchsorted inverts it.
        cdf = torch.cummax(cdf + torch.arange(m, dtype=torch.float64) * 1e-18, dim=0).values
        probs = torch.linspace(0.0, 1.0, self._JITTER_ICDF_KNOTS, dtype=torch.float64)
        idx = torch.searchsorted(cdf, probs).clamp(1, m - 1)
        c0, c1 = cdf[idx - 1], cdf[idx]
        t0, t1 = t[idx - 1], t[idx]
        w = ((probs - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0.0, 1.0)
        table = (t0 + w * (t1 - t0)).to(device=device, dtype=torch.float32)
        cache[key] = table
        return table

    def _sample_jitter_batch(self, eta_max: float, n: int, device=None, dtype=None) -> "torch.Tensor":
        """Draw ``n`` jitters in one shot, no host sync."""
        gen = self._generator
        gen_device = gen.device if gen is not None else None
        u = torch.rand((n,), device=gen_device, generator=gen)

        if self.probe_radius_jitter_dist == "uniform":
            return ((2.0 * u - 1.0) * eta_max).to(device=device, dtype=dtype)

        table = self._jitter_icdf(u.device)
        pos = u * (table.numel() - 1)
        lo = pos.floor().clamp_(0, table.numel() - 2).long()
        frac = pos - lo
        t0 = table.index_select(0, lo)
        t1 = table.index_select(0, lo + 1)
        return (torch.lerp(t0, t1, frac) * eta_max).to(device=device, dtype=dtype)

    def _sample_jitter_pooled(self, eta_max: float) -> float:
        """One scalar draw from a pool, so we sync once per ``_JITTER_POOL`` steps."""
        pool = getattr(self, "_jitter_pool", None)
        if not pool:
            drawn = self._sample_jitter_batch(1.0, self._JITTER_POOL)
            pool = self._jitter_pool = drawn.detach().to("cpu").tolist()
        return float(pool.pop()) * eta_max

    def _apply_particle_step_jitter(self, X: "torch.Tensor", X_bary: "torch.Tensor") -> "torch.Tensor":
        """Return ``X + (1 + eta) * (X_bary - X)`` with independent per-particle ``eta``."""
        eta_max = float(self.step_radius_jitter)
        if eta_max <= 0.0:
            return X_bary
        etas = self._sample_jitter_batch(eta_max, int(X.shape[0]), device=X_bary.device, dtype=X_bary.dtype).unsqueeze(
            -1
        )
        return X + (1.0 + etas) * (X_bary - X)

    def _get_ent_epsilon(self, iteration: int) -> Optional[float]:
        """Resolve ent_epsilon at current iteration."""
        if self.ent_epsilon is None:
            return None
        if hasattr(self.ent_epsilon, "at"):
            return self._check_temperature(self.ent_epsilon.at(iteration), "ent_epsilon")
        return self.ent_epsilon

    def _bf16_supported(self) -> bool:
        """Whether BF16 is supported on the model's device."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return False
        if device.type == "cuda":
            # Also covers emulated bf16 on older cards.
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif device.type == "cpu":
            return True  # may be slower without AMX
        else:
            return True  # let PyTorch error if unsupported

    def _invalidate_reuse_cache(self) -> None:
        """Drop cached cost rows, rotations, and probe state when the cost geometry changes."""
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
        """Whether evaluators should fusion-compile the vmapped forward."""
        return self._compile_evaluator

    @property
    def compile_forward(self) -> Optional[bool]:
        """Whether evaluators should CUDA-graph the in-place forward."""
        return self._compile_forward

    def register_evaluator(
        self,
        evaluator: "NNCostEvaluator",
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> None:
        """Register an evaluator + data for fused in-place evaluation.

        Call before each ``step()`` with the current mini-batch. The evaluator's
        ``loss_fn`` must match the closure's objective.

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
        # Rebuild the fast paths when the objective changes; they cache the model and
        # loss_fn they were built from.
        fastpath_key = (
            id(evaluator.model),
            id(evaluator.loss_fn),
            evaluator._inplace_forced,
            evaluator.autocast_dtype,
        )
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
        # These paths score outside the autocast frame; mixing precisions in one cost
        # matrix would rank candidates by precision.
        if evaluator.autocast_dtype is not None:
            if not getattr(self, "_warned_autocast_fastpath", False):
                self._warned_autocast_fastpath = True
                warnings.warn(
                    "candidate_autocast is on, so the site-aware and delta evaluators are "
                    "off: they score outside the autocast frame, and a cost matrix built "
                    "from both precisions ranks candidates partly by precision. Drop "
                    "candidate_autocast to get those paths back.",
                    stacklevel=2,
                )
            self._factored_evaluator = None
            self._sparse_delta_evaluator = None
            self._site_vmap_evaluator = None
            self._subspace_delta_evaluator = None
            return
        # use_inplace=True asks for the real forward, so it outranks the fast paths.
        if evaluator._inplace_forced:
            # Warn once, not once per step.
            if not getattr(self, "_warned_forced_inplace", False):
                self._warned_forced_inplace = True
                extra = (
                    " FactoredSubspace in particular scores candidates through the low-rank "
                    "identity, which never runs the model's forward at all, so the low-rank "
                    "speedup is off too."
                    if self._factored
                    else ""
                )
                warnings.warn(
                    "use_inplace=True asks for the model's real forward, so the site-aware and "
                    "delta evaluators are off and candidates are materialized through "
                    "reconstruct_batch instead." + extra + " Drop use_inplace=True to get "
                    "them back.",
                    stacklevel=2,
                )
            self._factored_evaluator = None
            self._sparse_delta_evaluator = None
            self._site_vmap_evaluator = None
            self._subspace_delta_evaluator = None
            return
        # FactoredSubspace can score through the low-rank identity, never building a
        # candidate weight.
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
        # Full-space candidates perturb one contiguous run, so only the perturbed tensor
        # needs batching.
        if self.subspace is None and not hasattr(self, "_sparse_delta_evaluator"):
            from .cost_nn import SiteVmapEvaluator, SparseDeltaEvaluator

            self._sparse_delta_evaluator = SparseDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.layout
            )
            # Batches only the perturbed tensor; covers what the delta path declines.
            self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)
        if self._hybrid and not hasattr(self, "_subspace_delta_evaluator"):
            from .cost_nn import SiteVmapEvaluator, SubspaceDeltaEvaluator

            self._subspace_delta_evaluator = SubspaceDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.subspace
            )
            # A per-layer block maps to one parameter, so the same site argument holds
            # in coordinate space.
            self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)

    def release_evaluator(self) -> None:
        """Drop the registered evaluator and its batch data."""
        self._cost_evaluator = None
        self._fused_inputs = None
        self._fused_targets = None

    def _screen_data(self, use_fused: bool):
        """Low-fidelity ``(inputs, targets)`` for the fused in-place screen pass."""
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
        """Wrap a full-fidelity batch into a cheap screening closure, or return None."""
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
        """Run one optimization step and update the model in-place.

        Args:
            closure: ``closure(batched_params) -> losses``, where ``batched_params``
                is ``{key: (N, *shape)}`` and ``losses`` has shape ``(N,)``.
            screen_closure: Cheap-fidelity closure for ``multifidelity_screen``.
            objective_token: Identity of the objective ``closure`` measures; when it
                changes, cached cost rows are not reused. Leave ``None`` for a
                stationary objective.

        Returns:
            Mean raw model cost for this step.
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
                    "objective, e.g. full-batch training. Turning it off for this run.",
                    stacklevel=2,
                )
                # A moving objective never reuses, so stop paying for the X clone and
                # the retained cost matrix every step.
                self._adaptive_probes = False
            self._invalidate_reuse_cache()
            # A prediction on the previous minibatch cannot be scored against this one.
            self._prev_predicted_improvement = None
            self._prev_pre_step_loss = None
            self._prev_objective_token = objective_token
        # Amortized OT: cheap momentum steps between full OT solves.
        if (
            self.amortize_steps > 1
            and self._amortize_counter % self.amortize_steps != 0
            and self._transport_direction_ema is not None
        ):
            # Coasting has nothing predicting the move, so clear the deferred
            # trust-region comparison.
            self._prev_predicted_improvement = None
            self._prev_pre_step_loss = None
            result = self._step_momentum(closure)
            self._amortize_counter += 1
            return result

        if self._subspace_blockwise:
            result = self._step_subspace_blockwise(closure)
        elif self._blocks is not None:
            result = self._step_blockwise(closure)
        else:
            result = self._step_monolithic(closure, screen_closure)

        self._amortize_counter += 1
        return result

    def _step_monolithic(self, closure: Callable, screen_closure: Optional[Callable] = None) -> float:
        """Monolithic step: one OT solve over all particles."""
        return _step_monolithic_fn(self, closure, screen_closure)

    def _step_blockwise(self, closure: Callable) -> float:
        """Block-wise step: per-block OT solve."""
        return _step_blockwise_fn(self, closure)

    def _step_subspace_blockwise(self, closure: Callable) -> float:
        """Combined subspace + block-wise step: per-block OT in subspace coords."""
        return _step_subspace_blockwise_fn(self, closure)

    def _step_momentum(self, closure: Callable) -> float:
        """Cheap step: reapply the last direction with decay, no forward passes."""
        return _step_momentum_fn(self, closure)

    def _transition_rank(self, new_rank: int) -> None:
        """Absorb the perturbation, then rebuild the subspace at ``new_rank``."""
        state = self._state

        old_subspace = self.subspace
        if not isinstance(old_subspace, HybridSubspace):
            import warnings

            warnings.warn(f"Rank transition not supported for {type(old_subspace).__name__}, skipping")
            return

        flat_sub = state.X.reshape(-1)[: old_subspace.subspace_dim]
        new_base, _ = old_subspace.absorb(state.hybrid_projections, state.base_params, flat_sub)
        state.base_params = new_base

        # Carry every non-structural config field across the rank change.
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
            f.name: getattr(old_subspace, f.name) for f in dataclass_fields(old_subspace) if f.name not in _structural
        }
        self.subspace = HybridSubspace.from_layout(
            self.layout,
            rank=new_rank,
            seed=old_subspace.seed,
            max_subspace_dim=getattr(old_subspace, "_max_subspace_dim", None),
            **carried,
        )

        state.subspace = self.subspace

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

        # Per-particle state carries the old count; reset to avoid a shape error or
        # one broadcast stale row.
        if state.velocity is not None:
            state.velocity = torch.zeros_like(new_X)
        if state.p_c is not None:
            state.p_c = torch.zeros(sub_dim, dtype=new_X.dtype, device=new_X.device)
        if state.p_sigma is not None:
            state.p_sigma = torch.zeros(sub_dim, dtype=new_X.dtype, device=new_X.device)
        if state.C_diag is not None:
            state.C_diag = torch.ones(sub_dim, dtype=new_X.dtype, device=new_X.device)

        if isinstance(self.subspace, HybridSubspace):
            state.displacement_history = torch.zeros(
                self.subspace.displacement_history_size,
                self.subspace.subspace_dim,
                dtype=state.X.dtype,
                device=state.X.device,
            )
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            state.hybrid_projections = self.subspace.init_projections(
                state.X.device,
                state.X.dtype,
            )
            # The fused matrix is per basis; rebuild it for the new rank's shape.
            if hasattr(self.subspace, "build_fused_projection"):
                self.subspace.build_fused_projection(state.hybrid_projections)
            self._hybrid_subspace = self.subspace

        num_points = state.X.shape[0]
        state.a = torch.ones(num_points, device=state.X.device, dtype=state.X.dtype) / num_points

        # Duals, directions, and the reuse cache were measured in the outgoing basis.
        invalidate_for_basis_change(self, state)

        # Record only once the rebuild succeeded.
        self._applied_rank = new_rank

        logger.info(
            f"Rank transition: rank={new_rank}, subspace_dim={self.subspace.subspace_dim}, particles={num_points}"
        )

    def _update_sampling_projection(self) -> None:
        """Cache the coord-to-param projection, covariance-scaled under CMA.

        Absorbs first when the scaling moved, since rescaling under a nonzero ``X``
        moves the represented parameters without any candidate scoring the move.
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

        # Compare by identity, not value, to skip work on unchanged steps.
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
                # Re-anchoring changes the metric, so cached directions index the old frame.
                invalidate_for_basis_change(self, state)
            else:
                # The coord-to-param map moved, so cached costs no longer describe their points.
                self._invalidate_reuse_cache()

        self._sampling_C_diag = state.C_diag
        self._sampling_proj_src = proj
        self._sampling_projection = self.subspace.apply_covariance_scaling(proj, state.C_diag)

    def resync_from_model(self) -> None:
        """Re-read particle state from the model's current weights, after external writes."""
        if self.subspace is not None:
            # Re-anchor the base and zero the coordinates; share one clone so an
            # absorb update reaches both.
            base = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self._base_params = base
            self._state.base_params = base
            self._state.X = torch.zeros_like(self._state.X)
            self._realign_subspace_to_model()
        else:
            self._state.X = self.layout.flatten(self.model).to(self._state.X.device, self._state.X.dtype)
        self._state.f = None
        self._state.g = None
        self._state.prev_prev_f = None
        self._state.prev_prev_g = None
        self._invalidate_reuse_cache()
        # These were measured at the old anchor; keep them and the next step coasts
        # on a stale direction.
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
        # Reset counters measured at the old weights, or the next step could fire a
        # spurious stagnation absorb.
        self._state.stagnation_count = 0
        self._state.prev_loss = float("inf")
        self._loss_decreasing_count = 0
        self._ot_step_costs.clear()

    def _realign_subspace_to_model(self) -> None:
        """Move projections and particles onto the model's current dtype/device."""
        ref = next(iter(self.model.parameters()), None)
        if ref is None:
            return
        state = self._state
        if state.X.device != ref.device or state.X.dtype != ref.dtype:
            state.X = state.X.to(device=ref.device, dtype=ref.dtype)
            if state.velocity is not None:
                state.velocity = state.velocity.to(device=ref.device, dtype=ref.dtype)

        def _move(p):
            return p.to(device=ref.device, dtype=ref.dtype) if isinstance(p, torch.Tensor) else p

        if state.hybrid_projections is not None:
            state.hybrid_projections = {k: _move(v) for k, v in state.hybrid_projections.items()}
            if hasattr(self.subspace, "build_fused_projection"):
                self.subspace.build_fused_projection(state.hybrid_projections)
        if state.projection is not None:
            state.projection = _move(state.projection)
        self._sampling_projection = None
        self._sampling_C_diag = None
        self._sampling_proj_src = None

    def _write_params(self, sd: dict) -> None:
        """Copy the parameter entries of ``sd`` into the live parameters, once per step.

        Non-parameter keys are skipped, so subspace snapshots do not revert live
        buffers like BatchNorm running stats.
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
            # Flatten X back to subspace coords, then rebuild params from base + perturb.
            X = state.X  # (num_sub_particles, particle_dim)
            flat_sub = X.reshape(-1)[: state.subspace.subspace_dim]
            if self._per_layer_projections:
                full_sd = state.subspace.apply_perturbation(
                    state.hybrid_projections,
                    state.base_params,
                    flat_sub,
                )
            elif self._adaptive:
                # Use the step's cached projection so the sync matches the probe metric.
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
            particles_2d = state.X
            if particles_2d.dim() == 1:
                particles_2d = particles_2d.reshape(-1, self._particle_dim)

            sd = self.layout.unflatten(particles_2d)
            self._write_params(sd)
