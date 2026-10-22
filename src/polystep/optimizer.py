"""Gradient-free optimizer with a closure-based step interface."""

from __future__ import annotations

import collections
import logging
import os
import warnings
from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from ._compiled import CompiledFunctions
from ._step_core import invalidate_for_basis_change
from .blockwise import (
    BlockConfig,
    create_grouped_blocks,
    create_per_layer_blocks,
)
from .cost_nn import (
    FactoredEvaluator,
    NNCostEvaluator,
    SiteVmapEvaluator,
    SparseDeltaEvaluator,
    SubspaceDeltaEvaluator,
    _first_input_tensor,
    _reuse_prefixes,
)
from .costs import resolve_cost_scale
from .epsilon import LinearEpsilon, ProgressiveEpsilon
from .geometry import POLYTOPE_MAP
from .projection import SparseRandomProjection
from .solvers import (
    KLSoftmaxSolver,
    MinCostGreedySolver,
    SinkhornSolver,
    SoftmaxSolver,
    TopKMeanSolver,
)
from .solver import SolverState
from .adaptive_subspace import AdaptiveSubspace
from .factored_subspace import FactoredSubspace
from .hybrid_subspace import HybridSubspace, _mixed_entry_dtypes
from ._serialization import SerializationMixin
from .transform import ParamLayout, _trainable_owners, create_generator

from ._step_monolithic import step_monolithic as _step_monolithic_fn
from ._step_blockwise import step_blockwise as _step_blockwise_fn
from ._step_momentum import step_momentum as _step_momentum_fn

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
            f"pool, where OpenMP spin-wait can dominate PolyStep's many small forwards, "
            f"most of all when other processes share the CPU. Small models often run as "
            f"fast on one thread: call torch.set_num_threads(1) before building the "
            f"optimizer, or run `python -m polystep.benchmarks.threads` to find the "
            f"fastest count on this machine. See docs/performance.md.",
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
        ent_epsilon: Separate solver temperature; if None, uses ``epsilon``. Not allowed
            with a ``ProgressiveEpsilon``.
        scale_cost: Cost scaling ('mean', 'max_cost', float, or None).
        step_radius: Step size. A scalar multiplies ``epsilon``; a schedule with
            ``.at()`` is the physical distance.
        probe_radius: Probe radius, same two parameterizations as ``step_radius``. Probe
            ``k`` sits at ``k/(num_probe+1)`` of it, so the default single probe is at half.
        num_probe: Probe points per direction; 1 is optimal.
        max_iterations: Maximum outer iterations.
        sinkhorn_max_iters: Maximum inner Sinkhorn iterations.
        chunk_size: Chunk size for cost-evaluation memory control.
        cost_batch_size: Mini-batch size for cost evaluation, read by step_batch() and train().
        compile: Compile hot-path tensor functions (default False).
        solver: OT solver. None auto-selects softmax for subspace modes, sinkhorn
            otherwise.
        kl_softmax_lam: KL penalty for 'kl_softmax'; 0 is softmax, inf is Sinkhorn.
        subspace: Subspace object, or None for full parameter space.
        subspace_particle_dim: OT polytope dimension in subspace mode (default 8).
        block_strategy: 'monolithic', 'per_layer', or 'grouped'. With a subspace, the
            last two only split Sinkhorn and KL solves into row segments.
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
        use_quadratic_model: Fit gradient and curvature from the vertex losses when
            biased_rotation or trust_region consumes the model. Costs one shared
            ``f(X)`` per step on every polytope.
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
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        # Jitter keeps probes off the discontinuity set; the convergence proof needs it > 0.
        # Default 0.0: it also makes the recorded cost noisy, which amortization misreads
        # as progress.
        probe_radius_jitter: float = 0.0,
        probe_radius_jitter_dist: str = "smooth",
        # Makes the one-step law of the iterate continuous within its plane.
        step_radius_jitter: float = 0.0,
        num_probe: int = 1,
        # None means "on wherever implemented", except under orthoplex Newton refinement.
        adaptive_num_probe: Optional[bool] = None,
        adaptive_probe_warmup: int = 20,
        max_iterations: int = 50,
        sinkhorn_max_iters: int = 2000,
        chunk_size: Optional[int] = None,
        cost_batch_size: Optional[int] = None,
        # Alternate full OT steps with cheap momentum steps.
        amortize_steps: int = 1,
        amortize_ema: float = 0.7,
        compile: bool = False,
        solver: Optional[str] = None,
        kl_softmax_lam: float = float("inf"),
        subspace: Optional[Union[AdaptiveSubspace, HybridSubspace, FactoredSubspace]] = None,
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
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        candidate_autocast: bool = False,
        projection_type: str = "dense",
        # Fusion-compile the vmapped forward (no CUDA graphs).
        compile_evaluator: bool = False,
        # CUDA-graph the in-place forward and the site-local vmap on CUDA. The in-place
        # path is reached whenever the evaluator selects it, which it auto-enables for
        # large GPU models, not only under an explicit use_inplace=True.
        compile_forward: Optional[bool] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"PolyStepOptimizer takes the nn.Module itself, not {type(model).__name__}; "
                f"freeze parameters with requires_grad_(False) before construction."
            )
        model = getattr(model, "_orig_mod", model)
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
        if subspace is not None and not isinstance(subspace, (AdaptiveSubspace, HybridSubspace, FactoredSubspace)):
            raise TypeError(
                f"subspace must be an AdaptiveSubspace, HybridSubspace or FactoredSubspace, "
                f"got {type(subspace).__name__}."
            )
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
        # the curvature comes from one shared f(X) evaluation.
        if polytope_type != "orthoplex":
            if newton_refinement:
                warnings.warn(
                    f"newton_refinement reads the orthoplex's antithetic vertex ordering, but "
                    f"polytope_type={polytope_type!r}. It will not take effect. Pass "
                    f"polytope_type='orthoplex' to use it.",
                    stacklevel=2,
                )

        if subspace is not None and particle_dim != 2:
            warnings.warn(
                f"particle_dim={particle_dim} is ignored in subspace mode. "
                f"The OT polytope uses subspace_particle_dim={subspace_particle_dim} instead. "
                f"Pass subspace_particle_dim={particle_dim} to control polytope geometry.",
                stacklevel=2,
            )

        _full_blockwise = subspace is None and block_strategy != "monolithic"

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

        self._progressive_epsilon = epsilon if isinstance(epsilon, ProgressiveEpsilon) else None
        if self._progressive_epsilon is not None and ent_epsilon is not None:
            raise ValueError("ProgressiveEpsilon adapts the solve epsilon; it cannot be combined with ent_epsilon.")
        resolve_cost_scale(torch.ones((), dtype=torch.float64, device="cpu"), scale_cost)
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
        _savings_default = not _full_blockwise
        # Reducing K needs a K above 1, so stay off at the default num_probe=1 and under
        # Newton refinement, which needs K>=2.
        _refines = _savings_default and newton_refinement and polytope_type == "orthoplex"
        if adaptive_num_probe is None:
            adaptive_num_probe = _savings_default and num_probe > 1 and not _refines
        elif adaptive_num_probe and _refines and num_probe > 1:
            warnings.warn(
                "adaptive_num_probe drops num_probe to 1 while costs keep falling, and "
                "newton_refinement needs num_probe>=2, so the refinement pauses on those "
                "steps. Pass adaptive_num_probe=False to keep it on.",
                stacklevel=2,
            )
        self.adaptive_num_probe = adaptive_num_probe
        self._adaptive_probe_warmup = adaptive_probe_warmup
        self._loss_decreasing_count = 0
        # OT-step costs only, to avoid mixing in momentum costs.
        self._ot_step_costs: collections.deque = collections.deque(maxlen=3)
        self._prev_objective_token: object = None
        self.max_iterations = max_iterations
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 or None, got {chunk_size}.")
        self.chunk_size = chunk_size
        if cost_batch_size is not None and cost_batch_size <= 0:
            # 0 slices an empty batch, so every candidate scores NaN and nothing trains.
            raise ValueError(f"cost_batch_size must be > 0 or None, got {cost_batch_size}.")
        self.cost_batch_size = cost_batch_size
        self.amortize_steps = max(1, amortize_steps)
        if not 0.0 <= amortize_ema <= 1.0:
            raise ValueError(f"amortize_ema must be in [0, 1], got {amortize_ema}.")
        self.amortize_ema = amortize_ema

        # Jitter makes the per-step cost noisy, which amortization reads as progress, so
        # disable it here for every caller.
        if self.probe_radius_jitter > 0.0 and self.amortize_steps > 1:
            self.amortize_steps = 1
            warnings.warn(
                f"probe_radius_jitter={self.probe_radius_jitter} disables amortize_steps: "
                f"jitter makes the per-step cost a noisy estimate, which amortization reads "
                f"as progress. Set probe_radius_jitter=0.0 to keep it.",
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
        self._prev_pre_step_loss = None  # f(X) at the centre
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
        if trust_region and _full_blockwise:
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
        if rank_schedule is not None and not isinstance(subspace, HybridSubspace):
            raise ValueError(f"rank_schedule requires a HybridSubspace, got {type(subspace).__name__}")
        # Rank the subspace was last rebuilt at; None until the first step.
        self._applied_rank = None
        self.block_strategy = block_strategy
        self.block_group_size = block_group_size

        # Read self., not the argument: newton_refinement and trust_region auto-enable
        # the quadratic model above.
        if self.use_quadratic_model and _full_blockwise:
            warnings.warn(
                f"use_quadratic_model=True is not supported with "
                f"block_strategy='{block_strategy}'. Quadratic model will be "
                f"silently disabled for block-wise steps.",
                stacklevel=2,
            )

        for _name, _value in (("momentum_init", momentum_init), ("momentum_final", momentum_final)):
            if not 0.0 <= _value < 1.0:
                raise ValueError(f"{_name} must be in [0, 1), got {_value}.")
        if not velocity_lr > 0:
            raise ValueError(f"velocity_lr must be > 0, got {velocity_lr}.")
        if not 0.0 < radius_min <= radius_max:
            raise ValueError(f"radius_min and radius_max must satisfy 0 < min <= max, got {radius_min}, {radius_max}.")
        if not radius_increase >= 1.0:
            raise ValueError(f"radius_increase must be >= 1, got {radius_increase}.")
        if not 0.0 < radius_decrease <= 1.0:
            raise ValueError(f"radius_decrease must be in (0, 1], got {radius_decrease}.")
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

        # Blockwise re-evaluates per block, so these save nothing there.
        if _full_blockwise:
            _ignored = [
                name
                for name, on in (
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
        if isinstance(subspace, AdaptiveSubspace) and 0 < subspace.rotation_interval <= 1:
            _cleared = [
                name
                for name, on in (
                    ("amortize_steps", self.amortize_steps > 1),
                    ("biased_rotation", biased_rotation),
                    ("use_momentum", use_momentum),
                )
                if on
            ]
            if _cleared:
                warnings.warn(
                    f"AdaptiveSubspace with rotation_interval={subspace.rotation_interval} changes its "
                    "basis every step, and each change clears the momentum velocity and the coasting "
                    f"and biased-rotation directions, so nothing from {', '.join(_cleared)} carries "
                    "over to the next step. Pass a larger rotation_interval to use them.",
                    stacklevel=2,
                )

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
        self._trainable_names = frozenset(n for n, p in model.named_parameters() if p.requires_grad)

        self._adaptive = isinstance(subspace, AdaptiveSubspace)

        self._hybrid = isinstance(subspace, HybridSubspace)

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

        # Polytope in particle_dim space (not full parameter space).
        self._polytope_vertices = POLYTOPE_MAP[polytope_type](
            particle_dim,
            device=device,
            dtype=X_init.dtype,
            radius=1.0,
        )

        # Interior probe positions.
        self._probes = torch.linspace(0, 1, num_probe + 2)[1 : num_probe + 1]

        _num_vertices = self._polytope_vertices.shape[0]
        _candidates = X_init.shape[0] * _num_vertices * num_probe
        if _candidates > 100_000:
            warnings.warn(
                f"Each step scores {_candidates:,} candidates ({X_init.shape[0]:,} particles x "
                f"{_num_vertices} vertices x {num_probe} probes). For a neural network, pass "
                "subspace=HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4, "
                "max_subspace_dim=...) to bound it.",
                stacklevel=2,
            )

        # With one particle, balanced OT pins a uniform plan and the step freezes.
        _single_particle = X_init.shape[0] == 1 and not _full_blockwise
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
            self.solver = SoftmaxSolver()
        elif solver == "sinkhorn":
            self.solver = SinkhornSolver(
                max_iterations=sinkhorn_max_iters,
                compile=compile,
                anderson_depth=anderson_depth,
                adaptive_omega=adaptive_omega,
            )
        elif solver == "min_cost_greedy":
            self.solver = MinCostGreedySolver()
        elif solver == "top_k_mean":
            self.solver = TopKMeanSolver()
        elif solver == "kl_softmax":
            self.solver = KLSoftmaxSolver(
                lam=kl_softmax_lam,
                max_iterations=sinkhorn_max_iters,
            )
        else:
            raise ValueError(
                f"Unknown solver: {solver!r}. Expected 'softmax', 'sinkhorn', "
                f"'min_cost_greedy', 'top_k_mean', 'kl_softmax', or None. For a fixed "
                f"softmax temperature, use solver='softmax' with ent_epsilon."
            )

        # KLSoftmax interpolates softmax (lam=0) to Sinkhorn (lam=inf), inheriting
        # each limit's guards.
        _one_shot = isinstance(self.solver, (SoftmaxSolver, MinCostGreedySolver, TopKMeanSolver)) or (
            isinstance(self.solver, KLSoftmaxSolver) and self.solver.lam == 0
        )
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

        self._compiled = CompiledFunctions(compile=compile)

        # Created early so AdaptiveSubspace init can use it.
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = create_generator(seed, device)

        self._state = SolverState(X=X_init.clone())

        if subspace is not None:
            self._state.base_params = {
                k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)
            }
            self._state.subspace = subspace

        self._init_subspace(subspace, seed, model_device)

        if use_momentum:
            self._state.velocity = torch.zeros_like(X_init)

        self._init_blocks(subspace, block_strategy, block_group_size, _one_shot)

        # Transition before the first step so the first sweep runs at the scheduled rank.
        if self._rank_schedule is not None:
            self._transition_rank(self._rank_schedule.at(0))
        if self._applied_rank is None:
            self._warn_frozen_segments()

    def _init_subspace(self, subspace, seed, model_device) -> None:
        """Initialize subspace projections and displacement history."""
        if subspace is None:
            return

        if self._adaptive:
            if self._actual_projection_type == "sparse":
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
            if self._hybrid:
                subspace._entry_dtypes = _mixed_entry_dtypes(self.layout)
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

    def _init_blocks(self, subspace, block_strategy, block_group_size, one_shot) -> None:
        """Initialize block-wise decomposition: blocks, duals, row segments."""
        self._blocks: Optional[List[BlockConfig]] = None
        self._row_segments: Optional[int] = None

        if subspace is not None:
            if block_strategy == "monolithic":
                return
            if one_shot or block_strategy == "grouped":
                detail = (
                    f"{type(self.solver).__name__} is one-sided, so every particle's row is "
                    "already solved on its own and the step is the monolithic one. Use "
                    "block_strategy='monolithic'."
                    if one_shot
                    else "Segments slice particles, not parameter entries, so Sinkhorn and KL "
                    "solves split exactly as under 'per_layer'. Use block_strategy='per_layer', "
                    "or drop the subspace to group parameter entries."
                )
                warnings.warn(
                    f"block_strategy={block_strategy!r} has no effect in subspace mode. {detail}",
                    stacklevel=3,
                )
            if not one_shot:
                self._row_segments = max(2, min(len(self.layout.entries), 8))

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

    def _warn_frozen_segments(self) -> None:
        """Warn when a row segment of the current particle count holds a single particle."""
        P = self._state.X.shape[0]
        segments = min(self._row_segments or 0, P)
        if not self._balanced_ot or segments < 2:
            return
        frozen = [P // segments + (i < P % segments) for i in range(segments)].count(1)
        if frozen:
            warnings.warn(
                f"{frozen} of {segments} subspace row segment(s) hold a single particle; "
                "with a balanced solver the column marginal forces a uniform plan there, "
                "so those particles will never move. Use solver='softmax', a larger "
                "subspace_dim, or a smaller subspace_particle_dim.",
                stacklevel=3,
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

    def _bf16_supported(self) -> bool:
        """Whether BF16 is supported on the model's device."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return False
        # Also covers emulated bf16 on older cards.
        return device.type != "cuda" or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    @property
    def compile_evaluator(self) -> bool:
        """Whether evaluators should fusion-compile the vmapped forward."""
        return self._compile_evaluator

    @property
    def compile_forward(self) -> Optional[bool]:
        """Whether evaluators should CUDA-graph the in-place and site-local forwards."""
        return self._compile_forward

    def register_evaluator(
        self,
        evaluator: "NNCostEvaluator",
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Callable:
        """Register an evaluator + data for fused in-place evaluation.

        Call before each ``step()`` with the current mini-batch. The evaluator's
        ``loss_fn`` must match the closure's objective.

        Args:
            evaluator: NNCostEvaluator instance.
            inputs: Current mini-batch inputs.
            targets: Current mini-batch targets (optional).

        Returns:
            Closure that scores this batch; pass it to ``step()``.
        """
        evaluator._refresh_recast_buffers()
        self._cost_evaluator = evaluator
        self._fused_inputs = inputs
        self._fused_targets = targets
        self._registration_fresh = True

        def closure(batched_params, _in=inputs, _tgt=targets):
            return evaluator.evaluate(batched_params, _in, _tgt)

        self._registration_closure = closure

        if self._candidate_autocast_dtype is not None:
            evaluator.autocast_dtype = self._candidate_autocast_dtype
        if self._compile_evaluator:
            evaluator._compile_vmap = True
        # None leaves the evaluator's own setting; only an explicit True/False overrides it.
        if self._compile_forward is not None:
            evaluator._compile_forward = self._compile_forward
        half = any(p.dtype in (torch.float16, torch.bfloat16) for p in evaluator.model.parameters())
        # Rebuild the fast paths when the objective changes; they cache the model and
        # loss_fn they were built from.
        fastpath_key = (
            id(evaluator),
            evaluator._reset_count,
            id(evaluator.model),
            id(evaluator.loss_fn),
            evaluator._inplace_forced,
            evaluator.autocast_dtype,
            half,
            inputs.dim() == 2 if isinstance(inputs, torch.Tensor) else None,
        )
        rebuild = getattr(self, "_fastpath_key", None) != fastpath_key
        if rebuild:
            self._fastpath_key = fastpath_key
            self._act_ratios = None
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
            return closure
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
                    "use_inplace=True asks for the model's real forward, so the vmap and delta "
                    "evaluators are off. A full-space candidate, or an unscreened HybridSubspace "
                    "candidate on FP32/FP64 weights, that changes one parameter is written into it in "
                    "place; other candidates are scored one full weight set at a time."
                    + extra
                    + " Drop use_inplace=True to get them back.",
                    stacklevel=2,
                )
            self._factored_evaluator = None
            self._sparse_delta_evaluator = None
            self._subspace_delta_evaluator = None
            if not hasattr(self, "_site_vmap_evaluator"):
                self._site_vmap_evaluator = (
                    SiteVmapEvaluator.try_build(evaluator, self.layout)
                    if self.subspace is None or self._hybrid
                    else None
                )
            return closure
        if rebuild:
            self._act_ratios = evaluator._activation_ratios(inputs)
        # FactoredSubspace can score through the low-rank identity, never building a
        # candidate weight.
        if self._factored and not hasattr(self, "_factored_evaluator"):
            self._factored_evaluator = FactoredEvaluator.try_build(evaluator.model, evaluator.loss_fn)
            if self._factored_evaluator is None:
                warnings.warn(
                    f"FactoredSubspace is in use but {type(evaluator.model).__name__} is not a "
                    "chain of Linear and activation layers, so candidates must be "
                    "materialized through reconstruct_batch and the low-rank speedup does not apply.",
                    stacklevel=2,
                )
        # Full-space candidates perturb one contiguous run, so only the perturbed tensor
        # needs batching.
        if self.subspace is None and not hasattr(self, "_sparse_delta_evaluator"):
            self._sparse_delta_evaluator = SparseDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.layout
            )
            # Batches only the perturbed tensor; covers what the delta path declines.
            self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)
        if self._per_layer_projections and not hasattr(self, "_subspace_delta_evaluator"):
            self._subspace_delta_evaluator = SubspaceDeltaEvaluator.try_build(
                evaluator.model, evaluator.loss_fn, self.subspace
            )
            # A per-layer block maps to one parameter, so the same site argument holds
            # in coordinate space.
            if self._hybrid:
                self._site_vmap_evaluator = SiteVmapEvaluator.try_build(evaluator, self.layout)
        if half:
            self._factored_evaluator = None
            self._subspace_delta_evaluator = None
        tensor = isinstance(inputs, torch.Tensor)
        if not tensor or inputs.dim() != 2:
            for attr in ("_factored_evaluator", "_sparse_delta_evaluator", "_subspace_delta_evaluator"):
                ev = getattr(self, attr, None)
                if ev is not None and not (tensor and any(tag == "flatten" for _, tag, _ in ev._layer_keys)):
                    setattr(self, attr, None)
        return closure

    def release_evaluator(self) -> None:
        """Drop the registered evaluator and its batch data."""
        for attr in ("_cost_evaluator", "_factored_evaluator", "_sparse_delta_evaluator", "_subspace_delta_evaluator"):
            ev = getattr(self, attr, None)
            if ev is not None:
                ev._cast_cache = None
        site = getattr(self, "_site_vmap_evaluator", None)
        if site is not None:
            site._drop_graphs()
        self._cost_evaluator = None
        self._fused_inputs = None
        self._fused_targets = None
        self._registration_fresh = False
        self._registration_closure = None

    def _screen_slice(self, inputs: torch.Tensor, targets=None):
        """Leading ``screen_fidelity`` rows of a batch and the batch share they realize."""
        if not isinstance(inputs, torch.Tensor):
            return inputs, targets, 1.0
        n = inputs.shape[0] if inputs.dim() else 1
        m = max(1, int(round(n * self.screen_fidelity)))
        if m >= n:
            return inputs, targets, 1.0
        return inputs[:m], (targets[:m] if targets is not None else None), m / n

    def _screen_data(self, screen_closure: Callable):
        """Low-fidelity ``(inputs, targets, share)`` the screen scores on."""
        inputs = getattr(self, "_fused_inputs", None)
        if inputs is None:
            return None, None, getattr(screen_closure, "fidelity", self.screen_fidelity)
        return self._screen_slice(inputs, getattr(self, "_fused_targets", None))

    def screen_closure_from(self, closure: Callable, inputs: torch.Tensor, targets=None) -> Callable:
        """Wrap a full-fidelity batch into a cheap screening closure, or return None.

        The screen calls ``closure(batched_params, inputs, targets)`` on the first
        ``screen_fidelity`` share of the batch, while ``step()`` calls
        ``closure(batched_params)``. One closure serves both when it takes the batch
        as default arguments, as in ``def closure(params, x=inputs, y=targets)``.
        """
        if not self.multifidelity_screen:
            return None
        sub_in, sub_tgt, share = self._screen_slice(inputs, targets)
        if sub_in is inputs:
            return None

        def screen(batched_params, _in=sub_in, _tgt=sub_tgt):
            return closure(batched_params, _in, _tgt)

        screen.fidelity = share
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
                is ``{key: (N, *shape)}`` and ``losses`` has shape ``(N,)``. The
                tensors may be expanded views or share storage that the next chunk
                overwrites, so treat them as read-only and clone any you keep after
                the closure returns.
            screen_closure: Cheap-fidelity closure for ``multifidelity_screen``.
            objective_token: Identity of the objective ``closure`` measures; a change
                drops the pending trust-region comparison. ``None`` or a constant means
                a stationary objective.

        Returns:
            Mean raw model cost for this step.
        """
        if frozenset(n for n, p in self.model.named_parameters() if p.requires_grad) != self._trainable_names:
            raise ValueError(
                "The set of requires_grad parameters changed after the optimizer was built, and "
                "PolyStepOptimizer keeps the layout it was built with. Build a new optimizer on the model."
            )
        if getattr(self, "_cost_evaluator", None) is not None:
            if (
                not self._registration_fresh
                and closure is not self._registration_closure
                and not getattr(self, "_warned_stale_registration", False)
            ):
                self._warned_stale_registration = True
                warnings.warn(
                    "register_evaluator was not called since the previous step, so the fast paths "
                    "may score the batch registered earlier. Register each batch, or pass step() "
                    "the closure register_evaluator returns.",
                    stacklevel=3,
                )
            self._registration_fresh = False
        if objective_token != self._prev_objective_token:
            # A prediction on the previous minibatch cannot be scored against this one.
            self._prev_predicted_improvement = None
            self._prev_pre_step_loss = None
            self._prev_objective_token = objective_token
        self._last_screen_savings = 0.0
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
            result = _step_momentum_fn(self)
            self._amortize_counter += 1
            return result

        if self._blocks is not None:
            result = self._step_blockwise(closure)
        else:
            result = _step_monolithic_fn(self, closure, screen_closure)

        self._amortize_counter += 1
        return result

    def step_batch(self, inputs, targets=None, *, loss_fn: Callable) -> float:
        """Step on one batch through an evaluator cached per ``loss_fn`` object; returns the mean candidate loss."""
        if self.trust_region:
            raise ValueError(
                "trust_region is not supported by step_batch() or train(): its ratio compares a "
                "prediction against the loss on the next batch, which is a different objective. "
                "Drop trust_region, or drive optimizer.step() on a fixed batch."
            )
        evaluator = getattr(self, "_step_batch_evaluator", None)
        if evaluator is None or evaluator.loss_fn is not loss_fn:
            evaluator = self._step_batch_evaluator = NNCostEvaluator(self.model, loss_fn)
        first = _first_input_tensor(inputs)
        cost_bs = self.cost_batch_size
        if cost_bs is not None and first.dim() and cost_bs < first.shape[0]:
            gen = self._generator
            if gen is not None:
                if gen.device == first.device:
                    idx = torch.randperm(first.shape[0], device=first.device, generator=gen)[:cost_bs]
                else:
                    # randperm needs the generator and tensor on one device; sample on the generator's device to keep the seed, then move the index.
                    idx = torch.randperm(first.shape[0], device=gen.device, generator=gen)[:cost_bs].to(first.device)
            else:
                idx = torch.randperm(first.shape[0], device=first.device)[:cost_bs]
            inputs, targets = tree_map(
                lambda t: t[idx] if isinstance(t, torch.Tensor) and t.shape[:1] == first.shape[:1] else t,
                (inputs, targets),
            )
        closure = self.register_evaluator(evaluator, inputs, targets)
        return self.step(closure, screen_closure=self.screen_closure_from(closure, inputs, targets))

    def _step_blockwise(self, closure: Callable) -> float:
        """Block-wise step: per-block OT solve."""
        with _reuse_prefixes(
            getattr(self, "_sparse_delta_evaluator", None), getattr(self, "_site_vmap_evaluator", None)
        ):
            return _step_blockwise_fn(self, closure)

    def _transition_rank(self, new_rank: int) -> None:
        """Absorb the perturbation, then rebuild the subspace at ``new_rank``."""
        state = self._state

        old_subspace = self.subspace
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
        self.subspace._entry_dtypes = {k: v.dtype for k, v in state.base_params.items()}

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
        self.subspace.build_fused_projection(state.hybrid_projections)

        num_points = state.X.shape[0]

        # Duals and directions were measured in the outgoing basis.
        invalidate_for_basis_change(self, state)

        # Record only once the rebuild succeeded.
        self._applied_rank = new_rank
        self._warn_frozen_segments()

        logger.info(
            f"Rank transition: rank={new_rank}, subspace_dim={self.subspace.subspace_dim}, particles={num_points}"
        )

    def resync_from_model(self) -> None:
        """Re-read particle state from the model's current weights, after external writes.

        Subspace projections follow a dtype cast, once ``evaluator.reset_vmap()`` has dropped the evaluator's
        cached input dtype. Full space and a global projection move their coordinates too; per-layer subspaces
        keep their coordinate dtype. Raises ``ValueError`` if the model moved to another device.
        """
        device = next(self.model.parameters()).device
        if device != self._state.X.device:
            raise ValueError(
                f"The model moved from {self._state.X.device} to {device}, and resync_from_model() "
                "does not move the optimizer state. Build a new optimizer on the moved model."
            )
        _trainable_owners(self.model)
        if self.subspace is not None:
            # Re-anchor the base and zero the coordinates.
            self._state.base_params = {
                k: v.detach().clone() for k, v in self.model.state_dict().items() if isinstance(v, torch.Tensor)
            }
            self._state.X = torch.zeros_like(self._state.X)
            self._realign_subspace_to_model()
        else:
            sd = self.model.state_dict()
            self.layout = replace(
                self.layout,
                entries=tuple(replace(e, dtype=sd[e.key].dtype) for e in self.layout.entries),
                dominant_dtype=ParamLayout.from_module(self.model).dominant_dtype,
            )
            self._state.X = self.layout.flatten(self.model)
        self._state.f = None
        self._state.g = None
        self._state.prev_prev_f = None
        self._state.prev_prev_g = None
        if self._state.displacement_history is not None:
            self._state.displacement_history = torch.zeros_like(self._state.displacement_history)
        self._state.displacement_history_full = None
        self._state.displacement_history_idx = 0
        self._state.displacement_history_count = 0
        # These were measured at the old anchor; keep them and the next step coasts
        # on a stale direction.
        self._amortize_counter = 0
        self._transport_direction_ema = None
        self._newton_direction = None
        self._prev_descent_direction = None
        self._prev_block_descent_directions = None
        self._prev_predicted_improvement = None
        self._prev_pre_step_loss = None
        if self._state.velocity is not None:
            self._state.velocity = torch.zeros_like(self._state.X)
        self._state._prev_prev_block_duals = None
        # Reset counters measured at the old weights, or the next step could fire a
        # spurious stagnation absorb.
        self._state.stagnation_count = 0
        self._state.prev_loss = float("inf")
        self._loss_decreasing_count = 0
        self._ot_step_costs.clear()
        self._step_batch_evaluator = None

    def _realign_subspace_to_model(self) -> None:
        """Cast per-layer projections to their parameters' dtypes, or a global projection and the coordinates to
        the dominant one."""
        state = self._state
        if self._per_layer_projections:
            base = state.base_params
            if self._hybrid:
                self.subspace._entry_dtypes = {k: v.dtype for k, v in base.items()}
            state.hybrid_projections = {
                k: v.to(base[k].dtype) if isinstance(v, torch.Tensor) else v
                for k, v in state.hybrid_projections.items()
            }
            if hasattr(self.subspace, "build_fused_projection"):
                self.subspace.build_fused_projection(state.hybrid_projections)
            return
        dtype = ParamLayout.from_module(self.model).dominant_dtype
        state.X = state.X.to(dtype)
        if state.velocity is not None:
            state.velocity = state.velocity.to(dtype)
        if isinstance(state.projection, torch.Tensor):
            state.projection = state.projection.to(dtype)

    def _write_params(self, sd: dict) -> None:
        """Copy the parameter entries of ``sd`` into the live parameters, once per step.

        Non-parameter keys are skipped, so subspace snapshots do not revert live
        buffers like BatchNorm running stats.
        """
        params = dict(self.model.named_parameters())
        keys = [k for k in sd if k in params]
        torch._foreach_copy_([params[k].data for k in keys], [sd[k] for k in keys])

    def _sync_model(self) -> None:
        """Write current particles back to the model."""
        state = self._state

        if self.subspace is not None:
            # Flatten X back to subspace coords, then rebuild params from base + perturb.
            X = state.X  # (num_sub_particles, particle_dim)
            flat_sub = X.reshape(-1)[: state.subspace.subspace_dim]
            proj = state.hybrid_projections if self._per_layer_projections else state.projection
            full_sd = state.subspace.apply_perturbation(proj, state.base_params, flat_sub)
            self._write_params(full_sd)
        else:
            particles_2d = state.X
            if particles_2d.dim() == 1:
                particles_2d = particles_2d.reshape(-1, self._particle_dim)

            sd = self.layout.unflatten(particles_2d)
            self._write_params(sd)
