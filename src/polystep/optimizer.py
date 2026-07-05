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
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .cost_nn import NNCostEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-selection helper for projection type
# ---------------------------------------------------------------------------


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

    Auto-selection thresholds are based on memory efficiency vs setup overhead.
    For small models, dense projection is more efficient. For large models,
    sparse projection reduces memory footprint significantly.

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
from .solvers import SinkhornSolver, SoftmaxSolver, MinCostGreedySolver, TopKMeanSolver, TemperedSoftmaxSolver
from .solver import SolverState
from .adaptive_subspace import AdaptiveSubspace
from .cma_subspace import CMAAdaptiveSubspace
from .subspace import LowRankSubspace, LinearSubspace
from .hybrid_subspace import HybridSubspace
from .transform import ParamLayout, create_generator

# Extracted step methods (decomposed for maintainability)
from ._step_monolithic import step_monolithic as _step_monolithic_fn
from ._step_blockwise import step_blockwise as _step_blockwise_fn
from ._step_blockwise import step_subspace_blockwise as _step_subspace_blockwise_fn
from ._step_momentum import (
    step_momentum as _step_momentum_fn,
    evaluate_current_loss as _evaluate_current_loss_fn,
)


# ---------------------------------------------------------------------------
# Rank schedule for progressive subspace expansion
# ---------------------------------------------------------------------------


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


class PolyStepOptimizer:
    """Gradient-free optimizer via entropic optimal transport.

    Wraps polytope sampling, OT solve, barycentric projection, momentum,
    and adaptive radius into a single ``step(closure)`` interface. Updates
    the model weights in-place after each step.

    The optimizer uses entropic regularization (controls smoothness of the
    transport plan; higher values give smoother but less precise solutions)
    to solve an optimal transport problem between current particle positions
    and candidate polytope vertices. The solution yields a transport plan
    whose weights are used for barycentric projection (updates particle
    positions as a weighted average of polytope vertices, with weights from
    the optimal transport plan). Between iterations, dual potentials
    (auxiliary variables from the Sinkhorn algorithm that encode the optimal
    transport solution) are warm-started for faster convergence.

    This is a standalone class (NOT a ``torch.optim.Optimizer`` subclass)
    because Sinkhorn Step is fundamentally gradient-free and does not use
    parameter groups or ``zero_grad()``.

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
            # batched_params: {key: (N, *shape)} -- N candidate param sets
            # Return losses tensor of shape (N,)
            ...

        cost = optimizer.step(closure)

    See Also:
        ``train()`` for a high-level training loop that builds closures
        automatically from a DataLoader and loss function.
        ``TrainConfig`` for training loop configuration.

    Args:
        model: The ``nn.Module`` to optimize. Weights are updated in-place.
        polytope_type: Polytope template ('orthoplex', 'simplex', 'cube').
        epsilon: Entropic regularization (float or LinearEpsilon schedule).
        ent_epsilon: Separate OT solver epsilon. If None, uses epsilon.
        scale_cost: Cost scaling strategy ('mean', 'max_cost', float, or None).
        step_radius: Base step radius multiplied by epsilon.
        probe_radius: Base probe radius multiplied by epsilon.
        num_probe: Number of probe points per direction (default 1).
            K=1 is optimal: multi-probe averaging is redundant when entropic
            regularization is active. K=1 gives ~3x speedup over K=3 with no
            accuracy loss.
        adaptive_probes: Enable adaptive probe count (default False). When
            True, stagnant particles (small displacement) reuse the
            previous step's cost row instead of recomputing, saving
            ``V * K`` forward passes each.
        adaptive_probes_threshold: Displacement squared norm below which a
            particle is considered stagnant (default ``1e-6``).
        max_iterations: Maximum outer iterations (for momentum warmup schedule).
        sinkhorn_max_iters: Maximum inner Sinkhorn iterations.
        chunk_size: Chunk size for cost evaluation memory control.
        cost_batch_size: Optional mini-batch size for cost matrix evaluation.
            Note: stored for the training loop closure to read; not used
            internally by the optimizer.
        compile: Whether to compile hot-path tensor functions. Defaults to
            False because ablation experiments show no measurable end-to-end
            speedup while incurring JIT warm-up overhead.
        solver: OT/weighting solver strategy. 'softmax' for direct softmax
            weighting (fast, no dual potentials), 'sinkhorn' for entropic OT
            solver (iterative, with warm-started duals). None (default)
            auto-selects: softmax for subspace modes, sinkhorn for full-space.
            ProgressiveEpsilon (auto_epsilon=True) is incompatible with softmax.
        subspace: Optional LowRankSubspace for subspace mode.
        subspace_particle_dim: Particle dimension for subspace mode (default 8).
            In subspace mode, this overrides ``particle_dim`` for the OT
            polytope dimension. Use this parameter (not ``particle_dim``)
            to control polytope geometry in subspace experiments.
            Higher values give more vertices (2*dim for orthoplex) and stronger
            per-step signal. Only used when subspace is not None.
        absorb_every: Periodic absorb interval (default 0 = disabled). When > 0
            and subspace is active, folds current perturbation into base weights
            every N steps, zeroing the subspace vector to explore new regions.
        block_strategy: 'monolithic', 'per_layer', or 'grouped'.
        block_group_size: Number of consecutive entries per block group.
        use_momentum: Enable momentum velocity accumulation.
        momentum_init: Starting momentum coefficient.
        momentum_final: Final momentum coefficient.
        velocity_lr: Learning rate for velocity update.
        use_adaptive_radius: Enable stagnation-based radius adaptation.
        stagnation_threshold: Relative change below which is stagnation.
        stagnation_patience: Stagnation iterations before radius boost.
        radius_increase: Multiplicative factor for radius boost.
        radius_decrease: Multiplicative factor for radius decay.
        radius_min: Minimum allowed radius multiplier.
        radius_max: Maximum allowed radius multiplier.
        use_covariance_adaptation: Enable diagonal CMA-ES covariance learning.
            Requires CMAAdaptiveSubspace. Learns per-dimension scaling of the
            search distribution.
        use_csa: Enable CSA (Cumulative Step-size Adaptation) from CMA-ES.
            Requires CMAAdaptiveSubspace. **Warning**: CSA may be unstable for
            OT-based optimization because OT displacement is ~1-5% of polytope
            size (vs ~100% in standard CMA-ES). Recommend use_adaptive_radius=True
            instead for stable step-size adaptation.
        seed: Optional seed for reproducible random rotations.
        mixed_precision: Enable BF16 model forward passes with FP32 Sinkhorn
            solver internals. Reduces memory usage (~50% for weights) while
            maintaining numerical stability. Requires GPU compute capability
            >= 7.0 (Volta) or CPU. Default False.
        projection_type: Type of projection for AdaptiveSubspace mode.
            'dense' uses QR-orthogonalized dense matrices (default).
            'sparse' uses SparseRandomProjection for memory efficiency.
            'auto' will auto-select based on model size.

    Hyperparameter quick reference (MNIST as a sanity check):

    - **Full-space (no subspace).** ``epsilon=LinearEpsilon(1.0 -> 0.1)``,
      ``step_radius=0.15``, ``probe_radius=0.12``.
    - **HybridSubspace.** ``rank=4``, ``rotation_interval=0``, decaying
      ``epsilon``, ``step_radius=4.5``, ``probe_radius=2.0``.
    - **LinearSubspace.** Same radii as HybridSubspace; decaying ``epsilon``.
    - **AdaptiveSubspace.** Large ``rank`` (e.g. 4096), *fixed* ``epsilon=0.5``,
      ``step_radius=10.0``, ``probe_radius=2.0``, ``use_adaptive_radius=True``.

    Rule of thumb: per-layer subspaces benefit from a decaying epsilon
    schedule; a single global projection prefers a fixed epsilon with
    adaptive radius. Global projections also need a larger ``step_radius``
    than per-layer ones.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        polytope_type: str = "orthoplex",
        particle_dim: int = 2,
        epsilon: Union[float, LinearEpsilon] = 0.1,
        ent_epsilon: Optional[Union[float, LinearEpsilon]] = None,
        # Auto-epsilon: ProgOT-inspired feedback-driven epsilon
        auto_epsilon: bool = False,
        auto_epsilon_config: Optional[dict] = None,
        scale_cost: Optional[Union[str, float]] = 1.0,
        step_radius: float = 1.0,
        probe_radius: float = 2.0,
        # Probe-radius jitter (Theorem 4.2 condition (iv)): per-step uniform
        # multiplicative perturbation eta_t ~ U[-eta_max, +eta_max] applied to
        # the resolved probe radius so the joint (rotation, jitter) probe
        # distribution is absolutely continuous on a positive-Lebesgue-measure
        # tube around the (d_p-1)-sphere. The convergence proof formally
        # requires probe_radius_jitter > 0; default 0.0 keeps existing
        # experiments bit-for-bit reproducible. Recommended value: 0.05.
        probe_radius_jitter: float = 0.0,
        num_probe: int = 1,
        # Adaptive probe count: reduce K during exploitation
        adaptive_num_probe: bool = False,
        adaptive_probe_warmup: int = 20,
        # Adaptive probes: reduce evaluations for stagnant particles
        adaptive_probes: bool = False,
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
        subspace: Optional[Union[LowRankSubspace, LinearSubspace, AdaptiveSubspace]] = None,
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
        # Multi-fidelity screening: dampen low-contrast vertex directions using
        # previous step's cost data to focus OT on informative directions
        multifidelity_screen: bool = False,
        screen_keep_ratio: float = 0.5,
        # CMA-ES configuration
        use_covariance_adaptation: bool = False,
        use_csa: bool = False,
        seed: Optional[int] = None,
        # Mixed precision
        mixed_precision: bool = False,
        # Projection type
        projection_type: str = "dense",
        # Compile vmap in NNCostEvaluator (torch.compile the vectorized forward)
        compile_evaluator: bool = False,
    ) -> None:
        # Projection type validation
        if projection_type not in ("dense", "sparse", "auto"):
            raise ValueError(f"Invalid projection_type: {projection_type!r}. Use 'dense', 'sparse', or 'auto'.")
        self._requested_projection_type = projection_type
        self._compile_evaluator = compile_evaluator

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

        # Warn if particle_dim is set but will be overridden by subspace_particle_dim
        if subspace is not None and particle_dim != 2:
            warnings.warn(
                f"particle_dim={particle_dim} is ignored in subspace mode. "
                f"The OT polytope uses subspace_particle_dim={subspace_particle_dim} instead. "
                f"Pass subspace_particle_dim={particle_dim} to control polytope geometry.",
                stacklevel=2,
            )

        # Combined subspace + block-wise mode
        # When both subspace and block_strategy are specified, operate in combined mode:
        # global projection compresses full params to subspace coords, then per-block OT
        # decomposes the subspace coordinate optimization.
        self._subspace_blockwise = subspace is not None and block_strategy != "monolithic"

        # Mixed precision config
        self._mixed_precision = mixed_precision
        self._model_dtype = torch.float32  # default, updated below if mixed_precision enabled

        # Store model and config
        self.model = model
        self.polytope_type = polytope_type
        self.epsilon = epsilon
        self.ent_epsilon = ent_epsilon

        # Auto-epsilon: ProgressiveEpsilon with solver feedback
        if auto_epsilon:
            from .epsilon import ProgressiveEpsilon

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
        self.adaptive_num_probe = adaptive_num_probe
        self._adaptive_probe_warmup = adaptive_probe_warmup
        self._loss_decreasing_count = 0
        # OT-step-only costs for adaptive_num_probe check (avoids mixing with momentum costs)
        self._ot_step_costs: collections.deque = collections.deque(maxlen=3)
        self._adaptive_probes = adaptive_probes
        self._adaptive_probes_threshold = adaptive_probes_threshold
        # Previous per-particle displacement squared norms for stagnation detection
        self._prev_displacement_sqnorms: Optional[torch.Tensor] = None
        # Previous cost matrix rows for reuse by stagnant particles
        self._prev_cost_matrix: Optional[torch.Tensor] = None
        # Previous rotation matrices: stagnant particles reuse theirs so a reused
        # cost row still describes the vertices it was evaluated at.
        self._prev_rot_mats: Optional[torch.Tensor] = None
        # Track K_eff and step_r to invalidate _prev_cost_matrix on change
        self._prev_k_eff: Optional[int] = None
        self._prev_step_r: Optional[float] = None
        self.max_iterations = max_iterations
        self.chunk_size = chunk_size
        self.cost_batch_size = cost_batch_size
        self.amortize_steps = max(1, amortize_steps)
        self.amortize_ema = amortize_ema

        # SNN-like models (LIF / Leaky / Spiking / ALIF cells) collapse
        # from ~93% to 10-47% accuracy when step_radius is on a cosine
        # schedule -- the discrete spike landscape is too chaotic for a
        # shrinking step. Warn loudly when we detect that combination.
        # Heuristic only -- matches substring of module class names.
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
        self._transport_direction = None
        self._transport_direction_ema = None
        # Transport-biased rotation: store previous OT descent direction
        self.biased_rotation = biased_rotation
        self._prev_descent_direction: Optional[torch.Tensor] = None
        self._prev_descent_direction_finite: bool = False
        # Quadratic model: FD gradient/Hessian extraction from cost evaluations
        self.use_quadratic_model = use_quadratic_model
        self._prev_losses_3d = None  # (P, V, K) retained for quadratic model
        # Newton refinement: post-OT correction using quadratic model
        self._newton_refinement = newton_refinement
        self._newton_refinement_alpha = newton_refinement_alpha
        if newton_refinement and not use_quadratic_model:
            self.use_quadratic_model = True
            logger.info(
                "newton_refinement=True auto-enables use_quadratic_model=True "
                "(needed to retain probe losses for Newton correction)"
            )

        self._newton_direction = None  # (P, pdim) Newton step in original space
        # Trust region: adapt step_radius via multiplier based on quadratic model
        self.trust_region = trust_region
        self._trust_region_multiplier = 1.0  # Multiplier on step_radius, range [0.1, 3.0]
        self._prev_predicted_improvement = None
        self._prev_pre_step_loss = None  # Per-particle min cost proxy
        # Multi-fidelity vertex screening: dampen low-contrast directions
        self.multifidelity_screen = multifidelity_screen
        self.screen_keep_ratio = screen_keep_ratio
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
        self.block_strategy = block_strategy
        self.block_group_size = block_group_size

        # Warn if quadratic model used with non-monolithic block strategy
        if use_quadratic_model and block_strategy != "monolithic":
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

        # Dual momentum config
        self._dual_momentum_beta = dual_momentum_beta

        # CMA-ES config
        # Detect CMA subspace mode early for validation
        self._cma_subspace = isinstance(subspace, CMAAdaptiveSubspace)

        # Validate: CMA features require CMAAdaptiveSubspace
        if (use_covariance_adaptation or use_csa) and not self._cma_subspace:
            warnings.warn(
                "use_covariance_adaptation or use_csa requires CMAAdaptiveSubspace. CMA features will be disabled."
            )
            use_covariance_adaptation = False
            use_csa = False

        # Validate: CSA replaces heuristic adaptive radius
        if use_csa and use_adaptive_radius:
            warnings.warn(
                "Both use_csa and use_adaptive_radius enabled. CSA will be used, heuristic radius adaptation disabled."
            )
            use_adaptive_radius = False
            self.use_adaptive_radius = False

        self.use_covariance_adaptation = use_covariance_adaptation
        self.use_csa = use_csa
        # Coord-to-param projection for the current step, covariance-scaled for
        # CMA. Cached so a step's probes and its sync share one metric.
        self._sampling_projection = None

        # Create layout from model
        # Thread particle_dim for full-space mode; subspace mode ignores this
        # (subspace uses subspace_particle_dim instead)
        self.layout = ParamLayout.from_module(model, particle_dim=self._full_space_particle_dim)

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

        # ------------------------------------------------------------------
        # Multi-particle architecture:
        # Parameters are reshaped to (num_particles, particle_dim) where
        # particle_dim is the layout's column count (typically 2). The OT
        # polytope operates in particle_dim space, giving a small number of
        # vertices (e.g., 4 for orthoplex in 2D, 3 for simplex).
        # ------------------------------------------------------------------

        # Detect adaptive subspace mode
        self._adaptive = isinstance(subspace, AdaptiveSubspace)

        # Detect hybrid subspace mode
        self._hybrid = isinstance(subspace, HybridSubspace)
        self._hybrid_subspace = subspace if self._hybrid else None

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
            self._sub_padded = padded_sub  # for unpadding later
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

        if solver is None:
            solver = "softmax" if subspace is not None else "sinkhorn"

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
        else:
            raise ValueError(
                f"Unknown solver: {solver!r}. Expected 'softmax', 'sinkhorn', "
                f"'min_cost_greedy', 'top_k_mean', 'tempered_softmax', or None."
            )

        _non_iterative_solvers = ("softmax", "min_cost_greedy", "top_k_mean", "tempered_softmax")
        if self._progressive_epsilon is not None and solver in _non_iterative_solvers:
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
        self._seed = seed
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = create_generator(seed, device)

        # Initialize solver state
        num_points = X_init.shape[0]
        a = torch.ones(num_points, device=device, dtype=X_init.dtype) / num_points
        self._state = SolverState(X=X_init.clone(), a=a)

        if subspace is not None:
            self._state.base_params = self._base_params
            self._state.subspace = subspace

        # Initialize subspace projections and displacement history
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

    # ------------------------------------------------------------------
    # Init helpers (split from __init__ for readability)
    # ------------------------------------------------------------------

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

        # HybridSubspace: initialize per-layer projections and displacement history
        if self._hybrid:
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

        if self.use_covariance_adaptation or self.use_csa:
            cma_state = subspace.init_cma_state(
                device=model_device,
                dtype=self._model_dtype,
            )
            self._state.p_c = cma_state["p_c"]
            self._state.p_sigma = cma_state["p_sigma"]
            self._state.C_diag = cma_state["C_diag"]
            self._state.sigma = 1.0
            self._state.generation = 0
            self._state.use_csa = self.use_csa
            self._cma_params = {
                "c_c": subspace.c_c,
                "c_sigma": subspace.c_sigma,
                "c_1": subspace.c_1,
                "c_mu": subspace.c_mu,
                "d_sigma": subspace.d_sigma,
                "expected_norm": subspace.expected_norm,
                "mu_eff": subspace.mu_eff,
                "cov_min": subspace.cov_min,
                "cov_max": subspace.cov_max,
            }

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

        Note: When 'auto' is requested, this returns the resolved type based
        on model size (sparse for >1M params on CPU, >2M on GPU).
        """
        return self._actual_projection_type

    # ------------------------------------------------------------------
    # Epsilon resolution
    # ------------------------------------------------------------------

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
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(device)
                return cap[0] >= 7  # Volta (7.0) and newer
            return False
        elif device.type == "cpu":
            return True  # CPU BF16 works, may be slower without AMX
        else:
            return True  # MPS, XLA, etc. - let PyTorch error if unsupported

    # ------------------------------------------------------------------
    # Adaptive probes
    # ------------------------------------------------------------------

    def _get_stagnant_mask(self, num_particles: int) -> Optional[torch.Tensor]:
        """Return boolean mask of stagnant particles based on displacement history.

        Returns:
            Tensor of shape (num_particles,) with True for stagnant particles,
            or None if adaptive_probes is disabled or no displacement history yet.
        """
        if not self._adaptive_probes or self._prev_displacement_sqnorms is None:
            return None

        stagnant = self._prev_displacement_sqnorms < self._adaptive_probes_threshold
        # Ensure mask length matches current particle count (may differ on first step)
        if stagnant.shape[0] != num_particles:
            return None
        return stagnant

    def _invalidate_reuse_cache(self) -> None:
        """Drop the cached cost rows, rotations, and probe state used by
        adaptive-probe reuse. Call whenever the cost geometry changes (subspace
        rotation or absorb, rank transition), since a reused row would then
        describe stale directions.
        """
        self._prev_cost_matrix = None
        self._prev_rot_mats = None
        self._prev_losses_3d = None
        self._prev_displacement_sqnorms = None
        self._prev_k_eff = None
        self._prev_step_r = None

    # ------------------------------------------------------------------
    # Fused inplace evaluation (EGGROLL-inspired)
    # ------------------------------------------------------------------

    def register_evaluator(
        self,
        evaluator: "NNCostEvaluator",
        inputs: torch.Tensor,
        targets: torch.Tensor = None,
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
        (auto-detected for GPU models >50K params).

        Args:
            evaluator: NNCostEvaluator instance.
            inputs: Current mini-batch inputs.
            targets: Current mini-batch targets (optional).
        """
        self._cost_evaluator = evaluator
        self._fused_inputs = inputs
        self._fused_targets = targets

    # ------------------------------------------------------------------
    # Step: entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def step(self, closure: Callable) -> float:
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

        Returns:
            Scalar OT entropic regularized cost (float).
        """
        # Amortized OT: cheap momentum steps between full OT solves
        if (
            self.amortize_steps > 1
            and self._amortize_counter % self.amortize_steps != 0
            and self._transport_direction_ema is not None
        ):
            result = self._step_momentum(closure)
            self._amortize_counter += 1
            return result

        # Full OT step
        # HybridSubspace now uses monolithic step with per-layer projections
        # (per-layer OT in _step_hybrid did not achieve target accuracy)
        if self._subspace_blockwise:
            # Combined subspace + block-wise mode
            result = self._step_subspace_blockwise(closure)
        elif self._blocks is not None:
            result = self._step_blockwise(closure)
        else:
            result = self._step_monolithic(closure)

        self._amortize_counter += 1
        return result

    # ------------------------------------------------------------------
    # Step methods: delegated to extracted modules for maintainability.
    # See _step_monolithic.py, _step_blockwise.py, _step_momentum.py.
    # ------------------------------------------------------------------

    def _step_monolithic(self, closure: Callable) -> float:
        """Monolithic step: single OT solve over all particles.

        Delegates to ``_step_monolithic.step_monolithic()``.
        """
        return _step_monolithic_fn(self, closure)

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
        """Cheap momentum step: apply EMA transport direction with decay.

        Delegates to ``_step_momentum.step_momentum()``.
        """
        return _step_momentum_fn(self, closure)

    def _evaluate_current_loss(self, closure: Callable) -> float:
        """Evaluate model loss at current particle position.

        Delegates to ``_step_momentum.evaluate_current_loss()``.
        """
        return _evaluate_current_loss_fn(self, closure)

    # ------------------------------------------------------------------
    # Rank transition
    # ------------------------------------------------------------------

    def _transition_rank(self, new_rank: int) -> None:
        """Transition subspace to new rank, preserving accumulated progress via absorb.

        Absorbs the current perturbation into base weights, then reconstructs
        the subspace at the new rank. Resets particles, duals, and displacement
        history to match the new subspace dimension.

        Args:
            new_rank: Target rank for the new subspace.
        """
        state = self._state

        # 1. Absorb current perturbation into base weights
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

        # 2. Reconstruct subspace at new rank
        if isinstance(old_subspace, HybridSubspace):
            self.subspace = HybridSubspace.from_layout(
                self.layout,
                rank=new_rank,
                seed=old_subspace.seed,
                rotation_mode=old_subspace.rotation_mode,
                max_subspace_dim=getattr(old_subspace, "_max_subspace_dim", None),
            )
        elif isinstance(old_subspace, LinearSubspace):
            self.subspace = LinearSubspace.from_layout(
                self.layout,
                rank=new_rank,
                seed=old_subspace.seed,
                max_subspace_dim=getattr(old_subspace, "_max_subspace_dim", None),
            )

        # Update state reference
        state.subspace = self.subspace

        # 3. Resize particle array for new subspace dimension
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

        # 4. Reset duals (shape changed)
        state.f = None
        state.g = None

        # 5. Re-initialize displacement history at new subspace dimension
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
            # Update hybrid subspace reference
            self._hybrid_subspace = self.subspace

        # Update uniform distribution to match new particle count
        num_points = state.X.shape[0]
        state.a = torch.ones(num_points, device=state.X.device, dtype=state.X.dtype) / num_points

        # Clear stale adaptive probe state (shape changed with new rank)
        self._invalidate_reuse_cache()
        self._transport_direction_ema = None
        self._transport_direction = None
        self._newton_direction = None

        logger.info(
            f"Rank transition: rank={new_rank}, subspace_dim={self.subspace.subspace_dim}, particles={num_points}"
        )

    # ------------------------------------------------------------------
    # Model synchronization
    # ------------------------------------------------------------------

    def _update_sampling_projection(self) -> None:
        """Cache the coord-to-param projection for this step.

        Scaled by sqrt(C_diag) when CMA covariance adaptation is on, else the
        plain projection. Cached at step start so a step's probes and its
        end-of-step sync share one metric (C_diag only changes at step end).
        """
        state = self._state
        proj = state.projection
        if (
            self._cma_subspace
            and self.use_covariance_adaptation
            and isinstance(proj, torch.Tensor)
            and state.C_diag is not None
        ):
            proj = self.subspace.apply_covariance_scaling(proj, state.C_diag)
        self._sampling_projection = proj

    def _sync_model(self) -> None:
        """Write current particles back to the model via load_state_dict."""
        state = self._state

        if self.subspace is not None:
            # Subspace: flatten multi-particle X back to subspace coords,
            # then reconstruct full params from base + perturbation.
            X = state.X  # (num_sub_particles, particle_dim)
            flat_sub = X.reshape(-1)[: state.subspace.subspace_dim]
            if self._hybrid:
                # HybridSubspace requires hybrid_projections dict
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
            self.model.load_state_dict(full_sd, strict=False)
        else:
            # Multi-particle mode: X is already (num_particles, particle_dim)
            particles_2d = state.X
            if particles_2d.dim() == 1:
                particles_2d = particles_2d.reshape(-1, self._particle_dim)

            sd = self.layout.unflatten(particles_2d)
            self.model.load_state_dict(sd, strict=False)
