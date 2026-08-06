"""polystep: PyTorch PolyStep Optimizer."""

__version__ = "0.10.1"

__all__ = [
    # Core solver
    "SolverResult",
    "SinkhornSolver",
    "SoftmaxSolver",
    "KLSoftmaxSolver",
    "TemperedSoftmaxSolver",
    "MinCostGreedySolver",
    "TopKMeanSolver",
    # Geometry
    "get_orthoplex_vertices",
    "get_simplex_vertices",
    "get_cube_vertices",
    "get_random_rotation_matrices",
    "POLYTOPE_MAP",
    # Cost & epsilon
    "compute_cost_matrix",
    "scale_cost_matrix",
    "LinearEpsilon",
    "CosineEpsilon",
    "ProgressiveEpsilon",
    "PowerDecay",
    # Solver
    "SolverState",
    # Transform
    "ParamEntry",
    "ParamLayout",
    "create_generator",
    # NN cost evaluation
    "NNCostEvaluator",
    "auto_detect_chunk_size",
    # Compilation
    "CompiledFunctions",
    "try_compile",
    # Subspace
    "ProjectionSpec",
    "AdaptiveSubspace",
    "CMAAdaptiveSubspace",
    # CMA
    "compute_cma_hyperparameters",
    # Blockwise
    "BlockConfig",
    "create_per_layer_blocks",
    "create_grouped_blocks",
    # Dynamics
    "apply_momentum",
    "update_radius_multiplier",
    "update_stagnation",
    "compute_momentum_coefficient",
    # Optimizer
    "PolyStepOptimizer",
    "RankSchedule",
    # Ask/tell adapter
    "PolyStepES",
    "minimize",
    # High-level API
    "train",
    "TrainConfig",
    "TrainCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "get_diagnostics",
    # Objectives
    "ObjectiveFn",
    "Ackley",
    "Rosenbrock",
    "Rastrigin",
    "Sphere",
    # Layers
    "VmapSafeMultiHeadAttention",
    "VmapSafeLSTM",
    # Projection
    "SparseRandomProjection",
    # Hybrid subspace
    "FactoredSubspace",
    "HybridSubspace",
    "LayerProjectionSpec",
]

from .solvers import (
    SolverResult,
    SinkhornSolver,
    SoftmaxSolver,
    KLSoftmaxSolver,
    TemperedSoftmaxSolver,
    MinCostGreedySolver,
    TopKMeanSolver,
)

from .geometry import (
    get_orthoplex_vertices,
    get_simplex_vertices,
    get_cube_vertices,
    get_random_rotation_matrices,
    POLYTOPE_MAP,
)

from .costs import compute_cost_matrix, scale_cost_matrix
from .epsilon import LinearEpsilon, CosineEpsilon, ProgressiveEpsilon, PowerDecay

from .solver import SolverState

from .transform import ParamEntry, ParamLayout, create_generator

from .cost_nn import NNCostEvaluator, auto_detect_chunk_size

from ._compiled import CompiledFunctions, try_compile

from .subspace import ProjectionSpec

from .adaptive_subspace import AdaptiveSubspace

from .cma_subspace import CMAAdaptiveSubspace
from .cma import (
    compute_cma_hyperparameters,
)

from .blockwise import BlockConfig, create_per_layer_blocks, create_grouped_blocks

from .dynamics import (
    apply_momentum,
    update_radius_multiplier,
    update_stagnation,
    compute_momentum_coefficient,
)

from .optimizer import PolyStepOptimizer, RankSchedule
from .ask_tell import PolyStepES, minimize

from .api import (
    train,
    TrainConfig,
    TrainCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    get_diagnostics,
)

# Objectives
from .objectives import (
    ObjectiveFn,
    Ackley,
    Rosenbrock,
    Rastrigin,
    Sphere,
)

# Vmap-safe layers
from .layers import VmapSafeMultiHeadAttention, VmapSafeLSTM

# Sparse projection for large-scale models
from .projection import SparseRandomProjection

# Hybrid subspace
from .factored_subspace import FactoredSubspace
from .hybrid_subspace import HybridSubspace, LayerProjectionSpec
