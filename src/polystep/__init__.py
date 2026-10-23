"""polystep: PyTorch PolyStep Optimizer."""

__version__ = "1.1.0"

__all__ = [
    # Core solver
    "SolverResult",
    "SinkhornSolver",
    "SoftmaxSolver",
    "KLSoftmaxSolver",
    "MinCostGreedySolver",
    "TopKMeanSolver",
    # Geometry
    "get_orthoplex_vertices",
    "get_simplex_vertices",
    "get_cube_vertices",
    "get_random_rotation_matrices",
    "POLYTOPE_MAP",
    # Epsilon
    "LinearEpsilon",
    "CosineEpsilon",
    "ProgressiveEpsilon",
    "PowerDecay",
    # Solver
    "SolverState",
    # Transform
    "ParamLayout",
    # NN cost evaluation
    "NNCostEvaluator",
    # Subspace
    "AdaptiveSubspace",
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
]

from .solvers import (
    SolverResult,
    SinkhornSolver,
    SoftmaxSolver,
    KLSoftmaxSolver,
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

from .epsilon import LinearEpsilon, CosineEpsilon, ProgressiveEpsilon, PowerDecay

from .solver import SolverState

from .transform import ParamLayout

from .cost_nn import NNCostEvaluator

from .adaptive_subspace import AdaptiveSubspace

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
from .hybrid_subspace import HybridSubspace
