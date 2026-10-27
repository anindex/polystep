"""polystep: PyTorch PolyStep Optimizer."""

__version__ = "1.1.1"

__all__ = [
    "SolverResult",
    "SinkhornSolver",
    "SoftmaxSolver",
    "KLSoftmaxSolver",
    "MinCostGreedySolver",
    "TopKMeanSolver",
    "get_orthoplex_vertices",
    "get_simplex_vertices",
    "get_cube_vertices",
    "get_random_rotation_matrices",
    "POLYTOPE_MAP",
    "LinearEpsilon",
    "CosineEpsilon",
    "ProgressiveEpsilon",
    "PowerDecay",
    "SolverState",
    "ParamLayout",
    "NNCostEvaluator",
    "AdaptiveSubspace",
    "PolyStepOptimizer",
    "RankSchedule",
    "PolyStepES",
    "minimize",
    "train",
    "TrainConfig",
    "TrainCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "get_diagnostics",
    "ObjectiveFn",
    "Ackley",
    "Rosenbrock",
    "Rastrigin",
    "Sphere",
    "VmapSafeMultiHeadAttention",
    "VmapSafeLSTM",
    "SparseRandomProjection",
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

from .objectives import (
    ObjectiveFn,
    Ackley,
    Rosenbrock,
    Rastrigin,
    Sphere,
)

from .layers import VmapSafeMultiHeadAttention, VmapSafeLSTM

from .projection import SparseRandomProjection

from .factored_subspace import FactoredSubspace
from .hybrid_subspace import HybridSubspace
