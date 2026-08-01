"""Gradient-free baselines on one shared protocol.

All six methods take the same :class:`Objective`, so they can be handed the same
subspace, the same probe radius, the same minibatch stream and the same
evaluation counter as PolyStep::

    from polystep.baselines import Objective, openai_es

    obj = Objective(fn, dim=64, budget=10_000)
    result = openai_es(obj, sigma=0.05, lr=0.1)
    result.best_loss, result.evals

In a subspace, only the objective changes::

    obj = Objective.from_subspace(hybrid, base_sd, loss_batch, budget=10_000)

See ``README.md`` ("Gradient-free baselines").
"""

from .core import BudgetExhausted, Objective, Result, centered_rank, zscore
from .methods import METHODS, cma_es, eggroll, mezo, openai_es, random_search, spsa

__all__ = [
    "METHODS",
    "BudgetExhausted",
    "Objective",
    "Result",
    "centered_rank",
    "cma_es",
    "eggroll",
    "mezo",
    "openai_es",
    "random_search",
    "spsa",
    "zscore",
]
