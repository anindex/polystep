"""Shared protocol for the gradient-free baselines.

Everything here exists so the baselines and PolyStep can be handed the *same*
search space, the same probe radius, the same minibatch stream and the same
evaluation counter.

One evaluation = one candidate scored
-------------------------------------
:class:`Objective` counts rows. A method that scores a population of 32 spends
32, whether it did so in one vmapped call or 32 python calls. This replaces the
three incompatible semantics in ``experiments/``:
``FunctionEvalCounter`` (closure calls), ``CountingClosure`` (``losses.shape[0]``,
the only one that already agreed with this) and ``sgd_baseline`` (samples).

One call = one generation = one minibatch
-----------------------------------------
Every method calls the objective once per iteration with its whole population.
A stochastic ``fn`` should therefore draw its minibatch per call, which gives
every candidate in a generation the same data, as PolyStep's closure does.

Subspace
--------
:meth:`Objective.from_subspace` puts the search in a
:class:`~polystep.hybrid_subspace.HybridSubspace`'s projected coordinates:
``dim`` becomes ``subspace_dim`` and candidates are reconstructed into full
parameters before the loss sees them. Nothing else in a method changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["BudgetExhausted", "Objective", "Result", "centered_rank", "zscore"]


class BudgetExhausted(RuntimeError):
    """A candidate batch would have exceeded the evaluation budget.

    Methods loop on :attr:`Objective.remaining` and stop before this fires; it
    is the backstop that makes "never more than ``budget``" true even if a
    method's own bookkeeping is wrong.
    """


@dataclass
class Result:
    """Outcome of one baseline run.

    Attributes:
        method: Method name.
        x: Final iterate (the method's own estimate, e.g. the ES mean).
        best_x: Best candidate ever evaluated.
        best_loss: Its loss.
        evals: Candidates evaluated. Always ``<= Objective.budget``.
        iters: Iterations/generations completed.
        history: ``(evals, best_loss)`` after each iteration.
    """

    method: str
    x: torch.Tensor
    best_x: torch.Tensor
    best_loss: float
    evals: int
    iters: int
    history: List[Tuple[int, float]] = field(default_factory=list)


class Objective:
    """A budgeted, optionally projected black-box objective.

    Args:
        fn: ``(N, dim) -> (N,)`` losses, lower is better. Called once per
            generation with the whole population.
        dim: Search-space dimension.
        budget: Total candidates the run may evaluate.
        shapes: How ``dim`` decomposes into parameter tensors. Only EGGROLL
            reads this, to know where the matrices are. Defaults to one flat
            ``(dim,)`` block.
        subspace: The subspace the coordinates live in, when there is one.
            Informational; :meth:`from_subspace` does the wiring.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        budget: int,
        *,
        shapes: Optional[Sequence[Sequence[int]]] = None,
        subspace: object = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}.")
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}.")
        self._fn = fn
        self.dim = int(dim)
        self.budget = int(budget)
        self.subspace = subspace
        self.shapes: Tuple[Tuple[int, ...], ...] = (
            tuple(tuple(int(d) for d in s) for s in shapes) if shapes else ((self.dim,),)
        )
        covered = sum(math.prod(s) for s in self.shapes)
        if covered != self.dim:
            raise ValueError(f"shapes cover {covered} entries but dim is {self.dim}.")
        self.evals = 0
        # Best-so-far is tracked ON DEVICE. Materializing it per call costs two
        # host syncs, which measured 6.17 ms against 0.49 ms for the loss itself:
        # 93% of wall-clock would be this bookkeeping. Worse, the cost is per
        # *call*, so it scales with generation count and silently flatters
        # whichever method batches more candidates per call. Read `best_loss` at
        # the end of a run; use `best_loss_t` inside a loop.
        self._best_loss_t: Optional[torch.Tensor] = None
        self._best_x_t: Optional[torch.Tensor] = None

    @classmethod
    def from_subspace(
        cls,
        subspace,
        base_sd: Dict[str, torch.Tensor],
        loss_batch: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        budget: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Objective":
        """Search in a subspace's projected coordinates.

        Args:
            subspace: A :class:`~polystep.hybrid_subspace.HybridSubspace` (or any
                object with ``specs``/``subspace_dim``/``init_projections``/
                ``reconstruct_batch``, e.g. ``FactoredSubspace``).
            base_sd: Base parameters the perturbation is added to.
            loss_batch: ``{key: (N, *shape)} -> (N,)`` losses.
            budget: Candidate budget.
            device: Device for the projection matrices. Defaults to ``base_sd``'s.
            dtype: Dtype for the projection matrices.

        Returns:
            An :class:`Objective` of dimension ``subspace.subspace_dim``.
        """
        if device is None:
            device = next(iter(base_sd.values())).device
        projections = subspace.init_projections(device, dtype)

        def fn(coords: torch.Tensor) -> torch.Tensor:
            return loss_batch(subspace.reconstruct_batch(projections, base_sd, coords))

        # Coordinates carry no matrix structure (HybridSubspace's projection is a QR'd
        # Gaussian), so each layer's chunk is one flat block. EGGROLL degenerates to
        # dense Gaussian ES here; see the README note.
        shapes = tuple((spec.num_coords,) for spec in subspace.specs)
        return cls(fn, subspace.subspace_dim, budget, shapes=shapes, subspace=subspace)

    @classmethod
    def from_layout(
        cls,
        layout,
        fn: Callable[[torch.Tensor], torch.Tensor],
        budget: int,
    ) -> "Objective":
        """Search in full flat parameter space, keeping the per-tensor shapes.

        Args:
            layout: A :class:`~polystep.transform.ParamLayout`.
            fn: ``(N, total_params) -> (N,)`` losses over flat parameter vectors.
            budget: Candidate budget.
        """
        return cls(fn, layout.total_params, budget, shapes=tuple(e.shape for e in layout.entries))

    @property
    def remaining(self) -> int:
        """Candidates still affordable."""
        return self.budget - self.evals

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Score ``(N, dim)`` candidates, charging ``N`` to the budget."""
        X = X.reshape(-1, X.shape[-1]) if X.dim() > 1 else X.reshape(1, -1)
        n = X.shape[0]
        if X.shape[1] != self.dim:
            raise ValueError(f"candidates have width {X.shape[1]}, expected {self.dim}.")
        if n > self.remaining:
            raise BudgetExhausted(f"{n} candidates requested, {self.remaining} of {self.budget} left.")
        losses = torch.as_tensor(self._fn(X)).reshape(n)
        self.evals += n
        # NaN would win argmin, -inf would win forever.
        finite = torch.nan_to_num(losses.detach(), nan=float("inf"), neginf=float("inf"))
        i = finite.argmin()                       # device tensor: no sync
        cand_loss = finite[i]
        cand_x = X[i].detach()
        if self._best_loss_t is None:
            self._best_loss_t = cand_loss.clone()
            self._best_x_t = cand_x.clone()
        else:
            better = cand_loss < self._best_loss_t
            self._best_loss_t = torch.where(better, cand_loss, self._best_loss_t)
            self._best_x_t = torch.where(better, cand_x, self._best_x_t)
        return losses

    @property
    def best_loss_t(self) -> torch.Tensor:
        """Best loss so far, on device. Free to read inside a loop."""
        if self._best_loss_t is None:
            return torch.tensor(float("inf"))
        return self._best_loss_t

    @property
    def best_loss(self) -> float:
        """Best loss so far as a Python float. Costs one host sync; call it once."""
        return float("inf") if self._best_loss_t is None else float(self._best_loss_t)

    @property
    def best_x(self) -> Optional[torch.Tensor]:
        """Best candidate so far, or None if nothing has been scored."""
        return self._best_x_t


def centered_rank(x: torch.Tensor) -> torch.Tensor:
    """Rank-based utilities in ``[-0.5, 0.5]``, ascending (Salimans et al. 2017)."""
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)
    return x.argsort().argsort().to(x.dtype) / (n - 1) - 0.5


def zscore(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance shaping; zeros when the population is flat."""
    s = x.std()
    if s < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / s
