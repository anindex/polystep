"""Shared protocol for the gradient-free baselines."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["BudgetExhausted", "Objective", "Result", "centered_rank", "zscore"]


class BudgetExhausted(RuntimeError):
    """Raised when a candidate batch would exceed the evaluation budget."""


@dataclass
class Result:
    """Outcome of one baseline run."""

    method: str
    x: torch.Tensor
    best_x: torch.Tensor
    best_loss: float
    evals: int
    iters: int
    history: List[Tuple[int, float]] = field(default_factory=list)


def _coord_shape(spec, factored: bool) -> Tuple[int, ...]:
    """Shape of one layer's coordinate block: ``(d_out, rank)`` for factored, flat otherwise."""
    if factored and getattr(spec, "is_projected", False):
        rows = spec.original_shape[0]
        if rows > 0 and spec.num_coords % rows == 0:
            return (rows, spec.num_coords // rows)
    return (spec.num_coords,)


class Objective:
    """A budgeted, optionally projected black-box objective."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        budget: int,
        *,
        shapes: Optional[Sequence[Sequence[int]]] = None,
        subspace: object = None,
        deadline_s: Optional[float] = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}.")
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}.")
        self._fn = fn
        self.dim = int(dim)
        self.budget = int(budget)
        self.subspace = subspace
        self.deadline_s = None if deadline_s is None else float(deadline_s)
        self._t0: Optional[float] = None
        self.shapes: Tuple[Tuple[int, ...], ...] = (
            tuple(tuple(int(d) for d in s) for s in shapes) if shapes else ((self.dim,),)
        )
        covered = sum(math.prod(s) for s in self.shapes)
        if covered != self.dim:
            raise ValueError(f"shapes cover {covered} entries but dim is {self.dim}.")
        self.evals = 0
        # Best-so-far is tracked on-device to avoid per-call host syncs. Read best_loss once at the end; use best_loss_t in a loop.
        self._best_loss_t: Optional[torch.Tensor] = None
        self._best_x_t: Optional[torch.Tensor] = None
        #: The method's own current iterate. This is NOT best_x: for a population method, best_x is a sampled candidate displaced from the mean. Score both and select on validation.
        self.iterate: Optional[torch.Tensor] = None

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
        deadline_s: Optional[float] = None,
    ) -> "Objective":
        """Search in a subspace's projected coordinates."""
        if device is None:
            device = next(iter(base_sd.values())).device
        projections = subspace.init_projections(device, dtype)

        def fn(coords: torch.Tensor) -> torch.Tensor:
            return loss_batch(subspace.reconstruct_batch(projections, base_sd, coords))

        # shapes tells EGGROLL's low-rank sampler where the matrix structure is in coordinate space: flat for HybridSubspace, (d_out, r) for FactoredSubspace.
        factored = bool(getattr(subspace, "coords_are_factored", False))
        shapes = tuple(_coord_shape(spec, factored) for spec in subspace.specs)
        return cls(fn, subspace.subspace_dim, budget, shapes=shapes, subspace=subspace, deadline_s=deadline_s)

    @classmethod
    def from_layout(
        cls,
        layout,
        fn: Callable[[torch.Tensor], torch.Tensor],
        budget: int,
        *,
        deadline_s: Optional[float] = None,
    ) -> "Objective":
        """Search in flat parameter space, keeping the per-tensor shapes."""
        return cls(
            fn,
            layout.total_params,
            budget,
            shapes=tuple(e.shape for e in layout.entries),
            deadline_s=deadline_s,
        )

    @property
    def elapsed_s(self) -> float:
        """Seconds since the first call. Zero before the run starts."""
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    @property
    def out_of_time(self) -> bool:
        """True once a wall-clock deadline has passed."""
        return self.deadline_s is not None and self.elapsed_s >= self.deadline_s

    @property
    def remaining(self) -> int:
        """Candidates still affordable, under whichever budget binds first."""
        if self.out_of_time:
            return 0
        return self.budget - self.evals

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Score ``(N, dim)`` candidates, charging ``N`` to the budget."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        X = X.reshape(-1, X.shape[-1]) if X.dim() > 1 else X.reshape(1, -1)
        n = X.shape[0]
        if X.shape[1] != self.dim:
            raise ValueError(f"candidates have width {X.shape[1]}, expected {self.dim}.")
        if n > self.remaining:
            if self.out_of_time:
                raise BudgetExhausted(
                    f"wall-clock deadline of {self.deadline_s:.0f}s passed at {self.elapsed_s:.0f}s "
                    f"after {self.evals} candidates."
                )
            raise BudgetExhausted(f"{n} candidates requested, {self.remaining} of {self.budget} left.")
        losses = torch.as_tensor(self._fn(X)).reshape(n)
        self.evals += n
        # NaN would win argmin, -inf would win forever.
        finite = torch.nan_to_num(losses.detach(), nan=float("inf"), neginf=float("inf"))
        # Take the value by reduction and the row by a device-side gather: indexing with argmin's result would sync.
        # The loss may be computed on a different device than the search space; normalize once here so scalars follow the candidates.
        if finite.device != X.device:
            finite = finite.to(X.device)
        i = finite.argmin()
        cand_loss = finite.amin()
        cand_x = X.index_select(0, i.view(1)).squeeze(0).detach()
        if self._best_loss_t is None:
            self._best_loss_t = cand_loss.clone()
            self._best_x_t = cand_x.clone()
        else:
            better = cand_loss < self._best_loss_t
            self._best_loss_t = torch.where(better, cand_loss, self._best_loss_t)
            self._best_x_t = torch.where(better, cand_x, self._best_x_t)
        # Return losses on the candidates' device; methods combine them with state that lives there.
        return losses.to(X.device) if losses.device != X.device else losses

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
