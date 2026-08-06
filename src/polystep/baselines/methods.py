"""Gradient-free baselines over the shared Objective."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .core import Objective, Result, centered_rank, zscore

__all__ = ["METHODS", "cma_es", "eggroll", "mezo", "openai_es", "random_search", "spsa"]


def _init(obj: Objective, x0: Optional[torch.Tensor]) -> torch.Tensor:
    """Starting point, defaulting to the origin of the search space."""
    if x0 is None:
        return torch.zeros(obj.dim)
    x = torch.as_tensor(x0).detach().clone().reshape(-1).float()
    if x.numel() != obj.dim:
        raise ValueError(f"x0 has {x.numel()} entries, objective dim is {obj.dim}.")
    return x


#: Cap on retained trace points; entries hold device tensors, so thinning keeps memory bounded.
_HISTORY_CAP = 4096


class _Trace(list):
    """Trace that thins itself geometrically instead of growing without bound."""

    __slots__ = ("stride",)

    def __init__(self, *args):
        super().__init__(*args)
        self.stride = 1

    def record(self, obj: Objective, k: int) -> None:
        if k % self.stride:
            return
        self.append((obj.evals, obj.best_loss_t))
        if len(self) > _HISTORY_CAP:
            del self[1::2]
            self.stride *= 2


def _record(history, obj: Objective, k: int) -> None:
    """Append a trace point, thinning if ``history`` is a :class:`_Trace`."""
    if isinstance(history, _Trace):
        history.record(obj, k)
    else:
        history.append((obj.evals, obj.best_loss_t))


def _result(name: str, obj: Objective, x: torch.Tensor, iters: int, history) -> Result:
    # History entries are device tensors so the per-generation append costs no
    # sync; materialize the whole trace once, here, at the end of the run.
    history = [(e, float(v)) for e, v in history]
    best_x = obj.best_x if obj.best_x is not None else x
    return Result(
        method=name,
        x=x,
        best_x=best_x,
        best_loss=obj.best_loss,
        evals=obj.evals,
        iters=iters,
        history=history,
    )


def _shaped(losses: torch.Tensor, shaping: str) -> torch.Tensor:
    """Utilities from losses (lower is better), so a larger weight means better."""
    if shaping == "rank":
        return centered_rank(-losses)
    if shaping == "zscore":
        return zscore(-losses)
    raise ValueError(f"shaping must be 'rank' or 'zscore', got {shaping!r}.")


def openai_es(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma: float = 0.02,
    lr: float = 0.01,
    popsize: int = 32,
    antithetic: bool = True,
    shaping: str = "rank",
    seed: int = 0,
) -> Result:
    """OpenAI Evolution Strategy (Salimans et al. 2017, arXiv:1703.03864). Antithetic sampling and fitness shaping; costs ``popsize`` evaluations per generation."""
    if popsize < 2:
        raise ValueError(f"popsize must be >= 2, got {popsize}.")
    if antithetic and popsize % 2:
        raise ValueError(f"popsize must be even when antithetic=True, got {popsize}.")

    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    half = popsize // 2 if antithetic else popsize
    history, it = _Trace(), 0

    while obj.remaining >= popsize:
        eps = torch.randn(half, obj.dim, generator=gen, device=x.device, dtype=x.dtype)
        if antithetic:
            eps = torch.cat([eps, -eps])
        losses = obj(x + sigma * eps)
        # g = (1 / (pop * sigma)) * eps^T @ utilities; ascent on utility = descent on loss.
        x = x + (lr / (popsize * sigma)) * (eps.t() @ _shaped(losses, shaping))
        obj.iterate = x
        it += 1
        _record(history, obj, it)

    return _result("openai_es", obj, x, it, history)


def spsa(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    a: float = 0.1,
    c: float = 0.1,
    A: Optional[float] = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: int = 0,
) -> Result:
    """SPSA (Spall 1992). Two evaluations per iteration; gains follow Spall's ``a_k = a/(A+k)^alpha``, ``c_k = c/k^gamma``."""
    x = _init(obj, x0)
    if A is None:
        A = 0.1 * (obj.budget // 2)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    history, k = _Trace(), 0

    while obj.remaining >= 2:
        k += 1
        a_k = a / ((A + k) ** alpha)
        c_k = c / (k**gamma)
        # Bernoulli +-1.
        delta = torch.randint(0, 2, (obj.dim,), generator=gen, device=x.device, dtype=x.dtype) * 2 - 1
        losses = obj(torch.stack([x + c_k * delta, x - c_k * delta]))
        # g_hat_i = (L+ - L-) / (2 c_k delta_i); keep the scalar a 0-dim tensor to avoid a sync per step.
        x = x - (a_k / (2.0 * c_k)) * (losses[0] - losses[1]) * delta
        obj.iterate = x
        _record(history, obj, k)

    return _result("spsa", obj, x, k, history)


def mezo(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-3,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> Result:
    """MeZO: memory-efficient zeroth-order SGD (Malladi et al. 2023, arXiv:2305.17333). Two evaluations per step; z is regenerated from its seed instead of stored."""
    x = _init(obj, x0)
    seeds = torch.Generator(device="cpu").manual_seed(seed)
    history, k = _Trace(), 0

    def z_from(step_seed: int) -> torch.Tensor:
        g = torch.Generator(device=x.device).manual_seed(step_seed)
        return torch.randn(obj.dim, generator=g, device=x.device, dtype=x.dtype)

    while obj.remaining >= 2:
        step_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=seeds).item())
        z = z_from(step_seed)
        candidates = torch.stack([x + eps * z, x - eps * z])
        del z  # regenerated below; never held across the evaluation.
        losses = obj(candidates)
        # 0-dim tensor, not a float: see the note in spsa.
        projected_grad = (losses[0] - losses[1]) / (2.0 * eps)
        z = z_from(step_seed)
        if weight_decay:
            x = x * (1.0 - lr * weight_decay)
        x = x - (lr * projected_grad) * z
        obj.iterate = x
        k += 1
        _record(history, obj, k)

    return _result("mezo", obj, x, k, history)


def random_search(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma: float = 0.1,
    seed: int = 0,
) -> Result:
    """Random search: one random direction per step, accepted if the loss drops."""
    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    fx = obj(x.unsqueeze(0))[0]
    history, k = _Trace([(obj.evals, obj.best_loss_t)]), 0

    while obj.remaining >= 1:
        cand = x + sigma * torch.randn(obj.dim, generator=gen, device=x.device, dtype=x.dtype)
        f = obj(cand.unsqueeze(0))[0]
        # Accept-on-improvement, resolved on-device to avoid a sync per candidate.
        better = f < fx
        x = torch.where(better, cand, x)
        obj.iterate = x
        fx = torch.where(better, f, fx)
        k += 1
        _record(history, obj, k)

    return _result("random_search", obj, x, k, history)


def _lowrank_noise(
    shapes: Sequence[Tuple[int, ...]],
    rank: int,
    n: int,
    dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build ``n`` flat perturbations from per-matrix ``E = A B^T / sqrt(r)`` (arXiv:2511.16652 Sec. 4.1); 1D entries get a dense Gaussian."""
    out = torch.empty(n, dim, device=device, dtype=dtype)
    off = 0
    for shape in shapes:
        numel = math.prod(shape)
        if len(shape) >= 2:
            m = shape[0]
            k = math.prod(shape[1:])
            r = max(1, min(rank, m, k))
            A = torch.randn(n, m, r, generator=generator, device=device, dtype=dtype)
            B = torch.randn(n, k, r, generator=generator, device=device, dtype=dtype)
            out[:, off : off + numel] = ((A @ B.transpose(-1, -2)) / math.sqrt(r)).reshape(n, numel)
        else:
            out[:, off : off + numel] = torch.randn(n, numel, generator=generator, device=device, dtype=dtype)
        off += numel
    return out


def eggroll(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma: float = 0.02,
    lr: float = 0.01,
    rank: int = 1,
    popsize: int = 32,
    shaping: str = "sign",
    seed: int = 0,
) -> Result:
    """EGGROLL: low-rank evolution strategies (Sarkar et al., arXiv:2511.16652). Each perturbation is ``E = A B^T / sqrt(r)``; the population is sampled in antithetic pairs."""
    if popsize < 2 or popsize % 2:
        raise ValueError(f"popsize must be even and >= 2, got {popsize}.")

    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    pairs = popsize // 2
    history, it = _Trace(), 0

    while obj.remaining >= popsize:
        E = _lowrank_noise(obj.shapes, rank, pairs, obj.dim, gen, x.device, x.dtype)
        losses = obj(torch.cat([x + sigma * E, x - sigma * E]))
        l_plus, l_minus = losses[:pairs], losses[pairs:]
        if shaping == "sign":
            # Fitness is -loss, so sign(s+ - s-) == sign(loss- - loss+).
            w = torch.sign(l_minus - l_plus)
        else:
            # Shape both halves; the -E half contributes with a flipped sign.
            u = _shaped(losses, shaping)
            w = u[:pairs] - u[pairs:]
        x = x + (lr / pairs) * (E.t() @ w)
        obj.iterate = x
        it += 1
        _record(history, obj, it)

    return _result("eggroll", obj, x, it, history)


def cma_es(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma0: float = 0.5,
    popsize: Optional[int] = None,
    diagonal: Optional[bool] = None,
    max_popsize: int = 256,
    seed: int = 0,
) -> Result:
    """CMA-ES via pycma (Hansen & Ostermeier 2001). Uses the diagonal variant above 1000 dimensions; may stop before the budget on pycma's own criteria."""
    try:
        import cma
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("CMA-ES needs pycma. Install with: pip install cma") from e
    if obj.dim < 2:
        raise ValueError(f"pycma needs dim >= 2, got {obj.dim}.")

    x = _init(obj, x0)
    if diagonal is None:
        diagonal = obj.dim >= 1000
    opts = {
        "popsize": popsize,
        # pycma reads seed 0/None as "seed from the clock".
        "seed": int(seed) + 1,
        "verbose": -9,
        "CMA_diagonal": bool(diagonal),
    }

    def _make(mean, sigma, pop):
        o = dict(opts)
        if pop is not None:
            o["popsize"] = pop
        return cma.CMAEvolutionStrategy(mean, sigma, {k: v for k, v in o.items() if v is not None})

    es = _make(x.double().tolist(), sigma0, popsize)
    history, it = _Trace(), 0
    best = torch.tensor(es.result.xfavorite, device=x.device, dtype=x.dtype)

    # IPOP restarts (Auger & Hansen 2005): pycma stops early on its own criteria, so double the population and resume from the incumbent on each stop.
    restarts = 0
    while obj.remaining >= es.popsize:
        while obj.remaining >= es.popsize and not es.stop():
            solutions = es.ask()
            # pycma hands back a list of numpy rows; stack once rather than per row.
            X = torch.from_numpy(np.asarray(solutions)).to(device=x.device, dtype=x.dtype)
            es.tell(solutions, obj(X).tolist())
            # pycma's distribution mean.
            obj.iterate = torch.as_tensor(es.result.xfavorite, device=x.device, dtype=x.dtype)
            it += 1
            _record(history, obj, it)

        best = torch.tensor(es.result.xfavorite, device=x.device, dtype=x.dtype)
        if obj.remaining < es.popsize:
            break
        restarts += 1
        # Budget cap stops a spin; max_popsize cap stops the doubling from exhausting GPU memory.
        pop = min(int(es.popsize) * 2, max_popsize, max(2, int(obj.remaining)))
        if pop <= es.popsize:
            # At the ceiling: restart at the same population to spend the remaining budget.
            pop = min(int(es.popsize), max(2, int(obj.remaining)))
            if pop < 2:
                break
        es = _make(best.double().tolist(), sigma0, pop)

    obj.iterate = best
    return _result("cma_es", obj, best, it, history)


#: Name -> method, for runners that select a baseline by string.
METHODS: Dict[str, Callable[..., Result]] = {
    "openai_es": openai_es,
    "spsa": spsa,
    "mezo": mezo,
    "random_search": random_search,
    "eggroll": eggroll,
    "cma_es": cma_es,
}
