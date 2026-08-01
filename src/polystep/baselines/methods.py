"""Gradient-free baselines over the shared :class:`~polystep.baselines.core.Objective`.

Every method has the same shape::

    result = method(objective, x0=None, **hyperparameters)

and the same stopping rule: loop while the objective can still afford one more
iteration. None of them can overspend, because the objective refuses.

Sources
-------
- :func:`openai_es`     ported from ``experiments/baselines/openai_es.py``
- :func:`spsa`          ported from ``experiments/baselines/spsa.py``
- :func:`cma_es`        pycma, replacing the four inline copies in ``experiments/runners/``
- :func:`random_search` trivial, new: the subspace-only control
- :func:`eggroll`       from arXiv:2511.16652 (Sarkar et al.), Alg. 1 / Eq. 6 / App. H.2
- :func:`mezo`          from arXiv:2305.17333 (Malladi et al. 2023), Alg. 1
"""

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
    """OpenAI Evolution Strategy (Salimans et al. 2017, arXiv:1703.03864).

    Antithetic sampling and fitness shaping, ported from
    ``experiments/baselines/openai_es.py``. Costs ``popsize`` evaluations per
    generation.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        sigma: Probe radius (noise std).
        lr: Step size on the estimated gradient.
        popsize: Perturbations per generation; must be even when ``antithetic``.
        antithetic: Mirror half the population for variance reduction.
        shaping: ``"rank"`` (Salimans) or ``"zscore"``.
        seed: Noise seed.
    """
    if popsize < 2:
        raise ValueError(f"popsize must be >= 2, got {popsize}.")
    if antithetic and popsize % 2:
        raise ValueError(f"popsize must be even when antithetic=True, got {popsize}.")

    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    half = popsize // 2 if antithetic else popsize
    history, it = [], 0

    while obj.remaining >= popsize:
        eps = torch.randn(half, obj.dim, generator=gen, device=x.device, dtype=x.dtype)
        if antithetic:
            eps = torch.cat([eps, -eps])
        losses = obj(x + sigma * eps)
        # g = (1 / (pop * sigma)) * eps^T @ utilities; ascent on utility = descent on loss.
        x = x + (lr / (popsize * sigma)) * (eps.t() @ _shaped(losses, shaping))
        it += 1
        history.append((obj.evals, obj.best_loss_t))

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
    """SPSA (Spall 1992). Two evaluations per iteration, whatever the dimension.

    Ported from ``experiments/baselines/spsa.py``; gains follow Spall's
    finite-sample recommendations ``a_k = a / (A + k)^alpha``,
    ``c_k = c / k^gamma``.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        a: Step-size gain.
        c: Probe radius gain.
        A: Stability constant, default 10% of the affordable iterations.
        alpha: Step-size decay exponent.
        gamma: Probe decay exponent (must stay below ``alpha``).
        seed: Perturbation seed.
    """
    x = _init(obj, x0)
    if A is None:
        A = 0.1 * (obj.budget // 2)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    history, k = [], 0

    while obj.remaining >= 2:
        k += 1
        a_k = a / ((A + k) ** alpha)
        c_k = c / (k**gamma)
        # Bernoulli +-1.
        delta = torch.randint(0, 2, (obj.dim,), generator=gen, device=x.device, dtype=x.dtype) * 2 - 1
        losses = obj(torch.stack([x + c_k * delta, x - c_k * delta]))
        # g_hat_i = (L+ - L-) / (2 c_k delta_i), and 1/delta_i == delta_i for +-1.
        x = x - (a_k * (losses[0] - losses[1]).item() / (2.0 * c_k)) * delta
        history.append((obj.evals, obj.best_loss_t))

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
    """MeZO: memory-efficient zeroth-order SGD (Malladi et al. 2023, arXiv:2305.17333).

    Two evaluations per step, like SPSA, but with Gaussian ``z`` and constant
    gains. The defining trick is memory: ``z`` is never stored, only the per-step
    seed is, and ``z`` is regenerated on demand. This implementation keeps that
    property -- ``z`` is dropped before the evaluation and regenerated from the
    same seed for the update.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        eps: Probe radius.
        lr: Step size.
        weight_decay: Decoupled L2, as in the paper's SGD variant.
        seed: Seed for the per-step seed stream.
    """
    x = _init(obj, x0)
    seeds = torch.Generator(device="cpu").manual_seed(seed)
    history, k = [], 0

    def z_from(step_seed: int) -> torch.Tensor:
        g = torch.Generator(device=x.device).manual_seed(step_seed)
        return torch.randn(obj.dim, generator=g, device=x.device, dtype=x.dtype)

    while obj.remaining >= 2:
        step_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=seeds).item())
        z = z_from(step_seed)
        candidates = torch.stack([x + eps * z, x - eps * z])
        del z  # regenerated below; never held across the evaluation
        losses = obj(candidates)
        projected_grad = (losses[0] - losses[1]).item() / (2.0 * eps)
        z = z_from(step_seed)
        if weight_decay:
            x = x * (1.0 - lr * weight_decay)
        x = x - (lr * projected_grad) * z
        k += 1
        history.append((obj.evals, obj.best_loss_t))

    return _result("mezo", obj, x, k, history)


def random_search(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma: float = 0.1,
    seed: int = 0,
) -> Result:
    """Random search: one random direction per step, accepted if the loss drops.

    The control for "how much of the gain is the subspace representation rather
    than the update rule": hand this the same :class:`Objective` as PolyStep and
    the only thing that differs is the update. One evaluation per step, plus one
    to score the starting point.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        sigma: Probe radius.
        seed: Direction seed.
    """
    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    fx = obj(x.unsqueeze(0))[0].item()
    history, k = [(obj.evals, obj.best_loss_t)], 0

    while obj.remaining >= 1:
        cand = x + sigma * torch.randn(obj.dim, generator=gen, device=x.device, dtype=x.dtype)
        f = obj(cand.unsqueeze(0))[0].item()
        if f < fx:
            x, fx = cand, f
        k += 1
        history.append((obj.evals, obj.best_loss_t))

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
    """``n`` flat perturbations built from per-matrix ``E = A B^T / sqrt(r)``.

    arXiv:2511.16652 Sec. 4.1: ``A in R^(m x r)``, ``B in R^(n x r)`` i.i.d.
    zero-mean unit-variance, the ``1/sqrt(r)`` keeping ``Var(E)`` bounded in ``r``.
    1D entries (biases, norms) have no matrix structure to factor and get a dense
    Gaussian, matching how ``FactoredSubspace`` treats them.
    """
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
    """EGGROLL: low-rank evolution strategies (Sarkar et al., arXiv:2511.16652).

    Per worker, ``E_i = A_i B_i^T / sqrt(r)`` with ``A_i in R^(m x r)``,
    ``B_i in R^(n x r)``; fitness at ``M + sigma E_i``; the mean moves along
    ``(alpha / N) sum_i E_i f_i`` (Eq. 6), which is full-rank once ``N r`` exceeds
    ``min(m, n)`` even though every perturbation is rank ``r``. Effective at
    ``r = 1``. Population sampled in antithetic pairs, so ``popsize`` must be even.

    Where the matrices are comes from :attr:`Objective.shapes`. Inside a
    ``HybridSubspace`` the coordinates are unstructured, so the perturbations are
    dense Gaussians and this degenerates to plain ES on the coordinates -- see the
    README.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        sigma: Probe radius. The paper notes the linearisation needs ``o(d^-1/2)``.
        lr: Step size; the paper absorbs ``1/sigma`` into it.
        rank: Perturbation rank ``r``, clipped per entry to ``min(m, n)``.
        popsize: Candidates per generation (``popsize / 2`` antithetic pairs).
        shaping: ``"sign"`` is the paper's antithetic-pair shaping
            ``sign(s+ - s-)`` (App. H.2); ``"rank"``/``"zscore"`` shape the whole
            population instead, matching :func:`openai_es`.
        seed: Noise seed.
    """
    if popsize < 2 or popsize % 2:
        raise ValueError(f"popsize must be even and >= 2, got {popsize}.")

    x = _init(obj, x0)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    pairs = popsize // 2
    history, it = [], 0

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
        it += 1
        history.append((obj.evals, obj.best_loss_t))

    return _result("eggroll", obj, x, it, history)


def cma_es(
    obj: Objective,
    x0: Optional[torch.Tensor] = None,
    *,
    sigma0: float = 0.5,
    popsize: Optional[int] = None,
    diagonal: Optional[bool] = None,
    seed: int = 0,
) -> Result:
    """CMA-ES via pycma (Hansen & Ostermeier 2001).

    The one implementation, replacing the four inline copies in
    ``run_elevation.py``, ``run_moe.py``, ``run_timeseries.py`` and
    ``run_maxsat.py``. Keeps their convention of switching to the diagonal
    (separable) variant above 1000 dimensions, where the full covariance is
    ``O(d^2)``.

    May stop before the budget is spent: pycma's own convergence criteria still
    apply. ``result.evals`` reports what was actually used.

    Args:
        obj: The objective.
        x0: Start point, default zeros.
        sigma0: Initial step size.
        popsize: Population size; ``None`` lets pycma pick ``4 + 3 ln d``.
        diagonal: Force the separable variant. ``None`` picks it for ``dim >= 1000``.
        seed: Seed.

    Raises:
        ImportError: If pycma is not installed (``pip install cma``).
    """
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
    es = cma.CMAEvolutionStrategy(x.double().tolist(), sigma0, {k: v for k, v in opts.items() if v is not None})
    history, it = [], 0

    while obj.remaining >= es.popsize and not es.stop():
        solutions = es.ask()
        # pycma hands back a list of numpy rows; stack once rather than per row.
        X = torch.from_numpy(np.asarray(solutions)).to(device=x.device, dtype=x.dtype)
        es.tell(solutions, obj(X).tolist())
        it += 1
        history.append((obj.evals, obj.best_loss_t))

    x = torch.tensor(es.result.xfavorite, device=x.device, dtype=x.dtype)
    return _result("cma_es", obj, x, it, history)


#: Name -> method, for runners that select a baseline by string.
METHODS: Dict[str, Callable[..., Result]] = {
    "openai_es": openai_es,
    "spsa": spsa,
    "mezo": mezo,
    "random_search": random_search,
    "eggroll": eggroll,
    "cma_es": cma_es,
}
