#!/usr/bin/env python
"""Fairness controls for the baseline tables. Three knobs:

1. **Matched representation.** PolyStep ran inside a ``HybridSubspace`` while every
   baseline ran in full parameter space. :func:`make_subspace` hands every method the
   same subspace class, rank and seed, and :func:`run_baseline` searches in its
   coordinates. The one documented exception is EGGROLL: its rank-``r`` ``A B^T``
   perturbation needs matrix-structured coordinates, which ``HybridSubspace`` does not
   have, so it gets :class:`~polystep.factored_subspace.FactoredSubspace` -- the
   subspace that *is* that parameterization -- and the result JSON records which
   subspace each method actually used.

2. **Matched budget.** Forward evaluations were counted three incompatible ways.
   Everything here counts candidates through
   :class:`~polystep.baselines.core.Objective`, and :func:`polystep_eval_budget`
   derives the shared budget from what PolyStep itself consumes for its configured
   number of epochs, so a fairness table is a fixed-budget comparison by construction.

3. **Matched tuning.** :data:`TUNING_GRID` gives every method a per-method grid of the
   same size, so "configurations tried x cost per configuration" is comparable.
   ``variant_sweep.py --baseline`` runs it and writes the cost out.

Every run also emits the per-generation trajectory against *cumulative candidate
evaluations*, which is what the accuracy-vs-evaluations figure plots.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import torch

from polystep.baselines import METHODS, Objective
from polystep.epsilon import PowerDecay, resolve_radius
from polystep.factored_subspace import FactoredSubspace
from polystep.hybrid_subspace import HybridSubspace

__all__ = [
    "DEFAULT_SELECTION_PATH",
    "FAIR_METHODS",
    "RADIUS_KEYS",
    "TUNING_GRID",
    "TestSplitTripwire",
    "apply_point",
    "apply_polystep_multipliers",
    "apply_theory_mode",
    "build_polystep",
    "fair_hyperparams",
    "load_selection",
    "make_subspace",
    "minibatch_loss",
    "SEQUENTIAL_METHODS",
    "STEP_CAPPED_SHOWCASES",
    "MATCH_AXIS",
    "budget_for_method",
    "matched_budget",
    "polystep_eval_budget",
    "polystep_wall_seconds",
    "step_matched_budget",
    "probe_scale_of",
    "run_baseline",
    "select_best",
    "subspace_tag",
    "tuning_cost",
    "write_selection",
]

#: The gradient-free methods a fairness table compares. Adam is gradient-based and is
#: not a peer; it is not in here.
FAIR_METHODS = ("openai_es", "spsa", "mezo", "random_search", "eggroll", "cma_es")

#: Population size every population method shares in fairness mode, so "one generation"
#: costs the same for openai_es, eggroll and cma_es.
#:
#: 32 is right for the step axis and wrong for the others. Measured on MoE, a generation
#: costs ~26 ms at any popsize up to 1024, so 32 candidates buy 1,072 evals/s where 2048
#: buy 49,428 -- a 46x throughput handicap that has nothing to do with the method. Set
#: ``POLYSTEP_FAIR_POPSIZE`` when matching on evaluations or wall-clock, where throughput
#: is the whole point.
FAIR_POPSIZE = int(os.environ.get("POLYSTEP_FAIR_POPSIZE", "32"))


# --- matched budget ---------------------------------------------------------------


def evals_per_step(opt) -> int:
    """Candidates PolyStep scores per ``step()``.

    ``num_particles x num_polytope_vertices x num_probe``: every particle probes every
    vertex of its polytope, ``K`` times. Read off the built optimizer rather than
    re-derived from the config, because the particle count depends on how the subspace
    dimension pads to ``particle_dim``.
    """
    return int(opt._state.X.shape[0]) * int(opt._polytope_vertices.shape[0]) * int(opt.num_probe)


def polystep_eval_budget(opt, total_steps: int) -> int:
    """The shared candidate budget: what PolyStep spends over ``total_steps`` steps.

    Only the *OT* steps evaluate probes. Under ``amortize_steps = a > 1`` the optimizer
    coasts on the transport EMA for ``a - 1`` steps out of every ``a``
    (:meth:`PolyStepOptimizer.step`), and ``_step_momentum`` calls the closure zero
    times, so those steps cost nothing. Charging ``evals_per_step * total_steps`` would
    hand the baselines ``a`` times what PolyStep spent, 3x on MNIST and on the time
    series.
    """
    a = max(1, int(getattr(opt, "amortize_steps", 1)))
    ot_steps = -(-int(total_steps) // a)  # ceil: counter % a == 0 is an OT step
    return evals_per_step(opt) * ot_steps


# --- matched representation -------------------------------------------------------


def make_subspace(layout, *, rank: int, seed: int, method: str = "polystep", **kwargs):
    """The subspace a method searches in.

    Same class, same rank, same seed for everything except EGGROLL, which gets
    ``FactoredSubspace``: its perturbations are ``A B^T`` on weight matrices, and
    ``HybridSubspace`` coordinates carry no matrix structure, so EGGROLL there
    degenerates to dense Gaussian ES (see ``README.md``). ``FactoredSubspace`` is that
    parameterization and takes the same ``rank``/``seed``.
    """
    cls = FactoredSubspace if method == "eggroll" else HybridSubspace
    return cls.from_layout(layout, rank=rank, seed=seed, **kwargs)


def build_polystep(model, seed: int, total_steps: int, cfg: dict, solver: str = "softmax"):
    """``(layout, subspace, optimizer)`` for one gallery config.

    Shared by each runner's training loop and by its ``fair_eval_budget``, which needs
    the built optimizer to know what a step costs.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.transform import ParamLayout

    def sched(key, flat_default):
        # Scheduled when the config carries both endpoints; flat otherwise, which is
        # also what --theory-mode leaves behind.
        if f"{key}_init" not in cfg:
            return cfg.get(key, flat_default)
        init, target = cfg[f"{key}_init"], cfg[f"{key}_target"]
        return CosineEpsilon(init=init, target=target, decay=(init - target) / max(1, total_steps))

    layout = ParamLayout.from_module(model)
    subspace = make_subspace(
        layout,
        rank=cfg["rank"],
        seed=seed,
        rotation_mode="random",
        rotation_interval=cfg["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=cfg["absorb_interval"],
    )
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=sched("epsilon", 0.5),
        step_radius=sched("step_radius", 1.0),
        probe_radius=sched("probe_radius", 1.0),
        num_probe=cfg["num_probe"],
        subspace=subspace,
        chunk_size=cfg.get("chunk_size", 1024),
        probe_radius_jitter=cfg.get("probe_radius_jitter", 0.0),
        probe_radius_jitter_dist=cfg.get("probe_radius_jitter_dist", "smooth"),
        step_radius_jitter=cfg.get("step_radius_jitter", 0.0),
        polytope_type=cfg.get("polytope_type", "simplex"),
        amortize_steps=cfg.get("amortize_steps", 1),
        amortize_ema=cfg.get("amortize_ema", 0.0),
        use_momentum=cfg.get("use_momentum", False),
        momentum_init=cfg.get("momentum_init", 0.5),
        momentum_final=cfg.get("momentum_final", 0.95),
        biased_rotation=cfg.get("biased_rotation", False),
        solver=solver,
    )
    return layout, subspace, optimizer


def subspace_tag(subspace, rank: Optional[int], *, shared_dim: Optional[int] = None) -> Dict[str, object]:
    """The keys that let a reader check a table was matched -- including where it is not.

    Rank is matched across the whole table. Dimension is not, for EGGROLL: matching
    dimension would force ``FactoredSubspace`` to rank 1, and a ``(d_out, 1)``
    coordinate matrix is already rank 1, so its ``A B^T`` sampler would have nothing
    to factor and it would silently become dense Gaussian ES -- the degeneracy the
    whole arrangement exists to avoid. Rank is matched and the dimension gap is
    reported: pass ``shared_dim`` (what every other method searched) and the gap
    lands in the JSON as ``subspace_dim_shared`` / ``subspace_dim_matched``.
    """
    dim = getattr(subspace, "subspace_dim", None) if subspace is not None else None
    tag = {
        "subspace_class": type(subspace).__name__ if subspace is not None else None,
        "subspace_rank": rank,
        "subspace_dim": dim,
    }
    if shared_dim is not None:
        tag["subspace_dim_shared"] = int(shared_dim)
        tag["subspace_dim_matched"] = dim == shared_dim
        tag["matched_on"] = "rank" if dim != shared_dim else "rank+dimension"
    return tag


# --- matched hyperparameters ------------------------------------------------------


def fair_hyperparams(method: str, *, probe_scale: float, seed: int, popsize: int = FAIR_POPSIZE) -> dict:
    """Default hyperparameters at a shared probe scale.

    ``probe_scale`` is the shared *per-coordinate* probe scale from
    :func:`probe_scale_of`, not PolyStep's raw probe radius. Every method that has a
    notion of "how far from the current iterate do I sample" gets that same number:
    ``sigma`` for the ES family, ``c`` for SPSA, ``eps`` for MeZO, ``sigma0`` for
    CMA-ES. Step sizes stay at each method's own published default, since they are not
    the same quantity.
    """
    hp = {
        "openai_es": dict(sigma=probe_scale, lr=0.01, popsize=popsize, shaping="rank"),
        "spsa": dict(a=0.1, c=probe_scale, alpha=0.602, gamma=0.101),
        "mezo": dict(eps=probe_scale, lr=1e-2),
        "random_search": dict(sigma=probe_scale),
        "eggroll": dict(sigma=probe_scale, lr=0.01, popsize=popsize, rank=1),
        "cma_es": dict(sigma0=probe_scale, popsize=popsize),
    }[method]
    return {**hp, "seed": seed}


#: Candidates a single generation of each method costs. The population methods draw
#: ``popsize``; the rest are sequential by construction and draw one or two whatever the
#: dimension. This is the number that makes a step budget and an evaluation budget
#: different questions.
EVALS_PER_GENERATION = {
    "openai_es": None,  # popsize
    "eggroll": None,  # popsize
    "cma_es": None,  # popsize
    "spsa": 2,  # f(x + c*delta), f(x - c*delta)
    "mezo": 2,  # the same two-point estimate
    "random_search": 1,
}

#: Showcases whose *population* methods are also budgeted on steps rather than
#: evaluations. moe alone: ``polystep_eval_budget`` there is 100,552,050 --- ten times
#: the next largest, because moe runs 6,330 steps in a 14,120-dimensional subspace ---
#: which at ``FAIR_POPSIZE`` is 3.14M generations and, measured, 35 hours per cell. That
#: is 4.5 GPU-days to fill three rows of the *appendix* table, on the secondary axis.
#: The step axis, which is the headline one, reads at 6,330 steps
#: and is unaffected. Every other showcase stays eval-matched; their population cells
#: total ~91 cell-hours, which is affordable.
STEP_CAPPED_SHOWCASES = ("moe", "argmax", "timeseries")

#: ``timeseries`` joins them for a reason that is not only cost. Its budget is 21,115,080
#: and the LSTM forward is slow enough that a population cell measured ~7 hours, 15 cells
#: -- but the deciding point is that ETTh1's baselines cannot be tuned at all: no
#: load_selection, and tune_gallery has no minimize-mode branch.
#: Spending a GPU-day to fill an eval-axis row that would have to be captioned "untuned"
#: buys nothing the step axis does not already give.

#: ``argmax`` joins moe for the same reason at a different scale. Its budget is
#: 42,014,160 -- the largest in the gallery -- and DiscreteAttentionNet's eight
#: hard-attention slots cost 59 ms per generation against ~7 ms for an MLP of the same
#: parameter count, measured mid-run. That is 21 hours per population cell, 15 cells.
#: The eval-axis appendix still covers snn, mnist, int8 and staircase; argmax and moe
#: report on the step axis, which is the headline one and is unaffected.

#: Methods whose generation costs at most two candidates. At a matched *evaluation*
#: budget these take budget/2 or budget sequential steps -- 7.77M and 15.5M on the SNN
#: against PolyStep's 3,180 -- and measuring that costs days per cell to fill one
#: appendix entry. They are budgeted on steps instead; see :func:`step_matched_budget`.
SEQUENTIAL_METHODS = ("spsa", "mezo", "random_search")


def step_matched_budget(method: str, polystep_steps: int, popsize: int = FAIR_POPSIZE) -> int:
    """Evaluations that buys ``method`` exactly ``polystep_steps`` of its own steps.

    The primary axis is accuracy at a matched number of optimizer steps (REBUTTAL.md 1b):
    a forward pass is cheap and parallel, a step is a sequential dependency. This is that
    axis expressed as the budget the method has to be given to reach it.
    """
    per = EVALS_PER_GENERATION.get(method)
    if per is None:
        per = int(popsize)
    return max(1, int(polystep_steps) * int(per))


#: So only the deadline binds on a wall-clock-matched arm.
WALLCLOCK_EVAL_CAP = 1 << 40


def polystep_wall_seconds(showcase: str, seed: int, results_dir: str) -> Optional[float]:
    """Seconds PolyStep spent on this exact cell, or None if it has not been run.

    Per seed, not averaged, so the deadline is the cost of the run the baseline is
    compared against.
    """
    name = f"{showcase}_polystep_{seed}.json"
    # The wall-clock arm writes to a subdirectory of the campaign it is matched against,
    # like results/revision/evalmatched does, so look one level up too.
    for path in (os.path.join(results_dir, name), os.path.join(results_dir, "..", name)):
        if os.path.exists(path):
            with open(path) as fh:
                payload = json.load(fh)
            value = (payload.get("metrics") or {}).get("wall_time_seconds")
            return None if value is None else float(value)
    return None


def budget_for_method(
    method: str,
    eval_budget: int,
    polystep_steps: int,
    popsize: int = FAIR_POPSIZE,
    showcase: Optional[str] = None,
    match: str = "auto",
):
    """``(budget, match_axis)`` for one baseline arm. Three axes, all recorded.

    ``evals``: population methods get PolyStep's evaluation budget. At popsize 32 that
    buys them 21x to 153x PolyStep's steps, so this axis favours them, not us.

    ``steps``: the sequential methods. A matched evaluation budget is 7.77M sequential
    steps on the SNN against PolyStep's 3,180, measured at 42 h per cell.

    ``wallclock``: everyone gets the seconds PolyStep spent on the same cell. Needed on
    argmax, MoE and timeseries, where PolyStep's evaluation budget (42.0M and 100.6M
    candidates) is unaffordable for a population method, and capping those on steps
    instead handed PolyStep 55x to 500x the evaluations. The deadline is enforced by
    ``Objective.remaining``, so no method changes.
    """
    if match == "wallclock":
        return WALLCLOCK_EVAL_CAP, "wallclock"
    if match == "evals":
        return int(eval_budget), "evals"
    if method in SEQUENTIAL_METHODS or showcase in STEP_CAPPED_SHOWCASES:
        return step_matched_budget(method, polystep_steps, popsize), "steps"
    return int(eval_budget), "evals"


#: Set ``POLYSTEP_MATCH_AXIS`` to ``wallclock`` or ``evals`` to force an axis on a
#: showcase the default rule would cap on steps. Default "auto" keeps the eval/step rule. The axis is written into every result file,
#: so it is never inferred from the environment after the fact.
MATCH_AXIS = os.environ.get("POLYSTEP_MATCH_AXIS", "auto")


def matched_budget(
    method: str,
    *,
    showcase: str,
    seed: int,
    polystep_steps: int,
    eval_budget: int,
    results_dir: str,
    match: Optional[str] = None,
):
    """``(budget, match_axis, deadline_s)`` for one baseline arm.

    Wraps :func:`budget_for_method` and resolves the wall-clock deadline from
    PolyStep's own run on the same cell. Raises if that run is missing, rather than
    falling back to a guessed deadline.
    """
    match = MATCH_AXIS if match is None else match
    budget, axis = budget_for_method(method, eval_budget, polystep_steps, showcase=showcase, match=match)
    if axis != "wallclock":
        return budget, axis, None
    deadline = polystep_wall_seconds(showcase, seed, results_dir)
    if deadline is None:
        raise FileNotFoundError(f"wall-clock matching needs {showcase}_polystep_{seed}.json in {results_dir}")
    return budget, axis, deadline


def probe_scale_of(cfg: dict, default: float = 1.0, *, dim: Optional[int] = None) -> float:
    """PolyStep's probe radius as one *per-coordinate* number, for the baselines.

    A scheduled probe radius has no single value; the floor is the one the run spends
    most of its steps at, so that is what transfers.

    ``dim`` is the dimension the baseline samples in. PolyStep's radius is the norm of
    a displacement along one orthoplex vertex; a baseline's ``sigma`` is a
    per-coordinate standard deviation, and a Gaussian with per-coordinate ``sigma`` in
    dimension ``dim`` has norm ``sigma*sqrt(dim)``. Transferring the radius verbatim
    inflates every baseline's step by ``sqrt(dim)``, 35x on MNIST; passing ``dim``
    divides it out, so the two probes have the same displacement norm. Omitting ``dim``
    is wrong for any method that samples isotropically and exists only for callers with
    no dimension to hand.

    The radius is read through the optimizer's own rule
    (:func:`polystep.epsilon.radius_epsilon_factor`): a bare ``probe_radius`` float is a
    *multiplier on epsilon*, not a physical radius, so it must be scaled by the epsilon
    the run realizes before it can be compared with a baseline's sigma. Reading the raw
    key handed the baselines 2x PolyStep's displacement on every config with a scalar
    probe radius and a non-unit epsilon (snn, moe).

    That rule has to be read off the object the *optimizer* ends up holding, not off the
    config key. ``probe_radius_target`` is a plain float, so ``radius_epsilon_factor``
    called on it returns epsilon -- but ``build_polystep`` turns an
    ``_init``/``_target`` pair into a ``CosineEpsilon``, and both step drivers read a
    schedule as the physical radius with no epsilon factor at all. Applying the scalar
    rule to a scheduled endpoint therefore handed the baselines ``radius * epsilon``
    where PolyStep realizes ``radius``: 10x down on MNIST and timeseries, 3.33x on int8
    and argmax. :func:`_physical_radius` already decides this correctly and is the one
    place that should.
    """
    radius = _physical_radius(cfg, "probe_radius", flat_epsilon(cfg), default=default, prefer="target")
    return radius / math.sqrt(dim) if dim else radius


def flat_epsilon(cfg: dict, default: float = 1.0) -> float:
    """The epsilon a scalar radius is a multiplier on.

    A scheduled epsilon has no single value; the floor is where the run spends most of
    its steps, and it is what the tuned configs quote as ``epsilon_target``.
    """
    eps = cfg.get("epsilon_target", cfg.get("epsilon", default))
    return float(eps.at(0)) if hasattr(eps, "at") else float(eps)


def minibatch_loss(evaluator, loader, device) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """``{key: (N, *shape)} -> (N,)`` on one fresh minibatch per call.

    One call per generation means every candidate in a generation sees the same data,
    which is exactly what PolyStep's closure does. Cycles the loader forever.
    """
    it = [iter(loader)]

    def loss_batch(stacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            data, targets = next(it[0])
        except StopIteration:
            it[0] = iter(loader)
            data, targets = next(it[0])
        return evaluator.evaluate(stacked, data.to(device), targets.to(device))

    return loss_batch


#: EGGROLL's three tuned axes, as offsets from its centre config. ``popsize`` must stay
#: even: the sampler draws antithetic pairs.
_EGGROLL_CENTRE = {"lr": 0.01, "sigma": 1.0, "popsize": 32}
_EGGROLL_AXES = {"lr": (0.003, 0.03), "sigma": (0.25, 4.0), "popsize": (16, 64)}

#: The radius axis spans 16x. The best displacement norm is not
#: a property of the transfer rule -- it moves with the landscape. Measured at a fixed
#: 400k-evaluation budget with validation selection, EGGROLL on MNIST peaks at norm
#: <=0.5x the probe radius while OpenAI-ES on the SNN peaks at 4x, and at the SNN
#: optimum it scores 68.4% against 19.1% one grid step away. A 4x span reaches neither
#: from a common centre, which is how a baseline ends up looking collapsed when it is
#: only mis-scaled.
_RADIUS_MULT = (0.25, 1.0, 4.0)

#: One grid per method, all the same size, so tuning cost is comparable across the
#: table. ponytail: deliberately coarse -- the point is a *recorded, equal* tuning
#: budget, not a good one.
TUNING_GRID: Dict[str, List[dict]] = {
    "openai_es": [dict(lr=lr, sigma=s) for lr in (0.003, 0.01, 0.03) for s in _RADIUS_MULT],
    "spsa": [dict(a=a, c=c) for a in (0.03, 0.1, 0.3) for c in _RADIUS_MULT],
    "mezo": [dict(lr=lr, eps=e) for lr in (0.003, 0.01, 0.03) for e in _RADIUS_MULT],
    "random_search": [dict(sigma=s) for s in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)],
    # EGGROLL has three knobs the paper has to show were tuned -- sigma, learning rate
    # and population -- and only nine slots, so it uses the same one-factor-at-a-time
    # plus diagonals design as ``polystep`` below rather than a 3x3 over two of them.
    "eggroll": (
        [dict(_EGGROLL_CENTRE)]
        + [{**_EGGROLL_CENTRE, axis: v} for axis, vs in _EGGROLL_AXES.items() for v in vs]
        + [dict(lr=0.003, sigma=0.25, popsize=16), dict(lr=0.03, sigma=4.0, popsize=64)]
    ),
    "cma_es": [dict(sigma0=s, popsize=p) for s in _RADIUS_MULT for p in (16, 32, 64)],
}
#: Grid entries are multiplicative on ``fair_hyperparams``' probe scale for the radius
#: keys; see :func:`tuning_configs`.
RADIUS_KEYS = ("sigma", "sigma0", "c", "eps")
_RADIUS_KEYS = RADIUS_KEYS  # back-compat for callers of the private name

#: PolyStep's grid, the same size as every baseline's so tuning cost stays comparable.
#: Entries are *multipliers* on whatever config the runner already holds, applied by
#: :func:`apply_polystep_multipliers` to the flat key and to both ends of a schedule,
#: so a scheduled radius keeps its shape and only changes scale. The design is
#: one-factor-at-a-time around the transplanted prior plus the two diagonals: 1 centre
#: + 3 axes x 2 multipliers + 2 diagonals = 9.
_PS_AXES = ("epsilon", "step_radius", "probe_radius")
_PS_MULT = (0.3, 3.0)
TUNING_GRID["polystep"] = (
    [dict.fromkeys(_PS_AXES, 1.0)]
    + [{**dict.fromkeys(_PS_AXES, 1.0), axis: m} for axis in _PS_AXES for m in _PS_MULT]
    + [dict.fromkeys(_PS_AXES, m) for m in _PS_MULT]
)


#: Geometric spacing of each tuning axis, used to recentre the grid for round two.
_AXIS_RATIO = {
    "lr": 10.0**0.5,
    "a": 10.0**0.5,
    "sigma": 4.0,
    "sigma0": 4.0,
    "c": 4.0,
    "eps": 4.0,
    "popsize": 2.0,
    "epsilon": 3.0,
    "step_radius": 3.0,
    "probe_radius": 3.0,
}
#: Population sizes stay integers and stay affordable.
_INT_AXES = {"popsize"}


def refine_grid(method: str, winner: dict) -> List[dict]:
    """Round two: the same number of cells, recentred on round one's winner.

    Round one is a fixed grid, so its winner can land on an edge, which means the
    optimum is outside the swept range: on the SNN that happens for five of the six
    tuned baselines. Reporting such a sweep as "tuned" understates the baseline.

    Each axis is re-swept at ``(w/f, w, w*f)`` about the winner ``w``, with the
    spacing chosen by where the winner landed. On an edge of the round-one range the
    spacing is that axis's full ratio, which walks the grid outward into values never
    swept. Strictly inside it, the optimum is already bracketed and there is nothing
    to extend to, so the spacing is ``sqrt(ratio)`` and the round refines instead.
    Without that split an interior winner reproduces the round-one grid exactly and
    the second round measures nothing, which is the case that matters most for
    PolyStep: its grid is centred on the transplanted config and it won round one at
    the centre. Cell count is unchanged either way, so the tuning budget stays equal
    across the table at two rounds per method.
    """
    grid = TUNING_GRID[method]
    axes = sorted({k for point in grid for k in point})
    # Ladder length per axis, so the product matches round one's cell count where a
    # full product is the shape used. A one-knob method (random_search sweeps only
    # sigma) otherwise got a three-cell round two against everyone else's nine, and
    # then had nine quoted for it.
    rungs = max(3, int(round(len(grid) ** (1.0 / max(1, len(axes))))))
    per_axis = {}
    for axis in axes:
        w = winner[axis]
        swept = sorted({point[axis] for point in grid if axis in point})
        on_edge = len(swept) > 1 and w in (swept[0], swept[-1])
        ratio = _AXIS_RATIO.get(axis, 3.0)
        f = ratio if on_edge else math.sqrt(ratio)
        half = (rungs - 1) / 2.0
        vals = [w * f ** (i - half) for i in range(rungs)]
        if axis in _INT_AXES:
            # Round to even: eggroll rejects an odd popsize outright, so the four
            # cells an odd value produced raised and were swallowed by the sweep's
            # per-cell except, costing that method four of its nine round-two cells.
            vals = sorted({max(4, 2 * int(round(v / 2))) for v in vals})
        per_axis[axis] = vals

    # Match round one's shape: a full product where it is small enough, otherwise the
    # one-factor-at-a-time-plus-diagonals design the three-knob methods use.
    n_full = 1
    for vals in per_axis.values():
        n_full *= len(vals)
    if n_full <= len(grid):
        out: List[dict] = [{}]
        for axis, vals in per_axis.items():
            out = [{**base, axis: v} for base in out for v in vals]
        return out
    centre = {axis: winner[axis] for axis in axes}
    out = [dict(centre)]
    for axis, vals in per_axis.items():
        out += [{**centre, axis: v} for v in vals if v != centre[axis]]
    lo = {axis: per_axis[axis][0] for axis in axes}
    hi = {axis: per_axis[axis][-1] for axis in axes}
    out += [lo, hi]
    return out[: len(grid)]


def tuning_configs(
    method: str, *, probe_scale: float, seed: int, popsize: int = FAIR_POPSIZE, points: Optional[List[dict]] = None
) -> List[dict]:
    """The method's grid, with radius entries scaled by the shared probe radius.

    ``points`` overrides the round-one grid, so :func:`refine_grid` can supply the
    recentred round-two points without a second code path.
    """
    base = fair_hyperparams(method, probe_scale=probe_scale, seed=seed, popsize=popsize)
    out = []
    for point in points if points is not None else TUNING_GRID[method]:
        cfg = dict(base)
        for k, v in point.items():
            cfg[k] = v * probe_scale if k in _RADIUS_KEYS else v
        out.append(cfg)
    return out


def tuning_cost(
    method: str,
    evals_per_config: int,
    seeds: int = 1,
    rounds: int = 1,
    configs: Optional[int] = None,
) -> dict:
    """Configurations tried x cost per configuration, for the paper to quote.

    ``rounds`` counts the recentring passes of :func:`refine_grid`. Every method gets
    the same number of rounds at the same cell count, so this stays comparable across
    the table, which is the whole point of quoting it.

    ``configs`` is the number of cells that actually ran. Pass it. Deriving the count
    from ``TUNING_GRID`` quotes the grid we intended rather than the sweep we did, and
    a cell that raises is dropped by the sweep's per-cell ``except`` without changing
    the quoted number.
    """
    n = int(configs) if configs is not None else len(TUNING_GRID[method]) * int(rounds)
    return {
        "method": method,
        "configs": n,
        "rounds": int(rounds),
        "seeds_per_config": seeds,
        "evals_per_config": int(evals_per_config),
        "tuning_evals": n * seeds * int(evals_per_config),
    }


def apply_point(hyper: dict, point: dict, probe_scale: float) -> dict:
    """Overlay one grid point on a baseline's hyperparameters.

    Radius keys (:data:`RADIUS_KEYS`) are multipliers on ``probe_scale`` -- the *shared*
    per-coordinate probe scale, i.e. PolyStep's probe radius divided by
    ``sqrt(subspace_dim)``, because PolyStep's radius is a norm and a baseline's sigma
    is per-coordinate. Everything else (learning rate, population) is literal.
    """
    return {**hyper, **{k: (v * probe_scale if k in RADIUS_KEYS else v) for k, v in point.items()}}


def apply_polystep_multipliers(cfg: dict, point: dict) -> dict:
    """Scale a runner's PolyStep config by one :data:`TUNING_GRID` ``polystep`` point.

    Multiplicative, and applied to ``x``, ``x_init`` and ``x_target`` alike, so a
    cosine-scheduled radius keeps its shape and only moves scale.
    """
    out = dict(cfg)
    for axis, mult in point.items():
        for key in (axis, f"{axis}_init", f"{axis}_target"):
            if key in out:
                out[key] = out[key] * mult
    return out


# --- validation-only sweeps -------------------------------------------------------


class TestSplitTripwire:
    """Stands in for the test split during a hyperparameter sweep.

    A sweep that reads the test set invalidates the headline numbers it selects, so
    rather than trust a code review, the sweep is handed this and any use of it --
    iterating a DataLoader, unpacking an ``(x, y)`` split -- raises. Both go through
    ``__iter__``.
    """

    def __iter__(self):
        raise AssertionError("a hyperparameter sweep touched the test split")

    def __len__(self) -> int:
        return 0


#: Where the scale sweeps write their picks, and where the runners read them back.
DEFAULT_SELECTION_PATH = os.path.join("experiments", "results", "tuning", "selected_configs.json")


def select_best(trials: List[dict]) -> dict:
    """The winning trial: highest ``val``, ties broken by earliest grid index.

    Deterministic in both arguments -- validation floats tie often on small splits,
    and "first in :data:`TUNING_GRID` order" is a rule a reader can re-apply.
    """
    return max(trials, key=lambda t: (t["val"], -t["index"]))


def write_selection(
    experiment: str,
    trials: List[dict],
    provenance: dict,
    costs: Dict[str, dict],
    path: str = DEFAULT_SELECTION_PATH,
) -> str:
    """Reduce a sweep's trials to one config per (showcase, method) and write them.

    Args:
        experiment: Key in the selection file, e.g. ``"elevation"``.
        trials: ``{"showcase", "method", "index", "name", "point", "val"}`` per run.
            Only ``val`` on the *validation* split may appear.
        provenance: What the paper has to cite: which sweep, when, at what budget.
        costs: ``method -> tuning_cost(...)``.
        path: Selection file. Merged into, so the two experiments can sweep separately.

    Returns:
        The path written.
    """
    stamped = {**provenance, "written": datetime.now(timezone.utc).isoformat()}
    selected: Dict[str, Dict[str, dict]] = {}
    for showcase in sorted({t["showcase"] for t in trials}):
        for method in sorted({t["method"] for t in trials if t["showcase"] == showcase}):
            group = [t for t in trials if t["showcase"] == showcase and t["method"] == method]
            win = select_best(group)
            distinct = len({t["val"] for t in group})
            selected.setdefault(showcase, {})[method] = {
                "point": win["point"],
                "val": win["val"],
                "grid_index": win["index"],
                "name": win["name"],
                "configs_tried": len(group),
                "distinct_val_scores": distinct,
                # A sweep whose configs all score the same has not chosen anything: the
                # tie-break returned the grid's first entry, which is the untuned prior.
                # Say so here rather than let the paper quote it as a tuned config.
                "informative": distinct > 1,
                # Next to the pick, not once per experiment: the experiment-wide key
                # was overwritten by whichever sweep finished last, so a table could
                # quote one sweep's seed and epochs for another sweep's config.
                "provenance": dict(stamped),
            }

    # Merge per (showcase, method), not per experiment. One sweep per method is the
    # only way to pack a grid across workers, and replacing ``blob[experiment]``
    # wholesale drops every method swept before this one -- serially as well as
    # concurrently. The lock covers the read-modify-write so parallel tuners do not
    # interleave; without it four concurrent finishes left one method in the file.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            text = f.read()
            blob = json.loads(text) if text.strip() else {}
            exp = blob.setdefault(experiment, {})
            for showcase, methods in selected.items():
                exp.setdefault("selected", {}).setdefault(showcase, {}).update(methods)
            # Two levels, like ``selected``. Keyed by method alone, a second showcase
            # overwrote the first showcase's budget record, so a multi-showcase sweep
            # quoted the last showcase's cost for all of them.
            tc = exp.setdefault("tuning_cost", {})
            for showcase, per_method in costs.items():
                if isinstance(per_method, dict):
                    tc.setdefault(showcase, {}).update(per_method)
                else:  # a flat {method: cost} from a single-showcase sweep
                    tc[showcase] = per_method
            keys = {(t["showcase"], t["method"]) for t in trials}
            kept = [t for t in exp.get("trials", []) if (t["showcase"], t["method"]) not in keys]
            exp["trials"] = kept + trials
            exp["provenance"] = dict(stamped)
            f.seek(0)
            f.truncate()
            json.dump(blob, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return path


def load_selection(experiment: str, showcase: str, method: str, path: str = DEFAULT_SELECTION_PATH):
    """``(selected_entry, provenance)`` for one swept method, or ``(None, None)``.

    Missing file, missing experiment and missing method are all "not swept yet": the
    runners fall back to their transplanted defaults and say so in the result JSON.

    A file that exists but does not parse is a different case and raises. Swallowing
    it silently untunes the run that reads it, which is indistinguishable in the
    result JSON from never having swept, and a sweep writing this file while a table
    reads it can produce exactly that.
    """
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            # Shared lock: write_selection holds LOCK_EX across its read-modify-write,
            # so an unlocked read could catch the file mid-truncate and raise the
            # "does not parse" error below on a file that is perfectly fine.
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                blob = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ValueError as e:
        raise RuntimeError(
            f"{path} exists but does not parse ({e}). Refusing to fall back to the "
            "untuned config: that would report an untuned run as a tuned one. If a "
            "sweep is writing this file, wait for it or point --selection-path "
            "somewhere else."
        ) from e
    except OSError:
        return None, None
    exp = blob.get(experiment) or {}
    entry = ((exp.get("selected") or {}).get(showcase) or {}).get(method)
    if not entry:
        return None, None
    # No fallback to the experiment-wide provenance. That key is whatever sweep wrote
    # last, so falling back to it stamps one benchmark's sweep record onto another
    # benchmark's config -- observed live: SNN runs carrying {"showcases": ["mnist"]}
    # because the SNN baselines were swept before per-pick provenance existed and the
    # MNIST sweep then overwrote the shared key. That is the very failure the per-pick
    # nesting was introduced to prevent. A missing record is reported as missing.
    return entry, entry.get("provenance")


# --- theory mode ------------------------------------------------------------------

#: gamma in ``r_t = r_0 (t+1)^-(1/2+gamma)``.
THEORY_GAMMA = 0.1
#: Applied to BOTH radii: probe jitter spreads the probe cloud within its plane, step
#: jitter spreads the realised displacement. Setting only one leaves the reference
#: configuration incomplete.
THEORY_JITTER = 0.05


def _physical_radius(cfg: dict, name: str, epsilon: float, *, default: float, prefer: str) -> float:
    """The radius ``name`` realizes in parameter units, whichever key the config used.

    ``{name}_init``/``{name}_target`` present means the runner builds a schedule, and a
    schedule is the physical radius itself.  A bare ``{name}`` float is a multiplier on
    epsilon.  Same rule as :func:`polystep.epsilon.resolve_radius` inside the optimizer,
    read at the flat epsilon theory mode runs.

    ``prefer`` picks which end of a scheduled pair to transplant.  The step radius keeps
    decaying under theory mode, so it transplants its ``init``; the probe radius is held
    flat, so it transplants its ``target`` -- the floor, which is where the tuned run
    spends most of its steps, the same choice the flat epsilon above makes.
    """
    keys = (f"{name}_init", f"{name}_target") if prefer == "init" else (f"{name}_target", f"{name}_init")
    for key in keys:
        if key in cfg:
            return float(cfg[key])
    return resolve_radius(cfg.get(name, default), 0, epsilon)


def apply_theory_mode(cfg: dict) -> dict:
    """Rewrite a tuned PolyStep config into the unaccelerated reference one.

    Exactly: jitter ``0.05`` on the probe AND step radii with the smooth mollifier
    density, independently sampled
    rotations (no biased rotation), a flat epsilon, the decaying step radius
    ``r_0 (t+1)^-(1/2+gamma)``, an orthoplex, and none of the acceleration -- no
    momentum, no amortized OT, no Anderson. The subspace stays ``HybridSubspace``;
    the runner builds it.

    Reporting the gap between this and the tuned config is the point, so this returns
    a new dict and leaves ``cfg`` alone.

    Args:
        cfg: A runner's PolyStep config, in that runner's ``*_init``/``*_target`` form.

    Returns:
        A config with the scheduled keys removed and flat/theory values in their place.
    """
    out = dict(cfg)
    for key in (
        "epsilon_init",
        "epsilon_target",
        "step_radius_init",
        "step_radius_target",
        "probe_radius_init",
        "probe_radius_target",
        "amortize_ema",
        "momentum_init",
        "momentum_final",
    ):
        out.pop(key, None)

    # Flat epsilon at the tuned schedule's floor: the analysis fixes epsilon, and the
    # floor is the value the tuned run spends most of its steps at.
    eps = float(cfg.get("epsilon", cfg.get("epsilon_target", 0.5)))
    out["epsilon"] = eps

    # The radii transplant in PHYSICAL units, not in whichever key the source config
    # happened to use.  A bare ``step_radius``/``probe_radius`` float is a multiplier on
    # epsilon; a ``*_init``/``*_target`` pair becomes a schedule, which is the physical
    # radius itself (polystep.epsilon.resolve_radius).  Copying the raw key across the
    # two parameterizations rescaled the realized radius by epsilon or by 1/epsilon --
    # 10x down on the MNIST probe radius, 2x up on the SNN step radius -- so the
    # reported gap between the tuned and the analysed configuration measured an
    # unintended radius change on top of the analysed restrictions.
    r_s_phys = _physical_radius(cfg, "step_radius", eps, default=1.0, prefer="init")
    r_p_phys = _physical_radius(cfg, "probe_radius", eps, default=1.0, prefer="target")

    # PowerDecay exposes ``.at()``, so it is read as a physical schedule: init is the
    # physical radius at t = 0.
    out["step_radius"] = PowerDecay(init=r_s_phys, gamma=THEORY_GAMMA)
    # Theory mode wants a *flat* probe radius and no schedule class expresses that, so
    # it stays a scalar -- which means a multiplier, and the physical value has to be
    # divided back out.
    out["probe_radius"] = r_p_phys / eps
    out["probe_radius_realized"] = r_p_phys
    out["step_radius_realized_t0"] = r_s_phys
    out["probe_radius_jitter"] = THEORY_JITTER
    out["probe_radius_jitter_dist"] = "smooth"
    out["step_radius_jitter"] = THEORY_JITTER
    out["polytope_type"] = "orthoplex"
    out["biased_rotation"] = False
    out["use_momentum"] = False
    out["amortize_steps"] = 1
    out["anderson_depth"] = 0
    out["theory_mode"] = True

    # A schedule and a bare float are read differently; check the physical radius.
    assert abs(resolve_radius(out["step_radius"], 0, eps) - r_s_phys) < 1e-9 * max(1.0, r_s_phys)
    assert abs(resolve_radius(out["probe_radius"], 0, eps) - r_p_phys) < 1e-9 * max(1.0, r_p_phys)
    return out


# --- the runner -------------------------------------------------------------------


def _load(model, stacked: Dict[str, torch.Tensor]) -> None:
    """Write the single candidate in ``{key: (1, *shape)}`` into ``model``."""
    model.load_state_dict({k: v[0] for k, v in stacked.items()}, strict=False)


def _log_schedule(budget: int, log_points: int, first: int = 1) -> List[int]:
    """``log_points`` evaluation counts, geometrically spaced over ``[first, budget]``.

    Precomputed rather than stepped, so the point count is exactly ``log_points``. A
    running ratio recomputed from the current position never reaches the budget in
    ``n`` samples and keeps emitting points -- and every point costs two full validation
    passes, so an unbounded schedule is not a cosmetic problem.
    """
    n = max(int(log_points), 1)
    if n < 2 or budget <= first:
        return [budget]
    ratio = (budget / first) ** (1.0 / (n - 1))
    pts, prev = [], 0
    for i in range(n):
        v = max(prev + 1, int(round(first * ratio**i)))
        pts.append(min(v, budget))
        prev = pts[-1]
        if prev >= budget:
            break
    return pts


def run_baseline(
    method: str,
    *,
    model,
    layout,
    loss_batch: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    budget: int,
    val_fn: Callable[[object], float],
    test_fn: Callable[[object], float],
    mode: str = "max",
    seed: int = 0,
    subspace=None,
    subspace_rank: Optional[int] = None,
    shared_subspace_dim: Optional[int] = None,
    hp: Optional[dict] = None,
    probe_scale: float = 1.0,
    log_points: int = 25,
    quality_key: str = "accuracy",
    deadline_s: Optional[float] = None,
) -> dict:
    """Run one :mod:`polystep.baselines` method and report it like a runner does.

    Selection is on ``val_fn``; ``test_fn`` is called for the trajectory and once more
    at the end on the selected iterate, which is the protocol every runner now follows.

    Args:
        method: A key of :data:`polystep.baselines.METHODS`.
        model: The module. Overwritten in place while probing; the search itself runs
            against a cloned base state dict, so this does not perturb it.
        layout: :class:`~polystep.transform.ParamLayout` for ``model``.
        loss_batch: ``{key: (N, *shape)} -> (N,)``. Called once per generation, so it
            should draw one minibatch per call -- that gives every candidate in a
            generation the same data, as PolyStep's closure does.
        budget: Candidate evaluations. The objective refuses to exceed it.
        val_fn: ``model -> quality`` on the validation split. Drives selection.
        test_fn: ``model -> quality`` on the test split. Never drives selection.
        mode: ``"max"`` (accuracy) or ``"min"`` (MSE).
        seed: Method seed.
        subspace: Search in this subspace's coordinates; ``None`` searches full space.
        subspace_rank: Recorded as ``subspace_rank``; informational.
        hp: Hyperparameter overrides on :func:`fair_hyperparams`.
        probe_scale: Shared probe radius the defaults are built from.
        log_points: Trajectory points, spread geometrically over the budget so the
            early steps -- where a 2-candidate method does all of its optimizing
            relative to PolyStep's step count -- are resolved.
        quality_key: Name for the recorded quality, e.g. ``"accuracy"`` or ``"mse"``.
        deadline_s: Wall-clock seconds the method may spend, or ``None``. Used by the
            wall-clock-matched arms; the objective stops the run when it passes.

    Returns:
        ``{"metrics", "hyperparameters", "epoch_logs", "step_logs"}``, ready to splat
        into :func:`experiments.runners.common.save_result`. ``step_logs`` is the
        accuracy-vs-cumulative-evaluations trajectory.
    """
    if method not in METHODS:
        raise ValueError(f"unknown baseline {method!r}; have {sorted(METHODS)}")
    better = (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)

    device = next(model.parameters()).device
    sd = model.state_dict()
    base_sd = {e.key: sd[e.key].detach().clone() for e in layout.entries}

    trajectory: List[dict] = []
    best_val = -float("inf") if mode == "max" else float("inf")
    best_coords: Optional[torch.Tensor] = None
    start = time.time()
    # Geometric, not even: the matched-steps axis reads each baseline at PolyStep's own
    # step count (~3,180), and even spacing over a multi-million-generation budget puts
    # its first sample far past that. Same 25 points, a third of them below step 3,000.
    log_at = _log_schedule(budget, log_points)
    next_log = [0]
    # Wall-clock-matched arms log on the clock, evenly, since evals are not the budget.
    time_at = [deadline_s * (i + 1) / log_points for i in range(log_points)] if deadline_s is not None else []
    next_time = [0.0]

    if subspace is not None:
        projections = subspace.init_projections(device, torch.float32)

        def write(coords: torch.Tensor) -> None:
            _load(model, subspace.reconstruct_batch(projections, base_sd, coords.unsqueeze(0)))
    else:

        def write(coords: torch.Tensor) -> None:
            _load(model, layout.batch_unflatten(coords.unsqueeze(0)))

    def probe(obj: Objective) -> None:
        """Score the method on validation, at both points it could reasonably mean.

        ``obj.best_x`` is the lowest-training-loss *candidate*; ``obj.iterate`` is the
        method's own estimate. For a population method these differ by ``sigma*eps``,
        whose norm is about one probe radius, so scoring only ``best_x`` reported every
        population baseline below the point it had actually found -- while PolyStep was
        scored at its true barycentre. Both are scored and validation picks between
        them, which is the same selection rule PolyStep gets.
        """
        nonlocal best_val, best_coords
        candidates = [("iterate", obj.iterate), ("best_x", obj.best_x)]
        scored = []
        for label, coords in candidates:
            if coords is None:
                continue
            write(coords)
            scored.append((val_fn(model), label, coords))
        if not scored:
            return
        # key=, not bare max: a tie would otherwise fall through to comparing the
        # tensors in the tuple, which raises.
        pick = max if mode == "max" else min
        v, label, coords = pick(scored, key=lambda s: s[0])
        write(coords)
        trajectory.append(
            {
                "evals": obj.evals,
                # The generation index, the axis the headline tables use. It cannot be
                # recovered from ``evals``: CMA-ES doubles its population on every IPOP
                # restart, so evals/popsize is off by whatever the restart schedule did.
                "step": generations[0],
                f"val_{quality_key}": v,
                # Best-so-far on validation, which is what the honest protocol selects.
                # The value at exactly step N is the *final* iterate at N, and for a
                # method whose validation curve turns over late -- PolyStep on MNIST ends
                # at 0.8867 having peaked at 0.9652 -- reading that instead of the
                # selected checkpoint understates it by eight points.
                f"best_val_{quality_key}": v
                if not trajectory
                else (pick(v, trajectory[-1][f"best_val_{quality_key}"])),
                # Which point validation preferred, so the choice is auditable rather
                # than an unstated convention.
                "scored_at": label,
                "loss": obj.best_loss,
                "wall_time": time.time() - start,
            }
        )
        if better(v, best_val):
            best_val, best_coords = v, coords.detach().clone()

    generations = [0]

    def counted(stacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Logging happens on entry, so it sees the previous generation's best; a
        # generation's own result lands on the next point. One point of lag on a
        # 25-point curve, and it keeps the hook out of Objective's accounting.
        if deadline_s is not None:
            # Evaluations are not the budget here, so log on the clock instead.
            if objective.elapsed_s >= next_time[0]:
                probe(objective)
                while time_at and time_at[0] <= objective.elapsed_s:
                    time_at.pop(0)
                next_time[0] = time_at[0] if time_at else deadline_s + 1.0
                print(
                    f"      {100.0 * objective.elapsed_s / deadline_s:5.1f}% of "
                    f"{deadline_s:.0f}s  evals={objective.evals:,}",
                    flush=True,
                )
            generations[0] += 1
            return loss_batch(stacked)
        if objective.evals >= next_log[0]:
            probe(objective)
            while log_at and log_at[0] <= objective.evals:
                log_at.pop(0)
            next_log[0] = log_at[0] if log_at else budget + 1
            # Print as well as record. The trajectory points go into the result file,
            # but a population baseline that logs nothing to stdout is indistinguishable
            # from a wedged one for the hour it runs -- the MNIST OpenAI-ES arm went 60
            # minutes without a line while holding 866 MiB of GPU. PYTHONUNBUFFERED
            # cannot help when nothing is written.
            pct = 100.0 * objective.evals / max(budget, 1)
            head = f"      {pct:5.1f}%  evals={objective.evals:,}/{budget:,}"
            q = trajectory[-1].get(quality_key) if trajectory else None
            print(head if q is None else f"{head}  {quality_key}={q:.4f}", flush=True)
        # One call == one generation, for every method: run_baseline's contract is that
        # loss_batch is "called once per generation". Incremented *after* the logging
        # block, because that block records the previous generation's result -- counting
        # first would label every point one generation ahead of the value it carries.
        generations[0] += 1
        return loss_batch(stacked)

    if subspace is not None:
        objective = Objective.from_subspace(subspace, base_sd, counted, budget, device=device, deadline_s=deadline_s)
        # The methods default x0 to CPU zeros; hand them the origin on the model's
        # device instead so every subsequent tensor they allocate lands there too.
        x0 = torch.zeros(subspace.subspace_dim, device=device)
    else:
        objective = Objective.from_layout(
            layout, lambda X: counted(layout.batch_unflatten(X)), budget, deadline_s=deadline_s
        )
        x0 = layout.flatten(model).reshape(-1)[: layout.total_params].to(torch.float32)

    kwargs = fair_hyperparams(method, probe_scale=probe_scale, seed=seed, popsize=FAIR_POPSIZE)
    kwargs.update(hp or {})
    # Sweep a knob without a runner flag, e.g. POLYSTEP_HP_OVERRIDE='{"popsize":2048}'.
    # Lands in the saved hyperparameters like any other setting.
    override = os.environ.get("POLYSTEP_HP_OVERRIDE")
    if override:
        kwargs.update(json.loads(override))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = METHODS[method](objective, x0=x0, **kwargs)
    probe(objective)

    # Test, once, on the checkpoint val selected. Falls back to the objective's best
    # candidate when the budget was too small for a single trajectory point.
    write(best_coords if best_coords is not None else result.best_x)
    final = test_fn(model)
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

    # NaN, not 0.0, where accuracy is not the metric. A 0.0 survives the table
    # generator's `is not None` filter and prints as a measured "0.0 +- 0.0" for every
    # baseline on a regression benchmark, which is indistinguishable from a method
    # that ran and scored nothing. PolyStep's own runner already writes NaN here.
    _na = float("nan")
    metrics = {
        "final_accuracy": final if mode == "max" else _na,
        "best_accuracy": best_val if (mode == "max" and trajectory) else _na,
        "test_accuracy_at_selected": final if mode == "max" else _na,
        f"final_{quality_key}": final,
        f"best_val_{quality_key}": best_val if trajectory else float("nan"),
        "wall_time_seconds": time.time() - start,
        "peak_gpu_memory_mb": peak_mb,
        "function_evals": result.evals,
        "total_steps": result.iters,
    }
    hyperparameters = {
        **kwargs,
        **subspace_tag(subspace, subspace_rank, shared_dim=shared_subspace_dim),
        "eval_budget": int(budget),
        "evals_used": int(result.evals),
        "probe_scale": probe_scale,
        "fair": True,
        "deadline_s": deadline_s,
    }
    return {
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "epoch_logs": trajectory,
        "step_logs": trajectory,
    }
