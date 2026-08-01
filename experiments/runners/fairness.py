#!/usr/bin/env python
"""Fairness controls for the baseline tables.

Three reviewer complaints, three knobs, one module:

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

import json
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import torch

from polystep.baselines import METHODS, Objective
from polystep.epsilon import PowerDecay
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
    "fair_hyperparams",
    "load_selection",
    "make_subspace",
    "minibatch_loss",
    "polystep_eval_budget",
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
FAIR_POPSIZE = 32


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
    """The shared candidate budget: what PolyStep spends over ``total_steps`` steps."""
    return evals_per_step(opt) * int(total_steps)


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

    ``probe_scale`` is PolyStep's probe radius. Every method that has a notion of
    "how far from the current iterate do I sample" gets that same number: ``sigma``
    for the ES family, ``c`` for SPSA, ``eps`` for MeZO, ``sigma0`` for CMA-ES. Step
    sizes stay at each method's own published default, since they are not the same
    quantity.
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


def probe_scale_of(cfg: dict, default: float = 1.0) -> float:
    """PolyStep's probe radius as one number, for the baselines to share.

    A scheduled probe radius has no single value; the floor is the one the run spends
    most of its steps at, so that is what transfers.
    """
    return float(cfg.get("probe_radius", cfg.get("probe_radius_target", default)))


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
_EGGROLL_AXES = {"lr": (0.003, 0.03), "sigma": (0.5, 2.0), "popsize": (16, 64)}

#: One grid per method, all the same size, so tuning cost is comparable across the
#: table. ponytail: deliberately coarse -- the point is a *recorded, equal* tuning
#: budget, not a good one. Widen only if the paper claims a tuned baseline.
TUNING_GRID: Dict[str, List[dict]] = {
    "openai_es": [dict(lr=lr, sigma=s) for lr in (0.003, 0.01, 0.03) for s in (0.5, 1.0, 2.0)],
    "spsa": [dict(a=a, c=c) for a in (0.03, 0.1, 0.3) for c in (0.5, 1.0, 2.0)],
    "mezo": [dict(lr=lr, eps=e) for lr in (0.003, 0.01, 0.03) for e in (0.5, 1.0, 2.0)],
    "random_search": [dict(sigma=s) for s in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)],
    # EGGROLL has three knobs the paper has to show were tuned -- sigma, learning rate
    # and population -- and only nine slots, so it uses the same one-factor-at-a-time
    # plus diagonals design as ``polystep`` below rather than a 3x3 over two of them.
    "eggroll": (
        [dict(_EGGROLL_CENTRE)]
        + [{**_EGGROLL_CENTRE, axis: v} for axis, vs in _EGGROLL_AXES.items() for v in vs]
        + [dict(lr=0.003, sigma=0.5, popsize=16), dict(lr=0.03, sigma=2.0, popsize=64)]
    ),
    "cma_es": [dict(sigma0=s, popsize=p) for s in (0.25, 0.5, 1.0) for p in (16, 32, 64)],
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


def tuning_configs(method: str, *, probe_scale: float, seed: int, popsize: int = FAIR_POPSIZE) -> List[dict]:
    """The method's grid, with radius entries scaled by the shared probe radius."""
    base = fair_hyperparams(method, probe_scale=probe_scale, seed=seed, popsize=popsize)
    out = []
    for point in TUNING_GRID[method]:
        cfg = dict(base)
        for k, v in point.items():
            cfg[k] = v * probe_scale if k in _RADIUS_KEYS else v
        out.append(cfg)
    return out


def tuning_cost(method: str, evals_per_config: int, seeds: int = 1) -> dict:
    """Configurations tried x cost per configuration, for the paper to quote."""
    n = len(TUNING_GRID[method])
    return {
        "method": method,
        "configs": n,
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
        experiment: Key in the selection file, e.g. ``"cifar"``.
        trials: ``{"showcase", "method", "index", "name", "point", "val"}`` per run.
            Only ``val`` on the *validation* split may appear.
        provenance: What the paper has to cite: which sweep, when, at what budget.
        costs: ``method -> tuning_cost(...)``.
        path: Selection file. Merged into, so the two experiments can sweep separately.

    Returns:
        The path written.
    """
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)

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
            }

    blob[experiment] = {
        "provenance": {**provenance, "written": datetime.now(timezone.utc).isoformat()},
        "tuning_cost": costs,
        "selected": selected,
        "trials": trials,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return path


def load_selection(experiment: str, showcase: str, method: str, path: str = DEFAULT_SELECTION_PATH):
    """``(selected_entry, provenance)`` for one swept method, or ``(None, None)``.

    Missing file, missing experiment and missing method are all "not swept yet": the
    runners fall back to their transplanted defaults and say so in the result JSON.
    """
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return None, None
    exp = blob.get(experiment) or {}
    entry = ((exp.get("selected") or {}).get(showcase) or {}).get(method)
    if not entry:
        return None, None
    return entry, exp.get("provenance")


# --- theory mode ------------------------------------------------------------------

#: gamma in ``r_t = r_0 (t+1)^-(1/2+gamma)``.
THEORY_GAMMA = 0.1
#: The jitter the convergence analysis needs; 0 breaks the transversality argument.
#: Applied to BOTH radii. Probe jitter makes the probe cloud absolutely continuous
#: within its plane (Lemma "smooth kernel"); step jitter makes the realised
#: displacement diffuse, which is what Lemma "blockwise section" uses to show the
#: iterate never lands on the discontinuity set. Setting only the first leaves the
#: analysed configuration missing a hypothesis it is supposed to satisfy.
THEORY_JITTER = 0.05


def apply_theory_mode(cfg: dict) -> dict:
    """Rewrite a tuned PolyStep config into the one Theorem 4.2 analyses.

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
    out["epsilon"] = cfg.get("epsilon", cfg.get("epsilon_target", 0.5))
    r0 = cfg.get("step_radius_init", cfg.get("step_radius", 1.0))
    out["step_radius"] = PowerDecay(init=float(r0), gamma=THEORY_GAMMA)
    out["probe_radius"] = cfg.get("probe_radius", cfg.get("probe_radius_target", 1.0))
    out["probe_radius_jitter"] = THEORY_JITTER
    out["probe_radius_jitter_dist"] = "smooth"
    out["step_radius_jitter"] = THEORY_JITTER
    out["polytope_type"] = "orthoplex"
    out["biased_rotation"] = False
    out["use_momentum"] = False
    out["amortize_steps"] = 1
    out["anderson_depth"] = 0
    out["theory_mode"] = True
    return out


# --- the runner -------------------------------------------------------------------


def _load(model, stacked: Dict[str, torch.Tensor]) -> None:
    """Write the single candidate in ``{key: (1, *shape)}`` into ``model``."""
    model.load_state_dict({k: v[0] for k, v in stacked.items()}, strict=False)


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
    hp: Optional[dict] = None,
    probe_scale: float = 1.0,
    log_points: int = 25,
    quality_key: str = "accuracy",
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
        log_points: Trajectory points, spread evenly over the budget.
        quality_key: Name for the recorded quality, e.g. ``"accuracy"`` or ``"mse"``.

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
    next_log = [0]

    if subspace is not None:
        projections = subspace.init_projections(device, torch.float32)

        def write(coords: torch.Tensor) -> None:
            _load(model, subspace.reconstruct_batch(projections, base_sd, coords.unsqueeze(0)))
    else:

        def write(coords: torch.Tensor) -> None:
            _load(model, layout.batch_unflatten(coords.unsqueeze(0)))

    def probe(obj: Objective) -> None:
        """Score the best-so-far iterate on val (selection) and test (the figure)."""
        nonlocal best_val, best_coords
        if obj.best_x is None:
            return
        write(obj.best_x)
        v, t = val_fn(model), test_fn(model)
        trajectory.append(
            {
                "evals": obj.evals,
                f"val_{quality_key}": v,
                f"test_{quality_key}": t,
                "loss": obj.best_loss,
                "wall_time": time.time() - start,
            }
        )
        if better(v, best_val):
            best_val, best_coords = v, obj.best_x.detach().clone()

    def counted(stacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Logging happens on entry, so it sees the previous generation's best; a
        # generation's own result lands on the next point. One point of lag on a
        # 25-point curve, and it keeps the hook out of Objective's accounting.
        if objective.evals >= next_log[0]:
            probe(objective)
            next_log[0] = objective.evals + max(1, budget // max(log_points, 1))
        return loss_batch(stacked)

    if subspace is not None:
        objective = Objective.from_subspace(subspace, base_sd, counted, budget, device=device)
        # The methods default x0 to CPU zeros; hand them the origin on the model's
        # device instead so every subsequent tensor they allocate lands there too.
        x0 = torch.zeros(subspace.subspace_dim, device=device)
    else:
        objective = Objective.from_layout(layout, lambda X: counted(layout.batch_unflatten(X)), budget)
        x0 = layout.flatten(model).reshape(-1)[: layout.total_params].to(torch.float32)

    kwargs = fair_hyperparams(method, probe_scale=probe_scale, seed=seed, popsize=FAIR_POPSIZE)
    kwargs.update(hp or {})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = METHODS[method](objective, x0=x0, **kwargs)
    probe(objective)

    # Test, once, on the checkpoint val selected. Falls back to the objective's best
    # candidate when the budget was too small for a single trajectory point.
    write(best_coords if best_coords is not None else result.best_x)
    final = test_fn(model)
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

    metrics = {
        "final_accuracy": final if mode == "max" else 0.0,
        "best_accuracy": best_val if (mode == "max" and trajectory) else 0.0,
        "test_accuracy_at_selected": final if mode == "max" else 0.0,
        f"final_{quality_key}": final,
        f"best_val_{quality_key}": best_val if trajectory else float("nan"),
        "wall_time_seconds": time.time() - start,
        "peak_gpu_memory_mb": peak_mb,
        "function_evals": result.evals,
        "total_steps": result.iters,
    }
    hyperparameters = {
        **kwargs,
        **subspace_tag(subspace, subspace_rank),
        "eval_budget": int(budget),
        "evals_used": int(result.evals),
        "probe_scale": probe_scale,
        "fair": True,
    }
    return {
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "epoch_logs": trajectory,
        "step_logs": trajectory,
    }
