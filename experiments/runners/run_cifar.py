#!/usr/bin/env python
"""CIFAR-10 at 5.6-6.7M parameters with a genuinely discontinuous forward pass.

Answers the "MNIST-centric, capped at 2.4M parameters" objection without
changing what is being demonstrated: every showcase here still has a forward
pass that jumps, and the comparison is still "black-box cost oracle vs methods
that need a gradient, real or estimated".

Showcases (see ``cifar_models.py``)::

    lif     hard-threshold spiking net, 8 timesteps      5,625,994 params
    int8    int8 round() weights, no STE                 5,625,994 params
    moe     argmax top-1 routing over 8 experts          6,680,210 params

Methods::

    polystep        PolyStep over a HybridSubspace
    eggroll         arXiv:2511.16652, over a FactoredSubspace (see below)
    openai_es       arXiv:1703.03864, same HybridSubspace as PolyStep
    cma_es          pycma, separable, same HybridSubspace
    random_search   accept-if-better, same HybridSubspace: the control that says
                    how much of the result is the subspace and not the update rule
    adam            Adam on the smooth twin: an accuracy ceiling, NOT budget-matched

Budget and fairness
-------------------
Every gradient-free method is charged in *candidate evaluations* --
``--eval-budget`` of them, counted by :class:`polystep.baselines.Objective`, and
PolyStep stops on the same counter. All of them search the same
:class:`~polystep.hybrid_subspace.HybridSubspace` at the same rank, with probe
radii derived from one ``--probe-radius`` so nobody is handed a different step
scale by accident.

EGGROLL is the documented exception. Its perturbation is ``A B^T`` per weight
*matrix*, and HybridSubspace coordinates are an unstructured projection with no
matrix to factor, so running it there would silently degrade it to dense
Gaussian ES. It gets :class:`~polystep.factored_subspace.FactoredSubspace`
instead, whose coordinates *are* the per-layer ``A`` factors, at the same rank,
so each layer's coordinates form a ``(d_out, rank)`` matrix its sampler can be
low-rank within. That leaves the two subspaces at different dimensions
(int8: 4,096 shared vs 12,530 factored), which every result records as
``subspace_class`` / ``subspace_dim`` rather than papering over.

Adam is the other exception: it uses gradients, so an evaluation budget does not
apply to it. Its results carry ``eval_budget: null`` and must not be plotted on
the same x-axis.

What the subspace cap costs
---------------------------
``--max-subspace-dim`` trades reachability for optimizer steps and the default
(4096) is on the aggressive end. At that cap the 4096x1024 layer gets ~780
coordinates from a sparse random projection, which touches roughly a third of
its 4.2M weights; the rest cannot move while that basis stands, and the runner
prints a warning saying so. Uncapped rank-4 reaches everything but costs 44k
evaluations per PolyStep step, i.e. 11 steps for the whole budget. Every
gradient-free method shares the cap, so the comparison is fair either way, but
"5.6M parameters" here means 5.6M parameters *in the model*, not 5.6M
independently searched directions -- say that in the paper.

Protocol: train on the train split, select the reported checkpoint on a held-out
10% validation slice, score the test set exactly once on that checkpoint. The
headline number is ``metrics.test_accuracy_at_selected``.

Usage::

    python experiments/runners/run_cifar.py --smoke
    python experiments/runners/run_cifar.py --showcases int8 --methods polystep eggroll
    python experiments/runners/run_cifar.py --seeds 42 123 456 789 1337
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.func import functional_call, vmap  # noqa: E402

from experiments.runners.common import (  # noqa: E402
    SEEDS,
    evaluate_accuracy,
    load_cifar10,
    make_train_val_split,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.cifar_models import (  # noqa: E402
    HardMoECIFARNet,
    QuantizedCIFARNet,
    SpikingCIFARNet,
)
from experiments.runners.fairness import (  # noqa: E402
    DEFAULT_SELECTION_PATH,
    TUNING_GRID,
    TestSplitTripwire,
    apply_point,
    apply_polystep_multipliers,
    load_selection,
    subspace_tag,
    tuning_cost,
    write_selection,
)
from polystep.baselines import METHODS, Objective  # noqa: E402
from polystep.factored_subspace import FactoredSubspace  # noqa: E402
from polystep.hybrid_subspace import HybridSubspace  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402


SHOWCASES: Dict[str, Dict] = {
    "cifar_lif": {"model_fn": SpikingCIFARNet, "discontinuity": "hard LIF threshold"},
    "cifar_int8": {"model_fn": QuantizedCIFARNet, "discontinuity": "int8 round(), no STE"},
    "cifar_moe": {"model_fn": HardMoECIFARNet, "discontinuity": "argmax top-1 routing"},
}

#: Candidate evaluations every gradient-free method gets. See ``--eval-budget``.
#: One evaluation is one forward pass of the whole net on one minibatch, and the
#: measured throughput is ~800/s, so this is ~10 min of GPU per gradient-free run.
EVAL_BUDGET = 500_000
BATCH_SIZE = 128
SUBSPACE_RANK = 4
#: PolyStep spends ~1.1 evaluations per subspace coordinate per step, so the
#: subspace dimension *is* the step cost. Uncapped rank-4 gives 44k coordinates
#: and 44k evaluations per step, i.e. ~11 steps for the whole budget. The cap is
#: what makes the budget buy optimizer steps instead of one enormous one.
MAX_SUBSPACE_DIM = 4096
PROBE_RADIUS = 1.0  # norm of a probe in subspace coordinates; per-coord sigma is derived
CHUNK = 32  # candidates materialized at once; the cap on peak VRAM
#: Epoch cap only. At these budgets the evaluation budget always binds first
#: (500k evals is well under one epoch of PolyStep steps).
EPOCHS = 1000
ADAM_EPOCHS = 30
ADAM_LR = 1e-3
PROBE_EVERY = 10_000  # evaluations between validation probes (the trajectory resolution)

# Untuned starting point, transplanted from the int8 showcase in run_elevation.py.
# Coarse-to-fine on all three radii; the sweep that earned these numbers was run at
# MNIST scale, so treat them as a prior, not a result.
POLYSTEP_CONFIG = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.3,
    "step_radius_init": 32.0,
    "step_radius_target": 8.0,
    "probe_radius_init": 2.0,
    "probe_radius_target": 0.5,
    "num_probe": 1,
    # 1024 (the MNIST-scale value) materializes 1024 x 5.6M floats = 23 GB and OOMs a
    # 32 GB card. This is the same VRAM knob as ``--chunk`` for the baselines.
    "chunk_size": 16,
    "amortize_steps": 1,
    "rotation_interval": 0,
    "absorb_interval": 0,
    "use_momentum": True,
    "momentum_init": 0.3,
    "momentum_final": 0.5,
}


# --- shared plumbing --------------------------------------------------------


def _loaders(seed: int, batch_size: int, max_batches: int = 0, hide_test: bool = False):
    """``(train, val, test)``. Val is a seeded 10% slice of train; test is untouched.

    ``hide_test`` swaps the test loader for a :class:`TestSplitTripwire`, which is how
    ``--tune`` guarantees rather than promises that a hyperparameter sweep never reads
    the test set: any attempt to iterate it raises.
    """
    train_loader, test_loader = load_cifar10(batch_size=batch_size)
    train_loader, val_loader = make_train_val_split(train_loader, val_frac=0.1, seed=seed)
    if max_batches:
        train_loader = _truncate(train_loader, max_batches)
        val_loader = _truncate(val_loader, max_batches)
        test_loader = _truncate(test_loader, max_batches)
    return train_loader, val_loader, (TestSplitTripwire() if hide_test else test_loader)


def _truncate(loader, n_batches: int):
    """First ``n_batches * batch_size`` samples, for smoke runs."""
    n = min(len(loader.dataset), n_batches * (loader.batch_size or 1))
    subset = torch.utils.data.Subset(loader.dataset, range(n))
    return torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=False)


def _cycle(loader):
    """Endless minibatch stream: one fresh batch per generation, as the contract wants."""
    while True:
        for batch in loader:
            yield batch


def _batched_loss(model, params: Dict[str, torch.Tensor], x, y) -> torch.Tensor:
    """Cross-entropy for a stack of parameter sets: ``{key: (N, *shape)} -> (N,)``."""

    def one(p):
        return F.cross_entropy(functional_call(model, p, (x,)), y)

    return vmap(one)(params)


def _param_dict(layout, model) -> Dict[str, torch.Tensor]:
    return {e.key: p.detach().clone() for e, p in zip(layout.entries, model.parameters())}


def _load_coords(model, subspace, projections, base_sd, coords: torch.Tensor) -> None:
    """Write one coordinate vector back into ``model`` so it can be evaluated normally."""
    model.load_state_dict(subspace.apply_perturbation(projections, base_sd, coords), strict=False)


def build_subspace(model, method: str, rank: int, max_dim: int, eggroll_rank: Optional[int] = None):
    """``(layout, subspace, shared_dim)`` -- the subspace a method searches.

    Everything shares one HybridSubspace except EGGROLL, which needs per-layer
    matrix structure in the coordinates (module docstring). Its FactoredSubspace
    keeps the same rank, which makes each layer's coordinates a ``(d_out, rank)``
    matrix that EGGROLL's rank-1 ``A B^T`` sampler can actually be low-rank
    *within*. Shrinking the rank to 1 to match the shared subspace's dimension
    would look fairer and be worse: a ``(d_out, 1)`` coordinate matrix is already
    rank 1, so the sampler would have nothing to factor and EGGROLL would
    silently become dense Gaussian ES -- the exact failure this whole
    arrangement exists to avoid.

    So: **rank is matched, dimension is not, and the gap is reported.**
    ``shared_dim`` is what every non-EGGROLL method searched, and every result
    records it next to EGGROLL's own ``subspace_dim`` plus a
    ``subspace_dim_matched: false`` flag, so a reader sees the asymmetry without
    having to diff two files. ``--eggroll-rank 1`` reproduces the degenerate
    dimension-matched control if a reviewer wants to see it.
    """
    layout = ParamLayout.from_module(model)
    shared = HybridSubspace.from_layout(layout, rank=rank, max_subspace_dim=max_dim)
    if method != "eggroll":
        return layout, shared, shared.subspace_dim
    return layout, FactoredSubspace.from_layout(layout, rank=eggroll_rank or rank), shared.subspace_dim


def objective_shapes(subspace) -> Tuple[Tuple[int, ...], ...]:
    """How ``dim`` decomposes for EGGROLL's ``A B^T`` sampler.

    FactoredSubspace coordinates for a projected entry *are* an ``(d_out, rank)``
    matrix, so say so. HybridSubspace coordinates are unstructured: one flat block
    per layer, which is what ``Objective.from_subspace`` would have produced.
    """
    shapes = []
    for spec in subspace.specs:
        r = getattr(subspace, "ranks", {}).get(spec.entry_key)
        if spec.is_projected and r and spec.num_coords == spec.original_shape[0] * r:
            shapes.append((spec.original_shape[0], r))
        else:
            shapes.append((spec.num_coords,))
    return tuple(shapes)


class _BudgetReached(BaseException):
    """Thrown from PolyStep's closure to abort a step that would overspend.

    ``BaseException`` on purpose: it has to escape the optimizer's own
    ``except Exception`` handlers rather than be swallowed as a failed step.
    """


class TrackedObjective(Objective):
    """An :class:`Objective` that probes the validation split as the budget burns.

    Gives the ``accuracy vs cumulative candidate evaluations`` trajectory, and
    the checkpoint selection: the reported model is the probed iterate with the
    best *validation* accuracy, never the best test accuracy.
    """

    def __init__(self, *args, on_probe: Callable[[int, torch.Tensor], None], probe_every: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_probe = on_probe
        self._probe_every = probe_every
        self._next = probe_every

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        out = super().__call__(X)
        if self.evals >= self._next and self.best_x is not None:
            self._next = self.evals + self._probe_every
            self._on_probe(self.evals, self.best_x)
        return out


# --- methods ----------------------------------------------------------------


def run_gradfree(showcase: str, method: str, seed: int, device, results_dir: str, args, point=None) -> float:
    """Run one of the shared gradient-free baselines under the matched budget.

    ``point`` is one :data:`fairness.TUNING_GRID` entry (radius keys as multipliers on
    the shared probe scale). ``None`` means "use the sweep's pick", which is what a
    headline run does. Returns the best *validation* accuracy.
    """
    set_seed(seed)
    model = SHOWCASES[showcase]["model_fn"]().to(device).eval()
    layout, subspace, shared_dim = build_subspace(model, method, args.rank, args.max_subspace_dim, args.eggroll_rank)
    base_sd = _param_dict(layout, model)
    projections = subspace.init_projections(device, torch.float32)

    train_loader, val_loader, test_loader = _loaders(seed, args.batch_size, args.max_batches, hide_test=args.tune)
    stream = _cycle(train_loader)

    # One probe radius for everybody. ``sigma`` is per-coordinate, so a probe of
    # norm ``probe_radius`` in ``dim`` dimensions means this much per coordinate.
    sigma = args.probe_radius / math.sqrt(subspace.subspace_dim)
    # The methods default x0 to CPU zeros; every candidate then lands on CPU.
    x0 = torch.zeros(subspace.subspace_dim, device=device)
    hyper = {
        "openai_es": dict(sigma=sigma, lr=args.lr, popsize=args.popsize, seed=seed),
        "eggroll": dict(sigma=sigma, lr=args.lr, popsize=args.popsize, rank=1, seed=seed),
        "random_search": dict(sigma=sigma, seed=seed),
        "cma_es": dict(sigma0=sigma, popsize=args.popsize, diagonal=True, seed=seed),
    }[method]
    point, provenance = _resolve(showcase, method, args, point)
    hyper = apply_point(hyper, point, sigma)

    trajectory: List[Dict] = []
    best = {"val": -1.0, "coords": None, "evals": 0}
    start = time.time()

    def probe(evals: int, coords: torch.Tensor) -> None:
        _load_coords(model, subspace, projections, base_sd, coords)
        acc = evaluate_accuracy(model, val_loader, device=device)
        trajectory.append({"evals": evals, "val_accuracy": acc, "wall_time": time.time() - start})
        if acc > best["val"]:
            best.update(val=acc, coords=coords.detach().clone(), evals=evals)

    def fn(coords: torch.Tensor) -> torch.Tensor:
        """One generation: one shared minibatch, candidates materialized in chunks."""
        x, y = next(stream)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = [
                _batched_loss(model, subspace.reconstruct_batch(projections, base_sd, coords[i : i + args.chunk]), x, y)
                for i in range(0, coords.shape[0], args.chunk)
            ]
        return torch.cat(out)

    obj = TrackedObjective(
        fn,
        subspace.subspace_dim,
        args.eval_budget,
        shapes=objective_shapes(subspace),
        subspace=subspace,
        on_probe=probe,
        probe_every=args.probe_every,
    )

    with track_gpu_memory() as mem:
        result = METHODS[method](obj, x0=x0, **hyper)
        # A run that never reached a probe still has to select something.
        if best["coords"] is None and obj.best_x is not None:
            probe(obj.evals, obj.best_x)

    wall = time.time() - start
    _load_coords(model, subspace, projections, base_sd, best["coords"])
    # Scored once, on the selection -- and not at all during a sweep, whose whole
    # point is that the test set stays untouched until the configs are frozen.
    test_acc = float("nan") if args.tune else evaluate_accuracy(model, test_loader, device=device)

    _save(
        showcase,
        method,
        seed,
        test_acc,
        best["val"],
        wall,
        mem,
        obj.evals,
        result.iters,
        trajectory,
        subspace,
        args,
        results_dir,
        extra_hyper={**hyper, "best_train_batch_loss": result.best_loss},
        shared_dim=shared_dim,
        tuning=_tuning_record(point, provenance, args),
    )
    return best["val"]


def run_polystep(showcase: str, seed: int, device, results_dir: str, args, point=None) -> float:
    """PolyStep on the same HybridSubspace, stopped by the same candidate budget.

    ``point`` is one ``TUNING_GRID["polystep"]`` entry -- multipliers on
    :data:`POLYSTEP_CONFIG`'s three radii. Returns the best validation accuracy.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.epsilon import CosineEpsilon
    from polystep.optimizer import PolyStepOptimizer

    set_seed(seed)
    model = SHOWCASES[showcase]["model_fn"]().to(device)
    train_loader, val_loader, test_loader = _loaders(seed, args.batch_size, args.max_batches, hide_test=args.tune)

    point, provenance = _resolve(showcase, "polystep", args, point)
    cfg = apply_polystep_multipliers(POLYSTEP_CONFIG, point)
    layout, subspace, shared_dim = build_subspace(model, "polystep", args.rank, args.max_subspace_dim)

    # The cosine schedules have to anneal over the run the budget actually buys,
    # not over ``epochs * batches``: the budget stops this loop long before an
    # epoch ends. A step costs ~1.1 evaluations per subspace coordinate.
    est_steps = max(1, int(args.eval_budget / (1.125 * subspace.subspace_dim)))

    def sched(key, fallback):
        if f"{key}_init" not in cfg:
            return cfg.get(key, fallback)
        init, target = cfg[f"{key}_init"], cfg[f"{key}_target"]
        return CosineEpsilon(init=init, target=target, decay=(init - target) / est_steps)

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=sched("epsilon", 0.5),
        step_radius=sched("step_radius", 1.0),
        probe_radius=sched("probe_radius", 1.0),
        num_probe=cfg["num_probe"],
        subspace=subspace,
        chunk_size=cfg["chunk_size"],
        amortize_steps=cfg["amortize_steps"],
        use_momentum=cfg["use_momentum"],
        momentum_init=cfg["momentum_init"],
        momentum_final=cfg["momentum_final"],
        solver="softmax",
    )
    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())

    trajectory: List[Dict] = []
    best_val, best_sd = -1.0, None
    evals = steps = 0
    next_probe = args.probe_every
    start = time.time()

    exhausted = False
    with track_gpu_memory() as mem:
        for epoch in range(args.epochs):
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                def closure(batched_params, _d=data, _t=targets):
                    nonlocal evals
                    n = next(iter(batched_params.values())).shape[0]
                    if evals + n > args.eval_budget:
                        raise _BudgetReached
                    evals += n
                    return evaluator.evaluate(batched_params, _d, _t)

                try:
                    optimizer.step(closure)
                except _BudgetReached:
                    # The aborted step never wrote its update, so the model still
                    # holds the last fully-paid-for iterate. That is what gets scored.
                    exhausted = True
                    break
                steps += 1

                if evals >= next_probe:
                    next_probe = evals + args.probe_every
                    acc = evaluate_accuracy(model, val_loader, device=device)
                    trajectory.append({"evals": evals, "val_accuracy": acc, "wall_time": time.time() - start})
                    if acc > best_val:
                        best_val, best_sd = acc, copy.deepcopy(model.state_dict())
            print(f"    epoch {epoch + 1}/{args.epochs} | evals {evals}/{args.eval_budget} | val {best_val * 100:.1f}%")
            if exhausted:
                break

    wall = time.time() - start
    if best_sd is None:  # budget too small to reach a probe
        best_val = evaluate_accuracy(model, val_loader, device=device)
        best_sd = copy.deepcopy(model.state_dict())
        trajectory.append({"evals": evals, "val_accuracy": best_val, "wall_time": wall})
    model.load_state_dict(best_sd)
    test_acc = float("nan") if args.tune else evaluate_accuracy(model, test_loader, device=device)

    _save(
        showcase,
        "polystep",
        seed,
        test_acc,
        best_val,
        wall,
        mem,
        evals,
        steps,
        trajectory,
        subspace,
        args,
        results_dir,
        extra_hyper={**cfg, "epochs": args.epochs},
        shared_dim=shared_dim,
        tuning=_tuning_record(point, provenance, args),
    )
    return best_val


def _resolve(showcase: str, method: str, args, point):
    """``(grid point, provenance)`` for one run.

    An explicit ``point`` is a sweep trial and carries no provenance. Otherwise a
    headline run picks up whatever ``--tune`` selected on the validation split, and
    falls back to the transplanted defaults (an empty point) if nothing was swept --
    recorded either way, so an untuned number is never mistaken for a tuned one.
    """
    if point is not None:
        return point, None
    if args.untuned:
        return {}, None
    entry, provenance = load_selection("cifar", showcase, method, args.selection)
    return (entry["point"], provenance) if entry else ({}, None)


def _tuning_record(point, provenance, args) -> Dict:
    """The provenance block every result JSON carries: which sweep chose this."""
    return {
        "grid_point": point,
        "tuned": provenance is not None,
        "role": "sweep_trial" if args.tune else "headline",
        "sweep": provenance,
    }


def run_adam(showcase: str, seed: int, device, results_dir: str, args) -> None:
    """Adam on the smooth twin: the ceiling. Uses gradients, so it has no eval budget."""
    set_seed(seed)
    model = SHOWCASES[showcase]["model_fn"](smooth=True).to(device)
    train_loader, val_loader, test_loader = _loaders(seed, args.batch_size, args.max_batches)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    loss_fn = nn.CrossEntropyLoss()

    trajectory: List[Dict] = []
    best_val, best_sd, steps = -1.0, None, 0
    start = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(args.adam_epochs):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                opt.zero_grad()
                loss_fn(model(data), targets).backward()
                opt.step()
                steps += 1
            acc = evaluate_accuracy(model, val_loader, device=device)
            trajectory.append({"evals": None, "steps": steps, "val_accuracy": acc, "wall_time": time.time() - start})
            if acc > best_val:
                best_val, best_sd = acc, copy.deepcopy(model.state_dict())
            print(f"    epoch {epoch + 1}/{args.adam_epochs} | val {acc * 100:.1f}%")

    wall = time.time() - start
    model.load_state_dict(best_sd)
    test_acc = evaluate_accuracy(model, test_loader, device=device)

    _save(
        showcase,
        "adam",
        seed,
        test_acc,
        best_val,
        wall,
        mem,
        0,
        steps,
        trajectory,
        None,
        args,
        results_dir,
        extra_hyper={"lr": ADAM_LR, "epochs": args.adam_epochs, "model": "smooth_twin"},
    )


def _save(
    showcase,
    method,
    seed,
    test_acc,
    best_val,
    wall,
    mem,
    evals,
    steps,
    trajectory,
    subspace,
    args,
    results_dir,
    extra_hyper,
    shared_dim=None,
    tuning=None,
):
    """One JSON per run, with everything needed to plot accuracy against evaluations."""
    path = save_result(
        benchmark=showcase,
        method=method,
        seed=seed,
        metrics={
            "final_accuracy": test_acc,
            "best_accuracy": best_val,
            "test_accuracy_at_selected": test_acc,
            "best_val_accuracy": best_val,
            "wall_time_seconds": wall,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": evals,
            "total_steps": steps,
            "eval_budget": None if method == "adam" else args.eval_budget,
            "evals_used": None if method == "adam" else evals,
        },
        hyperparameters={
            "showcase": showcase,
            "discontinuity": SHOWCASES[showcase]["discontinuity"],
            "n_params": sum(p.numel() for p in SHOWCASES[showcase]["model_fn"]().parameters()),
            # Rank is matched across the table; dimension is not, for EGGROLL. See
            # ``build_subspace``. ``subspace_dim_matched`` puts that in every file.
            **subspace_tag(subspace, args.rank if subspace is not None else None, shared_dim=shared_dim),
            "subspace_compression_ratio": subspace.compression_ratio if subspace is not None else None,
            "max_subspace_dim": args.max_subspace_dim if subspace is not None else None,
            "tuning": tuning,
            "eval_budget": None if method == "adam" else args.eval_budget,
            "batch_size": args.batch_size,
            "probe_radius": args.probe_radius,
            "gradient_based": method == "adam",
            **extra_hyper,
        },
        step_logs=trajectory,
        results_dir=results_dir,
    )
    print(f"    saved {path}  test@selected={test_acc * 100:.2f}%  val={best_val * 100:.2f}%  evals={evals}")


METHOD_LIST = ["polystep", "eggroll", "openai_es", "cma_es", "random_search", "adam"]
#: Everything ``--tune`` sweeps. Adam is gradient-based, has no evaluation budget and
#: is not a peer in this table, so it keeps its published lr and is not swept.
TUNABLE = [m for m in METHOD_LIST if m != "adam"]
#: ``--tune`` budget as a fraction of ``--eval-budget``. The cosine schedules are
#: derived from the budget, so a reduced-budget run is a scale model of the full one:
#: same schedule shape, fewer steps.
TUNE_DIVISOR = 20


def dispatch(showcase, method, seed, device, results_dir, args):
    if method == "polystep":
        run_polystep(showcase, seed, device, results_dir, args)
    elif method == "adam":
        run_adam(showcase, seed, device, results_dir, args)
    else:
        run_gradfree(showcase, method, seed, device, results_dir, args)


def tune(args, device) -> str:
    """Validation-only sweep: every gradient-free method over its equal-size grid.

    Nine configurations per method, one seed, one reduced budget shared by all of
    them, so "configurations x seeds x evaluations per configuration" is the same
    number for every method and the paper can quote it. The test split is a
    :class:`TestSplitTripwire` for the whole sweep, so a leak raises rather than
    silently selecting on test.
    """
    trials, seed = [], args.seeds[0]
    trial_dir = os.path.join(os.path.dirname(args.selection), "cifar_trials")
    for showcase in args.showcases:
        print(f"=== {showcase} ===")
        for method in args.methods:
            if method not in TUNABLE:
                print(f"  skip {method} (gradient-based; not budget-matched, not swept)")
                continue
            for i, point in enumerate(TUNING_GRID[method]):
                name = "_".join(f"{k}{v:g}" for k, v in sorted(point.items()))
                run = run_polystep if method == "polystep" else run_gradfree
                extra = () if method == "polystep" else (method,)
                try:
                    # One directory per grid point, or the nine trials of a method
                    # overwrite each other: the filename is showcase_method_seed.
                    val = run(showcase, *extra, seed, device, os.path.join(trial_dir, name), args, point=point)
                except Exception as e:
                    print(f"  ERROR {showcase}/{method}/{name}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                trials.append(
                    {
                        "showcase": showcase,
                        "method": method,
                        "index": i,
                        "name": name,
                        "point": point,
                        "val": val,
                    }
                )
                print(f"  {method:14s} {name:34s} val={val * 100:.2f}%")

    costs = {m: tuning_cost(m, args.eval_budget, seeds=1) for m in args.methods if m in TUNABLE}
    path = write_selection(
        "cifar",
        trials,
        {
            "sweep": "experiments/runners/run_cifar.py --tune",
            "split": "validation only (test split replaced by TestSplitTripwire)",
            "selection_metric": "best_val_accuracy",
            "tie_break": "lowest TUNING_GRID index",
            "seeds": [seed],
            "budget_per_config": args.eval_budget,
            "headline_budget": args.eval_budget * TUNE_DIVISOR,
            "budget_reduction_factor": TUNE_DIVISOR,
            "rank": args.rank,
            "max_subspace_dim": args.max_subspace_dim,
            "probe_radius": args.probe_radius,
            "batch_size": args.batch_size,
        },
        costs,
        args.selection,
    )
    print(f"\nwrote {len(trials)} trials and the selected configs to {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--showcases", nargs="+", default=list(SHOWCASES), choices=list(SHOWCASES))
    p.add_argument("--methods", nargs="+", default=METHOD_LIST, choices=METHOD_LIST)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-dir", default="experiments/results/cifar")
    p.add_argument("--eval-budget", type=int, default=EVAL_BUDGET, help="Candidate evaluations per gradient-free run")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="PolyStep epoch cap (the budget usually binds first)")
    p.add_argument("--adam-epochs", type=int, default=ADAM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--rank", type=int, default=SUBSPACE_RANK, help="Subspace rank, shared by every method")
    p.add_argument(
        "--eggroll-rank",
        type=int,
        default=None,
        help="FactoredSubspace rank for EGGROLL only. Defaults to --rank; must be >= 2 to keep its sampler low-rank.",
    )
    p.add_argument(
        "--max-subspace-dim",
        type=int,
        default=MAX_SUBSPACE_DIM,
        help="Cap on subspace coordinates. This is PolyStep's per-step cost; see the constant.",
    )
    p.add_argument("--probe-radius", type=float, default=PROBE_RADIUS, help="Probe norm; baseline sigma is derived")
    p.add_argument("--lr", type=float, default=0.05, help="Step size for openai_es / eggroll")
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--chunk", type=int, default=CHUNK, help="Candidates materialized at once (peak VRAM knob)")
    p.add_argument("--probe-every", type=int, default=PROBE_EVERY, help="Evaluations between validation probes")
    p.add_argument("--max-batches", type=int, default=0, help="Truncate every split, for smoke runs (0 = full)")
    p.add_argument("--force", action="store_true", help="Rerun even if the result JSON exists")
    p.add_argument(
        "--tune",
        action="store_true",
        help=(
            f"Validation-only hyperparameter sweep. Every gradient-free method over its "
            f"9-config grid at 1/{TUNE_DIVISOR} of --eval-budget, one seed; writes the picks "
            f"to --selection. The test split is unreadable for the whole sweep."
        ),
    )
    p.add_argument("--selection", default=DEFAULT_SELECTION_PATH, help="Where --tune writes and a run reads")
    p.add_argument(
        "--untuned",
        action="store_true",
        help="Ignore --selection and use the transplanted defaults (the pre-sweep numbers)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end check: 1 seed, 25k evaluations, 8 batches per split",
    )
    args = p.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.adam_epochs = 2
        # Enough to buy PolyStep a handful of steps at the default subspace cap,
        # which is what "every method ran end to end" needs to mean here.
        args.eval_budget = 25_000
        args.probe_every = 5_000
        args.max_batches = 8
        args.results_dir = os.path.join(args.results_dir, "smoke")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.tune:
        args.eval_budget = max(1, args.eval_budget // TUNE_DIVISOR)
        # Ten validation probes per trial. Selection only needs the best one, and at
        # 5,000 val images a probe is not free -- keeping the headline resolution here
        # would roughly double the sweep's wall-clock for no gain in the pick.
        args.probe_every = max(1, args.eval_budget // 10)
        n = sum(len(TUNING_GRID[m]) for m in args.methods if m in TUNABLE)
        print(f"CIFAR-10 tuning sweep (validation only) | {args.showcases} x {args.methods} x seed {args.seeds[0]}")
        print(f"  {n} configs per showcase, {args.eval_budget} evals each (1/{TUNE_DIVISOR} of the headline budget)\n")
        tune(args, torch.device(args.device))
        return

    print(f"CIFAR-10 non-differentiable showcases | {args.showcases} x {args.methods} x {args.seeds}")
    print(f"  budget={args.eval_budget} evals  rank={args.rank}  probe_radius={args.probe_radius}\n")

    for showcase in args.showcases:
        print(f"=== {showcase} ({SHOWCASES[showcase]['discontinuity']}) ===")
        for method in args.methods:
            for seed in args.seeds:
                out = os.path.join(args.results_dir, f"{showcase}_{method}_{seed}.json")
                if os.path.exists(out) and not args.force:
                    print(f"  skip {method} seed={seed} (exists)")
                    continue
                print(f"  {method} seed={seed}")
                try:
                    dispatch(showcase, method, seed, torch.device(args.device), args.results_dir, args)
                except Exception as e:
                    print(f"    ERROR: {method} seed={seed}: {e}")
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        print()

    print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
