#!/usr/bin/env python
"""Systematic variant sweep for PolyStep.

Runs one staged benchmark over the real behavior axes of the optimizer
(solver, search representation, block strategy, adaptation flags, schedules,
geometry) on a small task suite, and records enough diagnostics to tell which
variants help, hurt, or do nothing.

Design constraints (from the review that motivated this runner):
  - Vehicle is PolyStepOptimizer.step(closure); the ask/tell path skips
    subspace/CMA/momentum/radius so it cannot exercise the full surface.
  - solver is always set explicitly (solver=None couples to the subspace).
  - Configs are ranked by forward-eval budget, not step count, because
    variants change evals per step.
  - Every config runs a state-mutation self-check; a variant whose state never
    changed is flagged dead and excluded from ranking, not silently ranked.
  - softmax vs sinkhorn are only compared in full space, where the vertex
    marginal binds; the column-marginal violation is logged as the separator.

Run:
    python experiments/runners/variant_sweep.py --dry-run
    python experiments/runners/variant_sweep.py --stage all --seeds 42 123 456 789 1337 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

# Ensure the repo root is importable so `experiments.runners.common` resolves
# when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep.cost_nn import NNCostEvaluator
from polystep.epsilon import CosineEpsilon, LinearEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.objectives.synthetic import Ackley, Rastrigin, Rosenbrock, Sphere
from polystep.solvers import SinkhornSolver
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout

SEEDS = [42, 123, 456, 789, 1337]
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "variant_sweep",
)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


class VectorParam(nn.Module):
    """Holds a single optimizable vector as the model parameter."""

    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.theta = nn.Parameter(init.clone())

    def forward(self):  # unused; the closure reads the batched theta directly
        return self.theta


class SignNet(nn.Module):
    """Two-layer MLP with a hard sign activation (non-differentiable)."""

    def __init__(self, d_in: int = 2, hidden: int = 16):
        super().__init__()
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.sign(self.l1(x))
        return self.l2(h).squeeze(-1)


def _checkerboard(n: int, k: int = 3, noise: float = 0.05, seed: int = 0):
    """k x k XOR grid: a fragmented, non-linearly-separable boundary."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g) * k
    y = ((x[:, 0].floor().long() + x[:, 1].floor().long()) % 2).float()
    x = x + noise * torch.randn(n, 2, generator=g)
    return (x - x.mean(0)) / x.std(0), y


@dataclass
class Env:
    """A benchmark task: builds a model, closure, and a true-quality probe."""

    name: str
    kind: str  # 'synthetic' | 'classification'
    metric_mode: str  # 'min' (loss) or 'max' (accuracy)
    default_solver: str  # explicit full-space solver for the baseline
    budget: int  # forward-eval budget per run
    build: Callable  # (device) -> (model, closure, quality_fn)
    max_steps: int = 8000  # step cap so eval-efficient configs stay bounded in wall-clock
    quality_name: str = "quality"
    subspace_ok: bool = True  # whether representation/block axes apply
    # Subspace used by the baseline and by non-representation axes. None means the
    # env runs full space (synthetic, sign_net); a subspace factory means the env
    # needs compression to be practical (mnist).
    baseline_subspace: Optional[Callable] = None


def _synthetic_env(name, obj_cls, dim, x0_val, budget, cond=None):
    def build(device):
        obj = obj_cls(dim=dim)
        init = torch.full((dim,), float(x0_val), device=device)
        model = VectorParam(init).to(device)
        # Optional anisotropic reweighting for the vertex-contention probe.
        scale = None
        if cond is not None:
            exps = torch.linspace(0.0, 1.0, dim, device=device)
            scale = (cond**exps).sqrt()  # sqrt so the quadratic curvature ratio is `cond`

        def closure(batched_params):
            theta = batched_params["theta"]  # (N, dim)
            if scale is not None:
                theta = theta * scale
            return obj.evaluate(theta)

        def quality(m):
            with torch.no_grad():
                theta = m.theta.unsqueeze(0)
                if scale is not None:
                    theta = theta * scale
                return obj.evaluate(theta).item()

        return model, closure, quality

    # A subspace over a single parameter vector is degenerate, so representation /
    # block / cma axes are skipped here; these envs carry the solver, schedule,
    # geometry, and quadratic axes in full space.
    return Env(
        name=name,
        kind="synthetic",
        metric_mode="min",
        default_solver="sinkhorn",
        budget=budget,
        build=build,
        subspace_ok=False,
    )


def _sign_net_env(budget):
    def build(device):
        x, y = _checkerboard(400, seed=0)
        x, y = x.to(device), y.to(device)
        model = SignNet(d_in=2, hidden=16).to(device)

        def zero_one(output, targets):
            return ((output > 0).float() != targets).float().mean()

        evaluator = NNCostEvaluator(model, loss_fn=zero_one)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, x, y)

        def quality(m):
            with torch.no_grad():
                acc = ((m(x) > 0).float() == y).float().mean().item()
            return 100.0 * acc

        return model, closure, quality

    return Env(
        name="sign_net",
        kind="classification",
        metric_mode="max",
        default_solver="sinkhorn",
        budget=budget,
        build=build,
        quality_name="accuracy",
    )


def _mnist_env(budget, n_train=2000):
    def build(device):
        from experiments.runners.common import load_mnist  # lazy: pulls torchvision

        train_loader, test_loader = load_mnist(batch_size=n_train, data_dir=os.environ.get("POLYSTEP_DATA", "./data"))
        xb, yb = next(iter(train_loader))
        x, y = xb.reshape(xb.shape[0], -1)[:n_train].to(device), yb[:n_train].to(device)
        # Deterministic val split off the training shard for selection-free reporting.
        val_x, val_y = x[: n_train // 5], y[: n_train // 5]
        tr_x, tr_y = x[n_train // 5 :], y[n_train // 5 :]
        model = nn.Sequential(nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())

        def closure(batched_params):
            return evaluator.evaluate(batched_params, tr_x, tr_y)

        def quality(m):
            with torch.no_grad():
                acc = (m(val_x).argmax(-1) == val_y).float().mean().item()
            return 100.0 * acc

        return model, closure, quality

    return Env(
        name="mnist_mlp",
        kind="classification",
        metric_mode="max",
        default_solver="softmax",
        budget=budget,
        build=build,
        quality_name="accuracy",
        baseline_subspace=lambda model, layout: HybridSubspace.from_layout(layout, rank=8),
    )


def _maxsat_env(budget, n_vars=200, ratio=4.2, seed0=0):
    def build(device):
        g = torch.Generator().manual_seed(seed0)
        n_clauses = int(ratio * n_vars)
        lits = torch.randint(0, n_vars, (n_clauses, 3), generator=g).to(device)
        want = (torch.randint(0, 2, (n_clauses, 3), generator=g)).float().to(device)  # target bit per literal
        model = VectorParam(torch.zeros(n_vars)).to(device)

        def _pct_sat(assign):  # assign: (..., n_vars) in {0,1}
            v = assign[..., lits]  # (..., n_clauses, 3)
            clause_sat = (v == want).float().amax(dim=-1)  # (..., n_clauses)
            return clause_sat.mean(dim=-1)

        def closure(batched_params):
            assign = (batched_params["theta"] > 0).float()  # round() non-differentiability
            return 1.0 - _pct_sat(assign)  # fraction unsatisfied (minimized)

        def quality(m):
            with torch.no_grad():
                return 100.0 * _pct_sat((m.theta > 0).float()).item()

        return model, closure, quality

    return Env(
        name="maxsat",
        kind="synthetic",
        metric_mode="max",
        default_solver="sinkhorn",
        budget=budget,
        build=build,
        quality_name="pct_sat",
        subspace_ok=False,
    )


class TinySNN(nn.Module):
    """Compact hard-threshold LIF spiking net (non-differentiable spikes)."""

    def __init__(self, num_steps: int = 6, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        out = torch.zeros(x.shape[0], 10, device=x.device)
        for _ in range(self.num_steps):
            mem = mem + self.fc1(x)
            spk = (mem >= 1.0).float()  # hard LIF threshold, zero gradient a.e.
            mem = mem - spk  # reset by subtraction
            out = out + self.fc2(spk)
        return out


def _snn_env(budget, num_steps=6, n_train=1000):
    def build(device):
        from experiments.runners.common import load_mnist

        train_loader, _ = load_mnist(batch_size=n_train, data_dir=os.environ.get("POLYSTEP_DATA", "./data"))
        xb, yb = next(iter(train_loader))
        x, y = xb.reshape(xb.shape[0], -1)[:n_train].to(device), yb[:n_train].to(device)
        val_x, val_y = x[: n_train // 5], y[: n_train // 5]
        tr_x, tr_y = x[n_train // 5 :], y[n_train // 5 :]
        model = TinySNN(num_steps=num_steps).to(device)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss(), chunk_size=256)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, tr_x, tr_y)

        def quality(m):
            with torch.no_grad():
                acc = (m(val_x).argmax(-1) == val_y).float().mean().item()
            return 100.0 * acc

        return model, closure, quality

    return Env(
        name="snn",
        kind="classification",
        metric_mode="max",
        default_solver="softmax",
        budget=budget,
        build=build,
        quality_name="accuracy",
        baseline_subspace=lambda model, layout: HybridSubspace.from_layout(layout, rank=8),
    )


def build_envs(names, scale):
    """Return the requested envs. `scale` shrinks budgets for smoke runs."""
    reg = {
        "ackley": _synthetic_env("ackley", Ackley, 32, 3.0, int(1.5e5 * scale)),
        "rosenbrock": _synthetic_env("rosenbrock", Rosenbrock, 32, 1.5, int(1.5e5 * scale)),
        "rastrigin": _synthetic_env("rastrigin", Rastrigin, 32, 3.0, int(1.5e5 * scale)),
        "sphere": _synthetic_env("sphere", Sphere, 32, 3.0, int(1.2e5 * scale)),
        "aniso": _synthetic_env("aniso", Sphere, 16, 3.0, int(1.2e5 * scale), cond=1e4),
        "maxsat": _maxsat_env(int(1.5e5 * scale)),
        "sign_net": _sign_net_env(int(3e5 * scale)),
        "mnist_mlp": _mnist_env(int(8e5 * scale)),
        "snn": _snn_env(int(8e5 * scale)),
    }
    for _name in ("sign_net", "maxsat"):
        reg[_name].max_steps = 3000
    reg["mnist_mlp"].max_steps = 800
    reg["snn"].max_steps = 800
    fast = ["ackley", "rosenbrock", "rastrigin", "sphere", "aniso", "maxsat", "sign_net"]
    if names == ["fast"]:
        names = fast
    elif names == ["all"]:
        names = list(reg.keys())
    return [reg[n] for n in names if n in reg]


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------


@dataclass
class Config:
    axis: str
    name: str
    solver: str
    kwargs: dict = field(default_factory=dict)
    subspace: Optional[Callable] = None  # (model, layout) -> subspace, overrides baseline
    needs_subspace: bool = False  # skip on envs where representation does not apply
    full_space: bool = False  # force no subspace even when the env baseline has one


def _hybrid(rank):
    return lambda model, layout: HybridSubspace.from_layout(layout, rank=rank)


def _linear(rank):
    return lambda model, layout: LinearSubspace.from_layout(layout, rank=rank)


def _adaptive(model, layout):
    return AdaptiveSubspace.auto_from_params(model, min_rank=8, max_rank=64)


def _cma(model, layout):
    return CMAAdaptiveSubspace.auto_from_params(model, min_rank=8, max_rank=64)


def screen_configs(env: Env):
    """One-factor-at-a-time configs from the per-env baseline."""
    s = env.default_solver
    cfgs = [Config("baseline", "baseline", s)]

    # 1. Solver / weighting (full space, where the vertex marginal binds).
    for solver in ["softmax", "tempered_softmax", "min_cost_greedy", "top_k_mean", "sinkhorn"]:
        if solver == s:
            continue
        cfgs.append(Config("solver", f"solver={solver}", solver, full_space=True))

    # 2. Search representation (needs real layers).
    cfgs += [
        Config("representation", "hybrid_r8", "softmax", subspace=_hybrid(8), needs_subspace=True),
        Config("representation", "hybrid_r16", "softmax", subspace=_hybrid(16), needs_subspace=True),
        Config("representation", "linear_r8", "softmax", subspace=_linear(8), needs_subspace=True),
        Config("representation", "adaptive", "softmax", subspace=_adaptive, needs_subspace=True),
        Config("representation", "cma_subspace", "softmax", subspace=_cma, needs_subspace=True),
    ]

    # 3. Block strategy (full-space per-layer / grouped OT).
    cfgs += [
        Config("block", "per_layer", s, kwargs={"block_strategy": "per_layer"}, full_space=True),
        Config("block", "grouped", s, kwargs={"block_strategy": "grouped"}, full_space=True),
    ]

    # 4. Schedule (baseline uses CosineEpsilon; compare constant and linear).
    cfgs += [
        Config("schedule", "constant_eps", s, kwargs={"epsilon": 0.3}),
        Config("schedule", "linear_eps", s, kwargs={"epsilon": LinearEpsilon(init=1.0, target=0.1, decay=0.02)}),
    ]

    # 5. Radius adaptation, plus a fixed cosine-radius control (anneal without adapt).
    cfgs += [
        Config("radius", "adaptive_radius", s, kwargs={"use_adaptive_radius": True}),
        Config(
            "radius",
            "cosine_radius_control",
            s,
            kwargs={"step_radius": CosineEpsilon(init=1.0, target=0.3, decay=0.02)},
        ),
    ]

    # 6. Momentum family.
    cfgs += [
        Config("momentum", "use_momentum", s, kwargs={"use_momentum": True}),
        Config("momentum", "amortize", s, kwargs={"amortize_steps": 3}),
    ]
    if s == "sinkhorn":
        cfgs.append(Config("momentum", "dual_momentum", s, kwargs={"dual_momentum_beta": 0.5}))

    # 7. Sinkhorn acceleration (only meaningful for the sinkhorn solver).
    if s == "sinkhorn":
        cfgs += [
            Config("sinkhorn_accel", "anderson", s, kwargs={"anderson_depth": 5}),
            Config("sinkhorn_accel", "adaptive_omega", s, kwargs={"adaptive_omega": True}),
            Config("sinkhorn_accel", "ddi", s, kwargs={"data_dependent_init": True}),
        ]

    # 8. CMA (needs CMAAdaptiveSubspace + monolithic).
    cfgs += [
        Config(
            "cma",
            "covariance",
            "softmax",
            subspace=_cma,
            kwargs={"use_covariance_adaptation": True},
            needs_subspace=True,
        ),
        Config(
            "cma",
            "covariance_csa",
            "softmax",
            subspace=_cma,
            kwargs={"use_covariance_adaptation": True, "use_csa": True},
            needs_subspace=True,
        ),
    ]

    # 9. Quadratic cluster (orthoplex + num_probe>=2).
    cfgs += [
        Config("quadratic", "quadratic_model", s, kwargs={"num_probe": 2, "use_quadratic_model": True}),
        Config("quadratic", "trust_region", s, kwargs={"num_probe": 2, "trust_region": True}),
        Config("quadratic", "newton", s, kwargs={"num_probe": 2, "newton_refinement": True}),
        Config(
            "quadratic",
            "biased_rotation",
            s,
            kwargs={"num_probe": 2, "biased_rotation": True, "use_quadratic_model": True},
        ),
        Config("quadratic", "multifidelity", s, kwargs={"num_probe": 2, "multifidelity_screen": True}),
    ]

    # 10. Geometry.
    cfgs += [
        Config("geometry", "simplex", s, kwargs={"polytope_type": "simplex"}),
        Config("geometry", "num_probe_3", s, kwargs={"num_probe": 3}),
    ]
    return cfgs


def interaction_configs(env: Env):
    s = env.default_solver
    cfgs = []
    # solver x representation
    for solver in ["softmax", "sinkhorn"]:
        cfgs.append(Config("solver_x_repr", f"{solver}_hybrid", solver, subspace=_hybrid(8), needs_subspace=True))
    # CMA x AdaptiveSubspace vs plain adaptive
    cfgs.append(Config("cma_x_adaptive", "adaptive_no_cma", "softmax", subspace=_adaptive, needs_subspace=True))
    cfgs.append(
        Config(
            "cma_x_adaptive",
            "cma_covariance",
            "softmax",
            subspace=_cma,
            kwargs={"use_covariance_adaptation": True},
            needs_subspace=True,
        )
    )
    # block x representation
    cfgs.append(
        Config(
            "block_x_repr",
            "per_layer_hybrid",
            "softmax",
            subspace=_hybrid(8),
            kwargs={"block_strategy": "per_layer"},
            needs_subspace=True,
        )
    )
    # trust_region x quadratic at K=3
    cfgs.append(
        Config(
            "trust_x_quad",
            "trust_quad_k3",
            s,
            kwargs={"num_probe": 3, "trust_region": True, "use_quadratic_model": True},
        )
    )
    # adaptive_radius x schedule
    cfgs.append(
        Config("radius_x_sched", "adaptive_constant_eps", s, kwargs={"use_adaptive_radius": True, "epsilon": 0.3})
    )
    return cfgs


def configs_for(env: Env, stage: str):
    catalog = []
    if stage in ("screen", "all"):
        catalog += screen_configs(env)
    if stage in ("interactions", "all"):
        catalog += interaction_configs(env)
    if stage == "mechanism":
        # Mechanism isolation reuses the screen axes (solver, representation) on the
        # dedicated envs (aniso, sphere) so the marginal-violation and C_diag plots
        # have the full solver / subspace spread.
        catalog += screen_configs(env)
    out = []
    for c in catalog:
        if (c.needs_subspace or c.subspace is not None) and not env.subspace_ok:
            continue
        # Solver comparison belongs in full space; on envs that need a subspace
        # baseline (mnist, snn) it is not informative, so run it on the full-space envs.
        if c.axis == "solver" and env.baseline_subspace is not None:
            continue
        # Full-space configs are impractical on large models (100k+ params); the
        # subspace-baseline envs cover representation, schedule, radius, cma instead.
        if c.full_space and env.baseline_subspace is not None:
            continue
        out.append(c)
    # dedupe by name (interactions can repeat a baseline-like config)
    seen = set()
    uniq = []
    for c in out:
        key = (c.axis, c.name)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# ---------------------------------------------------------------------------
# Baseline kwargs and optimizer construction
# ---------------------------------------------------------------------------


def baseline_kwargs(env: Env, seed: int):
    # Horizon for the cosine anneal, in optimizer steps. Approximate (evals per
    # step vary a little by config); the schedule just needs to reach target.
    horizon = max(200, env.budget // 64)
    return dict(
        epsilon=CosineEpsilon(init=1.0, target=0.05, decay=0.02, total_steps=horizon),
        step_radius=0.8,
        probe_radius=1.5,
        num_probe=1,
        polytope_type="orthoplex",
        scale_cost="mean",
        compile=False,
        mixed_precision=False,
        seed=seed,
        max_iterations=horizon,
    )


def build_optimizer(env: Env, cfg: Config, model, seed):
    layout = ParamLayout.from_module(model)
    # Subspace resolution: explicit config subspace overrides; else full_space
    # forces none; else fall back to the env baseline subspace.
    if cfg.subspace is not None:
        subspace = cfg.subspace(model, layout)
    elif cfg.full_space or env.baseline_subspace is None:
        subspace = None
    else:
        subspace = env.baseline_subspace(model, layout)
    kwargs = baseline_kwargs(env, seed)
    kwargs.update(cfg.kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(model, subspace=subspace, solver=cfg.solver, **kwargs)
    return opt


# ---------------------------------------------------------------------------
# Solver instrumentation (n_iters + column-marginal violation)
# ---------------------------------------------------------------------------


def instrument_solver(opt, log):
    """Wrap the solver so each solve records n_iters and column-marginal error."""
    solver = getattr(opt, "solver", None)
    if solver is None or not hasattr(solver, "solve"):
        return
    orig = solver.solve

    def wrapped(*a, **k):
        r = orig(*a, **k)
        log["n_iters"].append(int(getattr(r, "n_iters", 0)))
        m = getattr(r, "matrix", None)
        if m is not None and m.dim() == 2:
            col = m.sum(dim=0)
            target = col.sum() / col.shape[0]
            log["marginal_viol"].append((col - target).abs().sum().item())
        return r

    solver.solve = wrapped


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _clamp_to_budget(curve, budget):
    # Drop the trailing point that overshoots the budget so variants with more
    # evals per step are scored at the same budget. Keep the first point if none fit.
    within = [p for p in curve if p[0] <= budget]
    return within if within else curve[:1]


def _best(mode, curve, budget):
    vals = [q for _, q in _clamp_to_budget(curve, budget)]
    if not vals:
        return float("nan")
    return min(vals) if mode == "min" else max(vals)


def _auc(mode, curve, budget):
    """Area under best-so-far quality vs eval fraction (higher is better for both
    modes after sign flip). Trapezoid over normalized evals, clamped to budget."""
    curve = _clamp_to_budget(curve, budget)
    if not curve:
        return 0.0
    best = math.inf if mode == "min" else -math.inf
    pts = []
    for evals, q in curve:
        best = min(best, q) if mode == "min" else max(best, q)
        signed = -best if mode == "min" else best
        pts.append((evals / max(budget, 1), signed))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area


def self_checks(cfg: Config, state, log):
    """Assert the variant's state actually mutated. Returns (dead, reasons)."""
    reasons = []
    kw = cfg.kwargs
    if kw.get("trust_region"):
        tm = state.trust_region_multipliers
        if len(tm) == 0 or all(v == 1.0 for v in tm):
            reasons.append("trust_region multiplier frozen")
    if kw.get("use_covariance_adaptation"):
        if state.generation == 0 or state.C_diag is None:
            reasons.append("CMA generation did not advance")
    if kw.get("use_csa"):
        if state.sigma <= 1e-6 or state.sigma >= 1e6:
            reasons.append("CSA sigma pinned at bound")
    if kw.get("use_adaptive_radius"):
        if abs(state.radius_multiplier - 1.0) < 1e-9:
            reasons.append("adaptive radius never moved")
    if kw.get("anderson_depth") or kw.get("adaptive_omega"):
        iters = log["n_iters"]
        if iters and max(iters) <= 1:
            reasons.append("sinkhorn ran <=1 iter (accel inactive)")
    return (len(reasons) > 0, reasons)


def run_one(env: Env, cfg: Config, seed: int, device: str):
    torch.manual_seed(seed)
    model, closure, quality_fn = env.build(device)
    opt = build_optimizer(env, cfg, model, seed)
    log = {"n_iters": [], "marginal_viol": []}
    instrument_solver(opt, log)

    curve = []  # (cumulative_evals, true quality)
    evals = 0
    steps = 0
    t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    probe_every = 1
    while evals < env.budget and steps < env.max_steps:

        def counting_closure(batched_params):
            nonlocal evals
            evals += next(iter(batched_params.values())).shape[0]
            return closure(batched_params)

        opt.step(counting_closure)
        steps += 1
        if steps % probe_every == 0:
            curve.append((evals, quality_fn(model)))

    wall = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
    state = opt._state
    dead, reasons = self_checks(cfg, state, log)

    best_q = _best(env.metric_mode, curve, env.budget)
    return {
        "benchmark": env.name,
        "method": f"{cfg.axis}:{cfg.name}",
        "axis": cfg.axis,
        "variant": cfg.name,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_mode": env.metric_mode,
        "quality_name": env.quality_name,
        "metrics": {
            "final_quality": curve[-1][1] if curve else float("nan"),
            "best_quality": best_q,
            "auc": _auc(env.metric_mode, curve, env.budget),
            "function_evals": evals,
            "total_steps": steps,
            "wall_time_seconds": wall,
            "peak_gpu_memory_mb": peak_mb,
            "mean_n_iters": (sum(log["n_iters"]) / len(log["n_iters"])) if log["n_iters"] else 0.0,
            "mean_marginal_viol": (sum(log["marginal_viol"]) / len(log["marginal_viol"]))
            if log["marginal_viol"]
            else 0.0,
        },
        "diagnostics": {
            "solver": cfg.solver,
            "radius_multiplier": float(state.radius_multiplier),
            "sigma": float(state.sigma),
            "generation": int(state.generation),
            "absorb_count": int(state.absorb_count),
            "n_trust_updates": len(state.trust_region_multipliers),
            "dead": dead,
            "dead_reasons": reasons,
        },
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["fast"], help="env names, or 'fast' / 'all'")
    ap.add_argument("--stage", choices=["screen", "interactions", "mechanism", "all"], default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="1 seed, tiny budget, fast smoke")
    args = ap.parse_args()

    scale = 0.02 if args.dry_run else 1.0
    seeds = [args.seeds[0]] if args.dry_run else args.seeds
    envs = build_envs(args.envs, scale)
    os.makedirs(args.results_dir, exist_ok=True)

    total = 0
    dead_total = 0
    for env in envs:
        cfgs = configs_for(env, args.stage)
        print(
            f"[{env.name}] {len(cfgs)} configs x {len(seeds)} seeds  (budget={env.budget} evals, device={args.device})"
        )
        for cfg in cfgs:
            for seed in seeds:
                try:
                    rec = run_one(env, cfg, seed, args.device)
                except Exception as e:
                    print(f"  ERROR {env.name}/{cfg.axis}:{cfg.name}/seed{seed}: {type(e).__name__}: {e}")
                    continue
                total += 1
                flag = ""
                if rec["diagnostics"]["dead"]:
                    dead_total += 1
                    flag = f"  DEAD: {', '.join(rec['diagnostics']['dead_reasons'])}"
                path = os.path.join(args.results_dir, f"{env.name}_{cfg.axis}_{cfg.name}_{seed}.json")
                with open(path, "w") as f:
                    json.dump(rec, f, indent=2)
                m = rec["metrics"]
                print(
                    f"  {cfg.axis:16s} {cfg.name:22s} seed={seed} best={m['best_quality']:.4f} evals={m['function_evals']} t={m['wall_time_seconds']:.1f}s{flag}"
                )

    print(f"\nwrote {total} runs to {args.results_dir}  ({dead_total} flagged dead)")


if __name__ == "__main__":
    main()
