"""Evaluation- and time-budget controls using the existing forward evaluator."""

from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runners.controlled_data import hard_model
from experiments.runners.run_controlled import (
    ROOT,
    RESULTS,
    atomic_json,
    block_candidates,
    config_id,
    grid,
    load_data,
    make_projection,
    reconstruct,
    source_manifest,
    hardware_info,
)
from polystep.cost_nn import NNCostEvaluator
from polystep.factored_subspace import FactoredSubspace
from polystep.transform import ParamLayout
from polystep.baselines.methods import _lowrank_noise, _shaped

OUT = RESULTS / "practical"
METHODS = (
    "polystep_tuned",
    "polystep",
    "openai_es",
    "spsa",
    "mezo",
    "random_search",
    "eggroll",
    "adam_full",
    "adam_subspace",
)


def points(task, method):
    if method == "polystep_tuned":
        for epsilon, probe, step in itertools.product((0.5, 1.0, 2.0), repeat=3):
            yield dict(epsilon_multiplier=epsilon, probe_multiplier=probe, step_multiplier=step)
    elif method == "polystep":
        yield from grid("softmax")
    elif method.startswith("adam"):
        for lr, wd, extra in itertools.product(
            (0.0001, 0.001, 0.01), (0.0, 0.00001, 0.001), (1.0, 10.0, None) if task == "int8" else (1.0, 5.0, 25.0)
        ):
            yield dict(lr=lr, weight_decay=wd, **({"clip": extra} if task == "int8" else {"slope": extra}))
    elif method in ("openai_es", "eggroll"):
        for population, rho, lr in itertools.product((32, 256, 2048), (0.001, 0.01, 0.1), (0.001, 0.01, 0.1)):
            yield dict(population=population, rho=rho, lr=lr)
    elif method == "random_search":
        # Random search has one proposal radius, not an independent learning rate.
        for k in range(27):
            yield dict(rho=10 ** (-4 + 4 * k / 26))
    else:
        for rho, k in itertools.product((0.001, 0.01, 0.1), range(9)):
            yield dict(rho=rho, lr=10 ** (-4 + k / 2))


def host_data(task):
    if task in ("snn", "int8"):
        train, val, _, test, _ = load_data("cpu")
        return train, val, test
    output = []
    for split in ("train", "validation", "test"):
        with np.load(ROOT / "data/dvs_controlled" / f"{split}.npz") as record:
            output.append((torch.from_numpy(record["x"].copy()), torch.from_numpy(record["y"].copy())))
    return tuple(output)


@torch.no_grad()
def measure(model, params, data):
    nclass = 11 if data[0].ndim == 5 else 10
    loss, correct, counts = 0.0, torch.zeros(nclass, device="cuda"), torch.zeros(nclass, device="cuda")
    x, y = data
    for first in range(0, len(y), 256):
        yy = y[first : first + 256]
        logits = torch.func.functional_call(model, params, (x[first : first + 256],))
        loss += F.cross_entropy(logits, yy, reduction="sum").item()
        counts += torch.bincount(yy, minlength=nclass)
        correct += torch.bincount(yy[logits.argmax(-1) == yy], minlength=nclass)
    return dict(
        accuracy=(correct.sum() / counts.sum()).item(),
        macro_accuracy=(correct / counts).mean().item(),
        loss=loss / len(y),
    )


def simplex(blocks, generator):
    # Deterministic Helmert basis of the centered hyperplane in R^9.
    basis = torch.zeros(9, 8, device="cuda")
    for k in range(8):
        basis[: k + 1, k] = 1 / math.sqrt((k + 1) * (k + 2))
        basis[k + 1, k] = -(k + 1) / math.sqrt((k + 1) * (k + 2))
    vertices = basis * math.sqrt(9 / 8)
    q, r = torch.linalg.qr(torch.randn(blocks, 8, 8, generator=generator, device="cuda"))
    q *= torch.where(r.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).unsqueeze(-2)
    # SO(8), as in the specified reference update.
    q[:, :, -1] *= torch.linalg.det(q).sign().unsqueeze(-1)
    return vertices.unsqueeze(0) @ q


class ForwardRun:
    """One fixed subspace, using PolyStep's batched or compiled evaluator."""

    def __init__(self, task, method, cfg, seed, chunk, compiled):
        self.task, self.method, self.cfg = task, method, cfg
        self.model = hard_model(task).cuda().eval()
        self.base = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        if method == "eggroll":
            self.sub = FactoredSubspace.from_layout(ParamLayout.from_module(self.model), rank=4, seed=seed + 10000)
            self.projections = self.sub.init_projections(torch.device("cuda"), torch.float32)
            self.active = self.sub.subspace_dim
        else:
            self.sub, self.projections, _ = make_projection(self.model, seed + 10000, "cuda")
            self.active = self.sub.subspace_dim // 8 * 8
        if hasattr(self.sub, "build_fused_projection"):
            self.sub.build_fused_projection(self.projections)
        self.z = torch.zeros(self.sub.subspace_dim, device="cuda")
        self.generator = torch.Generator(device="cuda").manual_seed(seed + 30000)
        self.chunk = chunk
        self.sequential = method in ("spsa", "mezo", "random_search")
        self.evaluator = NNCostEvaluator(
            self.model,
            nn.CrossEntropyLoss(),
            compile_vmap=compiled and not self.sequential,
            use_inplace=self.sequential,
            compile_forward=compiled and self.sequential,
        )
        self.shapes = (
            [
                (s.original_shape[0], self.sub.ranks[s.entry_key]) if s.is_projected else (s.num_coords,)
                for s in self.sub.specs
            ]
            if method == "eggroll"
            else [(self.active,)]
        )

    def params(self):
        return reconstruct(self.base, self.sub, self.projections, self.z)

    def evaluate(self, points, batch):
        if self.sequential:
            return self.evaluator.evaluate_subspace_inplace(self.sub, self.projections, self.base, points, *batch)
        out = []
        for first in range(0, len(points), self.chunk):
            params = self.sub.reconstruct_batch(self.projections, self.base, points[first : first + self.chunk])
            out.append(self.evaluator.evaluate(params, *batch))
        return torch.cat(out)

    @property
    def candidates_per_step(self):
        if self.method == "polystep":
            return self.active // 8 * 9
        return self.cfg.get("population", 2)

    @torch.no_grad()
    def step(self, batch):
        cfg, method, gen = self.cfg, self.method, self.generator
        if method == "polystep":
            u = simplex(self.active // 8, gen)
            base, costs = self.params(), []
            for first in range(0, len(u), max(1, self.chunk // 9)):
                directions = u[first : first + max(1, self.chunk // 9)]
                params = block_candidates(base, self.sub, self.projections, first, cfg["rho"] * directions)
                # Slice externally to avoid compiling an unrolled vmap chunk loop.
                n = len(directions) * 9
                cost = torch.cat(
                    [
                        self.evaluator.evaluate({k: v[j : j + self.chunk] for k, v in params.items()}, *batch)
                        for j in range(0, n, self.chunk)
                    ]
                ).reshape(-1, 9)
                costs.append(cost)
            cost = torch.cat(costs)
            w = torch.softmax(-(cost - cost.amin(-1, keepdim=True)) / cfg["tau"], -1)
            d = torch.einsum("pv,pvd->pd", w, u)
            d[cost.amax(-1) == cost.amin(-1)] = 0
            self.z[: self.active] += cfg["ell"] * d.flatten()
        else:
            pop = cfg.get("population", 2)
            if method == "eggroll":
                noise = _lowrank_noise(self.shapes, 1, pop // 2, self.active, gen, self.z.device, self.z.dtype)
            elif method == "spsa":
                noise = (2 * torch.randint(0, 2, (1, self.active), generator=gen, device="cuda") - 1).float()
            else:
                noise = torch.randn(pop // 2, self.active, generator=gen, device="cuda")
            # rho is sqrt(E||perturbation||^2), so the per-coordinate scale is rho/sqrt(d).
            sigma = cfg["rho"] / math.sqrt(self.active)
            wide = self.z.new_zeros(pop // 2, len(self.z))
            wide[:, : self.active] = noise
            if method == "random_search":
                candidate = self.z + sigma * wide[0]
                cost = self.evaluate(torch.stack((self.z, candidate)), batch)
                self.z = torch.where(cost[1] < cost[0], candidate, self.z)
            else:
                cost = self.evaluate(torch.cat((self.z + sigma * wide, self.z - sigma * wide)), batch)
                if method == "openai_es":
                    delta = cfg["lr"] / (pop * sigma) * torch.cat((wide, -wide)).T @ _shaped(cost, "rank")
                elif method == "eggroll":
                    delta = cfg["lr"] / (pop // 2) * wide.T @ torch.sign(cost[pop // 2 :] - cost[: pop // 2])
                else:
                    delta = -cfg["lr"] * (cost[0] - cost[1]) / (2 * sigma) * wide[0]
                self.z += delta
        if not torch.isfinite(cost).all() or not torch.isfinite(self.z).all():
            raise FloatingPointError("nonfinite candidate cost or update")
        return self.candidates_per_step


class TunedRun:
    """Existing PolyStep optimizer with task settings fixed before this campaign."""

    def __init__(self, task, method, cfg, seed, chunk, compiled):
        from polystep.optimizer import PolyStepOptimizer
        from polystep.hybrid_subspace import HybridSubspace

        self.task, self.method, self.cfg = task, method, cfg
        self.model = hard_model(task).cuda().eval()
        self.base = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        self.sub = HybridSubspace.from_layout(
            ParamLayout.from_module(self.model),
            rank=8 if task == "int8" else 4,
            seed=seed + 10000,
            rotation_interval=0,
            absorb_mode="periodic",
            absorb_interval=0 if task == "int8" else 20,
        )
        self.active = self.sub.subspace_dim
        self.optimizer = PolyStepOptimizer(
            self.model,
            subspace=self.sub,
            seed=seed + 30000,
            compile=False,
            chunk_size=chunk,
            num_probe=1,
            polytope_type="simplex",
            adaptive_probes=False,
            adaptive_num_probe=False,
            biased_rotation=task != "int8",
            use_momentum=task == "int8",
            momentum_init=0.4,
            momentum_final=0.4,
            amortize_steps=1,
        )
        self.chunk, self.sequential, self.progress = chunk, False, 0.0
        self.evaluator = NNCostEvaluator(
            self.model, nn.CrossEntropyLoss(), use_inplace=False, compile_vmap=compiled, compile_forward=False
        )
        self.evaluations = 0

    def params(self):
        return dict(self.model.named_parameters())

    @property
    def candidates_per_step(self):
        return int(self.optimizer._state.X.shape[0]) * 9

    @torch.no_grad()
    def step(self, batch):
        cfg = self.cfg
        f = (1 + math.cos(math.pi * min(1.0, self.progress))) / 2
        epsilon = ((0.3 + 4.7 * f) if self.task == "int8" else 0.5) * cfg["epsilon_multiplier"]
        # Actual probe norm is half the optimizer's physical probe radius at K=1.
        rho = ((0.25 + 0.75 * f) if self.task == "int8" else 0.25) * cfg["probe_multiplier"]
        ell = ((8 + 24 * f) if self.task == "int8" else 1.0) * cfg["step_multiplier"]
        self.optimizer.epsilon = epsilon
        self.optimizer.step_radius = ell / epsilon
        self.optimizer.probe_radius = 2 * rho / epsilon
        self.optimizer.chunk_size = self.chunk
        self.evaluations = 0

        def closure(params):
            self.evaluations += len(next(iter(params.values())))
            costs = self.evaluator.evaluate(params, *batch)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("nonfinite candidate cost")
            return costs

        self.optimizer.step(closure)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("nonfinite update")
        if self.evaluations != self.candidates_per_step:
            raise RuntimeError("optimizer candidate accounting mismatch")
        return self.evaluations


def make_engine(task, method, cfg, seed, chunk, compiled):
    cls = TunedRun if method == "polystep_tuned" else ForwardRun
    return cls(task, method, cfg, seed, chunk, compiled)


def run(task, method, cfg, seed, stage, axis, profile, *, smoke=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    train_host, val_host, test_host = host_data(task)
    budget = (2_000_000 if stage == "tune" else 10_000_000) if axis == "evals" else (720 if stage == "tune" else 3600)
    if smoke:
        budget = 20_000 if axis == "evals" else 20
    fullcfg = dict(task=task, method=method, axis=axis, **cfg)
    path = OUT / ("smoke" if smoke else stage) / task / axis / method / config_id(fullcfg) / f"{seed}.json"
    provenance = source_manifest()
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["config"] != fullcfg or existing["budget"] != budget or existing["profile"] != profile:
            raise ValueError(f"incompatible existing result: {path}")
        if existing.get("source_sha256") != provenance["source_sha256"]:
            raise ValueError(f"source changed since completed result: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    hardware = hardware_info()
    start = time.perf_counter()
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    train, val, test = [(x.cuda().float(), y.cuda()) for x, y in (train_host, val_host, test_host)]
    backward = method.startswith("adam")
    if backward:
        if axis != "seconds":
            raise ValueError("backward controls have wall-time budgets only")
        model = hard_model(task, slope=cfg.get("slope"), ste=task == "int8").cuda().eval()
        base = {k: p.detach().clone() for k, p in model.named_parameters()}
        if method == "adam_subspace":
            sub, projections, _ = make_projection(model, seed + 10000, "cuda")
            active = sub.subspace_dim // 8 * 8
            z = nn.Parameter(torch.zeros(active, device="cuda"))

            def get_params():
                return reconstruct(base, sub, projections, F.pad(z, (0, sub.subspace_dim - active)))

            optimized = [z]
        else:
            active = sum(p.numel() for p in model.parameters())
            optimized = list(model.parameters())

            def get_params():
                return dict(model.named_parameters())

        optimizer = torch.optim.Adam(
            optimized, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999), eps=1e-8
        )
        engine = None
    else:
        engine = make_engine(task, method, cfg, seed, profile["chunk"], profile["compiled"])
        model, active, get_params = engine.model, engine.active, engine.params
    objective = "macro_accuracy" if task == "dvs" else "accuracy"
    best = measure(model, get_params(), val)
    best_params = {k: v.detach().cpu().clone() for k, v in get_params().items()}
    torch.save(best_params, path.with_suffix(".pt"))
    torch.cuda.synchronize()
    logs = [dict(step=0, evals=0, wall_seconds=time.perf_counter() - start, **best)]
    last_finite_params, last_finite_metric, last_finite_step = best_params, best, 0
    best_step, steps, evals, examples, next_validation = 0, 0, 0, 0, 36 if axis == "seconds" else 100_000
    gen = torch.Generator().manual_seed(seed + 20000)
    order, position = torch.empty(0, dtype=torch.long), 0
    batch_size = 32 if task == "dvs" else 128
    failure = None
    while time.perf_counter() - start < budget if axis == "seconds" else evals + engine.candidates_per_step <= budget:
        if position >= len(order):
            order, position = torch.randperm(len(train[1]), generator=gen), 0
        idx = order[position : position + batch_size]
        position += len(idx)
        batch = (train[0][idx], train[1][idx])
        try:
            if backward:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(torch.func.functional_call(model, get_params(), (batch[0],)), batch[1])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                if cfg.get("clip") is not None:
                    torch.nn.utils.clip_grad_norm_(optimized, cfg["clip"])
                if not all(torch.isfinite(p.grad).all() for p in optimized if p.grad is not None):
                    raise FloatingPointError("nonfinite derivative")
                optimizer.step()
                spent = 0
            else:
                engine.progress = (time.perf_counter() - start if axis == "seconds" else evals) / budget
                spent = engine.step(batch)
        except FloatingPointError as exc:
            if not backward:
                charged = engine.evaluations if method == "polystep_tuned" else engine.candidates_per_step
                evals += charged
                examples += len(idx) * charged
            failure = str(exc)
            break
        steps += 1
        evals += spent
        examples += len(idx) * (1 if backward else spent)
        torch.cuda.synchronize()
        value = time.perf_counter() - start if axis == "seconds" else evals
        if value >= next_validation or (axis == "evals" and evals + engine.candidates_per_step > budget):
            measured = measure(model, get_params(), val)
            if not math.isfinite(measured["loss"]):
                failure = "nonfinite validation loss"
                break
            candidate_params = {k: v.detach().cpu().clone() for k, v in get_params().items()}
            checkpoint = path.with_suffix(".candidate.pt")
            torch.save(candidate_params, checkpoint)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            eligible = axis == "evals" or elapsed <= budget
            log = dict(
                step=steps,
                evals=evals,
                processed_examples=examples,
                wall_seconds=elapsed,
                eligible=eligible,
                **measured,
            )
            logs.append(log)
            if eligible and math.isfinite(measured["loss"]):
                last_finite_params, last_finite_metric, last_finite_step = candidate_params, measured, steps
            if (
                eligible
                and math.isfinite(measured["loss"])
                and (measured[objective], -measured["loss"]) > (best[objective], -best["loss"])
            ):
                best, best_params, best_step = measured, candidate_params, steps
                checkpoint.replace(path.with_suffix(".pt"))
            atomic_json(path.with_suffix(".progress.json"), dict(config=fullcfg, seed=seed, logs=logs))
            next_validation = (int(value // (36 if axis == "seconds" else 100_000)) + 1) * (
                36 if axis == "seconds" else 100_000
            )
            print(json.dumps(dict(task=task, method=method, seed=seed, **log)), flush=True)
    if failure:
        best_params, best, best_step = last_finite_params, last_finite_metric, last_finite_step
        torch.save(best_params, path.with_suffix(".pt"))
    torch.cuda.synchronize()
    result = dict(
        config=fullcfg,
        seed=seed,
        stage=stage,
        budget=budget,
        axis=axis,
        evals=evals,
        steps=steps,
        forward_examples=examples,
        backward_passes=steps if backward else 0,
        wall_seconds=time.perf_counter() - start,
        best_validation=best,
        selected_step=best_step,
        active_dimension=active,
        failure=failure,
        logs=logs,
        profile=profile,
        unused_evaluation_budget=budget - evals if axis == "evals" else None,
        per_coordinate_sigma=cfg["rho"] / math.sqrt(active)
        if not backward and method not in ("polystep", "polystep_tuned")
        else None,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        torch_version=torch.__version__,
        device=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        **provenance,
        **hardware,
        runner_sha256=provenance["source_files"]["experiments/runners/run_practical.py"],
        compiled_fallback=None
        if backward
        else engine.evaluator._compile_failed or engine.evaluator._compile_forward_failed,
        perturbation_norm_definition="sqrt(E ||delta z||^2); per-coordinate scale rho/sqrt(active_dimension)",
        subspace_class="full parameter" if method == "adam_full" else type(sub if backward else engine.sub).__name__,
        polystep_source=__import__("polystep").__file__,
        evaluator_sha256=hashlib.sha256(
            Path(__import__("polystep.cost_nn", fromlist=["_"]).__file__).read_bytes()
        ).hexdigest(),
    )
    if stage == "final":
        test_start = time.perf_counter()
        result["test"] = measure(model, {k: v.cuda() for k, v in best_params.items()}, test)
        torch.cuda.synchronize()
        result["test_seconds"] = time.perf_counter() - test_start
    atomic_json(path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task", choices=("snn", "dvs", "int8"), required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", choices=("tune", "final"), default="tune")
    p.add_argument("--axis", choices=("evals", "seconds"), default="evals")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(
        a.task,
        a.method,
        json.loads(a.config),
        a.seed,
        a.stage,
        a.axis,
        dict(chunk=a.chunk, compiled=a.compiled),
        smoke=a.smoke,
    )
