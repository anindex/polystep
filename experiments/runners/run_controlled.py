"""Fixed-protocol PolyStep mechanism experiments. Run from the repository root."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
import torch.nn.functional as F

from experiments.runners.nondiff_models import SpikingMNISTNet
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace, _stable_entry_seed
from polystep.transform import ParamLayout

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/controlled"
RULES = ("softmax", "linear", "rank", "greedy", "top4")
TUNING_SEEDS = (2027, 2028, 2029)
FINAL_SEEDS = tuple(range(1000, 1010))


def source_manifest():
    """Capture source bytes before a run and support extracted supplements."""
    paths = sorted((ROOT / "src/polystep").rglob("*.py"))
    paths += sorted((ROOT / "experiments/runners").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return dict(
        code_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        source_files=hashes,
        source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    )


def hardware_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,power.limit,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    return dict(
        gpu_configuration=result.stdout.strip(),
        gpu_query_error=result.stderr.strip() if result.returncode else None,
        precision="float32; highest matmul precision; TF32 disabled",
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def config_id(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def grid(rule):
    for rho in (0.001, 0.01, 0.1):
        if rule in ("greedy", "top4", "linear_normalized"):
            for k in range(-4, 5):
                yield dict(rho=rho, ell=rho * 10 ** (k / 4), tau=1.0)
        else:
            temperatures = (0.05, 0.2, 1.0) if rule == "rank" else (0.001, 0.01, 0.1)
            for ratio, tau in itertools.product((0.1, 1.0, 10.0), temperatures):
                yield dict(rho=rho, ell=rho * ratio, tau=tau)


def directions(cost, vertices, rule, tau, normalize=False):
    """Tie-invariant rules, including exactly zero constant-cost rows."""
    n = cost.shape[-1]
    if rule == "linear":
        w = -(cost - cost.mean(-1, keepdim=True)) / (n * tau)
    elif rule == "softmax":
        w = torch.softmax(-(cost - cost.amin(-1, keepdim=True)) / tau, -1)
    elif rule == "rank":
        less = (cost.unsqueeze(-2) < cost.unsqueeze(-1)).sum(-1)
        equal = (cost.unsqueeze(-2) == cost.unsqueeze(-1)).sum(-1)
        rank = (less.to(cost.dtype) + (equal.to(cost.dtype) - 1) / 2) / (n - 1)
        w = torch.softmax(-rank / tau, -1)
    elif rule == "greedy":
        w = (cost == cost.amin(-1, keepdim=True)).to(cost.dtype)
        w = w / w.sum(-1, keepdim=True)
    elif rule == "top4":
        cutoff = cost.kthvalue(4, dim=-1).values.unsqueeze(-1)
        below, tied = cost < cutoff, cost == cutoff
        fraction = (4 - below.sum(-1, keepdim=True)).to(cost.dtype) / tied.sum(-1, keepdim=True)
        w = (below.to(cost.dtype) + tied * fraction) / 4
    else:
        raise ValueError(rule)
    # Pair subtraction preserves exact cancellation for tied antipodes.
    d = torch.einsum("pv,pvd->pd", w[:, :8] - w[:, 8:], vertices[:, :8])
    # An explicit constant-row mask prevents roundoff from causing motion.
    d = torch.where((cost.amax(-1) == cost.amin(-1)).unsqueeze(-1), 0.0, d)
    if normalize:
        norm = d.norm(dim=-1, keepdim=True)
        d = d / torch.where(norm > 0, norm, 1.0)
    return d


def draw_vertices(blocks, geometry, generator, device):
    z = torch.randn(blocks, 8, 8, generator=generator, device=device)
    if geometry == "orthoplex":
        q, r = torch.linalg.qr(z)
        q = q * torch.where(r.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).unsqueeze(-2)
        # Rows of a Haar orthogonal matrix are an orthonormal frame too.
    elif geometry == "antipodal":
        q = z / z.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(geometry)
    return torch.cat((q, -q), dim=1)


def load_data(device):
    from torchvision.datasets import MNIST

    train = MNIST(ROOT / "data/mnist", train=True, download=False)
    test = MNIST(ROOT / "data/mnist", train=False, download=False)
    gen = torch.Generator().manual_seed(0)
    order = torch.randperm(60000, generator=gen)
    pool, val = order[:54000], order[54000:]
    diag_order = torch.randperm(54000, generator=gen)
    diag, update = pool[diag_order[:1024]], pool[diag_order[1024:]]
    split = dict(train=update.tolist(), validation=val.tolist(), diagnostic=diag.tolist())
    atomic_json(RESULTS / "mnist_split.json", split)
    x = ((train.data.to(device).float() / 255 - 0.1307) / 0.3081).reshape(-1, 784)
    y = train.targets.to(device)
    tx = ((test.data.to(device).float() / 255 - 0.1307) / 0.3081).reshape(-1, 784)
    return (x[update], y[update]), (x[val], y[val]), (x[diag], y[diag]), (tx, test.targets.to(device)), split


def make_projection(model, seed, device):
    sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4, seed=seed, sparse_threshold_bytes=10**15)
    projections = {}
    errors = {}
    for spec in sub.specs:
        if not spec.is_projected:
            continue
        gen = torch.Generator().manual_seed(_stable_entry_seed(seed, spec.entry_key, 0))
        raw = torch.randn(spec.num_params, spec.num_coords, generator=gen).to(device)
        q, _ = torch.linalg.qr(raw, mode="reduced")
        error = (q.T @ q - torch.eye(q.shape[1], device=device)).abs().max().item()
        if error > 2e-5:
            raise RuntimeError(f"nonorthogonal projection: {spec.entry_key}: {error}")
        projections[spec.entry_key] = q
        errors[spec.entry_key] = error
    return sub, projections, errors


def reconstruct(base, sub, projections, z):
    return sub.apply_perturbation(projections, base, z)


def block_candidates(base, sub, projections, first, offsets):
    """Reconstruct only the eight changed coordinates per block."""
    groups, vertices, dim = offsets.shape
    device = offsets.device
    indices = torch.arange(first * dim, (first + groups) * dim, device=device).reshape(groups, dim)
    out = {}
    for spec in sub.specs:
        mask = (indices >= spec.flat_start) & (indices < spec.flat_end)
        if (first + groups) * dim <= spec.flat_start or first * dim >= spec.flat_end:
            out[spec.entry_key] = base[spec.entry_key].expand(groups * vertices, *spec.original_shape)
            continue
        local = (indices - spec.flat_start).clamp(0, spec.num_coords - 1)
        values = offsets * mask.unsqueeze(1)
        if spec.is_projected:
            columns = projections[spec.entry_key].T[local]
            delta = torch.bmm(values, columns).reshape(groups * vertices, -1)
        else:
            delta = values.new_zeros(groups, vertices, spec.num_coords)
            delta.scatter_add_(2, local.unsqueeze(1).expand(-1, vertices, -1), values)
            delta = delta.flatten(0, 1)
        out[spec.entry_key] = (delta + base[spec.entry_key].reshape(1, -1)).reshape(
            groups * vertices, *spec.original_shape
        )
    return out


@torch.no_grad()
def metrics(model, params, data):
    loss, correct, total = 0.0, 0, 0
    x, y = data
    for start in range(0, len(y), 512):
        target = y[start : start + 512]
        output = torch.func.functional_call(model, params, (x[start : start + 512],))
        loss += F.cross_entropy(output, target, reduction="sum").item()
        correct += (output.argmax(-1) == target).sum().item()
        total += len(target)
    return dict(accuracy=correct / total, loss=loss / total)


def smooth_jitter(shape, generator, device):
    # Rejection sampling from the compactly supported smooth mollifier.
    result = torch.empty(shape, device=device)
    remaining = torch.ones(shape, dtype=torch.bool, device=device)
    while remaining.any():
        u = 2 * torch.rand(shape, generator=generator, device=device) - 1
        accept = torch.rand(shape, generator=generator, device=device) < torch.exp(1 - 1 / (1 - u.square()))
        take = remaining & accept
        result[take] = 0.05 * u[take]
        remaining &= ~take
    return result


@torch.no_grad()
def run(cfg, seed, stage, *, budget=None, chunk=128, compiled=False, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run with GPU access outside the sandbox")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    budget = budget or (2_000_000 if stage == "tune" else 10_000_000)
    path = RESULTS / stage / cfg["arm"] / config_id(cfg) / f"{seed}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if (
            record["budget"] != budget
            or record["config"] != cfg
            or record["chunk"] != chunk
            or record["compiled_requested"] != compiled
        ):
            raise ValueError(f"incompatible existing result: {path}")
        return record
    provenance = source_manifest()
    hardware = hardware_info() if device == "cuda" else {}
    started = time.perf_counter()
    train, val, diagnostic, test, split = load_data(device)
    model = SpikingMNISTNet().to(device).eval()
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    sub, projections, errors = make_projection(model, seed + 10_000, device)
    active = sub.subspace_dim // 8 * 8
    blocks = active // 8
    evals_per_step = blocks * 16
    steps = budget // evals_per_step
    if steps < 1:
        raise ValueError(f"budget must fit {evals_per_step} candidates")
    z = torch.zeros(sub.subspace_dim, device=device)
    data_gen = torch.Generator().manual_seed(seed + 20_000)
    probe_gen = torch.Generator(device=device).manual_seed(seed + 30_000)
    background_gen = torch.Generator(device=device).manual_seed(seed + 40_000)
    jitter_gen = torch.Generator(device=device).manual_seed(seed + 50_000)
    output_gen = torch.Generator().manual_seed(seed + 60_000)
    evaluator = NNCostEvaluator(
        model, torch.nn.CrossEntropyLoss(), compile_vmap=compiled, compile_forward=False, use_inplace=False
    )
    schedule = cfg.get("schedule")
    multipliers = (
        torch.arange(1, steps + 1, dtype=torch.float64).pow(-0.6)
        if schedule == "decay"
        else torch.ones(steps, dtype=torch.float64)
    )
    output_index = torch.multinomial(multipliers, 1, generator=output_gen).item()
    best = metrics(model, base, val)
    best_z, best_step = z.clone(), 0
    last_finite_z, last_finite_metric, last_finite_step = best_z, best, 0
    logs = [dict(step=0, evals=0, processed_examples=0, wall_seconds=time.perf_counter() - started, **best)]
    order, position, evals, processed = torch.empty(0, dtype=torch.long), 0, 0, 0
    next_val, snapshot_index = 100_000, 1
    snapshots, output_z = [], None
    failure = None
    torch.cuda.synchronize() if device == "cuda" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        if step == output_index:
            output_z = z.clone()
        if schedule:
            idx = torch.randint(len(train[1]), (128,), generator=data_gen)
        else:
            if position >= len(order):
                order, position = torch.randperm(len(train[1]), generator=data_gen), 0
            idx = order[position : position + 128]
            position += len(idx)
        x, y = train[0][idx], train[1][idx]
        u = draw_vertices(blocks, cfg["geometry"], probe_gen, device)
        rho, ell = cfg["rho"], cfg["ell"]
        if schedule:
            rho *= 1 + smooth_jitter((), jitter_gen, device)
            rs = cfg["a"] / math.sqrt(steps) if schedule == "constant" else cfg["a"] * (step + 1) ** -0.6
            ell = cfg["tau"] * rs * (1 + smooth_jitter((blocks, 1), jitter_gen, device))
        background = torch.zeros_like(z)
        if cfg.get("background"):
            g = torch.randn(blocks, 8, generator=background_gen, device=device)
            radius = torch.rand(blocks, 1, generator=background_gen, device=device).pow(1 / 8)
            background[:active] = (cfg["rho"] * radius * g / g.norm(dim=-1, keepdim=True)).flatten()
        params = reconstruct(base, sub, projections, z + background)
        costs = []
        groups_per_chunk = max(1, chunk // 16)
        for first in range(0, blocks, groups_per_chunk):
            count = min(groups_per_chunk, blocks - first)
            offsets = (
                rho * u[first : first + count] - background[:active].reshape(blocks, 8)[first : first + count, None]
            )
            candidate = block_candidates(params, sub, projections, first, offsets)
            costs.append(evaluator.evaluate(candidate, x, y).reshape(count, 16))
        cost = torch.cat(costs)
        evals += evals_per_step
        processed += evals_per_step * len(idx)
        if not torch.isfinite(cost).all():
            failure = "nonfinite candidate cost"
            break
        d = directions(cost, u, cfg["rule"], cfg["tau"], cfg["normalize"])
        new_z = z.clone()
        new_z[:active] += (ell * d).flatten()
        if not torch.isfinite(new_z).all():
            failure = "nonfinite update"
            break
        z = new_z
        if stage == "final" and (step + 1) * 20 >= snapshot_index * steps:
            snapshots.append(dict(step=step + 1, evals=evals, z=z.cpu()))
            snapshot_index += 1
        if evals >= next_val or step + 1 == steps:
            measured = metrics(model, reconstruct(base, sub, projections, z), val)
            if not math.isfinite(measured["loss"]):
                failure = "nonfinite validation loss"
                break
            last_finite_z, last_finite_metric, last_finite_step = z.clone(), measured, step + 1
            if (measured["accuracy"], -measured["loss"]) > (best["accuracy"], -best["loss"]):
                best, best_z, best_step = measured, z.clone(), step + 1
            log = dict(
                step=step + 1,
                evals=evals,
                processed_examples=processed,
                wall_seconds=time.perf_counter() - started,
                **measured,
            )
            logs.append(log)
            next_val = (evals // 100_000 + 1) * 100_000
            atomic_json(path.with_suffix(".progress.json"), dict(config=cfg, seed=seed, budget=budget, logs=logs))
            print(json.dumps(dict(arm=cfg["arm"], config=config_id(cfg), seed=seed, **log)), flush=True)
    if failure:
        best_z, best, best_step = last_finite_z, last_finite_metric, last_finite_step
    elapsed = time.perf_counter() - started
    record = dict(
        config=cfg,
        seed=seed,
        stage=stage,
        budget=budget,
        evals=evals,
        unused_budget=budget - evals,
        steps=evals // evals_per_step,
        processed_examples=processed,
        best_validation=best,
        selected_step=best_step,
        logs=logs,
        wall_seconds=elapsed,
        failure=failure,
        sampler="independent_with_replacement" if schedule else "shuffled_epochs",
        active_dimension=active,
        frozen_coordinates=sub.subspace_dim - active,
        projection_orthogonality_error=errors,
        chunk=chunk,
        compiled_requested=compiled,
        compiled_fallback=evaluator._compile_failed,
        torch_version=torch.__version__,
        **provenance,
        **hardware,
        runner_sha256=provenance["source_files"]["experiments/runners/run_controlled.py"],
        polystep_source=__import__("polystep").__file__,
        evaluator_sha256=hashlib.sha256(
            Path(__import__("polystep.cost_nn", fromlist=["_"]).__file__).read_bytes()
        ).hexdigest(),
        split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
    )
    if device == "cuda":
        record.update(
            device=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )
    if stage == "final":
        record["test"] = metrics(model, reconstruct(base, sub, projections, best_z), test)
        if output_z is not None and schedule:
            record["random_output_validation"] = metrics(model, reconstruct(base, sub, projections, output_z), val)
            record["random_output_step"] = output_index
        torch.save(dict(snapshots=snapshots, selected_z=best_z.cpu(), last_z=z.cpu()), path.with_suffix(".pt"))
    atomic_json(path, record)
    return record


def configs(arm):
    if arm == "joint_softmax_natural":
        for point in grid("softmax"):
            yield dict(arm=arm, geometry="orthoplex", rule="softmax", normalize=False, background=True, **point)
        return
    if arm in ("schedule_constant", "schedule_decay"):
        for tau, rho, a in itertools.product((0.01, 0.1, 1.0), (0.001, 0.01, 0.1), (0.1, 1.0, 10.0)):
            yield dict(
                arm=arm,
                geometry="orthoplex",
                rule="softmax",
                normalize=False,
                schedule=arm.split("_")[1],
                tau=tau,
                rho=rho,
                a=a,
                ell=1.0,
            )
        return
    geometry, rule, length = arm.split("_")
    normalized = length == "normalized"
    for point in grid("linear_normalized" if rule == "linear" and normalized else rule):
        yield dict(arm=arm, geometry=geometry, rule=rule, normalize=normalized, **point)


def campaign(arms, chunk, compiled):
    """Fresh process per run; completed records make interruption restartable."""
    for arm in arms:
        cells = [{**cfg, "execution": dict(chunk=chunk, compiled=compiled, source="checkout")} for cfg in configs(arm)]
        manifest = RESULTS / "configs" / f"{arm}.json"
        atomic_json(manifest, cells)
        for cfg, seed in itertools.product(cells, TUNING_SEEDS):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.runners.run_controlled",
                    "run",
                    "--config",
                    json.dumps(cfg),
                    "--seed",
                    str(seed),
                    "--stage",
                    "tune",
                    "--chunk",
                    str(chunk),
                ]
                + (["--compiled"] if compiled else []),
                cwd=ROOT,
                check=True,
            )

        def score(cfg):
            records = [
                json.loads((RESULTS / "tune" / arm / config_id(cfg) / f"{seed}.json").read_text())
                for seed in TUNING_SEEDS
            ]
            return (
                -sum(r["best_validation"]["accuracy"] for r in records) / 3,
                sum(r["best_validation"]["loss"] for r in records) / 3,
                config_id(cfg),
            )

        selected = min(cells, key=score)
        atomic_json(RESULTS / "selected" / f"{arm}.json", dict(config=selected, score=score(selected)))
        for seed in FINAL_SEEDS:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.runners.run_controlled",
                    "run",
                    "--config",
                    json.dumps(selected),
                    "--seed",
                    str(seed),
                    "--stage",
                    "final",
                    "--chunk",
                    str(chunk),
                ]
                + (["--compiled"] if compiled else []),
                cwd=ROOT,
                check=True,
            )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=("run", "campaign"))
    p.add_argument("--config")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--stage", choices=("tune", "final", "smoke"), default="tune")
    p.add_argument("--budget", type=int)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--arms", nargs="+", default=["orthoplex_softmax_natural"])
    a = p.parse_args()
    if a.command == "run":
        cfg = json.loads(a.config) if a.config else next(configs(a.arms[0]))
        run(cfg, a.seed, a.stage, budget=a.budget, chunk=a.chunk, compiled=a.compiled)
    else:
        campaign(a.arms, a.chunk, a.compiled)


if __name__ == "__main__":
    main()
