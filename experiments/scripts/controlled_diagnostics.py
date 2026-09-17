"""Paired weighting diagnostics at saved, validation-independent snapshots."""

from __future__ import annotations
import itertools
import json
import subprocess
import sys
import time

from experiments.runners.run_controlled import (
    ROOT,
    RESULTS,
    FINAL_SEEDS,
    RULES,
    atomic_json,
    block_candidates,
    config_id,
    directions,
    draw_vertices,
    load_data,
    make_projection,
    metrics,
    reconstruct,
    configs as arm_configs,
    source_manifest,
)
from experiments.runners.nondiff_models import SpikingMNISTNet
from polystep.cost_nn import NNCostEvaluator
import numpy as np
import torch


def ready():
    process = json.loads((RESULTS / "campaign_process.json").read_text())
    for arm in process["arms"]:
        path = RESULTS / "selected" / f"{arm}.json"
        if not path.exists():
            return False
        cfg = json.loads(path.read_text())["config"]
        if not all((RESULTS / "final" / arm / config_id(cfg) / f"{seed}.json").exists() for seed in FINAL_SEEDS):
            return False
    return True


@torch.inference_mode()
def run(geometry, normalization, seed, smoke=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    model = SpikingMNISTNet().cuda().eval()
    base = {k: p.detach().clone() for k, p in model.named_parameters()}
    sub, projections, _ = make_projection(model, seed + 10000, "cuda")
    _, _, data, _, _ = load_data("cuda")
    active = sub.subspace_dim // 8 * 8
    blocks = active // 8
    evaluator = NNCostEvaluator(model, torch.nn.CrossEntropyLoss(), use_inplace=False, compile_vmap=True)
    configs = {}
    for rule in RULES:
        arm = f"{geometry}_{rule}_{normalization}"
        selected = RESULTS / "selected" / f"{arm}.json"
        configs[rule] = next(arm_configs(arm)) if smoke else json.loads(selected.read_text())["config"]
    gen = torch.Generator(device="cuda").manual_seed(seed + 70000)
    background_gen = torch.Generator(device="cuda").manual_seed(seed + 80000)
    root = RESULTS / "diagnostics" / ("smoke" if smoke else f"{geometry}_{normalization}") / str(seed)
    root.mkdir(parents=True, exist_ok=True)

    def costs(z, vertices, rho, background):
        params = reconstruct(base, sub, projections, z + background)
        output = []
        for first in range(0, blocks, 32):
            offsets = (
                rho * vertices[first : first + 32] - background[:active].reshape(blocks, 8)[first : first + 32, None]
            )
            batch = block_candidates(params, sub, projections, first, offsets)
            output.append(evaluator.evaluate(batch, *data).reshape(-1, 16))
        return torch.cat(output)

    total_evals = 0
    draws = 2 if smoke else 64
    provenance = source_manifest()
    for source_index, source in enumerate(("softmax", "greedy")):
        cfg = configs[source]
        checkpoint = RESULTS / "final" / cfg["arm"] / config_id(cfg) / f"{seed}.pt"
        snapshots = (
            [dict(z=torch.zeros(sub.subspace_dim), step=0, evals=0)]
            if smoke
            else torch.load(checkpoint, map_location="cpu", weights_only=True)["snapshots"]
        )
        if not smoke and len(snapshots) != 20:
            raise ValueError(f"expected 20 snapshots: {checkpoint}")
        for index, snapshot in enumerate(snapshots):
            # Independent streams make each cell restartable without replay.
            gen.manual_seed(seed + 70000 + 10000 * source_index + 100 * index)
            background_gen.manual_seed(seed + 80000 + 10000 * source_index + 100 * index)
            dest = root / f"{source}_{index:02d}.npz"
            if dest.exists():
                total_evals += int(np.load(dest)["candidate_evaluations"])
                continue
            z = snapshot["z"].cuda()
            baseline = metrics(model, reconstruct(base, sub, projections, z), data)["loss"]
            norms, zeros, spreads, changes, updates = [], [], [], [], []
            background_updates, background_changes = [], []
            evaluations = 1
            for draw in range(draws):
                u = draw_vertices(blocks, geometry, gen, "cuda")
                c = costs(z, u, cfg["rho"], torch.zeros_like(z))
                evaluations += blocks * 16
                spreads.append((c.amax(-1) - c.amin(-1)).cpu().numpy())
                draw_norms, draw_zeros, draw_changes, draw_updates = [], [], [], []
                for rule in RULES:
                    raw = directions(c, u, rule, configs[rule]["tau"], False)
                    d = directions(c, u, rule, configs[rule]["tau"], normalization == "normalized")
                    next_z = z.clone()
                    next_z[:active] += cfg["ell"] * d.flatten()
                    loss = metrics(model, reconstruct(base, sub, projections, next_z), data)["loss"]
                    evaluations += 1
                    draw_norms.append(raw.norm(dim=-1).cpu().numpy())
                    draw_zeros.append((raw.norm(dim=-1) == 0).cpu().numpy())
                    draw_changes.append(loss - baseline)
                    draw_updates.append(d.cpu().numpy())
                norms.append(draw_norms)
                zeros.append(draw_zeros)
                changes.append(draw_changes)
                updates.append(draw_updates)
                if geometry == "orthoplex" and normalization == "natural":
                    g = torch.randn(blocks, 8, generator=background_gen, device="cuda")
                    radius = torch.rand(blocks, 1, generator=background_gen, device="cuda").pow(1 / 8)
                    b = torch.zeros_like(z)
                    b[:active] = (cfg["rho"] * radius * g / g.norm(dim=-1, keepdim=True)).flatten()
                    cb = costs(z, u, cfg["rho"], b)
                    d = directions(cb, u, "softmax", configs["softmax"]["tau"])
                    next_z = z.clone()
                    next_z[:active] += cfg["ell"] * d.flatten()
                    loss = metrics(model, reconstruct(base, sub, projections, next_z), data)["loss"]
                    evaluations += blocks * 16 + 1
                    background_updates.append(d.cpu().numpy())
                    background_changes.append(loss - baseline)
            if not np.isfinite(changes).all() or not np.isfinite(norms).all():
                raise FloatingPointError("nonfinite diagnostic measurement")
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                rules=np.array(RULES),
                norms=norms,
                zero_rows=zeros,
                cost_spreads=spreads,
                loss_changes=changes,
                directions=updates,
                background_directions=background_updates,
                background_loss_changes=background_changes,
                rho=cfg["rho"],
                ell=cfg["ell"],
                snapshot_step=snapshot["step"],
                snapshot_evals=snapshot["evals"],
                weighting_scales=[configs[r]["tau"] for r in RULES],
                source_config=json.dumps(cfg, sort_keys=True),
                source_sha256=provenance["source_sha256"],
                candidate_evaluations=evaluations,
                baseline_loss=baseline,
            )
            tmp.replace(dest)
            total_evals += evaluations
            print(geometry, normalization, seed, source, index, "diagnostic evaluations", evaluations, flush=True)
    atomic_json(
        root / "complete.json",
        dict(
            seed=seed,
            geometry=geometry,
            normalization=normalization,
            snapshots=2 if smoke else 40,
            draws_per_snapshot=draws,
            diagnostic_evaluations=total_evals,
            smoke=smoke,
            **provenance,
            step_and_probe_scales="source trajectory",
            weighting_scales="independently selected per rule",
        ),
    )


def campaign():
    while not ready():
        time.sleep(30)
    subprocess.run([sys.executable, "-m", "experiments.scripts.aggregate_controlled"], cwd=ROOT, check=True)
    for geometry, normalization, seed in itertools.product(
        ("orthoplex", "antipodal"), ("natural", "normalized"), FINAL_SEEDS
    ):
        done = RESULTS / "diagnostics" / f"{geometry}_{normalization}" / str(seed) / "complete.json"
        if not done.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.scripts.controlled_diagnostics",
                    geometry,
                    normalization,
                    str(seed),
                ],
                cwd=ROOT,
                check=True,
            )
    subprocess.run(
        [sys.executable, "-m", "experiments.scripts.aggregate_controlled", "--diagnostics"], cwd=ROOT, check=True
    )
    atomic_json(RESULTS / "diagnostics/complete.json", dict(cells=40, snapshots_per_cell=40, draws_per_snapshot=64))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        campaign()
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), smoke="--smoke" in sys.argv)
