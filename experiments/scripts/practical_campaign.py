"""Run all declared practical controls after the controlled campaign finishes."""

from __future__ import annotations
import argparse
import gc
import itertools
import json
import statistics
import subprocess
import sys
import time

from experiments.runners.run_controlled import (
    ROOT,
    RESULTS,
    TUNING_SEEDS,
    FINAL_SEEDS,
    atomic_json,
    config_id,
    source_manifest,
)
from experiments.runners.run_practical import OUT, make_engine, host_data, points
import torch


def groups():
    for task, axis, method in itertools.product(
        ("snn", "dvs"),
        ("evals", "seconds"),
        ("polystep_tuned", "polystep", "openai_es", "spsa", "mezo", "random_search", "eggroll"),
    ):
        yield task, axis, method
    for task, method in itertools.product(("snn", "dvs"), ("adam_full", "adam_subspace")):
        yield task, "seconds", method
    for method in ("polystep_tuned", "polystep", "mezo", "adam_full", "adam_subspace"):
        yield "int8", "seconds", method


def compute_pids():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}


def wait_free():
    while compute_pids():
        time.sleep(30)


def profile_key(task, method, cfg):
    return f"{task}_{method}_{cfg.get('population', 0)}"


@torch.inference_mode()
def profile(task, method, cfg):
    """Profile complete training updates, including candidate reconstruction."""
    path = OUT / "profiles" / f"{profile_key(task, method, cfg)}.json"
    provenance = source_manifest()
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("source_sha256") != provenance["source_sha256"]:
            raise ValueError(f"profiling source changed: {path}")
        return existing
    if method.startswith("adam"):
        # Candidate chunking does not apply to backward controls. Their forward
        # and backward work is measured by the actual time-budget runner.
        result = dict(
            selected=dict(chunk=1, compiled=False),
            rows=[],
            method=method,
            task=task,
            scope="eager exact hard-forward backward control",
            source_sha256=provenance["source_sha256"],
        )
        atomic_json(path, result)
        return result
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(2026)
    train, _, _ = host_data(task)
    batchsize = 32 if task == "dvs" else 128
    batch = (train[0][:batchsize].cuda().float(), train[1][:batchsize].cuda())
    rows = []
    start = time.perf_counter()
    for compiled, chunk in itertools.product((False, True), (1, 8, 32, 128, 512)):
        torch.manual_seed(2026)
        engine = make_engine(task, method, cfg, 2026, chunk, compiled)
        row = dict(chunk=chunk, compiled=compiled, warmup=20, timed=100)
        try:
            for _ in range(20):
                engine.step(batch)
            times = []
            for _ in range(100):
                torch.cuda.synchronize()
                before = time.perf_counter()
                engine.step(batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - before)
            row.update(
                median_seconds=statistics.median(times),
                compiled_fallback=engine.evaluator._compile_failed or engine.evaluator._compile_forward_failed,
                used_inplace=engine.sequential,
                used_batched_linear=engine.evaluator._batched_linear is not None,
            )
        except torch.cuda.OutOfMemoryError:
            row["failure"] = "out of memory"
        rows.append(row)
        atomic_json(path.with_suffix(".progress.json"), dict(task=task, method=method, config=cfg, rows=rows))
        print(row, flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    candidates = [r for r in rows if "failure" not in r]
    if not candidates:
        raise RuntimeError(f"no feasible backend: {task}/{method}")
    pick = min(candidates, key=lambda r: (r["median_seconds"], r["compiled"], r["chunk"]))
    result = dict(
        task=task,
        method=method,
        config=cfg,
        rows=rows,
        selected={k: pick[k] for k in ("chunk", "compiled")},
        total_seconds=time.perf_counter() - start,
        source=__import__("polystep").__file__,
        source_sha256=provenance["source_sha256"],
    )
    atomic_json(path, result)
    return result


def command_run(task, axis, method, cfg, seed, stage, setting):
    return [
        sys.executable,
        "-m",
        "experiments.runners.run_practical",
        "--task",
        task,
        "--method",
        method,
        "--axis",
        axis,
        "--stage",
        stage,
        "--config",
        json.dumps(cfg),
        "--seed",
        str(seed),
        "--chunk",
        str(setting["chunk"]),
    ] + (["--compiled"] if setting["compiled"] else [])


def result_path(task, axis, method, cfg, seed, stage):
    return (
        OUT
        / stage
        / task
        / axis
        / method
        / config_id(dict(task=task, method=method, axis=axis, **cfg))
        / f"{seed}.json"
    )


def campaign():
    manifest = [dict(task=t, axis=a, method=m, configs=list(points(t, m))) for t, a, m in groups()]
    atomic_json(OUT / "manifest.json", manifest)
    # First complete the already-running controlled campaign. A PID alone is not
    # a completion record, so a crashed predecessor leaves this queue waiting.
    controlled = json.loads((RESULTS / "campaign_process.json").read_text())
    while True:
        ready = True
        for arm in controlled["arms"]:
            selected = RESULTS / "selected" / f"{arm}.json"
            if not selected.exists():
                ready = False
                break
            cfg = json.loads(selected.read_text())["config"]
            if not all((RESULTS / "final" / arm / config_id(cfg) / f"{s}.json").exists() for s in FINAL_SEEDS):
                ready = False
                break
        if ready:
            break
        time.sleep(30)
    while not (RESULTS / "diagnostics/complete.json").exists():
        time.sleep(30)
    for group in manifest:
        task, axis, method = group["task"], group["axis"], group["method"]
        cells = group["configs"]
        for cfg in cells:
            key = profile_key(task, method, cfg)
            profile_path = OUT / "profiles" / f"{key}.json"
            if not profile_path.exists():
                wait_free()
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.scripts.practical_campaign",
                        "profile",
                        "--task",
                        task,
                        "--method",
                        method,
                        "--config",
                        json.dumps(cfg),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            profile_record = json.loads(profile_path.read_text())
            if profile_record.get("source_sha256") != source_manifest()["source_sha256"]:
                raise ValueError(f"profiling source changed: {profile_path}")
            setting = profile_record["selected"]
            for seed in TUNING_SEEDS:
                wait_free()
                subprocess.run(command_run(task, axis, method, cfg, seed, "tune", setting), cwd=ROOT, check=True)

        def score(cfg):
            rows = [json.loads(result_path(task, axis, method, cfg, seed, "tune").read_text()) for seed in TUNING_SEEDS]
            endpoint = "macro_accuracy" if task == "dvs" else "accuracy"
            return (
                -statistics.mean(r["best_validation"][endpoint] for r in rows),
                statistics.mean(r["best_validation"]["loss"] for r in rows),
                config_id(dict(task=task, method=method, axis=axis, **cfg)),
            )

        chosen = min(cells, key=score)
        atomic_json(OUT / "selected" / f"{task}_{axis}_{method}.json", dict(config=chosen, score=score(chosen)))
        setting = json.loads((OUT / "profiles" / f"{profile_key(task, method, chosen)}.json").read_text())["selected"]
        for seed in FINAL_SEEDS:
            wait_free()
            subprocess.run(command_run(task, axis, method, chosen, seed, "final", setting), cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "experiments.scripts.aggregate_practical"], cwd=ROOT, check=True)
    atomic_json(OUT / "complete.json", dict(comparisons=len(manifest), final_seeds=list(FINAL_SEEDS)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("command", choices=("campaign", "profile"))
    p.add_argument("--task")
    p.add_argument("--method")
    p.add_argument("--config")
    a = p.parse_args()
    if a.command == "profile":
        profile(a.task, a.method, json.loads(a.config))
    else:
        campaign()
