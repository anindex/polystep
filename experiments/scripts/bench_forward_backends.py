"""Candidate-forward parity, cold cost, throughput and memory on CPU or CUDA.

PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --device cpu
Add --compile to include compiler warmup; --prefix measures complete MLP steps.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from torch import nn

from experiments.runners.search_suite import WORKLOADS, synchronize, task
from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator, SiteVmapEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


def measure(fn, device, warmup, repeats):
    synchronize(device)
    before = time.perf_counter()
    fn()
    synchronize(device)
    cold = time.perf_counter() - before
    for _ in range(warmup):
        fn()
    synchronize(device)
    if torch.device(device).type == "cuda":
        base_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        before = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - before)
    q = statistics.quantiles(times, n=4)
    return dict(
        cold_seconds=cold,
        median_seconds=statistics.median(times),
        iqr_seconds=q[2] - q[0],
        peak_extra_bytes=torch.cuda.max_memory_allocated(device) - base_memory
        if torch.device(device).type == "cuda"
        else None,
    )


def profile_call(fn, device):
    """Profile outside timed samples; report executed operators and compiled regions."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        fn()
    events = prof.key_averages()
    return dict(
        compiled_regions=sum(e.count for e in events if "Torch-Compiled Region" in e.key),
        operators=[
            dict(
                name=e.key,
                calls=e.count,
                self_cpu_us=e.self_cpu_time_total,
                self_device_us=getattr(e, "self_device_time_total", 0),
                self_cpu_bytes=e.self_cpu_memory_usage,
            )
            for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:20]
        ],
    )


@torch.inference_mode()
def candidate_bench(args):
    rows = []
    for name in args.workloads or WORKLOADS:
        for seed in args.seeds:
            model, (train, _, _) = task(name, seed, args.device, args.batch)
            x, y = train
            layout = ParamLayout.from_module(model)
            base = {k: p.detach().clone() for k, p in model.named_parameters()}
            # Sample an early large site and the last site; prefix work differs sharply.
            keys = dict.fromkeys((max(base, key=lambda k: base[k].numel()), next(reversed(base))))
            for key in keys:
                site = base[key][None] + 0.02 * torch.randn(args.candidates, *base[key].shape, device=args.device)
                stacked = {k: v[None].expand(args.candidates, *v.shape) for k, v in base.items()} | {key: site}
                reference_ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
                expected = reference_ev._evaluate_loop(stacked, x, y)
                backends = ["evaluator", "site", "inplace"] + (
                    ["compiled_evaluator", "compiled_site"] if args.compile else []
                )
                for backend in backends:
                    # Independent backends must not exhaust a shared Dynamo code
                    # object's recompilation budget before the next trial starts.
                    if args.compile:
                        torch.compiler.reset()
                    ev = NNCostEvaluator(
                        model,
                        nn.CrossEntropyLoss(),
                        use_inplace=backend == "inplace",
                        compile_vmap=backend.startswith("compiled"),
                        compile_forward=False,
                    )
                    site_ev = SiteVmapEvaluator.try_build(ev, layout)

                    def fn():
                        if backend.endswith("site"):
                            return site_ev._vmap_over_site(key, base, site, x, y)
                        return ev.evaluate(stacked, x, y)

                    row = dict(
                        workload=name,
                        seed=seed,
                        site=key,
                        backend=backend,
                        device=args.device,
                        torch=torch.__version__,
                        candidates=args.candidates,
                        batch=args.batch,
                        threads=args.threads,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    try:
                        row.update(measure(fn, args.device, args.warmup, args.repeats))
                        got = fn()
                        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
                        row.update(status="ok", max_loss_error=float((got - expected).abs().max()))
                        row["compile_fallback"] = bool(
                            ev._compile_failed or (site_ev and site_ev._compile_failed_sites)
                        )
                        row["actual_path"] = (
                            "site-vmap"
                            if backend.endswith("site")
                            else "inplace"
                            if ev._use_inplace
                            else "bmm"
                            if ev._batched_linear is not None
                            else "loop"
                            if ev._vmap_failed
                            else "vmap"
                        )
                        row["compiled_callable"] = (
                            bool(site_ev._compiled_sites)
                            if backend.endswith("site")
                            else ev._compiled_vmap_fn is not None
                        )
                        if args.profile:
                            row["profile"] = profile_call(fn, args.device)
                    except Exception as error:
                        row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


@torch.inference_mode()
def prefix_bench(args):
    rows = []
    for width in (64, 256):
        for dimension in (0, 512):
            for seed in args.seeds:
                reference = None
                for reuse in (False, True):
                    torch.manual_seed(seed)
                    model = nn.Sequential(
                        nn.Linear(32, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4)
                    ).to(args.device)
                    x = torch.randn(args.batch, 32, device=args.device)
                    y = torch.randint(0, 4, (args.batch,), device=args.device)
                    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
                    sub = (
                        HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4, max_subspace_dim=dimension)
                        if dimension
                        else None
                    )
                    opt = PolyStepOptimizer(
                        model, subspace=sub, solver="softmax", chunk_size=72, compile=False, seed=seed
                    )
                    opt.register_evaluator(ev, x, y)

                    def fn():
                        return opt.step(lambda p: ev.evaluate(p, x, y))

                    context = (
                        nullcontext()
                        if reuse
                        else patch("polystep._step_monolithic._reuse_prefixes", lambda *a: lambda f: f)
                    )
                    with context:
                        row = measure(fn, args.device, args.warmup, args.repeats)
                        if args.profile:
                            row["profile"] = profile_call(fn, args.device)
                    if reference is None:
                        reference = opt.state.X.clone()
                    else:
                        torch.testing.assert_close(opt.state.X, reference, rtol=0, atol=0)
                    row.update(
                        workload="mlp_step",
                        width=width,
                        subspace=dimension,
                        seed=seed,
                        prefix_reuse=reuse,
                        device=args.device,
                        batch=args.batch,
                        threads=args.threads,
                        torch=torch.__version__,
                        status="ok",
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows


@torch.inference_mode()
def attention_bench(args):
    rows = []
    for seq in (6, 64, 128):
        torch.manual_seed(0)
        q = torch.randn(args.candidates, args.batch, 2, seq, 8, device=args.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def explicit(q, k, v):
            scores = (q @ k.transpose(-1, -2)) / 8**0.5
            return scores.softmax(-1) @ v

        reference = torch.vmap(explicit)(q, k, v)
        for name, kernel in (("explicit", explicit), ("sdpa", nn.functional.scaled_dot_product_attention)):

            def fn():
                return torch.vmap(kernel)(q, k, v)

            row = measure(fn, args.device, args.warmup, args.repeats)
            torch.testing.assert_close(fn(), reference, rtol=1e-5, atol=1e-6)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fn()
            kernels = [e.key for e in prof.key_averages() if "attention" in e.key]
            row.update(
                workload="attention_kernel",
                backend=name,
                sequence=seq,
                device=args.device,
                candidates=args.candidates,
                batch=args.batch,
                operators=kernels,
                torch=torch.__version__,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workloads", nargs="+", choices=WORKLOADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("experiments/results/benchmarks/forward.json"))
    args = ap.parse_args()
    if args.repeats < 2 or min(args.batch, args.candidates, args.threads) < 1 or args.warmup < 0:
        ap.error("repeats must be >=2; batch, candidates and threads positive; warmup nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; run on CPU or restore GPU access")
    torch.set_num_threads(args.threads)
    rows = prefix_bench(args) if args.prefix else attention_bench(args) if args.attention else candidate_bench(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if any(row.get("status") == "failed" for row in rows):
        raise SystemExit("Some backends failed; inspect the result file")


if __name__ == "__main__":
    main()
