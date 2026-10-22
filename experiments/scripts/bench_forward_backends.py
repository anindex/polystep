"""Candidate-forward parity, cold cost, throughput and memory on CPU or CUDA.

PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --device cpu
Add --compile to include compiler warmup; --prefix measures complete MLP steps (--dims
sets the layer widths, --parity checks one step against the closure-only path).
--step times complete optimizer steps on the search workloads, --kernels times solver
and geometry kernels (--dims sets the QR rows), and --projection measures sparse
projections at the full dimensions given by --dims.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import warnings

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from experiments.runners.search_suite import WORKLOADS, synchronize, task
from polystep import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cost_nn import (
    BatchedLinearEvaluator,
    FactoredEvaluator,
    NNCostEvaluator,
    SiteInplaceEvaluator,
    SiteVmapEvaluator,
    SparseDeltaEvaluator,
    SubspaceDeltaEvaluator,
)
from polystep.factored_subspace import FactoredSubspace
from polystep.geometry import apply_biased_rotation, get_random_rotation_matrices, get_simplex_vertices
from polystep.hybrid_subspace import HybridSubspace
from polystep.projection.sparse import SparseRandomProjection
from polystep.solver import PolyStep
from polystep.solvers._shared import thin_qr
from polystep.solvers.sinkhorn import SinkhornSolver
from polystep.transform import ParamLayout

SCORERS = (
    (NNCostEvaluator, "evaluate"),
    (NNCostEvaluator, "evaluate_subspace_inplace"),
    (BatchedLinearEvaluator, "evaluate"),
    (SiteVmapEvaluator, "evaluate"),
    (SiteVmapEvaluator, "evaluate_subspace"),
    (SiteInplaceEvaluator, "evaluate"),
    (SiteInplaceEvaluator, "evaluate_subspace"),
    (SparseDeltaEvaluator, "evaluate"),
    (SubspaceDeltaEvaluator, "evaluate"),
    (FactoredEvaluator, "evaluate"),
)


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


def checksum(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def count_syncs(fn, device):
    """Count host syncs in one call through CUDA's sync debug mode; None off CUDA."""
    if torch.device(device).type != "cuda":
        return None
    synchronize(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("called a synchronizing" in str(w.message) for w in caught)


@contextmanager
def trace_scorers(used):
    """Record which evaluator classes score candidates while active."""
    with ExitStack() as stack:
        for cls, name in SCORERS:

            def spy(self, *args, _original=getattr(cls, name), _cls=cls.__name__, **kwargs):
                used.add(_cls)
                return _original(self, *args, **kwargs)

            stack.enter_context(patch.object(cls, name, spy))
        yield


@contextmanager
def capture_costs(opt, captured):
    """Copy every cost matrix the step hands to the fused softmax or to solver.solve."""
    solve, fused = opt.solver.solve, opt._compiled.fused_softmax_project

    def solve_spy(**kwargs):
        captured.append(kwargs["cost_matrix"].clone())
        return solve(**kwargs)

    def fused_spy(cost, *args, **kwargs):
        captured.append(cost.clone())
        return fused(cost, *args, **kwargs)

    with patch.object(opt.solver, "solve", solve_spy), patch.object(opt._compiled, "fused_softmax_project", fused_spy):
        yield


def build_optimizer(model, kind, args, cap=None, **kwargs):
    layout = ParamLayout.from_module(model)
    rank = args.rank or 4
    if kind == "full":
        sub = None
    elif kind == "hybrid":
        sub = HybridSubspace.from_layout(layout, rank=rank, max_subspace_dim=cap)
    elif kind == "factored":
        sub = FactoredSubspace.from_layout(layout, rank=rank)
    else:
        sub = AdaptiveSubspace.from_layout(layout, rank=rank)
    if args.pdim is not None:
        kwargs["particle_dim" if sub is None else "subspace_particle_dim"] = args.pdim
    return PolyStepOptimizer(model, subspace=sub, **kwargs)


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
                entry = next(e for e in layout.entries if e.key == key)
                site = base[key][None] + 0.02 * torch.randn(args.candidates, *base[key].shape, device=args.device)
                stacked = {k: v[None].expand(args.candidates, *v.shape) for k, v in base.items()} | {key: site}
                reference_ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
                expected = reference_ev._evaluate_loop(stacked, x, y)
                backends = ["evaluator", "site", "site_inplace", "inplace"] + (
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
                        use_inplace=backend in ("inplace", "site_inplace"),
                        compile_vmap=backend.startswith("compiled"),
                        compile_forward=False,
                    )
                    site_ev = SiteVmapEvaluator.try_build(ev, layout)

                    def fn():
                        if backend == "site_inplace":
                            n = base[key].numel()
                            return site_ev.evaluate(
                                base,
                                entry,
                                torch.arange(n, device=args.device)[None],
                                site.reshape(1, args.candidates, n),
                                x,
                                y,
                            )
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
                            "site-inplace"
                            if backend == "site_inplace"
                            else "site-vmap"
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


def mlp_case(args, dims, kind, solver, seed, use_inplace, cap):
    torch.manual_seed(seed)
    model = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.ReLU())][:-1])
    model = model.to(args.device)
    x = torch.randn(args.batch, dims[0], device=args.device)
    y = torch.randint(0, dims[-1], (args.batch,), device=args.device)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=use_inplace)
    opt = build_optimizer(model, kind, args, cap, solver=solver, chunk_size=72, compile=args.compile, seed=seed)
    return model, opt, ev, x, y


def parity(args, dims, kind, solver, seed, cap):
    """One step from the same seed, registered fast path against the closure-only path."""
    costs, weights = [], []
    for registered in (True, False):
        model, opt, ev, x, y = mlp_case(
            args, dims, kind, solver, seed, (args.inplace or None) if registered else False, cap
        )
        if registered:
            opt.register_evaluator(ev, x, y)
        captured = []
        with capture_costs(opt, captured):
            opt.step(lambda p: ev.evaluate(p, x, y))
        if not captured:
            raise RuntimeError(f"{kind}/{solver}: no cost matrix reached the fused softmax or solver.solve")
        costs.append(torch.cat([c.flatten() for c in captured]))
        weights.append(parameters_to_vector(model.parameters()))
    torch.testing.assert_close(costs[0], costs[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights[0], weights[1], rtol=1e-5, atol=1e-6)
    return dict(
        parity_cost_error=float((costs[0] - costs[1]).abs().max()),
        parity_weight_error=float((weights[0] - weights[1]).abs().max()),
    )


@torch.inference_mode()
def prefix_bench(args):
    rows = []
    cap = None if args.kinds else 1024
    for dims in [args.dims] if args.dims else [(32, width, width, 4) for width in (64, 256)]:
        for kind in args.kinds or ("full", "hybrid"):
            for solver in args.solver or ("softmax",):
                for seed in args.seeds:
                    checked = parity(args, dims, kind, solver, seed, cap) if args.parity else {}
                    reference = None
                    for reuse in (False, True):
                        model, opt, ev, x, y = mlp_case(args, dims, kind, solver, seed, args.inplace or None, cap)
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
                            row["syncs"] = count_syncs(fn, args.device)
                        if reference is None:
                            reference = opt.state.X.clone()
                        else:
                            tolerance = 0 if torch.device(args.device).type == "cpu" else 1e-5
                            torch.testing.assert_close(opt.state.X, reference, rtol=tolerance, atol=tolerance)
                        row.update(
                            checked,
                            workload="mlp_step",
                            width=dims[1],
                            subspace=0 if kind == "full" else cap,
                            subspace_dim=opt.subspace.subspace_dim if opt.subspace is not None else 0,
                            dims=list(dims),
                            kind=kind,
                            rank=args.rank or 4,
                            pdim=args.pdim,
                            solver=solver,
                            inplace=args.inplace,
                            x_sha256=checksum(opt.state.X),
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
def step_bench(args):
    rows = []
    for name in args.workloads or WORKLOADS:
        for strategy in ("monolithic", "per_layer", "grouped"):
            for kind in args.kinds or ("full", "hybrid"):
                for solver in args.solver or ("softmax", "sinkhorn"):
                    for seed in args.seeds:
                        row = dict(
                            workload=name,
                            block_strategy=strategy,
                            kind=kind,
                            solver=solver,
                            seed=seed,
                            device=args.device,
                            batch=args.batch,
                            threads=args.threads,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            torch=torch.__version__,
                        )
                        try:
                            model, (train, _, _) = task(name, seed, args.device, args.batch)
                            x, y = train
                            ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=args.inplace or None)
                            opt = build_optimizer(
                                model,
                                kind,
                                args,
                                solver=solver,
                                block_strategy=strategy,
                                compile=args.compile,
                                seed=seed,
                            )
                            opt.register_evaluator(ev, x, y)
                            closure_candidates, used = [0], set()

                            def closure(params):
                                losses = ev.evaluate(params, x, y)
                                closure_candidates[0] += losses.shape[0]
                                return losses

                            def fn():
                                return opt.step(closure)

                            with trace_scorers(used):
                                row.update(measure(fn, args.device, args.warmup, args.repeats))
                            steps = 1 + args.warmup + args.repeats
                            row.update(
                                closure_candidates_per_step=closure_candidates[0] / steps,
                                candidate_evals_per_step=opt.candidate_evals / steps,
                                evaluators=sorted(used),
                                syncs=count_syncs(fn, args.device),
                                absorb_count=opt.state.absorb_count,
                                x_sha256=checksum(opt.state.X),
                                status="ok",
                            )
                        except Exception as error:
                            row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    return rows


def retained_bytes(obj):
    parts = []
    for value in vars(obj).values():
        if not isinstance(value, torch.Tensor):
            continue
        if value.layout == torch.sparse_coo:
            parts += [value._indices(), value._values()]
        elif value.layout == torch.sparse_csr:
            parts += [value.crow_indices(), value.col_indices(), value.values()]
        else:
            parts.append(value)
    return sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in parts}.values())


@torch.inference_mode()
def projection_bench(args):
    rows = []
    cuda = torch.device(args.device).type == "cuda"
    for full_dim in args.dims or (10**6, 10**7, 10**8):
        proj = SparseRandomProjection(full_dim, args.rank or 256, seed=0)
        coords = torch.randn(proj.subspace_dim, device=args.device)
        synchronize(args.device)
        if cuda:
            base_memory = torch.cuda.memory_allocated(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        before = time.perf_counter()
        proj.project(coords)
        synchronize(args.device)
        row = dict(
            workload="sparse_projection",
            full_dim=full_dim,
            subspace_dim=proj.subspace_dim,
            nnz=proj.nnz,
            first_build_seconds=time.perf_counter() - before,
            first_build_peak_bytes=torch.cuda.max_memory_allocated(args.device) - base_memory if cuda else None,
            retained_bytes=retained_bytes(proj),
            memory_bytes=proj.memory_bytes,
        )
        row.update(
            measure(lambda: proj.project(coords), args.device, args.warmup, args.repeats),
            device=args.device,
            threads=args.threads,
            torch=torch.__version__,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


@torch.inference_mode()
def kernel_bench(args):
    rows = []

    def record(fn, **fields):
        row = dict(fields, **measure(fn, args.device, args.warmup, args.repeats))
        row.update(
            syncs=count_syncs(fn, args.device), device=args.device, threads=args.threads, torch=torch.__version__
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    if "sinkhorn" in args.kernels:
        for shape in ((37634, 3), (1068, 9), (2048, 16), (256, 64)):
            torch.manual_seed(0)
            cost = torch.rand(*shape, device=args.device)
            warm = SinkhornSolver(epsilon=0.05).solve(cost)
            cost = cost + 0.01 * torch.rand_like(cost)
            for compile in (False, True) if args.compile else (False,):
                solver = SinkhornSolver(epsilon=0.05, compile=compile)
                for start, init in (("cold", {}), ("warm", dict(init_f=warm.f, init_g=warm.g))):
                    eager = SinkhornSolver(epsilon=0.05).solve(cost, **init).matrix
                    result = solver.solve(cost, **init)
                    record(
                        lambda: solver.solve(cost, **init),
                        kernel="sinkhorn",
                        shape=list(shape),
                        compile=compile,
                        start=start,
                        n_iters=result.n_iters,
                        converged=result.converged,
                        plan_error=float((result.matrix - eager).abs().max()),
                    )
    if "qr" in args.kernels:
        for height in args.dims or (784, 5000, 50000, 200000):
            for width in (16, 64, 256, 1024):
                if width > height:
                    continue
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    matrix = torch.randn(height, width, device=args.device)
                    for pinned in (True, False):
                        with patch("polystep.solvers._shared.QR_PIN_MAX_WORK", float("inf") if pinned else 0):
                            record(
                                lambda: thin_qr(matrix),
                                kernel="qr",
                                shape=[height, width],
                                work=height * width**2,
                                pinned=pinned,
                                seed=seed,
                            )
    if "rotation" in args.kernels:
        gen = torch.Generator(device=args.device).manual_seed(0)
        for dim in (2, 8, 32):
            rot = get_random_rotation_matrices(512, dim, device=args.device, generator=gen)
            bias = torch.randn(512, dim, device=args.device, generator=gen)
            verts = get_simplex_vertices(dim, device=args.device)
            out = apply_biased_rotation(rot, bias, verts)
            reach = torch.einsum("pji,vi,pj->pv", out, verts, bias / bias.norm(dim=-1, keepdim=True))
            record(
                lambda: apply_biased_rotation(rot, bias, verts),
                kernel="rotation",
                particles=512,
                dim=dim,
                vertex_error=float((reach.amax(dim=1) - 1).abs().max()),
                orthogonality_error=float((out.mT @ out - torch.eye(dim, device=args.device)).abs().max()),
                det_error=float((torch.linalg.det(out) - 1).abs().max()),
            )
    if "polystep" in args.kernels:
        for particles in (64, 1024):
            for dim in (2, 32):
                torch.manual_seed(0)
                solver = PolyStep.create(lambda x: x.square().sum(-1), dim=dim)
                state = solver.init_state(torch.randn(particles, dim, device=args.device))
                record(lambda: solver.step(state), kernel="polystep", particles=particles, dim=dim)
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
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--kernels", nargs="+", choices=("sinkhorn", "qr", "rotation", "polystep"))
    ap.add_argument("--projection", action="store_true")
    ap.add_argument("--kinds", nargs="+", choices=("full", "hybrid", "factored", "adaptive"))
    ap.add_argument("--dims", nargs="+", type=int)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--pdim", type=int)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--solver", nargs="+")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("experiments/results/benchmarks/forward.json"))
    args = ap.parse_args()
    if args.repeats < 2 or min(args.batch, args.candidates, args.threads) < 1 or args.warmup < 0:
        ap.error("repeats must be >=2; batch, candidates and threads positive; warmup nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    rows = (
        prefix_bench(args)
        if args.prefix
        else step_bench(args)
        if args.step
        else kernel_bench(args)
        if args.kernels
        else projection_bench(args)
        if args.projection
        else attention_bench(args)
        if args.attention
        else candidate_bench(args)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    if any(row.get("status") == "failed" for row in rows):
        raise SystemExit("Some backends failed; inspect the result file")


if __name__ == "__main__":
    main()
