"""Compare the existing candidate-forward backends on fixed training probes."""

import gc
import statistics
import time

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from polystep.cost_nn import NNCostEvaluator, SiteVmapEvaluator
from polystep.transform import ParamLayout
from experiments.runners.run_controlled import (
    RESULTS,
    atomic_json,
    block_candidates,
    draw_vertices,
    load_data,
    make_projection,
)
from experiments.runners.nondiff_models import SpikingMNISTNet


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(2026)
    model = SpikingMNISTNet().cuda().eval()
    train, *_ = load_data("cuda")
    x, y = train[0][:128], train[1][:128]
    sub, projections, _ = make_projection(model, 12026, "cuda")
    base = {k: p.detach() for k, p in model.named_parameters()}
    gen = torch.Generator(device="cuda").manual_seed(32026)
    u = draw_vertices(32, "orthoplex", gen, "cuda") * 0.1
    candidates = block_candidates(base, sub, projections, 0, u)
    reference = NNCostEvaluator(model, torch.nn.CrossEntropyLoss(), use_inplace=False).evaluate(candidates, x, y)
    rows = []
    # A common 512-candidate task makes latency comparable across chunk sizes.
    for backend in ("eager", "compiled", "site"):
        for chunk in (1, 8, 32, 128, 512):
            evaluator = NNCostEvaluator(
                model,
                torch.nn.CrossEntropyLoss(),
                chunk_size=None,
                use_inplace=False,
                compile_vmap=backend == "compiled",
            )
            site = SiteVmapEvaluator.try_build(evaluator, ParamLayout.from_module(model))
            if backend == "site":
                spec = sub.specs[0]
                assert spec.flat_start == 0 and spec.flat_end >= 256

                def call():
                    parts = []
                    for first in range(0, 32, max(1, chunk // 16)):
                        offsets = u[first : first + max(1, chunk // 16)]
                        parts.append(site.evaluate_subspace(projections, base, spec, first * 8, offsets, x, y))
                    return torch.cat(parts)
            else:

                def call():
                    return torch.cat(
                        [
                            evaluator.evaluate({k: v[start : start + chunk] for k, v in candidates.items()}, x, y)
                            for start in range(0, 512, chunk)
                        ]
                    )

            row = dict(backend=backend, chunk=chunk, candidates=512, batch=128, warmup=20, timed=100)
            started = time.perf_counter()
            try:
                for _ in range(20):
                    call()
                times = []
                for _ in range(100):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    output = call()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - before)
                row.update(
                    median_seconds=statistics.median(times),
                    max_loss_difference=(output - reference).abs().max().item(),
                    changed_costs=(output != reference).sum().item(),
                    compile_fallback=evaluator._compile_failed,
                    actual_site_chunk=max(16, chunk) if backend == "site" else None,
                )
            except torch.cuda.OutOfMemoryError:
                row["failure"] = "out of memory"
            row["total_seconds"] = time.perf_counter() - started
            rows.append(row)
            atomic_json(RESULTS / "forward_profile.json", rows)
            print(row, flush=True)
            evaluator = site = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
