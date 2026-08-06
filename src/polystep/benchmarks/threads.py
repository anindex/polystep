"""Measure step time against ``torch.set_num_threads``. Run with ``python -m polystep.benchmarks.threads``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_CHILD = """
import json, sys, time, torch, torch.nn as nn
torch.set_num_threads({threads})
from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator

torch.manual_seed(0)
model = nn.Sequential(nn.Flatten(), nn.Linear({d_in}, {hidden}), nn.ReLU(), nn.Linear({hidden}, 10))
inputs = torch.randn({batch}, {d_in})
targets = torch.randint(0, 10, ({batch},))
evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=1000, particle_dim={pdim})
opt.register_evaluator(evaluator, inputs, targets)
closure = lambda bp: evaluator.evaluate(bp, inputs, targets)

opt.step(closure)  # warm up lazily built buffers
start = time.perf_counter()
for _ in range({steps}):
    opt.step(closure)
print(json.dumps({{"ms": (time.perf_counter() - start) / {steps} * 1000}}))
"""


def sweep(thread_counts, d_in=784, hidden=64, batch=64, pdim=2, steps=3, timeout=600):
    """Return ``[(threads, ms_per_step_or_None), ...]``, one subprocess per count."""
    results = []
    for threads in thread_counts:
        source = _CHILD.format(threads=threads, d_in=d_in, hidden=hidden, batch=batch, pdim=pdim, steps=steps)
        try:
            done = subprocess.run(
                [sys.executable, "-c", source],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
            results.append((threads, json.loads(done.stdout.strip().splitlines()[-1])["ms"]))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, KeyError):
            results.append((threads, None))
    return results


def main() -> None:
    ncpu = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=None)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    counts = args.threads or sorted({1, 2, 4, 8, ncpu // 2, ncpu - 8, ncpu - 4, ncpu - 1, ncpu} - {0})
    counts = [c for c in counts if c >= 1]

    print(f"{ncpu} cores, torch default = {ncpu} threads")
    print(f"{'threads':>8}  {'ms/step':>10}")
    best = None
    for threads, ms in sweep(counts, hidden=args.hidden, batch=args.batch, steps=args.steps):
        print(f"{threads:>8}  {'timeout' if ms is None else f'{ms:10.1f}'}")
        if ms is not None and (best is None or ms < best[1]):
            best = (threads, ms)
    if best:
        print(f"\nfastest: {best[0]} threads at {best[1]:.1f} ms/step")
        print(f"torch.set_num_threads({best[0]})")


if __name__ == "__main__":
    main()
