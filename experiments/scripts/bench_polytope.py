"""Compare search rules at matched candidate or wall-clock budgets.

PYTHONPATH=src:. python experiments/scripts/bench_polytope.py --wall
"""

import argparse
import json
from pathlib import Path

import torch

from experiments.runners.search_suite import ARMS, WORKLOADS, run_search


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--suite", action="store_true", help=argparse.SUPPRESS)  # existing commands remain valid
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workloads", nargs="+", choices=WORKLOADS)
    ap.add_argument("--arms", nargs="+", choices=ARMS)
    ap.add_argument("--wall", action="store_true", help="Also compare at the baseline wall budget")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("experiments/results/benchmarks/algorithms.json"))
    args = ap.parse_args()
    if min(args.budget, args.batch, args.threads) < 1:
        ap.error("budget, batch and threads must be positive")
    torch.set_num_threads(args.threads)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.workloads or WORKLOADS:
        for seed in args.seeds:
            base = run_search(
                name,
                "hybrid",
                seed,
                device=args.device,
                budget=args.budget,
                batch=args.batch,
                streaming=args.streaming,
            )
            target = base["start_validation_loss"] - max(
                0.01, 0.5 * (base["start_validation_loss"] - base["best_validation_loss"])
            )
            base["target"] = target
            base["time_to_target"] = next(
                (h["seconds"] for h in base["history"] if h["validation_loss"] <= target), None
            )
            base["axis"] = "evaluations"
            rows.append(base)
            for arm in args.arms or ARMS:
                for axis in ("evaluations", "wall") if args.wall else ("evaluations",):
                    if arm == "hybrid" and axis == "evaluations":
                        continue
                    row = run_search(
                        name,
                        arm,
                        seed,
                        device=args.device,
                        budget=10**9 if axis == "wall" else args.budget,
                        seconds=base["seconds"] if axis == "wall" else None,
                        batch=args.batch,
                        target=target,
                        streaming=args.streaming,
                    )
                    row["axis"] = axis
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
