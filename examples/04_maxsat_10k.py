"""04 - MAX-SAT at scale: 10,000 variables.

Random 3-SAT at the phase-transition density (clause / variable ratio
4.27) solved by direct gradient-free optimization on the variable
assignment vector. The integer rounding step is treated as a black box
and the piecewise-constant SAT objective is optimized without
surrogate gradients.

This is the headline scaling result from the paper, reduced to a single
runnable script. The hyperparameters mirror the 10K row of
``experiments/runners/run_maxsat.py`` (sqrt-scaled from a 100K reference).

Hardware:
  Default: 10,000 variables, ~42,700 clauses. Best on a CUDA GPU with
  >=4 GB free; runs on CPU in a few minutes.
  ``--small``: 2,000 variables, ~8,500 clauses. Completes on CPU in <60s.

What you should see:
  SAT ratio climbs from ~0.86 (random assignment) past 0.98 within
  ~1000 steps and continues to creep upward. The default 1500-step
  budget gives comfortable margin above the 98%% threshold across
  seeds. Phase-transition 3-SAT is intrinsically hard; domain solvers
  like probSAT reach ~0.996.

Output:
  examples/figures/maxsat_10k.png

Run:
  python examples/04_maxsat_10k.py             # 10K vars, GPU recommended
  python examples/04_maxsat_10k.py --small     # 2K vars, CPU-friendly
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

# Allow running directly from a source checkout without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so experiments/runners/* is importable

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.epsilon import CosineEpsilon  # noqa: E402
from experiments.runners.nondiff_data import generate_maxsat_instance  # noqa: E402
from experiments.runners.nondiff_models import MaxSATModel  # noqa: E402


# Hyperparameter reference: sqrt-scaled from the 100K row of run_maxsat.py.
REFERENCE_NUM_VARS = 100_000
REFERENCE_STEP_RADIUS_INIT = 3000.0
REFERENCE_STEP_RADIUS_TARGET = 600.0
REFERENCE_PROBE_RADIUS_INIT = 100.0
REFERENCE_PROBE_RADIUS_TARGET = 20.0


def scaled_radii(num_vars: int):
    s = math.sqrt(num_vars / REFERENCE_NUM_VARS)
    return (
        REFERENCE_STEP_RADIUS_INIT * s, REFERENCE_STEP_RADIUS_TARGET * s,
        REFERENCE_PROBE_RADIUS_INIT * s, REFERENCE_PROBE_RADIUS_TARGET * s,
    )


@torch.no_grad()
def sat_ratio(model: MaxSATModel, clause_vars: torch.Tensor,
              clause_signs: torch.Tensor) -> float:
    hard = torch.round(torch.sigmoid(model.assignments))
    gathered = hard[clause_vars]
    literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
    satisfied = (literals > 0.5).any(dim=-1).float()
    return float(satisfied.mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--small", action="store_true",
                        help="Use 2,000 variables for a CPU-friendly run.")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Number of PolyStep iterations. 1500 gives "
                             "comfortable margin above 98%% SAT.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_vars = 2_000 if args.small else 10_000
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    print("=" * 60)
    print(f"MAX-SAT (3-SAT): {num_vars} vars at phase-transition density")
    print("=" * 60)

    instance = generate_maxsat_instance(num_vars=num_vars, ratio=4.27, seed=seed)
    print(f"  variables: {instance['num_vars']:,}")
    print(f"  clauses:   {instance['num_clauses']:,}")
    print(f"  device:    {device}")

    clause_vars = instance["clause_vars"].to(device)
    clause_signs = instance["clause_signs"].to(device)

    model = MaxSATModel(num_vars=num_vars).to(device)

    sr_init, sr_tgt, pr_init, pr_tgt = scaled_radii(num_vars)
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=CosineEpsilon(5.0, 0.5),
        step_radius=CosineEpsilon(sr_init, sr_tgt),
        probe_radius=CosineEpsilon(pr_init, pr_tgt),
        num_probe=1,
        chunk_size=256,
        amortize_steps=3,
        amortize_ema=0.7,
        use_momentum=True,
        momentum_init=0.5,
        momentum_final=0.95,
    )

    def closure(stacked_params):
        # The optimizer hands us a dict {"assignments": (N, num_vars)}.
        # Compute fraction of unsatisfied clauses per candidate.
        assignments = stacked_params["assignments"]
        soft = torch.sigmoid(assignments)
        hard = torch.round(soft)
        # Index along the variable axis: (N, C, k)
        gathered = hard[:, clause_vars]
        signs = clause_signs.unsqueeze(0).to(dtype=gathered.dtype)
        literals = gathered * signs + (1.0 - signs) * (1.0 - gathered)
        satisfied = (literals > 0.5).any(dim=-1).float()
        return 1.0 - satisfied.mean(dim=-1)

    print(f"  initial SAT ratio: {sat_ratio(model, clause_vars, clause_signs):.3f}")
    print()

    sat_log: list[float] = []
    step_log: list[int] = []
    best_sat = 0.0

    print(f"training {args.steps} steps...")
    start = time.time()
    for step in range(args.steps):
        optimizer.step(closure)
        if step % max(1, args.steps // 50) == 0 or step == args.steps - 1:
            r = sat_ratio(model, clause_vars, clause_signs)
            sat_log.append(r)
            step_log.append(step)
            best_sat = max(best_sat, r)
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                print(f"  step {step:4d} | sat={r:.4f} (best={best_sat:.4f})")
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"  final SAT ratio: {sat_log[-1]:.4f}")
    print(f"  best  SAT ratio: {best_sat:.4f}")
    print(f"  wallclock: {elapsed:.1f}s ({args.steps} steps)")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8), constrained_layout=True)
    ax.plot(step_log, sat_log, color="#0072B2", lw=1.5, marker="o",
            markersize=3, label="PolyStep")
    ax.axhline(1.0, color="#009E73", ls=":", lw=1.0, label="all clauses sat")
    ax.set_xlabel("PolyStep step")
    ax.set_ylabel("Fraction of clauses satisfied")
    ax.set_title(
        f"3-SAT phase transition ({num_vars:,} vars, "
        f"{instance['num_clauses']:,} clauses)",
        fontsize=9,
    )
    ax.set_ylim(min(0.85, sat_log[0] - 0.02), 1.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)

    out = Path(__file__).parent / "figures" / "maxsat_10k.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
