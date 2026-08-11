"""04 - MAX-SAT at scale: 10,000 variables.

Random 3-SAT at the phase-transition density (clause/variable ratio 4.27),
optimized gradient-free on the assignment vector; the integer rounding step
is treated as a black box. Hyperparameters are sqrt-scaled from the 100K row
of experiments/runners/run_maxsat.py.

Default: 10,000 variables (GPU recommended). ``--small``: 2,000 variables,
CPU-friendly.

Run:
  python examples/04_maxsat_10k.py
  python examples/04_maxsat_10k.py --small
"""

from __future__ import annotations

import argparse
import math
import random
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

import _env  # noqa: E402

_env.setup()

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.epsilon import CosineEpsilon  # noqa: E402


def generate_maxsat_instance(num_vars, k=3, ratio=4.27, seed=42):
    """A random k-SAT instance at the critical clause-to-variable ratio."""
    rng = random.Random(seed)
    num_clauses = int(num_vars * ratio)
    clause_vars, clause_signs = [], []
    for _ in range(num_clauses):
        chosen = rng.sample(range(num_vars), k)
        clause_vars.append(chosen)
        clause_signs.append([1.0 if rng.random() < 0.5 else 0.0 for _ in chosen])
    return {
        "clause_vars": torch.tensor(clause_vars, dtype=torch.long),
        "clause_signs": torch.tensor(clause_signs, dtype=torch.float),
        "num_vars": num_vars,
        "num_clauses": num_clauses,
    }


class MaxSATModel(nn.Module):
    """Continuous relaxation of a MAX-SAT assignment, hardened by round()."""

    def __init__(self, num_vars: int):
        super().__init__()
        self.assignments = nn.Parameter(torch.randn(num_vars) * 0.1)

    def forward(self, clause_vars: torch.Tensor, clause_signs: torch.Tensor) -> torch.Tensor:
        hard = torch.round(torch.sigmoid(self.assignments))  # non-differentiable
        gathered = hard[clause_vars]
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
        satisfied = (literals > 0.5).any(dim=-1).float()
        return 1.0 - satisfied.mean()


# Hyperparameter reference: sqrt-scaled from the 100K row of run_maxsat.py.
REFERENCE_NUM_VARS = 100_000
REFERENCE_STEP_RADIUS_INIT = 3000.0
REFERENCE_STEP_RADIUS_TARGET = 600.0
REFERENCE_PROBE_RADIUS_INIT = 100.0
REFERENCE_PROBE_RADIUS_TARGET = 20.0


def scaled_radii(num_vars: int):
    s = math.sqrt(num_vars / REFERENCE_NUM_VARS)
    return (
        REFERENCE_STEP_RADIUS_INIT * s,
        REFERENCE_STEP_RADIUS_TARGET * s,
        REFERENCE_PROBE_RADIUS_INIT * s,
        REFERENCE_PROBE_RADIUS_TARGET * s,
    )


def satisfied_clauses(assignments: torch.Tensor, clause_vars: torch.Tensor, clause_signs: torch.Tensor):
    """Which clauses each assignment satisfies, shape ``assignments.shape[:-1] + (C,)``.

    Kept as a boolean equality because the gathered ``(..., C, k)`` tensor is the
    largest allocation in the step and bool is a quarter the memory traffic.
    """
    hard = torch.sigmoid(assignments) > 0.5
    return (hard[..., clause_vars] == clause_signs).any(dim=-1)


@torch.no_grad()
def sat_ratio(model: MaxSATModel, clause_vars: torch.Tensor, clause_signs: torch.Tensor) -> float:
    satisfied = satisfied_clauses(model.assignments, clause_vars, clause_signs)
    return float(satisfied.sum().item()) / clause_vars.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--small", action="store_true", help="Use 2,000 variables for a CPU-friendly run.")
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of PolyStep iterations. 1500 gives comfortable margin above 98%% SAT.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_vars = 2_000 if args.small else 10_000
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)

    print("=" * 60)
    print(f"MAX-SAT (3-SAT): {num_vars} vars at phase-transition density")
    print("=" * 60)

    if importlib.util.find_spec("pysat") is None:
        raise SystemExit("this example generates its instance through pysat: pip install python-sat")

    instance = generate_maxsat_instance(num_vars=num_vars, ratio=4.27, seed=seed)
    print(f"  variables: {instance['num_vars']:,}")
    print(f"  clauses:   {instance['num_clauses']:,}")
    print(f"  device:    {device}")

    clause_vars = instance["clause_vars"].to(device)
    clause_signs = instance["clause_signs"].to(device).bool()  # 1 = positive literal

    model = MaxSATModel(num_vars=num_vars).to(device)

    sr_init, sr_tgt, pr_init, pr_tgt = scaled_radii(num_vars)
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=CosineEpsilon(init=5.0, target=0.5),
        step_radius=CosineEpsilon(init=sr_init, target=sr_tgt),
        probe_radius=CosineEpsilon(init=pr_init, target=pr_tgt),
        num_probe=1,
        chunk_size=256,
        amortize_steps=3,
        amortize_ema=0.7,
        use_momentum=True,
        momentum_init=0.5,
        momentum_final=0.95,
    )

    def closure(stacked_params):
        # cost = fraction of unsatisfied clauses per candidate
        satisfied = satisfied_clauses(stacked_params["assignments"], clause_vars, clause_signs)
        return 1.0 - satisfied.sum(dim=-1) / clause_vars.shape[0]

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

    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib not installed; skipping the figure (pip install matplotlib).")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8), constrained_layout=True)
    ax.plot(step_log, sat_log, color="#0072B2", lw=1.5, marker="o", markersize=3, label="PolyStep")
    ax.axhline(1.0, color="#009E73", ls=":", lw=1.0, label="all clauses sat")
    ax.set_xlabel("PolyStep step")
    ax.set_ylabel("Fraction of clauses satisfied")
    ax.set_title(
        f"3-SAT phase transition ({num_vars:,} vars, {instance['num_clauses']:,} clauses)",
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
