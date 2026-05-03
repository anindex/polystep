"""01 - Quickstart: PolyStep on a 2D non-smooth landscape.

Newcomers should run this first. It demonstrates the core mechanic of
PolyStep (sample polytope vertices around each particle, evaluate the
objective, and update via softmax-weighted optimal transport) on a
*visible* 2D problem so the polytope and the particle cloud are tangible.

What you should see:
  * 16 particles start spread around a 2D plane.
  * Over ~50 steps they contract toward the global minimum at the origin.
  * The objective is piecewise-constant ("staircase") in radius, so the
    landscape contours are sharp and non-smooth. Surrogate-gradient
    methods stall on this regime; PolyStep handles it directly.

Output:
  examples/figures/quickstart_2d.png

Run:
  python examples/01_quickstart_2d.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Allow running directly from a source checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polystep.solver import PolyStep  # noqa: E402


# A 2D non-smooth objective: staircase in radius. Minimum at the origin,
# piecewise-constant in concentric annuli, gradient zero almost everywhere.

def staircase_radial(X: torch.Tensor) -> torch.Tensor:
    """Piecewise-constant radial staircase. ``X`` shape: ``(..., 2)``."""
    r = torch.linalg.vector_norm(X, dim=-1)
    return torch.floor(r * 2.0)  # 0.5-wide concentric steps


class StaircaseObjective:
    """Wraps the staircase as a polystep-compatible callable."""
    dim = 2

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return staircase_radial(X)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return staircase_radial(X)


def main():
    seed = 42
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # 16 particles, 50 iterations. Probe radius wider than step radius so
    # probes reliably cross at least one staircase boundary per step.
    objective = StaircaseObjective()
    solver = PolyStep.create(
        objective,
        dim=2,
        epsilon=0.3,
        step_radius=0.5,
        probe_radius=1.5,
        num_probe=1,
        max_iterations=50,
        min_iterations=50,
        compile=False,
    )

    X_init = 4.0 * (torch.rand(16, 2, generator=generator) - 0.5)
    state = solver.init_state(X_init)

    cloud_history = [state.X.clone()]
    cost_history = []
    for _ in range(solver.max_iterations):
        state = solver.step(state, generator=generator)
        cloud_history.append(state.X.clone())
        cost_history.append(state.costs[-1])

    # Best-particle trajectory: lowest cost in each cloud.
    best_traj = []
    for c in cloud_history:
        best_traj.append(c[staircase_radial(c).argmin()])
    best_traj = torch.stack(best_traj)

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(7.0, 2.8),
        gridspec_kw={"width_ratios": [1.0, 1.05]},
        constrained_layout=True,
    )

    grid = np.linspace(-3.5, 3.5, 200)
    XX, YY = np.meshgrid(grid, grid)
    Z = staircase_radial(torch.from_numpy(np.stack([XX, YY], axis=-1)).float()).numpy()
    cf = ax_left.contourf(XX, YY, Z, levels=12, cmap="viridis", alpha=0.85)
    ax_left.set_aspect("equal")
    ax_left.set_xlim(-3.5, 3.5)
    ax_left.set_ylim(-3.5, 3.5)
    ax_left.set_xlabel(r"$x_1$")
    ax_left.set_ylabel(r"$x_2$")
    ax_left.set_title("Particles on a piecewise-constant 2D landscape", fontsize=9)
    plt.colorbar(cf, ax=ax_left, fraction=0.046, pad=0.03,
                 label=r"$\lfloor 2\,\|x\| \rfloor$")

    init_cloud = cloud_history[0].numpy()
    final_cloud = cloud_history[-1].numpy()
    ax_left.scatter(init_cloud[:, 0], init_cloud[:, 1],
                    c="white", edgecolors="black", s=18, alpha=0.55, label="init")
    ax_left.scatter(final_cloud[:, 0], final_cloud[:, 1],
                    c="#ffce4d", edgecolors="black", s=26, alpha=0.95,
                    label="final", zorder=5)
    bt = best_traj.numpy()
    ax_left.plot(bt[:, 0], bt[:, 1], color="#ffce4d", lw=1.0, alpha=0.6, zorder=4)
    ax_left.scatter([0.0], [0.0], marker="*", c="#e84040", s=110,
                    edgecolors="black", linewidths=0.5, label="optimum", zorder=6)
    ax_left.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9)

    ax_right.plot(range(1, len(cost_history) + 1), cost_history,
                  color="#0072B2", lw=1.4)
    ax_right.set_xlabel("PolyStep iteration")
    ax_right.set_ylabel("Entropic OT cost")
    ax_right.set_title("Solver cost decreases monotonically", fontsize=9)
    ax_right.grid(True, alpha=0.3)

    out = Path(__file__).parent / "figures" / "quickstart_2d.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    final_mean = float(staircase_radial(state.X).mean())
    init_mean = float(staircase_radial(X_init).mean())
    print(f"final mean cost = {final_mean:.3f} (started ~{init_mean:.3f})")
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
