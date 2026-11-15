"""12 - Particle gallery: 100,000 particles on six 2D test functions.

Each run draws particles uniformly over the objective's bounds and takes PolyStep
iterations with an annealed epsilon, which shrinks the probe and step radii. One
animation per objective is saved to examples/figures/gallery_<name>.gif. It shows the
particle density over the cost landscape, a ring at each global minimum, and the mean
cost above the optimum at every step.

Run:
  python examples/12_particle_gallery.py
  python examples/12_particle_gallery.py --particles 1000000 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import torch

import _env

_env.setup()

from polystep.epsilon import CosineEpsilon
from polystep.objectives import Ackley, ObjectiveFn, Rastrigin, Rosenbrock
from polystep.solver import PolyStep

STEPS = 50
BINS = 160
GRID = 320
COLORS = 64
INK, MUTED, RING, AMBER = "#05070d", "#8b96a8", "#f2f5fa", "#ffb061"


class Surface(ObjectiveFn):
    """A 2D test function from its formula ``f(x, y)``, the half-width of its square domain and its global minima."""

    def __init__(self, formula, half_width: float, minima):
        minima = torch.tensor(minima)
        super().__init__(
            dim=2,
            bounds=torch.tensor([[-half_width, half_width]] * 2),
            optimizers=minima,
            optimal_value=float(formula(minima[0, 0], minima[0, 1])),
        )
        self.formula = formula

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return self.formula(X[..., 0], X[..., 1])


def levy(x, y):
    u, v = 1 + (x - 1) / 4, 1 + (y - 1) / 4
    return (
        torch.sin(math.pi * u) ** 2
        + (u - 1) ** 2 * (1 + 10 * torch.sin(math.pi * u + 1) ** 2)
        + (v - 1) ** 2 * (1 + torch.sin(2 * math.pi * v) ** 2)
    )


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def eggholder(x, y):
    return -(y + 47) * torch.sin((x / 2 + y + 47).abs().sqrt()) - x * torch.sin((x - y - 47).abs().sqrt())


HIMMELBLAU_MINIMA = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
RUNS = {
    "ackley": (Ackley(dim=2), 2.0, 0.05, 1.0),
    "rastrigin": (Rastrigin(dim=2), 1.5, 0.05, 1.0),
    "rosenbrock": (Rosenbrock(dim=2), 1.0, 0.02, 1.0),
    "levy": (Surface(levy, 10.0, [[1.0, 1.0]]), 3.0, 0.05, 1.0),
    "himmelblau": (Surface(himmelblau, 5.0, HIMMELBLAU_MINIMA), 1.5, 0.03, 1.0),
    "eggholder": (Surface(eggholder, 512.0, [[512.0, 404.2319]]), 1.5, 0.05, 100.0),
}


def simulate(name: str, particles: int, device: str, seed: int):
    """Particle clouds after each iteration, the mean cost per iteration and the mean step time."""
    objective, eps_init, eps_target, reach = RUNS[name]
    low, high = objective.bounds[:, 0].to(device), objective.bounds[:, 1].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    solver = PolyStep.create(
        objective,
        dim=2,
        epsilon=CosineEpsilon(init=reach * eps_init, target=reach * eps_target, total_steps=STEPS),
        ent_epsilon=CosineEpsilon(init=eps_init, target=eps_target, total_steps=STEPS),
        scale_cost="mean",
        max_iterations=STEPS,
    )
    state = solver.init_state(low + (high - low) * torch.rand(particles, 2, generator=generator, device=device))
    clouds, costs = [state.X.cpu()], [float(objective(state.X).mean())]
    elapsed = 0.0
    for _ in range(STEPS):
        start = time.perf_counter()
        state = solver.step(state, generator=generator)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        clouds.append(state.X.cpu())
        costs.append(float(objective(state.X).mean()))
    return objective, clouds, costs, 1e3 * elapsed / STEPS


def animate(name: str, objective, clouds, costs, step_ms: float, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from PIL import Image

    (x0, x1), (y0, y1) = objective.bounds.tolist()
    extent = (x0, x1, y0, y1)
    xs, ys = torch.linspace(x0, x1, GRID), torch.linspace(y0, y1, GRID)
    height = objective(torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)) - objective.optimal_value
    relief = (height / height.flatten().quantile(0.97)).clamp(0, 1).sqrt().numpy()
    gap = np.maximum(np.asarray(costs) - objective.optimal_value, 1e-12)
    night = LinearSegmentedColormap.from_list("night", ["#05070d", "#0d1320", "#161f33", "#202c47", "#2b3a5c"])
    glow = LinearSegmentedColormap.from_list(
        "glow",
        [
            (0, (1, 0.35, 0.05, 0)),
            (0.25, (1, 0.42, 0.08, 0.4)),
            (0.55, (1, 0.62, 0.15, 0.9)),
            (0.8, (1, 0.85, 0.45, 1)),
            (1, (1, 1, 0.95, 1)),
        ],
    )

    kernel = torch.exp(-0.5 * (torch.arange(-4.0, 5.0) / 1.5) ** 2).double()
    kernel = (kernel / kernel.sum()).view(1, 1, -1, 1)

    def density(cloud):
        """Particle counts on the grid, smoothed by a Gaussian of 1.5 cells."""
        hist = np.histogram2d(cloud[:, 1].numpy(), cloud[:, 0].numpy(), bins=BINS, range=[[y0, y1], [x0, x1]])[0]
        hist = torch.nn.functional.conv2d(torch.from_numpy(hist)[None, None], kernel, padding=(4, 0))
        return torch.nn.functional.conv2d(hist, kernel.transpose(2, 3), padding=(0, 4))[0, 0].numpy()

    fig = plt.figure(figsize=(3.6, 4.5), dpi=100, facecolor=INK)
    ax = fig.add_axes([0, 56 / 450, 1, 360 / 450])
    bands = np.linspace(0, 1, 13)
    ax.contourf(xs, ys, relief, levels=bands, cmap=night)
    ax.contour(xs, ys, relief, levels=bands[1:-1], colors="white", linewidths=0.4, alpha=0.1)
    overlay = ax.imshow(
        density(clouds[0]),
        extent=extent,
        origin="lower",
        cmap=glow,
        norm=PowerNorm(0.45, vmin=0, vmax=60 * len(clouds[0]) / BINS**2),
        interpolation="gaussian",
        aspect="auto",
        zorder=3,
    )
    ax.scatter(
        *objective.optimizers.T,
        s=90,
        linewidths=1.2,
        facecolors="none",
        edgecolors=RING,
        path_effects=[patheffects.withStroke(linewidth=2.6, foreground=INK)],
        clip_on=False,
        zorder=4,
    )
    ax.set(xlim=(x0, x1), ylim=(y0, y1))
    ax.set_axis_off()

    strip = fig.add_axes([0.04, 6 / 450, 0.92, 30 / 450], facecolor=INK)
    strip.set(yscale="log", xlim=(-0.5, STEPS + 0.5), ylim=(gap.min() / 1.6, gap.max() * 1.6))
    strip.plot(gap, color=MUTED, linewidth=1, alpha=0.3)
    (trace,) = strip.plot([], [], color=AMBER, linewidth=1.6)
    (head,) = strip.plot([], [], "o", color="white", markersize=3.5)
    strip.set_axis_off()

    fig.text(0.04, 1 - 17 / 450, name.capitalize(), color="white", fontsize=11, fontweight="bold", va="center")
    counter = fig.text(0.96, 1 - 17 / 450, "", color=MUTED, fontsize=7.5, family="monospace", ha="right", va="center")
    fig.text(0.04, 46 / 450, "mean cost above optimum", color=MUTED, fontsize=7.5, va="center")
    readout = fig.text(0.96, 46 / 450, "", color=AMBER, fontsize=7.5, family="monospace", ha="right", va="center")

    frames = []
    for i, cloud in enumerate(clouds):
        overlay.set_data(density(cloud))
        counter.set_text(f"{len(cloud):,} particles · step {i:2d}/{STEPS}")
        trace.set_data(range(i + 1), gap[: i + 1])
        head.set_data([i], [gap[i]])
        readout.set_text(f"{gap[i]:,.0f}" if gap[i] >= 100 else f"{gap[i]:.3g}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)

    samples = [np.asarray(frames[i]) for i in (0, len(frames) // 6, len(frames) // 2, -1)]
    palette = Image.fromarray(np.hstack(samples)).quantize(COLORS, Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [600] + [80] * (len(frames) - 2) + [1800]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"  {name}: {step_ms:.1f} ms/step, mean cost {costs[0]:.2f} -> {costs[-1]:.2f}, saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=100_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    if args.particles < 2:
        parser.error("--particles must be at least 2")
    if importlib.util.find_spec("matplotlib") is None:
        parser.error("matplotlib is required (pip install matplotlib)")

    args.out.mkdir(exist_ok=True)
    print(f"{args.particles:,} particles on {args.device}, {STEPS} steps each")
    for name in RUNS:
        objective, clouds, costs, step_ms = simulate(name, args.particles, args.device, args.seed)
        animate(name, objective, clouds, costs, step_ms, args.out / f"gallery_{name}.gif")


if __name__ == "__main__":
    main()
