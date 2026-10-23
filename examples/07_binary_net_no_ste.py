"""STE-free training of a hard-threshold (sign-activation) net.

sign() has zero gradient almost everywhere, so backprop needs a straight-through
estimator whose forward/backward mismatch is biased, worst at 1-2 bit
(https://arxiv.org/abs/2505.18113, https://arxiv.org/abs/2601.22660). This
example skips gradients and minimizes 0-1 error directly, comparing PolyStepES
against OpenAI-ES on a matched budget, averaged over seeds.

Run:
    python examples/07_binary_net_no_ste.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import _env

_env.setup(default_threads=8)

from polystep import PolyStepES
from polystep.epsilon import CosineEpsilon

from _openai_es import OpenAIES, run

HIDDEN = 32
GENERATIONS = 200
SEEDS = 5


def make_moons(n=400, noise=0.15, seed=0):
    """Two interleaving half-moons (easy boundary)."""
    g = torch.Generator().manual_seed(seed)
    n_a = n // 2
    ta = math.pi * torch.rand(n_a, generator=g)
    tb = math.pi * torch.rand(n - n_a, generator=g)
    a = torch.stack([torch.cos(ta), torch.sin(ta)], dim=1)
    b = torch.stack([1.0 - torch.cos(tb), 1.0 - torch.sin(tb) - 0.5], dim=1)
    X = torch.cat([a, b], dim=0) + noise * torch.randn(n, 2, generator=g)
    y = torch.cat([torch.zeros(n_a), torch.ones(n - n_a)])
    return X, y


def make_checkerboard(n=400, k=3, noise=0.05, seed=0):
    """k x k XOR grid: a fragmented, non-linearly-separable boundary."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * k
    y = ((X[:, 0].floor().long() + X[:, 1].floor().long()) % 2).float()
    X = X + noise * torch.randn(n, 2, generator=g)
    return (X - X.mean(0)) / X.std(0), y


def param_dim(hidden=HIDDEN, d_in=2):
    return d_in * hidden + hidden + hidden + 1  # W1, b1, W2, b2


def error_rate(flat, X, y, hidden=HIDDEN):
    """0-1 error of a sign-activation MLP for a batch of flat param vectors.

    flat: (B, D) -> (B,) misclassification rates. Fully non-differentiable.
    """
    B, d_in = flat.shape[0], X.shape[1]
    W1, b1, W2, b2 = flat.split([d_in * hidden, hidden, hidden, 1], dim=1)
    W1 = W1.reshape(B, hidden, d_in).transpose(1, 2)
    h = torch.baddbmm(b1.unsqueeze(1), X.expand(B, -1, -1), W1).sign_()  # (B, N, H)
    out = torch.baddbmm(b2.unsqueeze(1), h, W2.unsqueeze(2)).squeeze(2)  # (B, N)
    return ((out > 0) != y.bool()).float().mean(dim=1)


def solve_task(X, y, seed):
    D = param_dim()
    x0 = 0.5 * torch.randn(D, generator=torch.Generator().manual_seed(seed + 1))

    def fit(flat):
        return error_rate(flat, X, y)

    radius = CosineEpsilon(init=3.0, target=0.5, total_steps=GENERATIONS)
    ps = PolyStepES(D, num_particles=1, epsilon=0.02, step_radius=radius.init, x0=x0, seed=seed)
    ps_curve = []
    for t in range(GENERATIONS):
        ps.step_radius = radius.at(t)
        ps.tell(fit(ps.ask()))
        ps_curve.append(100.0 * (1.0 - ps.best_fitness))
    es = OpenAIES(D, popsize=ps.popsize, x0=x0, sigma=0.1, lr=0.03, seed=seed)
    return ps_curve, run(es, fit, GENERATIONS), ps.popsize


def main():
    tasks = {
        "two-moons (easy)": make_moons,
        "XOR checkerboard (hard)": make_checkerboard,
    }
    D = param_dim()
    print("STE-free hard-threshold net: minimize 0-1 error, no gradients")
    print(f"  sign-activation MLP  params={D}  hidden={HIDDEN}  generations={GENERATIONS}  seeds={SEEDS}")
    print(f"  {'task':<26}{'PolyStepES':>12}{'OpenAI-ES':>12}{'gap':>8}")
    print("  " + "-" * 58)

    hard_curves = None
    for name, make in tasks.items():
        runs = [solve_task(*make(seed=seed), seed) for seed in range(SEEDS)]
        ps_curve = torch.tensor([r[0] for r in runs]).mean(0).tolist()
        es_curve = torch.tensor([r[1] for r in runs]).mean(0).tolist()
        ps_acc, es_acc = ps_curve[-1], es_curve[-1]
        print(f"  {name:<26}{ps_acc:>11.1f}%{es_acc:>11.1f}%{ps_acc - es_acc:>+7.1f}")
        if "hard" in name:
            hard_curves = (ps_curve, es_curve, runs[0][2])

    try:
        import matplotlib.pyplot as plt

        ps_curve, es_curve, popsize = hard_curves
        evals = [popsize * (i + 1) for i in range(GENERATIONS)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(evals, ps_curve, label=f"PolyStepES ({ps_curve[-1]:.1f}%)", lw=2)
        ax.plot(evals, es_curve, label=f"OpenAI-ES ({es_curve[-1]:.1f}%)", lw=2)
        ax.set_xlabel("forward evaluations")
        ax.set_ylabel(f"best accuracy (%), mean of {SEEDS} seeds")
        ax.set_title("STE-free binary net on XOR checkerboard")
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).parent / "figures" / "binary_net_no_ste.png"
        fig.savefig(out, dpi=110)
        print(f"  saved figure: {out}")
    except Exception as e:  # plotting is optional / headless-safe
        print(f"  (skipped figure: {e})")


if __name__ == "__main__":
    main()
