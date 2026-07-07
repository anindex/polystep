"""STE-free training of a hard-threshold (sign-activation) net.

Binary / 1-bit nets power quantized inference, but sign() has zero gradient
almost everywhere, so backprop needs a straight-through estimator whose
forward/backward mismatch is biased, worst at 1-2 bit
(https://arxiv.org/abs/2505.18113, https://arxiv.org/pdf/2601.22660).

This skips gradients and minimizes 0-1 error (doubly non-differentiable) with two
ask/tell optimizers on a matched budget: PolyStepES vs OpenAI-ES. On a
piecewise-constant loss OpenAI-ES averages to no signal and stalls once the
boundary fragments; PolyStep still descends. Easy boundary (moons): both win.
Hard boundary (XOR checkerboard): PolyStep leads by ~20 points.

Run:
    MPLBACKEND=Agg python examples/07_binary_net_no_ste.py
"""

from __future__ import annotations

import torch

from polystep import PolyStepES

HIDDEN = 32
GENERATIONS = 200


def make_moons(n=400, noise=0.15, seed=0):
    """Two interleaving half-moons (easy boundary)."""
    import math

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

    flat: (B, D) -> (B,) misclassification rates. Fully non-differentiable:
    sign hidden activations, a hard 0-threshold decision, then 0-1 loss.
    """
    B = flat.shape[0]
    d_in = X.shape[1]
    i = 0
    W1 = flat[:, i : i + d_in * hidden].reshape(B, hidden, d_in)
    i += d_in * hidden
    b1 = flat[:, i : i + hidden]
    i += hidden
    W2 = flat[:, i : i + hidden].reshape(B, 1, hidden)
    i += hidden
    b2 = flat[:, i : i + 1]
    h = torch.sign(torch.einsum("bhi,ni->bnh", W1, X) + b1[:, None, :])  # (B, N, H)
    out = torch.einsum("boh,bnh->bno", W2, h).squeeze(-1) + b2  # (B, N)
    return ((out > 0).float() != y[None, :]).float().mean(dim=1)


class OpenAIES:
    """OpenAI-ES ask/tell (Salimans et al., 2017): antithetic sampling, z-score
    fitness shaping, gradient estimate g = (1/(pop*sigma)) * sum(shaped * eps)."""

    def __init__(self, dim, popsize, x0, sigma=0.5, lr=0.2, seed=0):
        self.dim = dim
        self.popsize = popsize + (popsize % 2)  # even, for antithetic pairs
        self.sigma = sigma
        self.lr = lr
        self.mean = x0.clone()
        self.generator = torch.Generator().manual_seed(seed)
        self._eps = None
        self.best_fitness = float("inf")

    def ask(self):
        half = torch.randn(self.popsize // 2, self.dim, generator=self.generator)
        self._eps = torch.cat([half, -half], dim=0)
        return self.mean.unsqueeze(0) + self.sigma * self._eps

    def tell(self, fitness):
        self.best_fitness = min(self.best_fitness, fitness.min().item())
        adv = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        self.mean = self.mean - self.lr * (self._eps * adv.unsqueeze(1)).mean(dim=0) / self.sigma


def run(opt, fit_fn, generations):
    curve = []
    for _ in range(generations):
        opt.tell(fit_fn(opt.ask()))
        curve.append(100.0 * (1.0 - opt.best_fitness))  # best accuracy so far
    return curve


def solve_task(name, X, y):
    D = param_dim()
    popsize = 2 * D  # PolyStep orthoplex population for one particle
    x0 = 0.5 * torch.randn(D, generator=torch.Generator().manual_seed(1))
    ps = PolyStepES(D, num_particles=1, epsilon=0.02, step_radius=2.0, x0=x0, seed=0)
    es = OpenAIES(D, popsize=popsize, x0=x0, sigma=0.5, lr=0.2, seed=0)
    ps_curve = run(ps, lambda f: error_rate(f, X, y), GENERATIONS)
    es_curve = run(es, lambda f: error_rate(f, X, y), GENERATIONS)
    return ps_curve, es_curve, popsize


def main():
    torch.manual_seed(0)
    tasks = {
        "two-moons (easy)": make_moons(),
        "XOR checkerboard (hard)": make_checkerboard(),
    }
    D = param_dim()
    print("STE-free hard-threshold net: minimize 0-1 error, no gradients")
    print(f"  sign-activation MLP  params={D}  hidden={HIDDEN}  generations={GENERATIONS}")
    print(f"  {'task':<26}{'PolyStepES':>12}{'OpenAI-ES':>12}{'gap':>8}")
    print("  " + "-" * 58)

    hard_curves = None
    for name, (X, y) in tasks.items():
        ps_curve, es_curve, popsize = solve_task(name, X, y)
        ps_acc, es_acc = ps_curve[-1], es_curve[-1]
        print(f"  {name:<26}{ps_acc:>11.1f}%{es_acc:>11.1f}%{ps_acc - es_acc:>+7.1f}")
        if "hard" in name:
            hard_curves = (ps_curve, es_curve, popsize)

    try:
        import matplotlib.pyplot as plt

        ps_curve, es_curve, popsize = hard_curves
        evals = [popsize * (i + 1) for i in range(GENERATIONS)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(evals, ps_curve, label=f"PolyStepES ({ps_curve[-1]:.1f}%)", lw=2)
        ax.plot(evals, es_curve, label=f"OpenAI-ES ({es_curve[-1]:.1f}%)", lw=2)
        ax.set_xlabel("forward evaluations")
        ax.set_ylabel("best accuracy (%)")
        ax.set_title("STE-free binary net on XOR checkerboard")
        ax.legend()
        fig.tight_layout()
        out = "examples/figures/binary_net_no_ste.png"
        fig.savefig(out, dpi=110)
        print(f"  saved figure: {out}")
    except Exception as e:  # plotting is optional / headless-safe
        print(f"  (skipped figure: {e})")


if __name__ == "__main__":
    main()
