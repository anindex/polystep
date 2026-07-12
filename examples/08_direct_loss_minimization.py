"""Direct loss minimization: optimize the true task metric, no surrogate.

Many task metrics (F1, exact-match, edit distance) are non-decomposable and
piecewise-constant in the model parameters, so they carry no useful gradient.
The usual workaround trains a differentiable surrogate (cross-entropy), but the
surrogate optimum is not the metric optimum, so it leaves regret (Hazan,
Keshet, McAllester, NeurIPS 2010; Song, Schwing, Zemel, Urtasun, ICML 2016).

Here a small sign-activation net is optimized directly for F1 on a class-imbalanced
XOR checkerboard. The parameter count is small, so no subspace is used and the
comparison is fair: same parameters, same evaluation budget, averaged over seeds.

  Adam+STE on cross-entropy: the sign net has no true gradient, so Adam uses a
    straight-through estimator whose forward/backward mismatch is biased, and it
    optimizes cross-entropy rather than F1 (surrogate on both counts).
  OpenAI-ES on F1: the fragmented boundary makes F1 piecewise-constant with wide
    plateaus, so the isotropic-noise gradient estimate averages to no signal.
  PolyStepES on F1: the soft-argmin over directed probes descends the true metric.

Run:
    MPLBACKEND=Agg python examples/08_direct_loss_minimization.py
"""

from __future__ import annotations

import torch

from polystep import PolyStepES

HIDDEN = 8
GENERATIONS = 200
SEEDS = 5


def make_checkerboard_imbalanced(n=600, k=3, pos_frac=0.25, noise=0.05, seed=0):
    """k x k XOR grid with the positive class subsampled to a minority.

    The fragmented boundary makes F1 piecewise-constant with wide plateaus; the
    imbalance makes the cross-entropy 0-threshold F1-suboptimal.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * k
    lab = (X[:, 0].floor().long() + X[:, 1].floor().long()) % 2
    pos_idx = (lab == 1).nonzero().squeeze(1)
    neg_idx = (lab == 0).nonzero().squeeze(1)
    keep_pos = pos_idx[: int(len(neg_idx) * pos_frac / (1 - pos_frac))]
    idx = torch.cat([neg_idx, keep_pos])
    X = X[idx] + noise * torch.randn(len(idx), 2, generator=g)
    y = lab[idx].float()
    return (X - X.mean(0)) / X.std(0), y


def param_dim(hidden=HIDDEN, d_in=2):
    return d_in * hidden + hidden + hidden + 1  # W1, b1, W2, b2


def _signnet_logits(flat, X, hidden):
    """Sign-activation MLP output for a batch of flat param vectors: (B, N)."""
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
    h = torch.sign(torch.einsum("bhi,ni->bnh", W1, X) + b1[:, None, :])
    return torch.einsum("boh,bnh->bno", W2, h).squeeze(-1) + b2


def f1_error(flat, X, y, hidden=HIDDEN):
    """1 - F1 of a sign-activation MLP. Non-decomposable and piecewise-constant."""
    out = _signnet_logits(flat, X, hidden)
    pred = (out > 0).float()
    yb = y[None, :]
    tp = (pred * yb).sum(dim=1)
    fp = (pred * (1 - yb)).sum(dim=1)
    fn = ((1 - pred) * yb).sum(dim=1)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return 1.0 - f1


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
        curve.append(100.0 * (1.0 - opt.best_fitness))  # best F1 so far, percent
    return curve


class _STESign(torch.autograd.Function):
    """sign() forward, hardtanh straight-through gradient backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1).float()


def adam_ste_f1(X, y, hidden=HIDDEN, steps=1500, seed=0):
    """Train the same sign net with STE + cross-entropy, then report its F1."""
    torch.manual_seed(seed)
    W1 = torch.randn(hidden, X.shape[1], requires_grad=True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = torch.randn(1, hidden, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=0.03)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        out = (_STESign.apply(X @ W1.t() + b1) @ W2.t()).squeeze(-1) + b2
        loss_fn(out, y).backward()
        opt.step()
    with torch.no_grad():
        out = (torch.sign(X @ W1.t() + b1) @ W2.t()).squeeze(-1) + b2
    pred = (out > 0).float()
    tp = (pred * y).sum()
    fp = (pred * (1 - y)).sum()
    fn = ((1 - pred) * y).sum()
    return 100.0 * (2 * tp / (2 * tp + fp + fn + 1e-9)).item()


def solve_seed(seed):
    X, y = make_checkerboard_imbalanced(seed=seed)
    dim = param_dim()
    popsize = 2 * dim  # PolyStep orthoplex population for one particle
    x0 = 0.5 * torch.randn(dim, generator=torch.Generator().manual_seed(seed + 100))

    ps = PolyStepES(dim, num_particles=1, epsilon=0.05, step_radius=1.5, x0=x0, seed=seed)
    es = OpenAIES(dim, popsize=popsize, x0=x0, sigma=0.5, lr=0.2, seed=seed)
    ps_curve = run(ps, lambda f: f1_error(f, X, y), GENERATIONS)
    es_curve = run(es, lambda f: f1_error(f, X, y), GENERATIONS)
    adam_f1 = adam_ste_f1(X, y, seed=seed)
    return ps_curve, es_curve, adam_f1, popsize


def _mean_std(vals):
    t = torch.tensor(vals)
    return t.mean().item(), t.std(unbiased=False).item()


def main():
    print("Direct loss minimization: sign-activation net, maximize F1 on imbalanced checkerboard")
    print(f"  params={param_dim()}  hidden={HIDDEN}  generations={GENERATIONS}  seeds={SEEDS}  (no subspace)")

    ps_final, es_final, adam_final = [], [], []
    last_curves = None
    for seed in range(SEEDS):
        ps_curve, es_curve, adam_f1, popsize = solve_seed(seed)
        ps_final.append(ps_curve[-1])
        es_final.append(es_curve[-1])
        adam_final.append(adam_f1)
        last_curves = (ps_curve, es_curve, popsize)

    ps_m, ps_s = _mean_std(ps_final)
    es_m, es_s = _mean_std(es_final)
    ad_m, ad_s = _mean_std(adam_final)
    print(f"  {'method':<28}{'F1 (%)':>16}")
    print("  " + "-" * 44)
    print(f"  {'PolyStepES on F1':<28}{ps_m:>10.1f} +/- {ps_s:<4.1f}")
    print(f"  {'Adam+STE on cross-entropy':<28}{ad_m:>10.1f} +/- {ad_s:<4.1f}")
    print(f"  {'OpenAI-ES on F1':<28}{es_m:>10.1f} +/- {es_s:<4.1f}")
    print(f"  PolyStep over Adam+STE: {ps_m - ad_m:+.1f} pts   over OpenAI-ES: {ps_m - es_m:+.1f} pts")

    try:
        import matplotlib.pyplot as plt

        ps_curve, es_curve, popsize = last_curves
        evals = [popsize * (i + 1) for i in range(GENERATIONS)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(evals, ps_curve, label=f"PolyStepES on F1 ({ps_curve[-1]:.1f}%)", lw=2)
        ax.plot(evals, es_curve, label=f"OpenAI-ES on F1 ({es_curve[-1]:.1f}%)", lw=2)
        ax.axhline(adam_final[-1], ls="--", color="gray", label=f"Adam+STE ({adam_final[-1]:.1f}%)")
        ax.set_xlabel("forward evaluations")
        ax.set_ylabel("best F1 (%)")
        ax.set_title("Direct F1 minimization on imbalanced checkerboard")
        ax.legend()
        fig.tight_layout()
        out = "examples/figures/direct_loss_minimization.png"
        fig.savefig(out, dpi=110)
        print(f"  saved figure: {out}")
    except Exception as e:  # plotting is optional / headless-safe
        print(f"  (skipped figure: {e})")


if __name__ == "__main__":
    main()
