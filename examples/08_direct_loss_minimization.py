"""Direct loss minimization: optimize the true task metric, no surrogate.

Non-decomposable metrics like F1 are piecewise-constant in the parameters, so
the usual differentiable surrogate leaves regret (Hazan, Keshet, McAllester,
NeurIPS 2010; Song, Schwing, Zemel, Urtasun, ICML 2016). Here a small
sign-activation net is optimized directly for F1 on a class-imbalanced XOR
checkerboard, compared against Adam+STE on cross-entropy and OpenAI-ES on F1,
averaged over seeds. PolyStepES and OpenAI-ES share an evaluation budget;
Adam+STE uses a separate 1,500-step gradient baseline.

Run:
    python examples/08_direct_loss_minimization.py
"""

from __future__ import annotations

from pathlib import Path

import torch

import _env

_env.setup()

from polystep import PolyStepES

from _openai_es import OpenAIES, run

HIDDEN = 8
GENERATIONS = 200
SEEDS = 5


def make_checkerboard_imbalanced(n=600, k=3, pos_frac=0.25, noise=0.05, seed=0):
    """k x k XOR grid with the positive class subsampled to a minority."""
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
    d_in = X.shape[1]
    W1, b1, W2, b2 = flat.split([d_in * hidden, hidden, hidden, 1], dim=1)
    h = torch.sign(torch.einsum("bhi,ni->bnh", W1.reshape(-1, hidden, d_in), X) + b1[:, None, :])
    return torch.einsum("bh,bnh->bn", W2, h) + b2


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
    flat = torch.cat([W1.flatten(), b1, W2.flatten(), b2]).detach()
    return 100.0 * (1.0 - f1_error(flat[None], X, y, hidden)).item()


def solve_seed(seed):
    X, y = make_checkerboard_imbalanced(seed=seed)
    dim = param_dim()
    x0 = 0.5 * torch.randn(dim, generator=torch.Generator().manual_seed(seed + 100))

    ps = PolyStepES(dim, num_particles=1, epsilon=0.05, step_radius=1.5, x0=x0, seed=seed)
    es = OpenAIES(dim, popsize=ps.popsize, x0=x0, sigma=0.25, lr=0.5, seed=seed)
    ps_curve = run(ps, lambda f: f1_error(f, X, y), GENERATIONS)
    es_curve = run(es, lambda f: f1_error(f, X, y), GENERATIONS)
    adam_f1 = adam_ste_f1(X, y, seed=seed)
    return ps_curve, es_curve, adam_f1, ps.popsize


def _mean_std(vals):
    t = torch.tensor(vals)
    return t.mean().item(), t.std(unbiased=False).item()


def main():
    print("Direct loss minimization: sign-activation net, maximize F1 on imbalanced checkerboard")
    print(f"  params={param_dim()}  hidden={HIDDEN}  generations={GENERATIONS}  seeds={SEEDS}")

    ps_curves, es_curves, adam_final = [], [], []
    for seed in range(SEEDS):
        ps_curve, es_curve, adam_f1, popsize = solve_seed(seed)
        ps_curves.append(ps_curve)
        es_curves.append(es_curve)
        adam_final.append(adam_f1)

    ps_m, ps_s = _mean_std([c[-1] for c in ps_curves])
    es_m, es_s = _mean_std([c[-1] for c in es_curves])
    ad_m, ad_s = _mean_std(adam_final)
    print(f"  {'method':<28}{'F1 (%)':>16}")
    print("  " + "-" * 44)
    print(f"  {'PolyStepES on F1':<28}{ps_m:>10.1f} +/- {ps_s:<4.1f}")
    print(f"  {'Adam+STE on cross-entropy':<28}{ad_m:>10.1f} +/- {ad_s:<4.1f}")
    print(f"  {'OpenAI-ES on F1':<28}{es_m:>10.1f} +/- {es_s:<4.1f}")
    print(f"  PolyStep over Adam+STE: {ps_m - ad_m:+.1f} pts   over OpenAI-ES: {ps_m - es_m:+.1f} pts")

    try:
        import matplotlib.pyplot as plt

        evals = [popsize * (i + 1) for i in range(GENERATIONS)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(evals, torch.tensor(ps_curves).mean(0), label=f"PolyStepES on F1 ({ps_m:.1f}%)", lw=2)
        ax.plot(evals, torch.tensor(es_curves).mean(0), label=f"OpenAI-ES on F1 ({es_m:.1f}%)", lw=2)
        ax.axhline(ad_m, ls="--", color="gray", label=f"Adam+STE ({ad_m:.1f}%)")
        ax.set_xlabel("forward evaluations")
        ax.set_ylabel(f"best F1 (%), mean of {SEEDS} seeds")
        ax.set_title("Direct F1 optimization on imbalanced checkerboard")
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).parent / "figures" / "direct_loss_minimization.png"
        fig.savefig(out, dpi=110)
        print(f"  saved figure: {out}")
    except Exception as e:  # plotting is optional / headless-safe
        print(f"  (skipped figure: {e})")


if __name__ == "__main__":
    main()
