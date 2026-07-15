"""Training a hard oblique decision tree with no gradients and no relaxation.

An oblique decision tree routes each sample with a strict test at every internal
node: go right when w.x + b > 0, else left. The sample lands in exactly one leaf
and takes that leaf's label. The loss is piecewise-constant in the node weights:
its gradient is zero almost everywhere and undefined at the split boundaries.

That breaks the usual toolbox:

  * Adam cannot touch the hard tree, so the standard fix is a soft relaxation
    (sigmoid routing). The relaxed model trains, but hardening it back to real
    splits at inference loses much of the accuracy it appeared to reach.
  * OpenAI-ES and SPSA estimate a local gradient from small perturbations. On a
    flat piece the perturbed losses are equal, the estimate averages to zero, and
    they stall.

PolyStep is a randomized direct search: it probes a finite-radius polytope around
the current parameters and moves toward the best vertices by an optimal-transport
barycenter. A finite radius steps across split boundaries, so it optimizes the
hard tree directly.

The script trains the same hard tree with PolyStep, OpenAI-ES, and SPSA under a
matched forward-pass budget, plus an Adam soft-tree baseline scored after
hardening, and reports accuracy on a synthetic tabular task.

Run:
    MPLBACKEND=Agg python examples/09_hard_decision_tree.py
"""

from __future__ import annotations

import torch

from polystep import PolyStepES

DEPTH = 4  # complete binary tree: 15 internal nodes, 16 leaves
TOTAL_EVALS = 40_000  # matched forward-pass budget for the gradient-free methods


def make_checkerboard(n=600, k=3, noise=0.05, seed=0):
    """k x k XOR grid: a fragmented boundary a shallow tree can carve."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * k
    y = ((X[:, 0].floor().long() + X[:, 1].floor().long()) % 2).float()
    X = (X - X.mean(0)) / X.std(0) + noise * torch.randn(n, 2, generator=g)
    return X, y


def tree_shape(depth=DEPTH):
    n_internal = 2**depth - 1
    n_leaves = 2**depth
    return n_internal, n_leaves


def param_dim(d_in=2, depth=DEPTH):
    n_internal, n_leaves = tree_shape(depth)
    return n_internal * (d_in + 1) + n_leaves  # node (w, b) + one leaf logit each


def _unpack(flat, d_in, depth):
    """Split a (B, D) parameter batch into node weights, node biases, leaf logits."""
    n_internal, n_leaves = tree_shape(depth)
    B = flat.shape[0]
    i = 0
    W = flat[:, i : i + n_internal * d_in].reshape(B, n_internal, d_in)
    i += n_internal * d_in
    b = flat[:, i : i + n_internal]
    i += n_internal
    leaf = flat[:, i : i + n_leaves]
    return W, b, leaf


def hard_error(flat, X, y, depth=DEPTH):
    """0-1 error of a hard oblique tree for a batch of flat parameter vectors.

    flat: (B, D) -> (B,) misclassification rate. Routing is a strict > 0 test at
    each node, so the whole map is piecewise-constant in flat.
    """
    B, d_in = flat.shape[0], X.shape[1]
    N = X.shape[0]
    W, b, leaf = _unpack(flat, d_in, depth)
    logits = torch.einsum("bnd,md->bnm", W, X) + b[:, :, None]  # (B, n_internal, N)
    node = torch.zeros(B, N, dtype=torch.long)  # global node index, root = 0
    for _ in range(depth):
        dec = torch.gather(logits, 1, node.unsqueeze(1)).squeeze(1)  # (B, N)
        node = 2 * node + 1 + (dec > 0).long()
    leaf_idx = node - (2**depth - 1)  # (B, N) in [0, n_leaves)
    pred = torch.gather(leaf, 1, leaf_idx) > 0  # (B, N)
    return (pred.float() != y[None, :]).float().mean(dim=1)


class OpenAIES:
    """OpenAI-ES ask/tell (Salimans et al., 2017): antithetic sampling, z-scored
    fitness shaping, estimate g = (1/(pop*sigma)) sum(shaped * eps)."""

    def __init__(self, dim, popsize, x0, sigma=0.3, lr=0.15, seed=0):
        self.dim = dim
        self.popsize = popsize + (popsize % 2)
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


class SPSA:
    """SPSA ask/tell (Spall, 1992): a two-point Rademacher gradient estimate with
    decaying gain sequences a_k and c_k."""

    def __init__(self, dim, x0, a=0.2, c=0.2, alpha=0.602, gamma=0.101, seed=0):
        self.dim = dim
        self.theta = x0.clone()
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma
        self.k = 0
        self.generator = torch.Generator().manual_seed(seed)
        self._delta = None
        self._ck = None
        self.best_fitness = float("inf")

    def ask(self):
        self._ck = self.c / (self.k + 1) ** self.gamma
        self._delta = (torch.randint(0, 2, (self.dim,), generator=self.generator) * 2 - 1).float()
        return torch.stack([self.theta + self._ck * self._delta, self.theta - self._ck * self._delta])

    def tell(self, fitness):
        self.best_fitness = min(self.best_fitness, fitness.min().item())
        ak = self.a / (self.k + 1 + 10) ** self.alpha
        ghat = (fitness[0] - fitness[1]) / (2.0 * self._ck * self._delta)
        self.theta = self.theta - ak * ghat
        self.k += 1


def run_to_budget(opt, fit_fn, popsize, budget=TOTAL_EVALS):
    """Run ask/tell until the forward-pass budget is spent; return (evals, best-acc)."""
    evals, curve = [], []
    used = 0
    while used < budget:
        opt.tell(fit_fn(opt.ask()))
        used += popsize
        evals.append(used)
        curve.append(100.0 * (1.0 - opt.best_fitness))
    return evals, curve


def adam_soft_tree(X, y, depth=DEPTH, steps=800, temp=0.2, seed=0):
    """Adam on a sigmoid-relaxed tree; return accuracy after hardening the routes.

    Soft routing sends a fraction of each sample down both children with weight
    sigmoid((w.x + b) / temp). Adam optimizes that smooth surrogate; the returned
    number is the accuracy of the real hard tree built from the trained weights.
    """
    torch.manual_seed(seed)
    d_in = X.shape[1]
    D = param_dim(d_in, depth)
    n_internal, n_leaves = tree_shape(depth)
    flat = torch.nn.Parameter(0.3 * torch.randn(D))
    opt = torch.optim.Adam([flat], lr=0.05)
    y2 = y.long()

    for _ in range(steps):
        W, b, leaf = _unpack(flat.unsqueeze(0), d_in, depth)
        node_logits = (torch.einsum("bnd,md->bnm", W, X) + b[:, :, None]).squeeze(0)  # (n_internal, N)
        p_right = torch.sigmoid(node_logits / temp)  # (n_internal, N)
        leaf_probs = []  # per-leaf reach probability, kept out of place for autograd
        for leaf_id in range(n_leaves):
            node = 0
            p = torch.ones(X.shape[0])
            for level in range(depth):
                bit = (leaf_id >> (depth - 1 - level)) & 1
                pr = p_right[node]
                p = p * (pr if bit else (1.0 - pr))
                node = 2 * node + 1 + bit
            leaf_probs.append(p)
        prob = torch.stack(leaf_probs, 0)  # (n_leaves, N)
        leaf_logit = leaf.squeeze(0)  # (n_leaves,)
        soft_logit = (prob * leaf_logit[:, None]).sum(0)  # (N,)
        p1 = torch.sigmoid(soft_logit)
        loss = torch.nn.functional.binary_cross_entropy(p1, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    hard_acc = 100.0 * (1.0 - hard_error(flat.detach().unsqueeze(0), X, y, depth).item())
    return hard_acc, y2  # y2 unused, kept for clarity


def solve(X, y, seed=0):
    D = param_dim(X.shape[1])
    x0 = 0.3 * torch.randn(D, generator=torch.Generator().manual_seed(1))
    ps_pop = 2 * D  # one PolyStep particle: orthoplex has 2*D vertices
    ps = PolyStepES(D, num_particles=1, epsilon=0.02, step_radius=1.5, x0=x0, seed=seed)
    es = OpenAIES(D, popsize=ps_pop, x0=x0, sigma=0.3, lr=0.15, seed=seed)
    spsa = SPSA(D, x0=x0, seed=seed)

    def fit(flat):
        return hard_error(flat, X, y)

    ps_evals, ps_curve = run_to_budget(ps, fit, ps_pop)
    es_evals, es_curve = run_to_budget(es, fit, ps_pop)
    spsa_evals, spsa_curve = run_to_budget(spsa, fit, 2)
    adam_hard, _ = adam_soft_tree(X, y)
    return {
        "PolyStep": (ps_evals, ps_curve),
        "OpenAI-ES": (es_evals, es_curve),
        "SPSA": (spsa_evals, spsa_curve),
        "Adam (soft, hardened)": adam_hard,
    }


def main():
    torch.manual_seed(0)
    X, y = make_checkerboard()
    D = param_dim(X.shape[1])
    n_internal, n_leaves = tree_shape()
    print("Hard oblique decision tree: no gradients, no relaxation")
    print(f"  depth={DEPTH}  nodes={n_internal}  leaves={n_leaves}  params={D}  budget={TOTAL_EVALS} evals")
    res = solve(X, y)

    print(f"  {'method':<24}{'accuracy':>10}")
    print("  " + "-" * 34)
    for name in ("PolyStep", "OpenAI-ES", "SPSA"):
        print(f"  {name:<24}{res[name][1][-1]:>9.1f}%")
    print(f"  {'Adam (soft, hardened)':<24}{res['Adam (soft, hardened)']:>9.1f}%")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for name in ("PolyStep", "OpenAI-ES", "SPSA"):
            evals, curve = res[name]
            ax.plot(evals, curve, lw=2, label=f"{name} ({curve[-1]:.1f}%)")
        ax.axhline(
            res["Adam (soft, hardened)"],
            ls="--",
            color="gray",
            label=f"Adam soft, hardened ({res['Adam (soft, hardened)']:.1f}%)",
        )
        ax.set_xlabel("forward evaluations")
        ax.set_ylabel("best accuracy (%)")
        ax.set_title("Hard oblique decision tree on XOR checkerboard")
        ax.legend()
        fig.tight_layout()
        out = "examples/figures/hard_decision_tree.png"
        fig.savefig(out, dpi=110)
        print(f"  saved figure: {out}")
    except Exception as e:  # plotting is optional / headless-safe
        print(f"  (skipped figure: {e})")


if __name__ == "__main__":
    main()
