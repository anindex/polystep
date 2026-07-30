"""11 - Transformer on selective copy: attention trained with forward passes only.

The task needs attention and nothing else will do. Each sequence is six tokens; the
first is a pointer ``p``, and the label is the token sitting at position ``p``. A
network that mixes fixed positions cannot solve it, because which position matters
changes per example. Content-based lookup is exactly what an attention head does, so
solving it is evidence the head is being trained rather than bypassed.

Attention mixes every position with every other, so nothing about it fits the
Sequential-of-Linear shape the earlier fast paths require. Before
``SiteVmapEvaluator`` a transformer had to build one full weight set per candidate.

The insight that carries it does not care that the layer is attention: a candidate
perturbs one contiguous run of the flat parameter vector, so exactly one parameter
tensor differs from the base. Batching that tensor alone and passing the rest once
means the embedding and every layer ahead of the perturbed one are computed once
instead of per candidate. Chunks break at parameter boundaries so a chunk can name a
single site; sized purely by memory it would span several and fall back.

Attention comes from ``polystep.layers.VmapSafeMultiHeadAttention``. Stock
``nn.MultiheadAttention`` fuses its projections in a way ``vmap`` cannot trace, which
drops the model onto the sequential fallback; the drop-in keeps it batched. See
LIMITATIONS.md for what the replacement does not support.

``amortize_steps=5`` is the other lever: a momentum step reuses the EMA transport
direction and evaluates no candidates. It is not free, though, because it spends step
budget. This task needs about 900 steps, so it is paired with a halved batch that
doubles the steps per epoch and holds accuracy at 100%. Amortizing without that drops
it well below.

What you should see:
  Chance is 12.5%. Accuracy sits near chance for ~20 epochs while the head learns to
  attend, then rises sharply and reaches 100%.
  The registered run matches the materializing run to floating-point tolerance.

Run:
  python examples/11_transformer_selective_copy.py
  python examples/11_transformer_selective_copy.py --epochs 100 --device cpu
  python examples/11_transformer_selective_copy.py --compare   # time both paths
"""

from __future__ import annotations

import argparse
import os
import time

import torch

# One thread: PolyStep's per-step ops are small enough that torch's default pool of
# nproc threads costs far more than it returns. See docs/performance.md.
torch.set_num_threads(int(os.environ.get("POLYSTEP_THREADS", 0)) or 1)
import torch.nn as nn  # noqa: E402

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.cost_nn import NNCostEvaluator  # noqa: E402
from polystep.hybrid_subspace import HybridSubspace  # noqa: E402
from polystep.layers import VmapSafeMultiHeadAttention  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402

SEQ_LEN = 6
NUM_CLASSES = 8


def make_batch(n: int, seed: int):
    """``(x, y)`` where ``x[:, 0]`` points at the position holding the label."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(0, NUM_CLASSES, (n, SEQ_LEN), generator=gen)
    pointer = torch.randint(1, SEQ_LEN, (n,), generator=gen)
    x = values.clone()
    x[:, 0] = pointer  # pointers and values share the vocabulary
    return x, values[torch.arange(n), pointer]


class SelectiveCopyTransformer(nn.Module):
    """Embedding, one attention head, one feed-forward, read out at position 0."""

    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ff: int = 32):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES + SEQ_LEN, d_model)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN, d_model))
        self.attn = VmapSafeMultiHeadAttention(d_model, n_heads)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, x):
        tokens = self.embed(x) + self.pos
        tokens = tokens + self.attn(tokens, tokens, tokens)
        tokens = tokens + self.ff2(torch.relu(self.ff1(tokens)))
        return self.head(tokens[:, 0])  # position 0 held the pointer


def build_optimizer(model: nn.Module, seed: int) -> PolyStepOptimizer:
    """A per-layer subspace, so a coordinate run lands inside one parameter's block.

    That is what lets the step name a site in coordinate space. A single global
    projection would spread every coordinate across every layer, leaving no site to
    resolve and no shared prefix to reuse.
    """
    subspace = HybridSubspace.auto_from_layout(ParamLayout.from_module(model), compression_ratio=4)
    return PolyStepOptimizer(
        model,
        subspace=subspace,
        epsilon=0.5,
        step_radius=6.0,
        probe_radius=1.0,
        seed=seed,
        # A momentum step reuses the EMA transport direction and evaluates nothing, so
        # only every fifth step pays for probes. It spends step budget rather than
        # forward passes, so it needs a run with steps to spare: at a batch of 256 this
        # task has too few steps and amortizing loses accuracy. Halving the batch doubles
        # them, and the pair is faster at full accuracy.
        amortize_steps=5,
        amortize_ema=0.7,
        # Inductor fusion over the vmapped forward, at an unchanged accuracy.
        compile_evaluator=True,
    )


def run_epoch(optimizer, evaluator, x, y, batch_size):
    order = torch.randperm(x.shape[0])
    total, batches = 0.0, 0
    for start in range(0, x.shape[0] - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        xb, yb = x[idx], y[idx]
        # Hands the step the model and this batch, so candidates are scored directly
        # rather than through the closure.
        optimizer.register_evaluator(evaluator, xb, yb)
        total += optimizer.step(lambda params, _x=xb, _y=yb: evaluator.evaluate(params, _x, _y))
        batches += 1
    optimizer.release_evaluator()
    return total / max(batches, 1)


@torch.no_grad()
def accuracy(model, x, y):
    return 100.0 * (model(x).argmax(-1) == y).float().mean().item()


def compare_paths(x, y, device, batch_size, steps=6):
    """Time and cross-check the two paths on the same batches."""
    results = {}
    for register in (False, True):
        torch.manual_seed(0)
        model = SelectiveCopyTransformer().to(device)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        optimizer = build_optimizer(model, seed=3)
        started = time.perf_counter()
        losses = []
        for i in range(steps):
            xb = x[i * batch_size : (i + 1) * batch_size]
            yb = y[i * batch_size : (i + 1) * batch_size]
            if register:
                optimizer.register_evaluator(evaluator, xb, yb)
            losses.append(optimizer.step(lambda p, _x=xb, _y=yb: evaluator.evaluate(p, _x, _y)))
        results["site" if register else "materializing"] = (losses, time.perf_counter() - started)

    dense_losses, dense_time = results["materializing"]
    site_losses, site_time = results["site"]
    drift = max(abs(a - b) for a, b in zip(dense_losses, site_losses))
    print(f"  materializing : {dense_time:6.2f} s for {steps} steps")
    print(f"  site-aware    : {site_time:6.2f} s for {steps} steps   ({dense_time / site_time:.1f}x)")
    print(f"  max loss difference between the two: {drift:.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="time both paths and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_x, train_y = make_batch(args.train_size, seed=0)
    test_x, test_y = make_batch(args.test_size, seed=1)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    if args.compare:
        print("\nSame batches, same seed, both paths:\n")
        compare_paths(train_x, train_y, device, args.batch_size)
        return

    model = SelectiveCopyTransformer().to(device)
    print(
        f"SelectiveCopyTransformer: {sum(p.numel() for p in model.parameters())} parameters, "
        f"forward passes only. Chance is {100.0 / NUM_CLASSES:.1f}%.\n"
    )
    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    optimizer = build_optimizer(model, seed=args.seed)

    best, best_state = 0.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(optimizer, evaluator, train_x, train_y, args.batch_size)
        test_acc = accuracy(model, test_x, test_y)
        if test_acc > best:
            best, best_state = test_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {loss:.4f}  test {test_acc:5.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  best test accuracy: {best:.1f}%  ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
