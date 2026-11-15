"""11 - Transformer on selective copy: attention trained with forward passes only.

Each sequence is six tokens; the first is a pointer ``p``, and the label is the
token at position ``p``. The attention layer learns to retrieve that token.

As in example 10, the site-aware path applies: a candidate perturbs one
contiguous run of the flat parameter vector, so exactly one parameter tensor
differs from the base, and everything ahead of it is computed once per chunk of
candidates instead of once per candidate.

Attention uses ``polystep.layers.VmapSafeMultiHeadAttention``: ``vmap`` has no
batching rule for the fused kernel of stock ``nn.MultiheadAttention``, so it
runs that kernel once per candidate, which makes a CUDA step two to three times
slower. See LIMITATIONS.md.

Run:
  python examples/11_transformer_selective_copy.py
  python examples/11_transformer_selective_copy.py --device cpu
  python examples/11_transformer_selective_copy.py --compare   # time both paths
"""

from __future__ import annotations

import argparse
import time

import torch

import _env

_env.setup()
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.hybrid_subspace import HybridSubspace
from polystep.layers import VmapSafeMultiHeadAttention
from polystep.transform import ParamLayout

from _site_paths import compare_paths, run_epoch

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
    """Embedding, one attention layer with two heads, one feed-forward, read out at position 0."""

    def __init__(self, d_model: int = 16, n_heads: int = 2, d_ff: int = 32):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, d_model)
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


def build_optimizer(model: nn.Module, seed: int, amortize_steps: int = 5) -> PolyStepOptimizer:
    """A per-layer subspace, so a coordinate run lands inside one parameter's block.

    That is what lets the step name a site in coordinate space; a single global
    projection would spread every coordinate across every layer.
    """
    subspace = HybridSubspace.auto_from_layout(ParamLayout.from_module(model), compression_ratio=4)
    return PolyStepOptimizer(
        model,
        subspace=subspace,
        epsilon=0.5,
        step_radius=6.0,
        probe_radius=3.0,
        seed=seed,
        # A momentum step reuses the EMA transport direction and evaluates nothing,
        # so only every fifth step pays for probes.
        amortize_steps=amortize_steps,
    )


@torch.no_grad()
def accuracy(model, x, y):
    return 100.0 * (model(x).argmax(-1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="time both paths and exit")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_size, args.test_size) < 1:
        parser.error("epochs, batch size and split sizes must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_x, train_y = make_batch(args.train_size, seed=0)
    test_x, test_y = make_batch(args.test_size, seed=1)
    val_x, val_y = make_batch(args.test_size, seed=2)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    if args.compare:
        print("\nSame batches, same seed, both paths:\n")
        compare_paths(
            SelectiveCopyTransformer,
            lambda model: build_optimizer(model, seed=3, amortize_steps=1),
            train_x,
            train_y,
            device,
            args.batch_size,
            steps=6,
        )
        return

    model = SelectiveCopyTransformer().to(device)
    print(
        f"SelectiveCopyTransformer: {sum(p.numel() for p in model.parameters())} parameters, "
        f"forward passes only. Chance is {100.0 / NUM_CLASSES:.1f}%.\n"
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, seed=args.seed)

    best, best_state = 0.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(optimizer, loss_fn, train_x, train_y, args.batch_size)
        val_acc = accuracy(model, val_x, val_y)
        if val_acc >= best:
            best, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss {loss:.4f}  val {val_acc:5.1f}%")
    optimizer.release_evaluator()

    model.load_state_dict(best_state)
    print(f"\n  test accuracy: {accuracy(model, test_x, test_y):.1f}%  ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
