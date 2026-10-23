"""05 - MNIST: train a 2-layer MLP with PolyStep.

Recommended configuration: a ``HybridSubspace`` with cosine-scheduled epsilon,
step_radius, and probe_radius, plus validation-based checkpoint selection. MNIST is
downloaded directly (no torchvision) and stays on the device, so each epoch reads
the loss back once. On CUDA the model and data run in BF16, which speeds up each
step; CPU keeps FP32, where BF16 is slower. The first layer's basis is a sparse
projection, so CUDA runs repeat exactly only under deterministic algorithms (see
docs/determinism.md); otherwise repeated runs differ by a few tenths of a point in
test accuracy.

Run:
  python examples/05_mnist.py
  python examples/05_mnist.py --device cuda --epochs 30
  python examples/05_mnist.py --device cpu --train-size 2000 --epochs 2
"""

from __future__ import annotations

import argparse

import torch

import _env

_env.setup()
import torch.nn as nn
from torch.utils.data import random_split

from polystep import PolyStepOptimizer
from polystep.benchmarks import MNISTNet
from polystep.epsilon import CosineEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

from _mnist_data import get_mnist_tensors


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return (model(x).argmax(-1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="MNIST with PolyStep")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (30 reach about 95.5%% test).")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=0, help="Cap MNIST training samples (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden < 1:
        parser.error("--epochs and --hidden must be positive")
    if args.train_size < 0 or 0 < args.train_size < 10:
        parser.error("--train-size must be 0 or at least 10")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("MNIST Training with PolyStep (HybridSubspace + Softmax)")
    print("=" * 60)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    train_x, train_y, test_x, test_y = get_mnist_tensors(args.train_size)
    train_x, test_x = train_x.to(device, dtype), test_x.to(device, dtype)
    train_y, test_y = train_y.to(device), test_y.to(device)
    train, val = random_split(range(len(train_x)), [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed))
    val_x, val_y = train_x[val.indices], train_y[val.indices]
    train_x, train_y = train_x[train.indices], train_y[train.indices]
    model = MNISTNet(hidden=args.hidden).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    batch_size = 512
    steps_per_epoch = (len(train_x) + batch_size - 1) // batch_size

    # Compress each weight matrix to a rank-8 search subspace.
    total_steps = args.epochs * steps_per_epoch
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=8)

    # eps_target 0.5, not 0.1: a plan concentrated near argmax takes the full
    # step_radius, so a small target makes the effective step grow late and diverge.
    eps_init, eps_target = 10.0, 0.5
    sr_init, sr_target = 5.0, 1.0
    pr_init, pr_target = 10.0, 2.0

    optimizer = PolyStepOptimizer(
        model,
        seed=args.seed,
        subspace=subspace,
        solver="softmax",
        epsilon=CosineEpsilon(init=eps_init, target=eps_target, total_steps=total_steps),
        step_radius=CosineEpsilon(init=sr_init, target=sr_target, total_steps=total_steps),
        probe_radius=CosineEpsilon(init=pr_init, target=pr_target, total_steps=total_steps),
        # Only every fifth step pays for probes; momentum steps need no forward pass.
        amortize_steps=5,
        mixed_precision=dtype == torch.bfloat16,
    )

    print(f"  params: {num_params:,}  device: {device}  epochs: {args.epochs}")
    print(f"  subspace: HybridSubspace rank=8, {subspace.subspace_dim:,} coordinates  solver: softmax")
    print(f"  eps: {eps_init}->{eps_target}  sr: {sr_init}->{sr_target}  pr: {pr_init}->{pr_target}")
    print()

    print(f"  initial validation accuracy: {100 * accuracy(model, val_x, val_y):.1f}%")
    print()

    # Select checkpoints on held-out training data; test once after selection.
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None

    print("training...")
    for epoch in range(args.epochs):
        epoch_loss = torch.zeros((), device=device)
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), batch_size):
            idx = order[start : start + batch_size]
            inputs, targets = train_x[idx], train_y[idx]
            optimizer.step_batch(inputs, targets, loss_fn=loss_fn)

            with torch.no_grad():
                epoch_loss += loss_fn(model(inputs), targets)

        avg_loss = epoch_loss.item() / steps_per_epoch
        val_acc = accuracy(model, val_x, val_y)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:2d} | loss={avg_loss:.4f} | val={100 * val_acc:.1f}% | best={100 * best_acc:.1f}%")

    model.load_state_dict(best_state)
    final_acc = accuracy(model, test_x, test_y)
    print()
    print("=" * 60)
    print(f"  final test accuracy: {100 * final_acc:.1f}% (validation-selected checkpoint)")
    print("=" * 60)


if __name__ == "__main__":
    main()
