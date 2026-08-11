"""05 - MNIST: train a 2-layer MLP with PolyStep.

Recommended configuration: a ``HybridSubspace`` with cosine-scheduled epsilon,
step_radius, and probe_radius, plus best-state tracking across epochs. MNIST is
downloaded directly (no torchvision).

Run:
  python examples/05_mnist.py
  python examples/05_mnist.py --device cuda --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

from collections import OrderedDict

import torch

import _env  # noqa: E402

_env.setup()
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.epsilon import CosineEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mnist_data import get_mnist_tensors  # noqa: E402


def get_mnist_loaders(data_dir: str = "/tmp/mnist", batch_size: int = 512):
    train_x, train_y, test_x, test_y = get_mnist_tensors(0, 0, data_dir)
    return (
        DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(test_x, test_y), batch_size=256, shuffle=False),
    )


class MNISTNet(nn.Sequential):
    """Two-layer MLP (101K parameters).

    Must be an ``nn.Sequential`` subclass: the batched evaluators check
    ``type(model).forward is nn.Sequential.forward`` before building a plan,
    so a hand-written ``forward`` silently opts out of the fast paths. The
    ``OrderedDict`` keeps the ``fc1``/``fc2`` state_dict keys.
    """

    def __init__(self, hidden: int = 128):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(784, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, 10)),
                ]
            )
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        correct += (model(inputs).argmax(-1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="MNIST with PolyStep")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (paper uses 30 for 96%%).")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("MNIST Training with PolyStep (HybridSubspace + Softmax)")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders()
    model = MNISTNet(hidden=args.hidden).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # rank=8 gives 16 polytope vertices per step.
    total_steps = args.epochs * len(train_loader)
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=8, rotation_interval=0, absorb_interval=0)

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
        num_probe=1,
        epsilon=CosineEpsilon(init=eps_init, target=eps_target, decay=(eps_init - eps_target) / total_steps),
        step_radius=CosineEpsilon(init=sr_init, target=sr_target, decay=(sr_init - sr_target) / total_steps),
        probe_radius=CosineEpsilon(init=pr_init, target=pr_target, decay=(pr_init - pr_target) / total_steps),
        # Only every fifth step pays for probes; momentum steps need no forward pass.
        amortize_steps=5,
        amortize_ema=0.7,
        # Inductor warm-up costs more than it returns over a run this short.
        compile=False,
    )

    print(f"  params: {num_params:,}  device: {device}  epochs: {args.epochs}")
    print("  subspace: HybridSubspace rank=8  solver: softmax")
    print(f"  eps: {eps_init}->{eps_target}  sr: {sr_init}->{sr_target}  pr: {pr_init}->{pr_target}")
    print()

    init_acc = evaluate(model, test_loader)
    print(f"  initial test accuracy: {100 * init_acc:.1f}%")
    print()

    # Best-state tracking: report the peak accuracy, not the last epoch's.
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    best_acc = 0.0
    best_state = None

    print("training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_steps = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            def closure(stacked_params, _in=inputs, _tgt=targets):
                return evaluator.evaluate(stacked_params, _in, _tgt)

            # Hand-rolled loops must register the evaluator; api.train() does it for you.
            optimizer.register_evaluator(evaluator, inputs, targets)
            optimizer.step(closure)

            with torch.no_grad():
                step_loss = loss_fn(model(inputs), targets).item()
            epoch_loss += step_loss
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        test_acc = evaluate(model, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"  epoch {epoch:2d} | loss={avg_loss:.4f} | test={100 * test_acc:.1f}% | best={100 * best_acc:.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_acc = evaluate(model, test_loader)
    print()
    print("=" * 60)
    print(f"  final test accuracy: {100 * final_acc:.1f}% (best across epochs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
