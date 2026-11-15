"""10 - LeNet-5 on MNIST: a classical CNN trained with forward passes only.

LeCun et al. 1998, in the usual modern form (ReLU and max-pool in place of the
original tanh and average-pool): two convolutions, two hidden fully-connected
layers, 61,706 parameters. No gradients are computed anywhere.

A plain ``nn.Module`` with convolutions and a hand-written ``forward`` runs on
the site-aware path: a candidate perturbs one contiguous coordinate run, so
``SiteVmapEvaluator`` batches only the one parameter tensor that differs and
runs the layers ahead of it once per chunk of candidates instead of once per
candidate. Chunks break at parameter boundaries so a chunk can name a single site.

Only the probe radius is scheduled, cosine from 10 to 2, sized from ``--epochs``.
Epsilon stays at 10 and ``step_radius=0.5``, which the optimizer multiplies by
epsilon, giving an effective radius of 5. Checkpoints are selected on a held-out
tenth of the training data, then evaluated on the test set.

Run:
  python examples/10_cnn_mnist.py --epochs 10
  python examples/10_cnn_mnist.py --device cpu
  python examples/10_cnn_mnist.py --compare   # time both evaluation paths
"""

from __future__ import annotations

import argparse
import time

import torch

import _env

_env.setup()
import torch.nn as nn
from torch.utils.data import random_split

from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

from _mnist_data import LeNet5, get_mnist_tensors
from _site_paths import compare_paths, run_epoch


def build_optimizer(model: nn.Module, total_steps: int, seed: int, amortize_steps: int = 5) -> PolyStepOptimizer:
    """A per-layer subspace, flat epsilon and step radius, one annealed probe radius.

    Per-layer is required for the site argument: each layer owns a coordinate block,
    so a candidate's run lands inside one block and moves one parameter. A global
    projection would mix every layer into every coordinate, leaving no site to resolve.
    """
    # rank=4, not 8: halves the coordinates and the candidates per step; the conv
    # descent direction is low-rank, so the smaller perturbation carries less variance.
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4)
    return PolyStepOptimizer(
        model,
        seed=seed,
        subspace=subspace,
        epsilon=10.0,
        step_radius=0.5,
        probe_radius=CosineEpsilon(init=10.0, target=2.0, total_steps=total_steps),
        # Four momentum steps between OT steps, so only every fifth pays for probes.
        amortize_steps=amortize_steps,
    )


@torch.no_grad()
def accuracy(model, x, y, batch_size=512):
    correct = 0
    for start in range(0, x.shape[0], batch_size):
        logits = model(x[start : start + batch_size])
        correct += (logits.argmax(-1) == y[start : start + batch_size]).sum().item()
    return 100.0 * correct / x.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="time both paths and exit")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.train_size < 0 or args.test_size < 0 or 0 < args.train_size < 10:
        parser.error(
            "epochs and batch size must be positive; train size must be 0 or at least 10; test size must be >= 0"
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = get_mnist_tensors(args.train_size, args.test_size)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    if args.compare:
        print("\nSame batches, same seed, both paths:\n")
        compare_paths(
            LeNet5,
            lambda model: build_optimizer(model, total_steps=2, seed=3, amortize_steps=1),
            train_x,
            train_y,
            device,
            args.batch_size,
            steps=2,
        )
        return

    train, val = random_split(range(len(train_x)), [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed))
    val_x, val_y = train_x[val.indices], train_y[val.indices]
    train_x, train_y = train_x[train.indices], train_y[train.indices]
    model = LeNet5().to(device)
    steps_per_epoch = (len(train_x) + args.batch_size - 1) // args.batch_size
    optimizer = build_optimizer(model, total_steps=args.epochs * steps_per_epoch, seed=args.seed)
    loss_fn = nn.CrossEntropyLoss()
    print(
        f"LeNet-5: {sum(p.numel() for p in model.parameters()):,} parameters, forward passes only.\n"
        f"  subspace dim {optimizer.subspace.subspace_dim}, {steps_per_epoch} steps per epoch\n"
    )

    best, best_state = -1.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(optimizer, loss_fn, train_x, train_y, args.batch_size)
        val_acc = accuracy(model, val_x, val_y)
        if val_acc > best:
            best, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch:2d}  loss {loss:.4f}  val {val_acc:5.2f}%  "
            f"best {best:5.2f}%  ({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
    optimizer.release_evaluator()

    model.load_state_dict(best_state)
    print(f"\n  test accuracy: {accuracy(model, test_x, test_y):.2f}%  ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
