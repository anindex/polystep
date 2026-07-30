"""10 - LeNet-5 on MNIST: a classical CNN trained with forward passes only.

LeCun et al. 1998, in the usual modern form (ReLU and max-pool in place of the
original tanh and average-pool): two convolutions, two hidden fully-connected layers,
61,706 parameters. No gradients are computed anywhere.

Every fast path before ``SiteVmapEvaluator`` needed an ``nn.Sequential`` of ``Linear``
and elementwise layers, so a convolution fell back to building one full weight set per
candidate. This example is the counter-case: a plain ``nn.Module`` with convolutions
and a hand-written ``forward``.

Three insights carry it. The first two are automatic once the evaluator is
registered:

  1. A candidate perturbs one contiguous coordinate run, so it differs from the base
     inside a single parameter tensor. ``SiteVmapEvaluator`` batches only that tensor
     and passes the rest once, so the layers ahead of it run once rather than per
     candidate. It asks nothing of the module set beyond being traceable by
     ``torch.func``.
  2. Chunks break at parameter boundaries. Sized purely by memory a chunk spans
     several parameters, resolves to no site, and falls back.

``register_evaluator`` turns both on. Without it the step reaches the model only
through ``closure()`` and has no choice but to materialize.

  3. A smaller rank and more amortization are both cheaper and, here, better. rank=4
     halves the coordinates, and ``amortize_steps=5`` means only every fifth step pays
     for probes. Not "smaller is always better": rank=2 or amortize_steps=8 swings
     tens of points between seeds, and on the MLP of example 05 rank=4 costs accuracy.

``compile_evaluator`` and ``candidate_autocast`` are the wrong levers for this model.
Both target the vmap path, and this net runs on the site-aware path, where
``compile_evaluator`` shrinks the auto chunk for headroom it never uses. Both are
slower here.

Only the probe radius is scheduled, cosine from 10 to 2, sized from ``--epochs``.
Epsilon and the step radius stay flat. Step count matters more than any radius: halving
the batch doubles the steps per epoch and is worth several points.

What you should see (defaults: 20k images, batch 256, 40 epochs):
  93-95% by epoch 40. A zeroth-order test curve moves several points between epochs
  and a couple across seeds, so the best checkpoint is kept.

Run:
  python examples/10_cnn_mnist.py --epochs 10
  python examples/10_cnn_mnist.py --device cpu
  python examples/10_cnn_mnist.py --compare   # time both evaluation paths
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# One thread: PolyStep's per-step ops are small enough that torch's default pool of
# nproc threads costs far more than it returns. See docs/performance.md.
torch.set_num_threads(int(os.environ.get("POLYSTEP_THREADS", 0)) or 1)
import torch.nn as nn  # noqa: E402

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.cost_nn import NNCostEvaluator  # noqa: E402
from polystep.epsilon import CosineEpsilon, LinearEpsilon  # noqa: E402
from polystep.hybrid_subspace import HybridSubspace  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mnist_data import get_mnist_tensors  # noqa: E402


class LeNet5(nn.Module):
    """LeCun et al. 1998, with ReLU and max-pool. 61,706 parameters.

    A plain ``nn.Module`` with its own ``forward``, which is what
    the Sequential-only fast paths decline, and what the site-aware path handles.
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)  # 1 @ 28x28 -> 6 @ 28x28
        self.c3 = nn.Conv2d(6, 16, 5)  # 6 @ 14x14 -> 16 @ 10x10
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        pool = nn.functional.max_pool2d
        x = pool(torch.relu(self.c1(x)), 2)
        x = pool(torch.relu(self.c3(x)), 2)
        x = torch.relu(self.f5(x.flatten(1)))
        return self.out(torch.relu(self.f6(x)))


def build_optimizer(model: nn.Module, total_steps: int, seed: int) -> PolyStepOptimizer:
    """A per-layer subspace, flat epsilon and step radius, one annealed probe radius.

    A per-layer subspace is also what makes the site argument hold on coordinates:
    each layer owns a coordinate block, so a candidate's run lands inside one block and
    moves one parameter. A single global projection would mix every layer into every
    coordinate, leaving no site to resolve and no shared prefix to reuse.
    """
    # rank=4, not 8. Halves the coordinates and so the candidates per step, and scores
    # better here: the conv descent direction is low-rank, so the smaller perturbation
    # carries less variance per evaluation. Not free everywhere; on the MLP of
    # example 05 the same cut costs accuracy.
    subspace = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=4, rotation_interval=0, absorb_interval=0
    )
    return PolyStepOptimizer(
        model,
        seed=seed,
        subspace=subspace,
        solver="softmax",
        num_probe=1,
        # Annealing epsilon collapses the run to chance mid-training, so it stays flat.
        # step_radius is a schedule only because epsilon multiplies float radii and not
        # scheduled ones: a bare 5.0 would step at 50.
        epsilon=10.0,
        step_radius=LinearEpsilon(init=5.0, target=5.0, decay=0.0),
        probe_radius=CosineEpsilon(init=10.0, target=2.0, total_steps=total_steps),
        # Four momentum steps between OT steps, so only every fifth pays for probes.
        # Cliff just past this: amortize_steps=8 or rank=2 swing wildly between seeds.
        amortize_steps=5,
        amortize_ema=0.7,
    )


def run_epoch(optimizer, evaluator, x, y, batch_size, register=True):
    order = torch.randperm(x.shape[0])
    total, batches = 0.0, 0
    for start in range(0, x.shape[0] - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        xb, yb = x[idx], y[idx]
        if register:
            # Hands the step the model and this batch, so candidates are scored
            # directly instead of through the closure.
            optimizer.register_evaluator(evaluator, xb, yb)
        total += optimizer.step(lambda params, _x=xb, _y=yb: evaluator.evaluate(params, _x, _y))
        batches += 1
    if register:
        optimizer.release_evaluator()
    return total / max(batches, 1)


@torch.no_grad()
def accuracy(model, x, y, batch_size=512):
    correct = 0
    for start in range(0, x.shape[0], batch_size):
        logits = model(x[start : start + batch_size])
        correct += (logits.argmax(-1) == y[start : start + batch_size]).sum().item()
    return 100.0 * correct / x.shape[0]


def compare_paths(x, y, device, batch_size, steps=4):
    """Time and cross-check the two evaluation paths on the same batches."""
    results = {}
    for register in (False, True):
        torch.manual_seed(0)
        model = LeNet5().to(device)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        optimizer = build_optimizer(model, total_steps=steps, seed=3)
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="time both paths and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = get_mnist_tensors(args.train_size, args.test_size)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    if args.compare:
        print("\nSame batches, same seed, both paths:\n")
        compare_paths(train_x, train_y, device, args.batch_size)
        return

    model = LeNet5().to(device)
    steps_per_epoch = args.train_size // args.batch_size
    optimizer = build_optimizer(model, total_steps=args.epochs * steps_per_epoch, seed=args.seed)
    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    print(
        f"LeNet-5: {sum(p.numel() for p in model.parameters()):,} parameters, forward passes only.\n"
        f"  subspace dim {optimizer.subspace.subspace_dim}, {steps_per_epoch} steps per epoch\n"
    )

    best, best_state = 0.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(optimizer, evaluator, train_x, train_y, args.batch_size)
        test_acc = accuracy(model, test_x, test_y)
        if test_acc > best:
            best, best_state = test_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch:2d}  loss {loss:.4f}  test {test_acc:5.2f}%  "
            f"best {best:5.2f}%  ({time.perf_counter() - started:.0f} s)",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  best test accuracy: {best:.2f}%  ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
