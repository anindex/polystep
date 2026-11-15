"""13 - ProbeGradient: train MNIST from forward passes with any torch optimizer.

``ProbeGradient`` moves each sample's layer outputs by a finite radius, reads the change in that
sample's loss and writes the resulting gradient estimate into ``.grad``. The optimizer is plain
``torch.optim.SGD``. The ``sign`` and ``spiking`` models have hard thresholds, where backpropagation
returns a zero gradient and a probe wider than a unit's margin still sees the step.

Run:
  python examples/13_probe_gradient.py
  python examples/13_probe_gradient.py --model spiking
  python examples/13_probe_gradient.py --model lenet --device cuda
  python examples/13_probe_gradient.py --device cpu --train-size 4000 --epochs 2
"""

from __future__ import annotations

import argparse
import time

import torch

import _env

_env.setup()
import torch.nn as nn
from torch.utils.data import random_split

from polystep import ProbeGradient

from _mnist_data import get_mnist_tensors


class Sign(nn.Module):
    def forward(self, x):
        return torch.sign(x)


class Spiking(nn.Module):
    """One layer of leaky integrate-and-fire neurons with a hard threshold, read out as spike counts."""

    def __init__(self, hidden: int = 128, steps: int = 8):
        super().__init__()
        self.fc1, self.fc2, self.steps = nn.Linear(784, hidden), nn.Linear(hidden, 10), steps

    def forward(self, x):
        current = self.fc1(x.flatten(1))
        membrane = torch.zeros_like(current)
        out = 0
        for _ in range(self.steps):
            membrane = 0.9 * membrane + current
            spikes = (membrane > 1.0).to(current.dtype)
            membrane = membrane - spikes
            out = out + self.fc2(spikes)
        return out / self.steps


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c3 = nn.Conv2d(1, 6, 5, padding=2), nn.Conv2d(6, 16, 5)
        self.f5, self.f6, self.out = nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 10)

    def forward(self, x):
        pool = nn.functional.max_pool2d
        x = pool(torch.relu(self.c1(x)), 2)
        x = pool(torch.relu(self.c3(x)), 2)
        return self.out(torch.relu(self.f6(torch.relu(self.f5(x.flatten(1))))))


MODELS = {
    "mlp": (
        lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)),
        128,
        0.1,
        0.1,
        512,
        20,
    ),
    "sign": (
        lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 128), Sign(), nn.Linear(128, 10)),
        128,
        0.3,
        0.1,
        512,
        20,
    ),
    "spiking": (Spiking, 128, 0.5, 0.1, 512, 15),
    "lenet": (LeNet, 64, 0.1, 0.05, 256, 4),
}


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    hits = sum((model(x[i : i + 4096]).argmax(-1) == y[i : i + 4096]).sum().item() for i in range(0, len(x), 4096))
    return hits / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="mlp")
    parser.add_argument("--epochs", type=int, default=0, help="Training epochs (0 = the model's default).")
    parser.add_argument("--train-size", type=int, default=0, help="Cap MNIST training samples (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 0 or args.train_size < 0 or 0 < args.train_size < 1000:
        parser.error("--epochs must be non-negative and --train-size 0 or at least 1000")

    build, probes, radius, lr, batch_size, default_epochs = MODELS[args.model]
    epochs = args.epochs or default_epochs
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_x, train_y, test_x, test_y = (t.to(device) for t in get_mnist_tensors(args.train_size))
    train, val = random_split(range(len(train_x)), [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed))
    val_x, val_y = train_x[val.indices], train_y[val.indices]
    train_x, train_y = train_x[train.indices], train_y[train.indices]

    model = build().to(device)
    steps = epochs * (len(train_x) // batch_size)
    estimator = ProbeGradient(model, probes=probes, probe_radius=radius, seed=args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.1 * lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}, forward passes only")
    print(f"  probes={probes} probe_radius={radius} lr={lr} batch={batch_size} epochs={epochs}\n")
    best, best_state = -1.0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_x), device=device)
        total = torch.zeros((), device=device)
        for start in range(0, len(train_x) - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            total += estimator(train_x[idx], train_y[idx], loss_fn=loss_fn)
            optimizer.step()
            schedule.step()
        val_acc = accuracy(model, val_x, val_y)
        if val_acc > best:
            best, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        mean_loss = total.item() / (len(train_x) // batch_size)
        print(
            f"  epoch {epoch:2d}  loss {mean_loss:.4f}  val {100 * val_acc:5.2f}%  "
            f"({time.perf_counter() - started:.1f} s)",
            flush=True,
        )

    model.load_state_dict(best_state)
    print(f"\n  test accuracy: {100 * accuracy(model, test_x, test_y):.2f}%  ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
