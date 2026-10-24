"""02 - SNN starter: train a tiny spiking network without gradients.

A small spiking neural network with hard-threshold LIF spikes trained via
PolyStep, without surrogate gradients. The radii sit above
the defaults because the LIF dynamics make the loss rugged on the scale of a
few weight units, and momentum carries the step direction across minibatches.
Saves a loss/accuracy figure and training animation to examples/figures/.

Run: python examples/02_snn_starter.py
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import torch

import _env

_env.setup()
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer
from polystep.benchmarks import SpikingNet, evaluate_accuracy


def make_loader(num_samples: int, seed: int, shuffle: bool) -> DataLoader:
    """Noisy copies of four class templates, each owning a block of four "on" features."""
    rng = torch.Generator().manual_seed(seed)
    templates = (torch.eye(4).repeat_interleave(4, dim=1) + 0.2 * torch.randn(4, 16, generator=rng)).clamp(0, 1)
    labels = torch.randint(0, 4, (num_samples,), generator=rng)
    data = (templates[labels] + 0.15 * torch.randn(num_samples, 16, generator=rng)).clamp(0, 1)
    return DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    seed = 42
    torch.manual_seed(seed)

    print("=" * 60)
    print("Tiny SNN with hard-threshold LIF spikes (PolyStep training)")
    print("=" * 60)

    train_loader, test_loader = make_loader(256, seed, shuffle=True), make_loader(64, seed + 1, shuffle=False)
    model = SpikingNet(input_dim=16, hidden=24, output=4, num_steps=20).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")
    print(f"  train/test: {len(train_loader.dataset)}/{len(test_loader.dataset)} samples")

    optimizer = PolyStepOptimizer(
        model,
        seed=seed,
        epsilon=0.5,
        step_radius=2.0,
        probe_radius=4.0,
        use_momentum=True,
        solver="softmax",
    )
    loss_fn = nn.CrossEntropyLoss()

    init_acc = evaluate_accuracy(model, test_loader)
    print(f"  initial test accuracy: {100 * init_acc:.1f}%")
    print()

    loss_log: list[float] = []
    acc_log: list[float] = []
    step_log: list[int] = []

    start = time.time()
    global_step = 0
    target_steps = 60

    print("training...")
    while global_step < target_steps:
        for inputs, targets in train_loader:
            if global_step >= target_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.step_batch(inputs, targets, loss_fn=loss_fn)

            with torch.no_grad():
                logits = model(inputs)
                step_loss = loss_fn(logits, targets).item()
                step_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

            global_step += 1
            loss_log.append(step_loss)
            acc_log.append(step_acc)
            step_log.append(global_step)

            if global_step % 10 == 0 or global_step == 1:
                print(f"  step {global_step:3d} | loss={step_loss:.3f} batch_acc={100 * step_acc:5.1f}%")

    elapsed = time.time() - start
    final_acc = evaluate_accuracy(model, test_loader)

    print()
    print("=" * 60)
    print(f"  initial test acc: {100 * init_acc:.1f}%")
    print(f"  final   test acc: {100 * final_acc:.1f}%")
    print(f"  wallclock: {elapsed:.1f}s ({target_steps} steps)")
    print("=" * 60)

    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib not installed; skipping the figure (pip install matplotlib).")
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax_loss, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(6.0, 4.0),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle("Hard LIF spikes · no surrogate gradients", fontsize=12)
    (loss_line,) = ax_loss.plot(step_log, loss_log, color="#0072B2", lw=1.6)
    ax_loss.set_ylabel("Cross-entropy loss")
    progress = ax_loss.set_title(f"Training batch · step {target_steps}/{target_steps}", fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    accuracy = [100 * a for a in acc_log]
    (acc_line,) = ax_acc.plot(step_log, accuracy, color="#0072B2", lw=1.6, label="train batch")
    ax_acc.axhline(25, color="#999999", ls="--", lw=1.0, label="chance (4 classes)")
    ax_acc.set_xlabel("PolyStep step")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlim(0, target_steps)
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=8)

    out = Path(__file__).parent / "figures" / "snn_starter.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved figure: {out}")

    def animate(count):
        loss_line.set_data(step_log[:count], loss_log[:count])
        acc_line.set_data(step_log[:count], accuracy[:count])
        progress.set_text(f"Training batch · step {count}/{target_steps}")

    animation = FuncAnimation(fig, animate, frames=[*range(1, target_steps + 1), *([target_steps] * 10)])
    gif = out.with_suffix(".gif")
    animation.save(gif, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    print(f"saved animation: {gif}")


if __name__ == "__main__":
    main()
