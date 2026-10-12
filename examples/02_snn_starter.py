"""02 - SNN starter: train a tiny spiking network without gradients.

A small spiking neural network with hard-threshold LIF spikes (zero gradient
almost everywhere, so no backprop) trained via PolyStep. The radii sit above
the defaults because the LIF dynamics make the loss rugged on the scale of a
few weight units, and momentum carries the step direction across minibatches.
Saves a loss/accuracy figure to examples/figures/snn_starter.png.

Run: python examples/02_snn_starter.py
"""

from __future__ import annotations

import importlib.util
import argparse
import time
from pathlib import Path

import torch

import _env

_env.setup()
import torch.nn as nn

from _snn_demo import TinySNN, evaluate_accuracy, make_loaders
from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


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

    train_loader, test_loader = make_loaders(seed=seed)
    model = TinySNN().to(device)
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
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

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

            def closure(stacked_params, _in=inputs, _tgt=targets):
                return evaluator.evaluate(stacked_params, _in, _tgt)

            # Registering scores each parameter's candidates in its own forward: faster
            # on CPU, slower on CUDA than one batched closure forward for this net.
            if device.type == "cpu":
                optimizer.register_evaluator(evaluator, inputs, targets)
            optimizer.step(closure)

            with torch.no_grad():
                logits = model(inputs)
                step_loss = loss_fn(logits, targets).item()
                step_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

            loss_log.append(step_loss)
            acc_log.append(step_acc)
            step_log.append(global_step)

            if global_step % 10 == 0:
                print(f"  step {global_step:3d} | loss={step_loss:.3f} batch_acc={100 * step_acc:5.1f}%")
            global_step += 1

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

    fig, (ax_loss, ax_acc) = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.6),
        constrained_layout=True,
    )
    ax_loss.plot(step_log, loss_log, color="#0072B2", lw=1.4)
    ax_loss.set_xlabel("PolyStep step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training loss", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(step_log, [100 * a for a in acc_log], color="#0072B2", lw=1.4, label="train batch")
    ax_acc.axhline(100 * init_acc, color="#999999", ls="--", lw=1.0, label=f"initial test {100 * init_acc:.1f}%")
    ax_acc.axhline(100 * final_acc, color="#e84040", ls="--", lw=1.0, label=f"final test {100 * final_acc:.1f}%")
    ax_acc.set_xlabel("PolyStep step")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Classification accuracy", fontsize=9)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=7)

    out = Path(__file__).parent / "figures" / "snn_starter.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
