"""02 - SNN starter: train a tiny spiking network without gradients.

A small spiking neural network with hard-threshold LIF spikes (genuinely
non-differentiable) trained via PolyStep in under a minute on CPU. This is
the simplest end-to-end demonstration of PolyStep's core value proposition.

Why gradient-free for SNNs?
  Hard LIF spikes have ``d(spike)/d(mem) == 0`` almost everywhere, so
  backprop gives zero gradients. Standard practice is to introduce a
  *surrogate* gradient (a smooth approximation), which is an engineering
  hack. PolyStep evaluates the SNN forward only, so the spikes stay as-is.

What you should see:
  Training accuracy climbs from ~25% (chance, 4 classes) to >70% over 60
  steps. A 2-panel figure shows the loss + accuracy curves.

Output:
  examples/figures/snn_starter.png

Run:
  python examples/02_snn_starter.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow running directly from a source checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _snn_demo import (  # noqa: E402  (sys.path mutation above)
    OUTPUT_SCALE,
    TinySNN,
    evaluate_accuracy,
    make_loaders,
    make_optimizer,
)
from polystep.cost_nn import NNCostEvaluator  # noqa: E402


def main():
    seed = 42
    torch.manual_seed(seed)

    print("=" * 60)
    print("Tiny SNN with hard-threshold LIF spikes (PolyStep training)")
    print("=" * 60)

    train_loader, test_loader = make_loaders(seed=seed)
    model = TinySNN()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")
    print(f"  train/test: {len(train_loader.dataset)}/{len(test_loader.dataset)} samples")

    optimizer = make_optimizer(model, seed=seed)
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

            def closure(stacked_params, _in=inputs, _tgt=targets):
                return evaluator.evaluate(stacked_params, _in, _tgt)

            optimizer.step(closure)

            with torch.no_grad():
                logits = model(inputs) * OUTPUT_SCALE
                step_loss = loss_fn(logits, targets).item()
                step_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

            loss_log.append(step_loss)
            acc_log.append(step_acc)
            step_log.append(global_step)

            if global_step % 10 == 0:
                print(
                    f"  step {global_step:3d} | "
                    f"loss={step_loss:.3f} batch_acc={100 * step_acc:5.1f}%"
                )
            global_step += 1

    elapsed = time.time() - start
    final_acc = evaluate_accuracy(model, test_loader)

    print()
    print("=" * 60)
    print(f"  initial test acc: {100 * init_acc:.1f}%")
    print(f"  final   test acc: {100 * final_acc:.1f}%")
    print(f"  wallclock: {elapsed:.1f}s ({target_steps} steps)")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(
        1, 2, figsize=(7.0, 2.6), constrained_layout=True,
    )
    ax_loss.plot(step_log, loss_log, color="#0072B2", lw=1.4)
    ax_loss.set_xlabel("PolyStep step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training loss", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(step_log, [100 * a for a in acc_log], color="#0072B2", lw=1.4,
                label="train batch")
    ax_acc.axhline(100 * init_acc, color="#999999", ls="--", lw=1.0,
                   label=f"initial test {100 * init_acc:.1f}%")
    ax_acc.axhline(100 * final_acc, color="#e84040", ls="--", lw=1.0,
                   label=f"final test {100 * final_acc:.1f}%")
    ax_acc.set_xlabel("PolyStep step")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Classification accuracy", fontsize=9)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=7)

    out = Path(__file__).parent / "figures" / "snn_starter.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
