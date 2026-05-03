"""SGD/Adam gradient-based baseline (ceiling comparison).

Implements standard PyTorch gradient-based training as the "ceiling" baseline
for comparison with gradient-free methods. This shows the accuracy gap that
gradient-free methods must close.

Supports:
- Adam optimizer (default, recommended for most tasks)
- SGD with momentum=0.9

The training loop is straightforward: forward pass, loss, backward, step.
For SNN models with snnTorch surrogate gradients, the same loop works
because snnTorch provides surrogate gradient functions for autograd.

Usage:
    from experiments.baselines.sgd_baseline import train_sgd

    result = train_sgd(
        model, train_loader, test_loader,
        optimizer_name="adam", lr=0.001, epochs=20, device="cuda",
    )
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.runners.common import (
    evaluate_accuracy,
    set_seed,
    track_gpu_memory,
)


def train_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module = None,
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 20,
    device: str = "cuda",
    seed: int = 42,
    cosine_lr: bool = True,
) -> Dict[str, Any]:
    """Train a model using standard gradient-based optimization.

    Standard PyTorch training loop: for each epoch, iterate over
    train_loader batches, compute loss, backpropagate, step optimizer.
    After each epoch, evaluate full test accuracy.

    This serves as the "ceiling" baseline -- gradient methods have
    access to exact gradient information and should achieve the highest
    accuracy. The paper honestly reports the gap between gradient-free
    methods and this ceiling.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        test_loader: Test data loader for accuracy evaluation.
        loss_fn: Loss function. Defaults to CrossEntropyLoss.
        optimizer_name: Optimizer to use, "adam" or "sgd".
        lr: Learning rate.
        weight_decay: L2 regularization weight (default 0.0).
        epochs: Number of training epochs.
        device: Device for training.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: benchmark, method, seed, hyperparameters,
        metrics (final_accuracy, best_accuracy, wall_time_seconds,
        peak_gpu_memory_mb, function_evals, total_steps), epoch_logs.

    Raises:
        ValueError: If optimizer_name is not "adam" or "sgd".
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower not in ("adam", "sgd"):
        raise ValueError(
            f"optimizer_name must be 'adam' or 'sgd', got '{optimizer_name}'"
        )

    set_seed(seed)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Create optimizer
    if optimizer_name_lower == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    scheduler = None
    if cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_logs: List[Dict[str, Any]] = []
    best_accuracy = 0.0
    total_samples_seen = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(1, epochs + 1):
            # --- Training ---
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_correct = 0
            epoch_total = 0

            for batch in train_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                epoch_correct += (outputs.argmax(dim=1) == targets).sum().item()
                epoch_total += targets.size(0)
                total_samples_seen += inputs.size(0)

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(epoch_batches, 1)
            train_acc = epoch_correct / max(epoch_total, 1)

            # --- Evaluation ---
            test_acc = evaluate_accuracy(model, test_loader, device=device)
            elapsed = time.time() - start_time

            if test_acc > best_accuracy:
                best_accuracy = test_acc

            epoch_logs.append({
                "epoch": epoch,
                "accuracy": test_acc,
                "train_accuracy": train_acc,
                "loss": avg_loss,
                "time": elapsed,
            })

    wall_time = time.time() - start_time
    final_accuracy = evaluate_accuracy(model, test_loader, device=device)
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy

    # function_evals = total forward passes = total training samples seen
    function_evals = total_samples_seen

    return {
        "benchmark": "unknown",
        "method": optimizer_name_lower,
        "seed": seed,
        "hyperparameters": {
            "optimizer": optimizer_name_lower,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
        },
        "metrics": {
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": function_evals,
            "total_steps": epochs,
        },
        "epoch_logs": epoch_logs,
    }
