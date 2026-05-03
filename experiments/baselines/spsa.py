"""SPSA (Simultaneous Perturbation Stochastic Approximation) baseline.

Implements the SPSA algorithm from Spall 1992 for gradient-free neural
network training. The key advantage of SPSA is that it requires only
2 function evaluations per iteration regardless of parameter dimension,
making it highly efficient for high-dimensional problems.

Reference: Spall, J.C., "Multivariate Stochastic Approximation Using a
Simultaneous Perturbation Gradient Approximation", IEEE Transactions on
Automatic Control, 37(3):332-341, 1992.

Gain sequences (Spall's practical recommendations):
    a_k = a / (A + k)^alpha     (step size, decays faster)
    c_k = c / k^gamma           (perturbation size, decays slower)

    alpha = 0.602, gamma = 0.101  (finite-sample optimal)
    A = 10% of max_iters          (stability constant)

Note: The gain sequence a_k can become very small for large k. If alpha
is too high or a is too small, the algorithm may stall. Tuning a and c
per task is recommended.

Usage:
    from experiments.baselines.spsa import train_spsa

    result = train_spsa(
        model, train_loader, test_loader,
        a=0.1, c=0.1, max_iters=5000, device="cuda",
    )
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.runners.common import (
    evaluate_accuracy,
    load_flat_params,
    set_flat_params,
    set_seed,
    track_gpu_memory,
)


def train_spsa(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module = None,
    a: float = 0.1,
    c: float = 0.1,
    A: Optional[float] = None,
    alpha: float = 0.602,
    gamma: float = 0.101,
    max_iters: int = 5000,
    device: str = "cuda",
    seed: int = 42,
    log_interval: int = 100,
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train a model using SPSA (Simultaneous Perturbation Stochastic Approximation).

    Per iteration k (starting from k=1):
    1. Compute gain sequences: a_k = a / (A + k)^alpha, c_k = c / k^gamma
    2. Generate Bernoulli perturbation: delta_i ~ {-1, +1} with equal probability
    3. Evaluate: loss_plus = L(params + c_k * delta), loss_minus = L(params - c_k * delta)
    4. Approximate gradient: g_hat = (loss_plus - loss_minus) / (2 * c_k * delta)
    5. Update: params = params - a_k * g_hat

    Only 2 function evaluations per iteration regardless of parameter dimension.
    This is SPSA's defining property.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader (cycles through batches).
        test_loader: Test data loader for accuracy evaluation.
        loss_fn: Loss function. Defaults to CrossEntropyLoss.
        a: Initial step size gain. Controls learning rate magnitude.
        c: Initial perturbation gain. Controls gradient estimation noise.
        A: Stability constant. Defaults to 10% of max_iters (Spall's recommendation).
        alpha: Step size decay exponent (0.602 = finite-sample optimal).
        gamma: Perturbation decay exponent (0.101 = finite-sample optimal).
            Must satisfy gamma < alpha for consistency.
        max_iters: Maximum number of SPSA iterations.
        device: Device for model evaluation.
        seed: Random seed for reproducibility.
        log_interval: Evaluate test accuracy every N iterations.

    Returns:
        Dict with keys: benchmark, method, seed, hyperparameters,
        metrics (final_accuracy, best_accuracy, wall_time_seconds,
        peak_gpu_memory_mb, function_evals, total_steps), epoch_logs.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if A is None:
        A = 0.1 * max_iters

    set_seed(seed)
    model = model.to(device)
    model.eval()
    loss_fn = loss_fn.to(device)

    # Get initial flat parameters
    params = load_flat_params(model).to(device)
    n_params = params.numel()

    # Create an iterator that cycles through train_loader batches
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return batch

    epoch_logs: List[Dict[str, Any]] = []
    best_accuracy = 0.0
    start_time = time.time()

    # Pre-allocate reusable buffers to avoid 200K+ tensor allocations in hot loop
    delta = torch.empty(n_params, device=device)
    params_plus = torch.empty(n_params, device=device)
    params_minus = torch.empty(n_params, device=device)

    with track_gpu_memory() as mem:
        for k in range(1, max_iters + 1):
            # --- 1. Gain sequences ---
            a_k = a / ((A + k) ** alpha)
            c_k = c / (k ** gamma)

            # --- 2. Bernoulli perturbation: each element +1 or -1 ---
            delta.bernoulli_(0.5).mul_(2.0).sub_(1.0)

            # --- 3. Evaluate at perturbed points (same batch for both) ---
            torch.add(params, delta, alpha=c_k, out=params_plus)
            torch.sub(params, delta, alpha=c_k, out=params_minus)

            # Get one batch and reuse for both perturbations to avoid
            # injecting batch variance into the gradient estimate
            batch = get_batch()
            batch_inputs = batch[0].to(device)
            batch_targets = batch[1].to(device)

            set_flat_params(model, params_plus)
            with torch.no_grad():
                loss_plus = loss_fn(model(batch_inputs), batch_targets).item()

            set_flat_params(model, params_minus)
            with torch.no_grad():
                loss_minus = loss_fn(model(batch_inputs), batch_targets).item()

            # --- 4. Approximate gradient & 5. Update parameters (fused) ---
            # g_hat_i = (loss_plus - loss_minus) / (2 * c_k * delta_i)
            # params -= a_k * g_hat = a_k * (L+ - L-) / (2*c_k) * (1/delta)
            # Since delta ∈ {-1, +1}, 1/delta = delta, so:
            grad_scale = a_k * (loss_plus - loss_minus) / (2.0 * c_k)
            params.sub_(delta, alpha=grad_scale)

            # Restore params to model for evaluation
            set_flat_params(model, params)

            # --- Logging ---
            if k % log_interval == 0 or k == max_iters:
                elapsed = time.time() - start_time
                avg_loss = (loss_plus + loss_minus) / 2.0

                log_entry = {
                    "epoch": k,
                    "loss": avg_loss,
                    "time": elapsed,
                    "a_k": a_k,
                    "c_k": c_k,
                }
                if eval_fn is not None:
                    eval_result = eval_fn(model)
                    log_entry.update(eval_result)
                else:
                    test_acc = evaluate_accuracy(model, test_loader, device=device)
                    if test_acc > best_accuracy:
                        best_accuracy = test_acc
                    log_entry["accuracy"] = test_acc

                epoch_logs.append(log_entry)

    wall_time = time.time() - start_time
    if eval_fn is None:
        final_accuracy = evaluate_accuracy(model, test_loader, device=device)
        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
    else:
        final_accuracy = best_accuracy  # Caller handles regression metrics separately

    function_evals = 2 * max_iters  # Exactly 2 evals per iteration

    return {
        "method": "spsa",
        "seed": seed,
        "hyperparameters": {
            "a": a,
            "c": c,
            "A": A,
            "alpha": alpha,
            "gamma": gamma,
            "max_iters": max_iters,
        },
        "metrics": {
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": function_evals,
            "total_steps": max_iters,
        },
        "epoch_logs": epoch_logs,
    }
