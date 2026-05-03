"""OpenAI Evolution Strategy baseline (Salimans et al. 2017).

Implements the OpenAI ES algorithm for gradient-free neural network training.
Key features:
- Antithetic (mirror) sampling for variance reduction
- Fitness shaping (rank-normalized rewards)
- Per-generation batch evaluation (one batch per perturbation)
- Flat parameter interface via common.py utilities

Reference: Salimans et al., "Evolution Strategies as a Scalable Alternative
to Reinforcement Learning", arXiv:1703.03864, 2017.

Usage:
    from experiments.baselines.openai_es import train_openai_es

    result = train_openai_es(
        model, train_loader, test_loader,
        sigma=0.02, lr=0.01, population_size=50,
        generations=200, device="cuda",
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


def _fitness_shaping(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize rewards to zero mean and unit variance (fitness shaping).

    This reduces sensitivity to reward scale and outliers, stabilizing
    the gradient estimate. If all rewards are identical (zero std),
    returns zeros to avoid division by zero.

    Args:
        rewards: 1D tensor of fitness values, shape (population_size,).

    Returns:
        Shaped rewards with zero mean and unit variance.
    """
    std = rewards.std()
    if std < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / std


def _rank_fitness_shaping(rewards: torch.Tensor) -> torch.Tensor:
    """Rank-based fitness shaping (Salimans et al. 2017).

    Assigns utilities based on rank rather than raw reward values.
    More robust to outliers than z-score normalization.

    Args:
        rewards: 1D tensor of fitness values, shape (population_size,).

    Returns:
        Rank-based utilities centered around 0, range [-0.5, 0.5].
    """
    n = len(rewards)
    if n <= 1:
        return torch.zeros_like(rewards)
    ranks = torch.zeros_like(rewards)
    sorted_idx = rewards.argsort()
    for i, idx in enumerate(sorted_idx):
        ranks[idx] = i
    ranks = ranks / (n - 1) - 0.5  # center around 0
    return ranks


def train_openai_es(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module = None,
    sigma: float = 0.02,
    lr: float = 0.01,
    population_size: int = 50,
    generations: int = 200,
    device: str = "cuda",
    seed: int = 42,
    antithetic: bool = True,
    log_interval: int = 10,
    lr_decay: bool = False,
    weight_decay: float = 0.0,
    fitness_shaping: str = "zscore",
    eval_fn: Optional[Callable[[nn.Module], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Train a model using OpenAI Evolution Strategy.

    Per generation:
    1. Sample noise vectors (antithetic: half sampled, half mirrored).
    2. For each perturbation, evaluate loss on one training batch.
    3. Apply fitness shaping (zero mean, unit variance or rank-based).
    4. Estimate gradient: g = (1 / (pop * sigma)) * (epsilon^T @ rewards).
    5. Update parameters: params -= lr * g (minimize loss).
    6. Optionally apply L2 weight decay.

    Noise vectors are kept on CPU to save GPU memory with large populations.
    Only one perturbation is loaded into the model at a time.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader (cycles through batches).
        test_loader: Test data loader for accuracy evaluation.
        loss_fn: Loss function. Defaults to CrossEntropyLoss.
        sigma: Noise standard deviation (exploration radius).
        lr: Learning rate for parameter update.
        population_size: Number of perturbations per generation.
            Must be even when antithetic=True.
        generations: Number of ES generations.
        device: Device for model evaluation.
        seed: Random seed for reproducibility.
        antithetic: Use antithetic (mirror) sampling for variance reduction.
        log_interval: Evaluate test accuracy every N generations.
        lr_decay: If True, linearly decay lr from initial value to 0 over
            all generations (Salimans et al. 2017). Default False preserves
            current behavior.
        weight_decay: L2 weight decay coefficient. Applied after gradient
            update as multiplicative factor: params *= (1 - lr * wd).
            Default 0.0 preserves current behavior.
        fitness_shaping: Fitness shaping method. "zscore" for zero-mean
            unit-variance normalization (default), "rank" for rank-based
            utilities (Salimans et al. 2017).

    Returns:
        Dict with keys: benchmark, method, seed, hyperparameters,
        metrics (final_accuracy, best_accuracy, wall_time_seconds,
        peak_gpu_memory_mb, function_evals, total_steps), epoch_logs.

    Raises:
        ValueError: If antithetic=True and population_size is odd.
        ValueError: If fitness_shaping is not "zscore" or "rank".
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if antithetic and population_size % 2 != 0:
        raise ValueError(
            f"population_size must be even when antithetic=True, got {population_size}"
        )

    if fitness_shaping not in ("zscore", "rank"):
        raise ValueError(
            f"fitness_shaping must be 'zscore' or 'rank', got '{fitness_shaping}'"
        )

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

    # Pre-allocate reusable buffers to avoid 500K+ tensor allocations
    epsilon = torch.empty(population_size, n_params)  # CPU noise matrix
    eps_row_gpu = torch.empty(n_params, device=device)  # GPU buffer for one row
    perturbed = torch.empty(n_params, device=device)  # GPU buffer for perturbed params
    rewards = torch.zeros(population_size)
    half = population_size // 2 if antithetic else 0

    with track_gpu_memory() as mem:
        for gen in range(1, generations + 1):
            # --- 1. Sample noise vectors on CPU ---
            if antithetic:
                epsilon[:half].normal_()
                epsilon[half:].copy_(epsilon[:half]).neg_()
            else:
                epsilon.normal_()

            # --- 2. Evaluate each perturbation on the SAME batch ---
            # Using one batch per generation ensures the gradient estimate
            # reflects perturbation quality, not batch variance
            batch = get_batch()
            batch_inputs = batch[0].to(device)
            batch_targets = batch[1].to(device)

            for i in range(population_size):
                eps_row_gpu.copy_(epsilon[i])
                torch.add(params, eps_row_gpu, alpha=sigma, out=perturbed)
                set_flat_params(model, perturbed)
                with torch.no_grad():
                    outputs = model(batch_inputs)
                    loss_val = loss_fn(outputs, batch_targets).item()
                # Negate loss: lower loss = higher reward for fitness shaping
                rewards[i] = -loss_val

            # --- 3. Fitness shaping ---
            if fitness_shaping == "rank":
                shaped_rewards = _rank_fitness_shaping(rewards)
            else:
                shaped_rewards = _fitness_shaping(rewards)

            # --- 4. Gradient estimate ---
            # g = (1 / (pop * sigma)) * epsilon^T @ shaped_rewards
            # epsilon: (pop, n_params), shaped_rewards: (pop,)
            grad = (1.0 / (population_size * sigma)) * (
                epsilon.t() @ shaped_rewards
            ).to(device)

            # --- 5. Update params (gradient ascent on reward = descent on loss) ---
            current_lr = lr * (1.0 - gen / generations) if lr_decay else lr
            params.add_(grad, alpha=current_lr)

            # --- 5b. L2 weight decay (Salimans et al. 2017) ---
            if weight_decay > 0:
                params.mul_(1.0 - current_lr * weight_decay)

            # Restore current best params to model
            set_flat_params(model, params)

            # --- Logging ---
            if gen % log_interval == 0 or gen == generations:
                elapsed = time.time() - start_time
                log_entry = {
                    "epoch": gen,
                    "loss": -rewards.mean().item(),  # Average loss this generation
                    "time": elapsed,
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

    function_evals = generations * population_size

    return {
        "method": "openai_es",
        "seed": seed,
        "hyperparameters": {
            "sigma": sigma,
            "lr": lr,
            "population_size": population_size,
            "generations": generations,
            "antithetic": antithetic,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "fitness_shaping": fitness_shaping,
        },
        "metrics": {
            "final_accuracy": final_accuracy,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": function_evals,
            "total_steps": generations,
        },
        "epoch_logs": epoch_logs,
    }
