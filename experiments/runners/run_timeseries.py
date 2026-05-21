#!/usr/bin/env python
"""Run all methods and seeds for ETTh1 time-series forecasting benchmark.

Methods: polystep, adam, cmaes, openai_es, spsa, persistence
Model: TimeSeriesLSTM (VmapSafeLSTM hidden=64, Linear head, 23,392 params)
Data: ETTh1 univariate OT (oil temperature), hourly, Informer-standard split

Hyperparameters are hardcoded constants for reproducibility.
Results are saved as JSON files in experiments/results/softmax/main/.

Usage:
    python experiments/runners/run_timeseries.py
    python experiments/runners/run_timeseries.py --methods polystep adam --seeds 42 123
    python experiments/runners/run_timeseries.py --methods adam --seeds 42 --epochs-override 2
    python experiments/runners/run_timeseries.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import urllib.request

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiments.runners.common import (
    SEEDS,
    save_result,
    set_seed,
    track_gpu_memory,
)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_etth1() -> str:
    """Download ETTh1.csv if not already cached.

    Returns:
        Path to the cached CSV file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "ETTh1.csv")
    if not os.path.exists(filepath):
        print(f"Downloading ETTh1.csv to {filepath}...")
        urllib.request.urlretrieve(ETTH1_URL, filepath)
        print(f"Downloaded ({os.path.getsize(filepath)} bytes)")
    return filepath


def load_etth1():
    """Load and preprocess ETTh1 dataset.

    Parses the CSV, extracts the OT (oil temperature) column,
    splits into train/val/test per Informer convention (8640/2880/2880),
    and applies z-score normalization using train statistics ONLY.

    Returns:
        Tuple of (train, val, test, scaler_dict) where:
        - train: np.ndarray of shape (8640,) -- z-score normalized
        - val: np.ndarray of shape (2880,) -- z-score normalized
        - test: np.ndarray of shape (2880,) -- z-score normalized
        - scaler_dict: dict with 'mean' and 'std' keys (train statistics)
    """
    filepath = download_etth1()

    # Parse CSV with stdlib csv module (no pandas dependency)
    ot_values = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ot_values.append(float(row["OT"]))

    data = np.array(ot_values, dtype=np.float32)

    # Informer-standard split: 8640/2880/2880
    train_raw = data[:8640]
    val_raw = data[8640:8640 + 2880]
    test_raw = data[8640 + 2880:8640 + 2880 + 2880]

    # Z-score normalization using train statistics ONLY (no data leakage)
    train_mean = float(train_raw.mean())
    train_std = float(train_raw.std())

    train = (train_raw - train_mean) / train_std
    val = (val_raw - train_mean) / train_std
    test = (test_raw - train_mean) / train_std

    scaler = {"mean": train_mean, "std": train_std}

    return train, val, test, scaler


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for univariate time-series forecasting.

    Creates (input, target) pairs using a sliding window approach:
    - Input: data[i : i + seq_len] reshaped to (seq_len, 1)
    - Target: data[i + seq_len : i + seq_len + pred_len] as (pred_len,)

    Stride is 1 (every possible window).

    Args:
        data: 1D numpy array of z-score normalized values.
        seq_len: Lookback window length (default: 96).
        pred_len: Prediction horizon (default: 96).
    """

    def __init__(self, data: np.ndarray, seq_len: int = 96, pred_len: int = 96):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len].reshape(-1, 1)  # (seq_len, 1)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # (pred_len,)
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimeSeriesLSTM(nn.Module):
    """LSTM for univariate time-series forecasting.

    Architecture: input -> VmapSafeLSTM(input_size, hidden_size) -> Linear(hidden_size, pred_len)
    Uses last hidden state for direct multi-step prediction.

    Parameters: 23,392 (with hidden_size=64, pred_len=96)
    - lstm.cells.0.W_i: (256, 1) + (256,) = 512
    - lstm.cells.0.W_h: (256, 64) + (256,) = 16,640
    - fc: (96, 64) + (96,) = 6,240
    Total = 23,392

    Args:
        input_size: Number of input features (1 for univariate).
        hidden_size: LSTM hidden state dimension.
        pred_len: Number of future steps to predict.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, pred_len: int = 96):
        super().__init__()
        from polystep.layers import VmapSafeLSTM
        self.lstm = VmapSafeLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]    # (batch, hidden_size)
        return self.fc(last_hidden)    # (batch, pred_len)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    data_array: np.ndarray,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 256,
    device=None,
) -> dict:
    """Evaluate regression model on time-series data.

    Computes MSE and MAE over all sliding windows in data_array.

    Args:
        model: TimeSeriesLSTM model.
        data_array: 1D z-score normalized numpy array.
        seq_len: Lookback window length.
        pred_len: Prediction horizon.
        batch_size: Batch size for evaluation.
        device: Device for evaluation.

    Returns:
        Dict with 'mse' and 'mae' keys.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dataset = TimeSeriesDataset(data_array, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)
        mse = ((preds - targets) ** 2).sum().item()
        mae = (preds - targets).abs().sum().item()
        n = targets.numel()
        total_mse += mse
        total_mae += mae
        total_samples += n

    model.train()

    return {
        "mse": total_mse / max(total_samples, 1),
        "mae": total_mae / max(total_samples, 1),
    }


def compute_persistence_baseline(
    data_array: np.ndarray,
    seq_len: int = 96,
    pred_len: int = 96,
) -> dict:
    """Compute naive persistence baseline for time-series forecasting.

    The persistence forecast predicts the last observed value for all
    future steps: pred[t+1:t+H] = data[t] (repeat last value).

    This is the floor that any learned model must beat.

    Args:
        data_array: 1D z-score normalized numpy array.
        seq_len: Lookback window length.
        pred_len: Prediction horizon.

    Returns:
        Dict with 'mse' and 'mae' keys.
    """
    n_samples = len(data_array) - seq_len - pred_len + 1
    total_mse = 0.0
    total_mae = 0.0
    total_elements = 0

    for i in range(n_samples):
        last_value = data_array[i + seq_len - 1]
        target = data_array[i + seq_len:i + seq_len + pred_len]
        diff = target - last_value
        total_mse += float((diff ** 2).sum())
        total_mae += float(np.abs(diff).sum())
        total_elements += pred_len

    return {
        "mse": total_mse / max(total_elements, 1),
        "mae": total_mae / max(total_elements, 1),
    }


# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

BENCHMARK = "timeseries"
BATCH_SIZE = 64
EPOCHS = 30        # polystep epochs (30 for production)
ADAM_EPOCHS = 50   # Adam epochs (gradient-based ceiling)
SEQ_LEN = 96
PRED_LEN = 96

# polystep hyperparameters (HybridSubspace)
# Best softmax configuration (HybridSubspace + cosine schedules)
#   - Wider probe radius (10->2) is the key improvement over baseline (5->1)
#   - Momentum essential: no_mom collapses to MSE 0.44
#   - 20 epochs standardized (was 30, now matched to sweep budget)
PSTORCH_CONFIG = {
    "rank": 8,
    "epsilon_init": 10.0,
    "epsilon_target": 0.1,
    "step_radius_init": 5.0,
    "step_radius_target": 1.0,
    "probe_radius_init": 10.0,
    "probe_radius_target": 2.0,
    "num_probe": 1,  # K>1 adds zero benefit for softmax
    "rotation_interval": 0,
    "absorb_interval": 0,
    "chunk_size": 1024,
    "amortize_steps": 3,
    "amortize_ema": 0.7,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.95,
}

# OpenAI ES hyperparameters (compute budget matched)
OPENAI_ES_CONFIG = {
    "sigma": 0.02,
    "lr": 0.01,
    "population_size": 50,
    "generations": 2000,
    "lr_decay": True,
    "weight_decay": 0.01,
    "fitness_shaping": "rank",
}

# SPSA hyperparameters
SPSA_CONFIG = {
    "a": 0.1,
    "c": 0.1,
    "alpha": 0.602,
    "gamma": 0.101,
    "max_iters": 10000,
}

# CMA-ES hyperparameters
CMAES_CONFIG = {
    "generations": 2000,
    "popsize": 16,
    "stdev_init": 0.5,
}

# Adam hyperparameters
ADAM_CONFIG = {
    "lr": 0.001,
    "epochs": ADAM_EPOCHS,
}


# ---------------------------------------------------------------------------
# Helper: build train/test tensors for baselines that need raw tensor input
# ---------------------------------------------------------------------------

def _build_window_tensors(data_array, seq_len=SEQ_LEN, pred_len=PRED_LEN, device="cuda"):
    """Build sliding window input/target tensors for baselines.

    Returns:
        Tuple of (inputs, targets) tensors:
        - inputs: (N, seq_len, 1)
        - targets: (N, pred_len)
    """
    ds = TimeSeriesDataset(data_array, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    inputs, targets = next(iter(loader))
    return inputs.to(device), targets.to(device)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_polystep(seed, device, train_data, val_data, test_data, results_dir,
                epochs_override=None, solver=None, audit_no_leakage: bool = True):
    """Train time-series LSTM with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses validation MSE (from
    the Informer-standard val split) instead of test MSE (honest
    protocol). Set ``audit_no_leakage=False`` to revert to legacy.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    epochs = epochs_override if epochs_override is not None else EPOCHS
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_steps = epochs * len(train_loader)
    epsilon_decay = (
        (PSTORCH_CONFIG["epsilon_init"] - PSTORCH_CONFIG["epsilon_target"])
        / max(1, total_steps)
    )
    sr_decay = (PSTORCH_CONFIG["step_radius_init"] - PSTORCH_CONFIG["step_radius_target"]) / max(1, total_steps)
    pr_decay = (PSTORCH_CONFIG["probe_radius_init"] - PSTORCH_CONFIG["probe_radius_target"]) / max(1, total_steps)

    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=PSTORCH_CONFIG["rank"],
        rotation_mode="random",
        rotation_interval=PSTORCH_CONFIG["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=PSTORCH_CONFIG["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=CosineEpsilon(
            init=PSTORCH_CONFIG["epsilon_init"],
            target=PSTORCH_CONFIG["epsilon_target"],
            decay=epsilon_decay,
        ),
        step_radius=CosineEpsilon(
            init=PSTORCH_CONFIG["step_radius_init"],
            target=PSTORCH_CONFIG["step_radius_target"],
            decay=sr_decay,
        ),
        probe_radius=CosineEpsilon(
            init=PSTORCH_CONFIG["probe_radius_init"],
            target=PSTORCH_CONFIG["probe_radius_target"],
            decay=pr_decay,
        ),
        num_probe=PSTORCH_CONFIG["num_probe"],
        subspace=subspace,
        chunk_size=PSTORCH_CONFIG.get("chunk_size", 1024),
        amortize_steps=PSTORCH_CONFIG.get("amortize_steps", 0),
        amortize_ema=PSTORCH_CONFIG.get("amortize_ema", 0.0),
        use_momentum=PSTORCH_CONFIG.get("use_momentum", False),
        momentum_init=PSTORCH_CONFIG.get("momentum_init", 0.5),
        momentum_final=PSTORCH_CONFIG.get("momentum_final", 0.95),
        solver=solver,
    )

    import copy

    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    epoch_logs = []
    step_logs = []
    best_mse = float("inf")
    best_mae = float("inf")
    best_state_dict = None
    step_count = 0
    fwd_pass_count = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                def closure(batched_params, _data=data, _targets=targets):
                    nonlocal fwd_pass_count
                    fwd_pass_count += next(iter(batched_params.values())).shape[0]
                    return evaluator.evaluate(batched_params, _data, _targets)

                optimizer.step(closure)

                with torch.no_grad():
                    output = model(data)
                    loss = loss_fn(output, targets).item()
                epoch_loss += loss
                step_count += 1

                # Per-20-step fine-grained tracking
                if step_count % 20 == 0:
                    step_metrics = evaluate_regression(model, test_data, device=device)
                    step_logs.append({
                        "step": step_count,
                        "epoch": epoch + 1,
                        "test_mse": step_metrics["mse"],
                        "test_mae": step_metrics["mae"],
                        "loss": loss,
                        "wall_time": time.time() - start_time,
                    })

            # Evaluate on val and test sets
            val_metrics = evaluate_regression(model, val_data, device=device)
            test_metrics = evaluate_regression(model, test_data, device=device)
            # Select best checkpoint on validation or test MSE
            selection_metrics = val_metrics if audit_no_leakage else test_metrics
            if selection_metrics["mse"] < best_mse:
                best_mse = selection_metrics["mse"]
                best_mae = selection_metrics["mae"]
                best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(train_loader)

            epoch_logs.append({
                "epoch": epoch + 1,
                "train_mse": avg_loss,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "loss": avg_loss,
                "time": epoch_time,
                "wall_time": time.time() - start_time,
            })
            print(f"    Epoch {epoch+1}/{epochs} | train={avg_loss:.4f} | val={val_metrics['mse']:.4f} | test={test_metrics['mse']:.4f}")

    wall_time = time.time() - start_time
    last_metrics = evaluate_regression(model, test_data, device=device)
    last_val_metrics = evaluate_regression(model, val_data, device=device) if audit_no_leakage else None
    last_epoch_mse = last_metrics["mse"]
    last_epoch_mae = last_metrics["mae"]
    last_selection_mse = last_val_metrics["mse"] if audit_no_leakage else last_epoch_mse
    if last_selection_mse < best_mse:
        best_mse = last_selection_mse
        best_mae = last_val_metrics["mae"] if audit_no_leakage else last_epoch_mae
        best_state_dict = copy.deepcopy(model.state_dict())
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_metrics = evaluate_regression(model, test_data, device=device)
    final_mse = final_metrics["mse"]
    final_mae = final_metrics["mae"]

    filepath = save_result(
        benchmark=BENCHMARK,
        method="polystep",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": final_mse,
            "best_mse": best_mse,
            "final_mae": final_mae,
            "best_mae": best_mae,
            "last_epoch_mse": last_epoch_mse,
            "last_epoch_mae": last_epoch_mae,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_pass_count,
            "total_steps": step_count,
        },
        hyperparameters=PSTORCH_CONFIG,
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_adam(seed, device, train_data, val_data, test_data, results_dir,
             epochs_override=None):
    """Train time-series LSTM with Adam optimizer (gradient-based ceiling)."""
    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    epochs = epochs_override if epochs_override is not None else ADAM_EPOCHS
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_CONFIG["lr"])

    epoch_logs = []
    best_mse = float("inf")
    best_mae = float("inf")
    step_count = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            model.train()

            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step_count += 1

            # Evaluate on val and test sets
            val_metrics = evaluate_regression(model, val_data, device=device)
            test_metrics = evaluate_regression(model, test_data, device=device)
            if test_metrics["mse"] < best_mse:
                best_mse = test_metrics["mse"]
                best_mae = test_metrics["mae"]
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(train_loader)

            epoch_logs.append({
                "epoch": epoch + 1,
                "train_mse": avg_loss,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "loss": avg_loss,
                "time": epoch_time,
                "wall_time": time.time() - start_time,
            })
            print(f"    Epoch {epoch+1}/{epochs} | train={avg_loss:.4f} | val={val_metrics['mse']:.4f} | test={test_metrics['mse']:.4f}")

    wall_time = time.time() - start_time
    final_metrics = evaluate_regression(model, test_data, device=device)
    final_mse = final_metrics["mse"]
    final_mae = final_metrics["mae"]
    if final_mse < best_mse:
        best_mse = final_mse
        best_mae = final_mae

    # Function evals: each batch forward pass = 1 eval per epoch batch
    function_evals = step_count  # 1 forward pass per step

    filepath = save_result(
        benchmark=BENCHMARK,
        method="adam",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": final_mse,
            "best_mse": best_mse,
            "final_mae": final_mae,
            "best_mae": best_mae,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": function_evals,
            "total_steps": step_count,
        },
        hyperparameters=ADAM_CONFIG,
        epoch_logs=epoch_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_cmaes(seed, device, train_data, val_data, test_data, results_dir,
              epochs_override=None):
    """Train time-series LSTM with CMA-ES (EvoTorch) using negative MSE fitness."""
    try:
        from polystep.benchmarks.baselines import has_evotorch
        if not has_evotorch():
            print("    Skipping cmaes (EvoTorch not installed)")
            return
        from evotorch import Problem
        from evotorch.algorithms import CMAES
    except ImportError:
        print("    Skipping cmaes (EvoTorch not installed)")
        return

    from experiments.runners.common import load_flat_params, set_flat_params

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    # Build full sliding-window tensors for CMA-ES
    train_inputs, train_targets = _build_window_tensors(train_data, device=device)
    param_count = sum(p.numel() for p in model.parameters())

    class RegressionProblem(Problem):
        """EvoTorch Problem for regression using negative MSE fitness."""

        def __init__(self):
            super().__init__(
                objective_sense="max",  # Maximize negative MSE
                solution_length=param_count,
                dtype=torch.float32,
                device=device,
                initial_bounds=(-1.0, 1.0),
            )

        def _evaluate(self, solution):
            values = solution.values
            eval_model = TimeSeriesLSTM().to(device)
            eval_model.eval()
            offset = 0
            with torch.no_grad():
                for p in eval_model.parameters():
                    numel = p.numel()
                    p.data.copy_(values[offset:offset + numel].view(p.shape))
                    offset += numel
            # Evaluate on a random batch
            batch_size = min(512, len(train_inputs))
            indices = torch.randperm(len(train_inputs), device=device)[:batch_size]
            batch_data = train_inputs[indices]
            batch_labels = train_targets[indices]
            with torch.no_grad():
                outputs = eval_model(batch_data)
                mse = loss_fn(outputs, batch_labels).item()
            # Return negative MSE as fitness (maximize = minimize MSE)
            solution.set_evaluation(-mse)

    problem = RegressionProblem()
    use_separable = param_count > 10000
    if use_separable:
        print(f"  Using separable (diagonal) CMA-ES for {param_count:,} params")

    searcher = CMAES(
        problem,
        popsize=CMAES_CONFIG["popsize"],
        stdev_init=CMAES_CONFIG["stdev_init"],
        separable=use_separable,
    )

    generations = CMAES_CONFIG["generations"]
    epoch_logs = []
    best_mse = float("inf")
    start_time = time.time()

    with track_gpu_memory() as mem:
        for gen in range(generations):
            searcher.step()
            status = searcher.status
            pop_best_fitness = float(status.get("pop_best_eval", 0.0))

            if (gen + 1) % 100 == 0 or gen == generations - 1:
                # Load best solution into model for evaluation
                pop_best_sol = status.get("pop_best", None)
                if pop_best_sol is not None:
                    values = pop_best_sol.values if hasattr(pop_best_sol, 'values') else pop_best_sol
                    offset = 0
                    with torch.no_grad():
                        for p in model.parameters():
                            numel = p.numel()
                            p.data.copy_(values[offset:offset + numel].view(p.shape))
                            offset += numel

                metrics = evaluate_regression(model, test_data, device=device)
                if metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]

                epoch_logs.append({
                    "epoch": gen + 1,
                    "generation": gen + 1,
                    "pop_best_fitness": pop_best_fitness,
                    "test_mse": metrics["mse"],
                    "test_mae": metrics["mae"],
                    "loss": metrics["mse"],
                    "time": time.time() - start_time,
                })
                print(f"    Gen {gen+1}/{generations} | MSE={metrics['mse']:.4f} | MAE={metrics['mae']:.4f} | fitness={pop_best_fitness:.6f}")

    wall_time = time.time() - start_time

    # Re-evaluate final regression metrics on test data
    final_metrics = evaluate_regression(model, test_data, device=device)

    filepath = save_result(
        benchmark=BENCHMARK,
        method="cmaes",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": final_metrics["mse"],
            "best_mse": min(best_mse, final_metrics["mse"]),
            "final_mae": final_metrics["mae"],
            "best_mae": final_metrics["mae"],
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": generations * CMAES_CONFIG["popsize"],
            "total_steps": generations,
        },
        hyperparameters=CMAES_CONFIG,
        epoch_logs=epoch_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_openai_es(seed, device, train_data, val_data, test_data, results_dir,
                  epochs_override=None):
    """Train time-series LSTM with OpenAI Evolution Strategy."""
    from experiments.baselines.openai_es import train_openai_es

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    # Build DataLoaders for OpenAI-ES (uses train_loader/test_loader interface)
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TimeSeriesDataset(test_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Regression eval callback: evaluates MSE/MAE instead of classification accuracy
    def regression_eval(m):
        metrics = evaluate_regression(m, test_data, device=device)
        return {"test_mse": metrics["mse"], "test_mae": metrics["mae"]}

    result = train_openai_es(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        sigma=OPENAI_ES_CONFIG["sigma"],
        lr=OPENAI_ES_CONFIG["lr"],
        population_size=OPENAI_ES_CONFIG["population_size"],
        generations=OPENAI_ES_CONFIG["generations"],
        lr_decay=OPENAI_ES_CONFIG.get("lr_decay", False),
        weight_decay=OPENAI_ES_CONFIG.get("weight_decay", 0.0),
        fitness_shaping=OPENAI_ES_CONFIG.get("fitness_shaping", "zscore"),
        device=device,
        seed=seed,
        eval_fn=regression_eval,
    )

    # Re-evaluate final regression metrics
    final_metrics = evaluate_regression(model, test_data, device=device)

    # Track best_mse from epoch_logs (baseline doesn't track it internally for regression)
    best_mse = final_metrics["mse"]
    best_mae = final_metrics["mae"]
    for log in result["epoch_logs"]:
        if "test_mse" in log and log["test_mse"] < best_mse:
            best_mse = log["test_mse"]
        if "test_mae" in log and log["test_mae"] < best_mae:
            best_mae = log["test_mae"]

    filepath = save_result(
        benchmark=BENCHMARK,
        method="openai_es",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": final_metrics["mse"],
            "best_mse": best_mse,
            "final_mae": final_metrics["mae"],
            "best_mae": best_mae,
            "wall_time_seconds": result["metrics"]["wall_time_seconds"],
            "peak_gpu_memory_mb": result["metrics"]["peak_gpu_memory_mb"],
            "function_evals": result["metrics"]["function_evals"],
            "total_steps": result["metrics"]["total_steps"],
        },
        hyperparameters=OPENAI_ES_CONFIG,
        epoch_logs=result["epoch_logs"],
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_spsa(seed, device, train_data, val_data, test_data, results_dir,
             epochs_override=None):
    """Train time-series LSTM with SPSA."""
    from experiments.baselines.spsa import train_spsa

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    # Build DataLoaders for SPSA (uses train_loader/test_loader interface)
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TimeSeriesDataset(test_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Regression eval callback: evaluates MSE/MAE instead of classification accuracy
    def regression_eval(m):
        metrics = evaluate_regression(m, test_data, device=device)
        return {"test_mse": metrics["mse"], "test_mae": metrics["mae"]}

    result = train_spsa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        a=SPSA_CONFIG["a"],
        c=SPSA_CONFIG["c"],
        alpha=SPSA_CONFIG.get("alpha", 0.602),
        gamma=SPSA_CONFIG.get("gamma", 0.101),
        max_iters=SPSA_CONFIG["max_iters"],
        device=device,
        seed=seed,
        eval_fn=regression_eval,
    )

    # Re-evaluate final regression metrics
    final_metrics = evaluate_regression(model, test_data, device=device)

    # Track best_mse from epoch_logs (baseline doesn't track it internally for regression)
    best_mse = final_metrics["mse"]
    best_mae = final_metrics["mae"]
    for log in result["epoch_logs"]:
        if "test_mse" in log and log["test_mse"] < best_mse:
            best_mse = log["test_mse"]
        if "test_mae" in log and log["test_mae"] < best_mae:
            best_mae = log["test_mae"]

    filepath = save_result(
        benchmark=BENCHMARK,
        method="spsa",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": final_metrics["mse"],
            "best_mse": best_mse,
            "final_mae": final_metrics["mae"],
            "best_mae": best_mae,
            "wall_time_seconds": result["metrics"]["wall_time_seconds"],
            "peak_gpu_memory_mb": result["metrics"]["peak_gpu_memory_mb"],
            "function_evals": result["metrics"]["function_evals"],
            "total_steps": result["metrics"]["total_steps"],
        },
        hyperparameters=SPSA_CONFIG,
        epoch_logs=result["epoch_logs"],
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_persistence(seed, device, train_data, val_data, test_data, results_dir,
                    epochs_override=None):
    """Compute persistence (naive) baseline for time-series forecasting.

    Persistence forecast: predict the last observed value for all H=96
    future steps. This is the floor that any learned model must beat.
    """
    start_time = time.time()
    result = compute_persistence_baseline(test_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    wall_time = time.time() - start_time

    filepath = save_result(
        benchmark=BENCHMARK,
        method="persistence",
        seed=seed,
        metrics={
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "final_mse": result["mse"],
            "best_mse": result["mse"],
            "final_mae": result["mae"],
            "best_mae": result["mae"],
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": 0.0,
            "function_evals": 0,
            "total_steps": 0,
        },
        hyperparameters={"method": "persistence", "seq_len": SEQ_LEN, "pred_len": PRED_LEN},
        epoch_logs=[],
        results_dir=results_dir,
    )
    print(f"    Persistence MSE={result['mse']:.4f}, MAE={result['mae']:.4f}")
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "polystep": run_polystep,
    "adam": run_adam,
    "cmaes": run_cmaes,
    "openai_es": run_openai_es,
    "spsa": run_spsa,
    "persistence": run_persistence,
}


def run_method(method, seed, device, results_dir, epochs_override=None, solver=None,
               audit_no_leakage: bool = True):
    """Run a single method+seed combination.

    Loads ETTh1 data, then delegates to the method-specific runner.
    """
    train_data, val_data, test_data, scaler = load_etth1()
    runner = METHOD_RUNNERS.get(method)
    if runner is None:
        print(f"    Unknown method: {method}")
        return
    if method == 'polystep':
        runner(seed, device, train_data, val_data, test_data, results_dir,
               epochs_override=epochs_override, solver=solver,
               audit_no_leakage=audit_no_leakage)
    else:
        runner(seed, device, train_data, val_data, test_data, results_dir,
               epochs_override=epochs_override)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ETTh1 time-series benchmark: all methods x all seeds"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["polystep", "adam", "cmaes", "openai_es", "spsa", "persistence"],
        help="Methods to run (default: all 6)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to run (default: 42 123 456 789 1337)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--results-dir", default="experiments/results/softmax/main", help="Results directory")
    parser.add_argument(
        "--epochs-override", type=int, default=None,
        help="Override epoch count for polystep and Adam (for smoke testing)",
    )
    parser.add_argument(
        "--solver", choices=["softmax", "sinkhorn"], default="softmax",
        help="Solver backend: softmax (default, used with subspace) or sinkhorn (full-space).",
    )
    parser.add_argument(
        "--allow-test-leakage", action="store_true",
        help=(
            "Legacy mode: select best_state_dict on test MSE instead of "
            "validation MSE. Default is honest protocol (val-selected). "
            "Use only for bit-for-bit reproduction of earlier results."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("ETTh1 Time-Series Benchmark")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    if args.epochs_override is not None:
        print(f"  Epochs override: {args.epochs_override}")
    print()

    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{method}_{seed}.json"
            )
            if os.path.exists(output_file):
                print(f"Skipping {method} seed={seed} (result exists)")
                continue
            print(f"Running {method} seed={seed}...")
            try:
                run_method(method, seed, args.device, args.results_dir,
                           epochs_override=args.epochs_override, solver=args.solver,
                           audit_no_leakage=not args.allow_test_leakage)
            except Exception as e:
                print(f"  ERROR: {method} seed={seed} failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nDone. Results in experiments/results/softmax/main/{BENCHMARK}_*.json")


if __name__ == "__main__":
    main()
