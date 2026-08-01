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
import copy
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
from experiments.runners.fairness import (
    FAIR_METHODS,
    apply_theory_mode,
    make_subspace,
    minibatch_loss,
    polystep_eval_budget,
    probe_scale_of,
    run_baseline,
    subspace_tag,
)


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
        - train: np.ndarray of shape (8640,): z-score normalized
        - val: np.ndarray of shape (2880,): z-score normalized
        - test: np.ndarray of shape (2880,): z-score normalized
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
    val_raw = data[8640 : 8640 + 2880]
    test_raw = data[8640 + 2880 : 8640 + 2880 + 2880]

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
        x = self.data[idx : idx + self.seq_len].reshape(-1, 1)  # (seq_len, 1)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len,)
        return torch.from_numpy(x), torch.from_numpy(y)


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
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, pred_len)


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
        target = data_array[i + seq_len : i + seq_len + pred_len]
        diff = target - last_value
        total_mse += float((diff**2).sum())
        total_mae += float(np.abs(diff).sum())
        total_elements += pred_len

    return {
        "mse": total_mse / max(total_elements, 1),
        "mae": total_mae / max(total_elements, 1),
    }


class ValSelected:
    """Validation-based checkpoint selection for a regression run.

    Every method in this runner routes its periodic evaluation through
    one of these. Calling it scores the model on the *validation* split
    and keeps a copy of the weights whenever val MSE improves; the test
    split is only ever logged, never used to choose. ``metrics()`` then
    loads the selected weights back and scores test once, producing the
    metric dict handed to :func:`save_result`.

    The instance is directly usable as the ``eval_fn`` of the ES/SPSA
    baselines: they restore the current parameters into ``model`` before
    calling it, and merge the returned dict into their epoch logs.

    Args:
        model: The model being trained (read live, not copied).
        val_data: 1D normalized validation series.
        test_data: 1D normalized test series.
        device: Device for evaluation.
        audit_no_leakage: False reverts to legacy test-set selection.
    """

    def __init__(self, model, val_data, test_data, device, audit_no_leakage: bool = True):
        self.model = model
        self.test_data = test_data
        self.device = device
        self.select_data = val_data if audit_no_leakage else test_data
        self.best_mse = float("inf")
        self.best_mae = float("inf")
        self.best_state = None

    def __call__(self, _model=None) -> dict:
        """Score the selection split, snapshot on improvement, log both splits."""
        sel = evaluate_regression(self.model, self.select_data, device=self.device)
        if sel["mse"] < self.best_mse:
            self.best_mse = sel["mse"]
            self.best_mae = sel["mae"]
            self.best_state = copy.deepcopy(self.model.state_dict())
        # Logged for convergence curves only; selection never reads it.
        test = (
            sel
            if self.select_data is self.test_data
            else evaluate_regression(self.model, self.test_data, device=self.device)
        )
        return {
            "val_mse": sel["mse"],
            "val_mae": sel["mae"],
            "test_mse": test["mse"],
            "test_mae": test["mae"],
        }

    def metrics(self, **extra) -> dict:
        """Restore the selected checkpoint, score test once, build the metric dict."""
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        test = evaluate_regression(self.model, self.test_data, device=self.device)
        return regression_metrics(test, self.best_mse, self.best_mae, **extra)


def regression_metrics(test: dict, val_mse: float, val_mae: float, **extra) -> dict:
    """Build the ``save_result`` metric dict for a regression run.

    Args:
        test: MSE/MAE of the selected checkpoint on the test split.
        val_mse: Validation MSE that selected the checkpoint.
        val_mae: Validation MAE at the selected checkpoint.
        **extra: Remaining required keys (wall_time_seconds, etc.).
    """
    return {
        # ETTh1 is regression: there is no accuracy to report. NaN rather
        # than 0.0 so the aggregator's mean_accuracy column reads "not
        # applicable" instead of "0%" -- the mirror image of the NaN it
        # already writes into mean_mse for classification benchmarks.
        "final_accuracy": float("nan"),
        "best_accuracy": float("nan"),
        "test_accuracy_at_selected": float("nan"),
        # aggregate_results derives mean_mse/std_mse from best_mse, so
        # best_mse has to carry the headline number: test MSE of the
        # val-selected checkpoint. A min-over-epochs test MSE here would
        # be exactly the leak this protocol removes.
        "final_mse": test["mse"],
        "best_mse": test["mse"],
        "final_mae": test["mae"],
        "best_mae": test["mae"],
        "test_mse_at_selected": test["mse"],
        "test_mae_at_selected": test["mae"],
        "val_mse_at_selected": val_mse,
        "val_mae_at_selected": val_mae,
        **extra,
    }


BENCHMARK = "timeseries"
BATCH_SIZE = 64
EPOCHS = 30  # polystep epochs (30 for production)
ADAM_EPOCHS = 50  # Adam epochs (gradient-based ceiling)
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
    # Per-step multiplicative jitter on the probe radius (transversality).
    # Jitter makes the probe radius differ every step, so nothing measured
    # at the previous step's radius is valid at this one. Amortized OT
    # (amortize_steps=3, ema=0.7) would coast on a stale transport
    # direction and adaptive_probes would reuse stale cost rows, so both
    # are off while jitter is on. The optimizer already refuses cost-row
    # reuse under jitter; this makes the intent explicit at the call site.
    "amortize_steps": 3,
    "amortize_ema": 0.7,
    "adaptive_probes": False,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.95,
}

# Free-running (non-fair) candidate budgets, preserving what the previous ad-hoc
# implementations spent: ES 2000 gen x 50 pop, CMA-ES 2000 gen x 16 pop, SPSA 10000
# iters x 2. `--fair` replaces all of these with one budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 100_000,
    "eggroll": 100_000,
    "cma_es": 32_000,
    "spsa": 20_000,
    "mezo": 20_000,
    "random_search": 20_000,
}

# Adam hyperparameters
ADAM_CONFIG = {
    "lr": 0.001,
    "epochs": ADAM_EPOCHS,
}


def run_polystep(
    seed,
    device,
    train_data,
    val_data,
    test_data,
    results_dir,
    epochs_override=None,
    solver=None,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
):
    """Train time-series LSTM with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses validation MSE (from
    the Informer-standard val split) instead of test MSE (honest
    protocol). Set ``audit_no_leakage=False`` to revert to legacy.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    epochs = epochs_override if epochs_override is not None else EPOCHS
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_steps = epochs * len(train_loader)
    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)

    def sched(key, flat_default):
        # Scheduled when the config carries both endpoints; flat otherwise, which is
        # also what --theory-mode leaves behind.
        if f"{key}_init" not in cfg:
            return cfg.get(key, flat_default)
        init, target = cfg[f"{key}_init"], cfg[f"{key}_target"]
        return CosineEpsilon(init=init, target=target, decay=(init - target) / max(1, total_steps))

    layout = ParamLayout.from_module(model)
    subspace = make_subspace(
        layout,
        rank=cfg["rank"],
        seed=seed,
        rotation_mode="random",
        rotation_interval=cfg["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=cfg["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=sched("epsilon", 0.5),
        step_radius=sched("step_radius", 1.0),
        probe_radius=sched("probe_radius", 1.0),
        num_probe=cfg["num_probe"],
        subspace=subspace,
        chunk_size=cfg.get("chunk_size", 1024),
        probe_radius_jitter=cfg.get("probe_radius_jitter", 0.0),
        probe_radius_jitter_dist=cfg.get("probe_radius_jitter_dist", "smooth"),
        step_radius_jitter=cfg.get("step_radius_jitter", 0.0),
        polytope_type=cfg.get("polytope_type", "simplex"),
        adaptive_probes=cfg.get("adaptive_probes", None),
        amortize_steps=cfg.get("amortize_steps", 1),
        amortize_ema=cfg.get("amortize_ema", 0.0),
        use_momentum=cfg.get("use_momentum", False),
        momentum_init=cfg.get("momentum_init", 0.5),
        momentum_final=cfg.get("momentum_final", 0.95),
        biased_rotation=cfg.get("biased_rotation", False),
        solver=solver,
    )
    eval_budget = polystep_eval_budget(optimizer, total_steps)

    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    selector = ValSelected(model, val_data, test_data, device, audit_no_leakage=audit_no_leakage)
    epoch_logs = []
    step_logs = []
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
                    step_logs.append(
                        {
                            "step": step_count,
                            "epoch": epoch + 1,
                            # Cumulative candidate evaluations: the x-axis of the
                            # quality-vs-evaluations figure, shared with the baselines.
                            "evals": fwd_pass_count,
                            "test_mse": step_metrics["mse"],
                            "test_mae": step_metrics["mae"],
                            "loss": loss,
                            "wall_time": time.time() - start_time,
                        }
                    )

            # Score the selection split and snapshot if it improved.
            ev = selector()
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(train_loader)

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_mse": avg_loss,
                    **ev,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={avg_loss:.4f} | val={ev['val_mse']:.4f} | test={ev['test_mse']:.4f}"
            )

    wall_time = time.time() - start_time
    # The epoch loop already scored the final weights on the selection
    # split, so there is nothing left to select; test is scored once, in
    # metrics(), on whatever checkpoint won.
    filepath = save_result(
        benchmark=BENCHMARK,
        method="polystep",
        seed=seed,
        metrics=selector.metrics(
            last_epoch_mse=ev["test_mse"],
            last_epoch_mae=ev["test_mae"],
            wall_time_seconds=wall_time,
            peak_gpu_memory_mb=mem["peak_gpu_memory_mb"],
            function_evals=fwd_pass_count,
            total_steps=step_count,
        ),
        hyperparameters={
            **cfg,
            "epochs": epochs,
            # The four keys that let a reader check the table was matched.
            **subspace_tag(subspace, cfg["rank"]),
            "eval_budget": eval_budget,
            "evals_used": fwd_pass_count,
            "fair": fair,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
    )
    print(f"    Saved: {filepath}")


def run_adam(
    seed, device, train_data, val_data, test_data, results_dir, epochs_override=None, audit_no_leakage: bool = True
):
    """Train time-series LSTM with Adam optimizer (gradient-based ceiling)."""
    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    loss_fn = nn.MSELoss()

    epochs = epochs_override if epochs_override is not None else ADAM_EPOCHS
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_CONFIG["lr"])

    selector = ValSelected(model, val_data, test_data, device, audit_no_leakage=audit_no_leakage)
    epoch_logs = []
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

            # Score the selection split and snapshot if it improved.
            ev = selector()
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(train_loader)

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_mse": avg_loss,
                    **ev,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={avg_loss:.4f} | val={ev['val_mse']:.4f} | test={ev['test_mse']:.4f}"
            )

    wall_time = time.time() - start_time

    filepath = save_result(
        benchmark=BENCHMARK,
        method="adam",
        seed=seed,
        metrics=selector.metrics(
            wall_time_seconds=wall_time,
            peak_gpu_memory_mb=mem["peak_gpu_memory_mb"],
            # 1 forward pass per step
            function_evals=step_count,
            total_steps=step_count,
        ),
        hyperparameters=ADAM_CONFIG,
        epoch_logs=epoch_logs,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
    )
    print(f"    Saved: {filepath}")


def _fair_eval_budget(seed, device, train_loader, epochs, cfg, solver="softmax"):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    from polystep.optimizer import PolyStepOptimizer
    from polystep.transform import ParamLayout

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    layout = ParamLayout.from_module(model)
    total_steps = epochs * len(train_loader)
    subspace = make_subspace(
        layout,
        rank=cfg["rank"],
        seed=seed,
        rotation_mode="random",
        rotation_interval=cfg["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=cfg["absorb_interval"],
    )
    opt = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        num_probe=cfg["num_probe"],
        subspace=subspace,
        probe_radius_jitter=cfg.get("probe_radius_jitter", 0.0),
        step_radius_jitter=cfg.get("step_radius_jitter", 0.0),
        polytope_type=cfg.get("polytope_type", "simplex"),
        solver=solver,
    )
    return polystep_eval_budget(opt, total_steps)


def run_gradient_free(
    method,
    seed,
    device,
    train_data,
    val_data,
    test_data,
    results_dir,
    epochs_override=None,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    budget: int = None,
):
    """Run one ``polystep.baselines`` method on the ETTh1 LSTM.

    Replaces the inline EvoTorch CMA-ES copy and the two
    ``experiments/baselines`` wrappers, which each counted evaluations
    differently and searched full parameter space while PolyStep searched a
    subspace. Selection is on validation MSE; test is scored once.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)
    epochs = epochs_override if epochs_override is not None else EPOCHS

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    layout = ParamLayout.from_module(model)
    train_loader = DataLoader(
        TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        if budget is None:
            budget = _fair_eval_budget(seed, device, train_loader, epochs, cfg)
    if budget is None:
        budget = LEGACY_BUDGETS[method]

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.MSELoss()), train_loader, device)
    # Legacy mode selected on test; the honest protocol selects on val.
    selection_data = test_data if not audit_no_leakage else val_data

    out = run_baseline(
        method,
        model=model,
        layout=layout,
        loss_batch=loss_batch,
        budget=budget,
        val_fn=lambda m: evaluate_regression(m, selection_data, device=device)["mse"],
        test_fn=lambda m: evaluate_regression(m, test_data, device=device)["mse"],
        mode="min",
        seed=seed,
        subspace=subspace,
        subspace_rank=cfg["rank"] if fair else None,
        probe_scale=probe_scale_of(cfg),
        quality_key="mse",
    )
    # run_baseline leaves the val-selected checkpoint loaded, so this scores the
    # same weights the headline metric reports, on test, once.
    final = evaluate_regression(model, test_data, device=device)
    out["metrics"].update(
        final_mse=final["mse"],
        best_mse=final["mse"],
        final_mae=final["mae"],
        best_mae=final["mae"],
        test_mse_at_selected=final["mse"],
        test_mae_at_selected=final["mae"],
        val_mse_at_selected=out["metrics"]["best_val_mse"],
    )
    out["hyperparameters"]["fair"] = fair
    filepath = save_result(
        benchmark=BENCHMARK,
        method=method,
        seed=seed,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
        **out,
    )
    print(f"    Saved: {filepath}")


def run_persistence(
    seed, device, train_data, val_data, test_data, results_dir, epochs_override=None, audit_no_leakage: bool = True
):
    """Compute persistence (naive) baseline for time-series forecasting.

    Persistence forecast: predict the last observed value for all H=96
    future steps. This is the floor that any learned model must beat.

    It has no parameters and no checkpoints, so there is nothing to
    select; the val split is still scored, for the same reason the other
    methods report it. Test is scored once, as everywhere else.
    """
    start_time = time.time()
    val = compute_persistence_baseline(val_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test = compute_persistence_baseline(test_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    wall_time = time.time() - start_time

    filepath = save_result(
        benchmark=BENCHMARK,
        method="persistence",
        seed=seed,
        metrics=regression_metrics(
            test,
            val["mse"],
            val["mae"],
            wall_time_seconds=wall_time,
            peak_gpu_memory_mb=0.0,
            function_evals=0,
            total_steps=0,
        ),
        hyperparameters={"method": "persistence", "seq_len": SEQ_LEN, "pred_len": PRED_LEN},
        epoch_logs=[],
        results_dir=results_dir,
        leaked=not audit_no_leakage,
    )
    print(f"    Persistence MSE={test['mse']:.4f}, MAE={test['mae']:.4f}")
    print(f"    Saved: {filepath}")


# polystep, adam and persistence have their own loops; every gradient-free method
# goes through the one shared runner.
METHOD_RUNNERS = {"polystep": run_polystep, "adam": run_adam, "persistence": run_persistence}
ALL_METHODS = ("polystep", "adam", "persistence", *FAIR_METHODS)
#: The old name for cma_es, kept so existing result files and scripts still resolve.
ALIASES = {"cmaes": "cma_es"}


def run_method(
    method,
    seed,
    device,
    results_dir,
    epochs_override=None,
    solver=None,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    budget: int = None,
):
    """Run a single method+seed combination.

    Loads ETTh1 data, then delegates to the method-specific runner.
    """
    train_data, val_data, test_data, scaler = load_etth1()
    method = ALIASES.get(method, method)
    # Every method gets the val split and the leakage guard; only the
    # solver choice is polystep-specific.
    kwargs = {"epochs_override": epochs_override, "audit_no_leakage": audit_no_leakage}
    if method in FAIR_METHODS:
        run_gradient_free(
            method,
            seed,
            device,
            train_data,
            val_data,
            test_data,
            results_dir,
            fair=fair,
            theory_mode=theory_mode,
            budget=budget,
            **kwargs,
        )
        return
    runner = METHOD_RUNNERS.get(method)
    if runner is None:
        print(f"    Unknown method: {method}")
        return
    if method == "polystep":
        kwargs.update(solver=solver, fair=fair, theory_mode=theory_mode)
    runner(seed, device, train_data, val_data, test_data, results_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run ETTh1 time-series benchmark: all methods x all seeds")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        help=f"Methods to run (default: all of {list(ALL_METHODS)})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to run (default: 42 123 456 789 1337)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--results-dir", default="experiments/results/softmax/main", help="Results directory")
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Override epoch count for polystep and Adam (for smoke testing)",
    )
    parser.add_argument(
        "--solver",
        choices=["softmax", "sinkhorn"],
        default="softmax",
        help="Solver backend: softmax (default, used with subspace) or sinkhorn (full-space).",
    )
    parser.add_argument(
        "--allow-test-leakage",
        action="store_true",
        help=(
            "Legacy mode: select best_state_dict on test MSE instead of "
            "validation MSE. Default is honest protocol (val-selected). "
            "Use only for bit-for-bit reproduction of earlier results."
        ),
    )
    parser.add_argument(
        "--fair",
        action="store_true",
        help=(
            "Matched-budget, matched-representation table: every gradient-free "
            "method gets the same subspace (FactoredSubspace for eggroll), the same "
            "candidate budget derived from PolyStep, and the same probe radius."
        ),
    )
    parser.add_argument(
        "--theory-mode",
        action="store_true",
        help=(
            "Run the configuration Theorem 4.2 analyses: probe_radius_jitter=0.05 "
            "with the smooth density, independent rotations, flat epsilon, step "
            "radius r0*(t+1)^-(1/2+0.1), orthoplex, HybridSubspace, no momentum / "
            "amortization / Anderson."
        ),
    )
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
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
            output_file = os.path.join(args.results_dir, f"{BENCHMARK}_{ALIASES.get(method, method)}_{seed}.json")
            if os.path.exists(output_file):
                print(f"Skipping {method} seed={seed} (result exists)")
                continue
            print(f"Running {method} seed={seed}...")
            try:
                run_method(
                    method,
                    seed,
                    args.device,
                    args.results_dir,
                    epochs_override=args.epochs_override,
                    solver=args.solver,
                    audit_no_leakage=not args.allow_test_leakage,
                    fair=args.fair,
                    theory_mode=args.theory_mode,
                    budget=args.budget,
                )
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
