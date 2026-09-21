#!/usr/bin/env python
"""Run all methods and seeds for the ETTh1 time-series forecasting benchmark.

Trains a TimeSeriesLSTM (VmapSafeLSTM hidden=64, Linear head) with polystep,
Adam, gradient-free baselines, and a persistence floor, on ETTh1 univariate OT
(oil temperature) with the Informer-standard split. Results are saved as JSON
under experiments/results/softmax/main/.
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

from polystep.benchmarks.utils import seeded_loader_kwargs
from experiments.runners.common import (
    METHOD_ALIASES,
    SEEDS,
    reseed_loaders,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.fairness import (
    FAIR_METHODS,
    matched_budget,
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
    """Download ETTh1.csv if not already cached; return its path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "ETTh1.csv")
    if not os.path.exists(filepath):
        print(f"Downloading ETTh1.csv to {filepath}...")
        urllib.request.urlretrieve(ETTH1_URL, filepath)
        print(f"Downloaded ({os.path.getsize(filepath)} bytes)")
    return filepath


def load_etth1():
    """Load ETTh1: Informer-standard 8640/2880/2880 split, z-scored with train statistics only.

    Returns (train, val, test, scaler_dict).
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
    """Sliding-window (input, target) pairs over a 1D series, stride 1.

    Input: data[i : i + seq_len] as (seq_len, 1); target: the next pred_len values.
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
    """VmapSafeLSTM + Linear head; predicts all pred_len steps from the last hidden state."""

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
    """MSE and MAE over all sliding windows in data_array."""
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
    """Naive persistence forecast (repeat the last observed value): the floor any learned model must beat."""
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

    Scores the validation split, snapshots the weights whenever val MSE improves,
    and never lets test drive selection. Usable directly as the ES/SPSA
    ``eval_fn``: they restore the current parameters into ``model`` before
    calling it.
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
    """Build the ``save_result`` metric dict for a regression run."""
    return {
        # Regression has no accuracy: NaN reads as "not applicable" in the
        # aggregator, not "0%".
        "final_accuracy": float("nan"),
        "best_accuracy": float("nan"),
        "test_accuracy_at_selected": float("nan"),
        # best_mse carries the reported number: test MSE of the val-selected
        # checkpoint. A min-over-epochs test MSE here would leak.
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
EPOCHS = 30  # polystep epochs
ADAM_EPOCHS = 50  # Adam epochs (gradient-based ceiling)
SEQ_LEN = 96
PRED_LEN = 96

# polystep hyperparameters (HybridSubspace)
POLYSTEP_CONFIG = {
    "rank": 8,
    "epsilon_init": 10.0,
    "epsilon_target": 0.1,
    "step_radius_init": 5.0,
    "step_radius_target": 1.0,
    "probe_radius_init": 10.0,
    "probe_radius_target": 2.0,
    "num_probe": 1,
    "rotation_interval": 0,
    "absorb_interval": 0,
    "chunk_size": 1024,
    # Probe-radius jitter makes anything measured at the previous step's radius
    # stale, so amortized OT and adaptive probes stay off while jitter is on.
    "amortize_steps": 3,
    "amortize_ema": 0.7,
    "adaptive_probes": False,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.95,
}

# Free-running (non-fair) candidate budgets per method. --fair replaces them
# with one budget derived from PolyStep.
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
    """Train the time-series LSTM with PolyStepOptimizer + HybridSubspace.

    Best-checkpoint selection uses validation MSE, not test MSE.
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **seeded_loader_kwargs(seed))
    reseed_loaders(seed, train_loader)

    total_steps = epochs * len(train_loader)
    cfg = apply_theory_mode(POLYSTEP_CONFIG) if theory_mode else dict(POLYSTEP_CONFIG)

    def sched(key, flat_default):
        # Scheduled when the config carries both endpoints; flat otherwise.
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
                            # Cumulative candidate evaluations: the shared x-axis
                            # of the quality-vs-evaluations figure.
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
    # The epoch loop already selected on the selection split; metrics() scores
    # test once on the winning checkpoint.
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
            # Keys that let a reader check the table was matched.
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **seeded_loader_kwargs(seed))
    reseed_loaders(seed, train_loader)

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

    Selection is on validation MSE; test is scored once on the selected weights.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    cfg = apply_theory_mode(POLYSTEP_CONFIG) if theory_mode else dict(POLYSTEP_CONFIG)
    epochs = epochs_override if epochs_override is not None else EPOCHS

    set_seed(seed)
    model = TimeSeriesLSTM().to(device)
    match_axis = "evals"
    deadline_s = None
    layout = ParamLayout.from_module(model)
    # A seeded generator keeps the minibatch stream identical across
    # gradient-free methods; drawing from the global RNG would diverge per method.
    train_loader = DataLoader(
        TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        **seeded_loader_kwargs(seed),
    )
    reseed_loaders(seed, train_loader)

    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        # EGGROLL alone gets FactoredSubspace, whose dimension is smaller by
        # construction; record the shared dim so the table shows the rank match
        # and the dimension gap.
        shared_dim = make_subspace(layout, rank=cfg["rank"], seed=seed, method="polystep").subspace_dim
        if budget is None:
            budget = _fair_eval_budget(seed, device, train_loader, epochs, cfg)
            budget, match_axis, deadline_s = matched_budget(
                method,
                showcase=BENCHMARK,
                seed=seed,
                polystep_steps=epochs * len(train_loader),
                eval_budget=budget,
                results_dir=results_dir,
            )
    if budget is None:
        budget = LEGACY_BUDGETS[method]

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.MSELoss()), train_loader, device)
    # Honest protocol selects on val; test drives nothing.
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
        shared_subspace_dim=shared_dim if fair else None,
        probe_scale=probe_scale_of(cfg, dim=subspace.subspace_dim if subspace else layout.total_params),
        quality_key="mse",
        deadline_s=deadline_s,
    )
    # Score the validation-selected checkpoint on test once.
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
    out["hyperparameters"]["match_axis"] = match_axis
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
    """Persistence baseline: predict the last observed value for all 96 steps.

    No parameters, so nothing to select; val and test are each scored once.
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
    """Run a single method+seed combination."""
    train_data, val_data, test_data, scaler = load_etth1()
    method = METHOD_ALIASES.get(method, method)
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
            "Run the unaccelerated reference configuration: probe_radius_jitter=0.05 "
            "with the smooth density, independent rotations, flat epsilon, step "
            "radius r0*(t+1)^-(1/2+0.1), orthoplex, HybridSubspace, no momentum / "
            "amortization / Anderson."
        ),
    )
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repeat runs even when result files exist.",
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

    failures: list = []
    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{METHOD_ALIASES.get(method, method)}_{seed}.json"
            )
            if os.path.exists(output_file) and not args.force:
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
                failures.append((method, seed, repr(e)))
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for method, seed, err in failures:
            print(f"  {method} seed={seed}: {err}")
        return 1
    print(f"\nDone. Results in {args.results_dir}/{BENCHMARK}_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
