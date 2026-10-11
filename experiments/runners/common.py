"""Shared experiment utilities for paper experiments.

Seed management, result saving (JSON), environment info, accuracy evaluation,
GPU memory tracking, and dataset loading (MNIST, Fashion-MNIST).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# cuBLAS reads this once, when the CUDA context is created, so it must be set at
# import time before anything touches CUDA. setdefault preserves an
# operator-supplied value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# OpenMP spin-wait dominates per-step cost when the thread pool and the main
# thread oversubscribe; PASSIVE avoids it. Must be set before torch initializes
# its thread pool, hence import time.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch  # noqa: E402
import torch.nn as nn
from torch.utils.data import DataLoader

from polystep.benchmarks.utils import (
    seeded_loader_kwargs,
    get_environment_info as _base_get_environment_info,
    get_mnist_loaders as _base_get_mnist_loaders,
    MNISTNet,
)


SEEDS: List[int] = [42, 123, 456, 789, 1337]

#: `run_maxsat.py` writes `cmaes`; every other runner writes `cma_es`. Readers accept both.
METHOD_ALIASES = {"cmaes": "cma_es"}


_DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "softmax",
    "main",
)


def get_environment_info() -> Dict[str, Any]:
    """Environment info for reproducibility, plus peak GPU memory in MB."""
    info = _base_get_environment_info()

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        info["peak_gpu_memory_mb"] = round(peak_bytes / (1024 * 1024), 2)
    else:
        info["peak_gpu_memory_mb"] = 0.0

    return info


@contextmanager
def track_gpu_memory():
    """Record peak GPU memory usage of the wrapped block.

    Yields a dict populated with 'peak_gpu_memory_mb' on exit (0.0 without CUDA).
    """
    result: Dict[str, float] = {"peak_gpu_memory_mb": 0.0}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            result["peak_gpu_memory_mb"] = round(peak_bytes / (1024 * 1024), 2)


def save_result(
    benchmark: str,
    method: str,
    seed: int,
    metrics: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    epoch_logs: Optional[List[Dict[str, Any]]] = None,
    step_logs: Optional[List[Dict[str, Any]]] = None,
    results_dir: Optional[str] = None,
    leaked: bool = False,
) -> str:
    """Save a single experiment run result to JSON as {benchmark}_{method}_{seed}.json.

    ``leaked=True`` stamps runs that selected on the test set; the aggregator
    refuses those files. Raises ValueError if required metric keys are missing.
    """
    required_keys = {
        "final_accuracy",
        "best_accuracy",
        "wall_time_seconds",
        "peak_gpu_memory_mb",
        "function_evals",
        "total_steps",
    }
    missing = required_keys - set(metrics.keys())
    if missing:
        raise ValueError(f"Missing required metric keys: {missing}")

    metrics = dict(metrics)
    # Reported metric; runners evaluate the selected checkpoint on test as their
    # last act, so final_accuracy is the same number when not passed explicitly.
    metrics.setdefault("test_accuracy_at_selected", metrics["final_accuracy"])

    if results_dir is None:
        results_dir = _DEFAULT_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result_metrics = {
        "final_accuracy": float(metrics["final_accuracy"]),
        "best_accuracy": float(metrics["best_accuracy"]),
        "test_accuracy_at_selected": float(metrics["test_accuracy_at_selected"]),
        "wall_time_seconds": float(metrics["wall_time_seconds"]),
        "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
        # None passes through: gradient baselines (Adam, PPO, DQN) evaluate no
        # candidates, recorded as null rather than an invented count.
        "function_evals": (None if metrics["function_evals"] is None else int(metrics["function_evals"])),
        "total_steps": int(metrics["total_steps"]),
    }
    for k, v in metrics.items():
        if k not in result_metrics:
            result_metrics[k] = v

    result = {
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "timestamp": timestamp,
        "environment": get_environment_info(),
        "hyperparameters": hyperparameters or {},
        "metrics": result_metrics,
        "epoch_logs": epoch_logs or [],
        "step_logs": step_logs or [],
        "leaked": bool(leaked),
    }

    filename = f"{benchmark}_{method}_{seed}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        # default=str serializes scheduler objects in `hyperparameters` as their
        # dataclass repr instead of aborting the run after training is done.
        json.dump(result, f, indent=2, default=str)

    return filepath


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Classification accuracy on a test DataLoader.

    SNN models (those with a 'num_steps' attribute) accumulate spikes across
    timesteps; SST-2 style 3-element batches are also handled.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    correct = 0
    total = 0

    for batch in test_loader:
        if len(batch) == 2:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
        elif len(batch) == 3:
            # SST-2 format: (input_ids, attention_mask, labels)
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        preds = outputs.argmax(dim=-1)
        # Regression task: classification accuracy is meaningless, return 0.0.
        if preds.shape != targets.shape:
            model.train()
            return 0.0
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


def pin_threads() -> None:
    """Keep the intra-op pool clear of the core count.

    At the full core count the pool and the main thread oversubscribe and OpenMP
    spin-wait dominates PolyStep's many small forwards; two below nproc avoids
    it. ``POLYSTEP_THREADS`` overrides.
    """
    requested = os.environ.get("POLYSTEP_THREADS")
    if requested is not None:
        n = int(requested)
    else:
        n = max(1, (os.cpu_count() or 4) - 2)
    torch.set_num_threads(n)


def set_deterministic(warn_only: bool = True) -> None:
    """Pin every knob that makes a CUDA run drift between repeats.

    Deterministic kernel selection, the cuBLAS workspace (set at import time),
    and TF32 off, whose reduced mantissa makes sums kernel-dependent.
    ``warn_only=True`` because a few experiment ops have no deterministic CUDA
    implementation; see ``docs/determinism.md``.
    """
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 off: on by default on Ampere+, silently changes results and their
    # run-to-run stability.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def reseed_loaders(seed: int, *loaders) -> None:
    """Rewind each DataLoader's shuffle generator to ``seed``.

    A loader's generator advances with each epoch, so without this the minibatch
    stream differs between the first and later runs in one process, and
    ``set_seed`` cannot fix it because the loader does not draw from the global
    RNG. Call after ``set_seed``, before each run that consumes the loaders.
    """
    for loader in loaders:
        g = getattr(loader, "generator", None)
        if g is not None:
            g.manual_seed(seed)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.

    Also applies deterministic algorithms and thread pinning. Does not touch
    DataLoader shuffle generators; see :func:`reseed_loaders`.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_deterministic()
    pin_threads()


def make_train_val_split(
    train_loader: DataLoader,
    val_frac: float = 0.1,
    seed: int = 42,
    val_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Carve a deterministic validation subset out of a train DataLoader.

    Returns ``(new_train_loader, val_loader)``: a seed-controlled held-out subset
    of ``val_frac`` of the data and its complement, keeping the input loader's
    batching settings.
    """
    dataset = train_loader.dataset
    n_total = len(dataset)
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=g,
    )

    batch_size = getattr(train_loader, "batch_size", 64) or 64
    num_workers = getattr(train_loader, "num_workers", 0)
    shuffle_train = True

    new_train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=getattr(train_loader, "pin_memory", False),
        **seeded_loader_kwargs(seed),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size if val_batch_size is not None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(train_loader, "pin_memory", False),
        **seeded_loader_kwargs(seed),
    )
    return new_train_loader, val_loader


def load_mnist(
    data_dir: str = "data/",
    batch_size: int = 512,
    max_train: int = 0,
    max_test: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """MNIST train/test DataLoaders, stored under ``{data_dir}/mnist``."""
    mnist_dir = os.path.join(data_dir, "mnist")
    return _base_get_mnist_loaders(
        data_dir=mnist_dir,
        batch_size=batch_size,
        normalize=True,
        max_train=max_train,
        max_test=max_test,
    )


def load_fashion_mnist(
    data_dir: str = "data/",
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Fashion-MNIST train/test DataLoaders, with its own normalization stats."""
    from torchvision import datasets, transforms

    fmnist_dir = os.path.join(data_dir, "fashion_mnist")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_ds = datasets.FashionMNIST(fmnist_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(fmnist_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, **seeded_loader_kwargs())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, **seeded_loader_kwargs())
    return train_loader, test_loader


__all__ = [
    "SEEDS",
    "save_result",
    "get_environment_info",
    "evaluate_accuracy",
    "track_gpu_memory",
    "set_seed",
    "MNISTNet",
    "load_mnist",
    "load_fashion_mnist",
]
