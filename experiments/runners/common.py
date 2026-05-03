"""Shared experiment utilities for paper experiments.

Provides seed management, result saving (JSON), environment info collection,
accuracy evaluation, parameter flattening, GPU memory tracking, and dataset
loading (MNIST, CIFAR-10, DVS-Gesture, N-MNIST, SHD via Tonic).

All experiment runner scripts import from this module to ensure consistent
result format and reproducibility across benchmarks.

JSON schema per run:
    {
        "benchmark": str,
        "method": str,
        "seed": int,
        "timestamp": str (ISO 8601),
        "environment": dict,
        "hyperparameters": dict,
        "metrics": {
            "final_accuracy": float,
            "best_accuracy": float,
            "wall_time_seconds": float,
            "peak_gpu_memory_mb": float,
            "function_evals": int,
            "total_steps": int,
        },
        "epoch_logs": [{"epoch": int, "accuracy": float, "loss": float, "time": float}],
    }
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from polystep.benchmarks.utils import (
    BenchmarkResult,
    get_environment_info as _base_get_environment_info,
    get_mnist_loaders as _base_get_mnist_loaders,
    get_cifar10_loaders as _base_get_cifar10_loaders,
    MNISTNet,
    CIFAR10Net,
)


# ---------------------------------------------------------------------------
# Tonic availability check (for neuromorphic datasets)
# ---------------------------------------------------------------------------

_HAS_TONIC = False
try:
    import tonic
    import tonic.transforms as tonic_transforms
    _HAS_TONIC = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Seeds for publication rigor (5 seeds, extending the 3-seed benchmarks list)
# ---------------------------------------------------------------------------

SEEDS: List[int] = [42, 123, 456, 789, 1337]


# ---------------------------------------------------------------------------
# Default results directory (relative to repo root)
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "softmax",
    "main",
)


# ---------------------------------------------------------------------------
# Environment info (extends benchmarks/utils.py version)
# ---------------------------------------------------------------------------

def get_environment_info() -> Dict[str, Any]:
    """Collect environment info for reproducibility.

    Wraps polystep.benchmarks.utils.get_environment_info() and adds
    peak GPU memory tracking via torch.cuda.max_memory_allocated().

    Returns:
        Dict with torch_version, cuda_version, gpu_model, python_version,
        platform, and peak_gpu_memory_mb (if CUDA available).
    """
    info = _base_get_environment_info()

    if torch.cuda.is_available():
        # Record current peak GPU memory in MB
        peak_bytes = torch.cuda.max_memory_allocated()
        info["peak_gpu_memory_mb"] = round(peak_bytes / (1024 * 1024), 2)
    else:
        info["peak_gpu_memory_mb"] = 0.0

    return info


# ---------------------------------------------------------------------------
# GPU memory tracking
# ---------------------------------------------------------------------------

@contextmanager
def track_gpu_memory():
    """Context manager that records peak GPU memory usage.

    Resets CUDA memory stats on entry, records peak allocation on exit.
    Yields a dict that will be populated with 'peak_gpu_memory_mb' on exit.

    Usage::

        with track_gpu_memory() as mem:
            # ... run training ...
        print(f"Peak GPU: {mem['peak_gpu_memory_mb']:.1f} MB")

    If CUDA is not available, peak_gpu_memory_mb will be 0.0.
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


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(
    benchmark: str,
    method: str,
    seed: int,
    metrics: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    epoch_logs: Optional[List[Dict[str, Any]]] = None,
    step_logs: Optional[List[Dict[str, Any]]] = None,
    results_dir: Optional[str] = None,
) -> str:
    """Save a single experiment run result to JSON.

    File naming convention: {benchmark}_{method}_{seed}.json
    Saved to experiments/results/ by default.

    Args:
        benchmark: Benchmark name (e.g., 'mnist', 'dvs_gesture').
        method: Method name (e.g., 'polystep', 'cmaes', 'sgd').
        seed: Random seed used for this run.
        metrics: Dict with at least:
            - final_accuracy (float)
            - best_accuracy (float)
            - wall_time_seconds (float)
            - peak_gpu_memory_mb (float)
            - function_evals (int)
            - total_steps (int)
        hyperparameters: Optional dict of hyperparameters used.
        epoch_logs: Optional list of per-epoch metric dicts, each with
            at minimum 'epoch', 'accuracy', 'loss', 'time' keys.
        step_logs: Optional list of per-N-step metric dicts for fine-grained
            tracking. Each entry should have 'step', 'epoch', and relevant
            metrics (accuracy/mse, loss, wall_time).
        results_dir: Directory to save results. Defaults to experiments/results/.

    Returns:
        Path to the saved JSON file.

    Raises:
        ValueError: If required metric keys are missing.
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

    if results_dir is None:
        results_dir = _DEFAULT_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build metrics dict: required keys first, then any extras
    result_metrics = {
        "final_accuracy": float(metrics["final_accuracy"]),
        "best_accuracy": float(metrics["best_accuracy"]),
        "wall_time_seconds": float(metrics["wall_time_seconds"]),
        "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
        "function_evals": int(metrics["function_evals"]),
        "total_steps": int(metrics["total_steps"]),
    }
    # Preserve extra keys beyond standard metrics
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
    }

    filename = f"{benchmark}_{method}_{seed}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    return filepath


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate classification accuracy on a test DataLoader.

    Handles both regular models and SNN models:
    - Regular models: output logits of shape (batch, num_classes).
    - SNN models: accumulate spike outputs across timesteps. Detected
      by checking if model has a 'num_steps' attribute.

    Also handles SST-2 style 3-element batches (input_ids, attention_mask, labels).

    Args:
        model: The model to evaluate.
        test_loader: DataLoader yielding (inputs, labels) or
            (input_ids, attention_mask, labels).
        device: Device to evaluate on. If None, inferred from model parameters.

    Returns:
        Accuracy as a float in [0, 1].
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
        # Guard: if preds and targets shapes differ, this is a regression
        # task where classification accuracy is meaningless - return 0.0.
        if preds.shape != targets.shape:
            model.train()
            return 0.0
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Parameter flattening (for gradient-free baselines: OpenAI ES, SPSA)
# ---------------------------------------------------------------------------

def load_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single 1D tensor.

    Args:
        model: The model whose parameters to flatten.

    Returns:
        1D tensor containing all parameters concatenated.
    """
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a flat 1D tensor.

    Args:
        model: The model whose parameters to set.
        flat_params: 1D tensor with the same total number of elements
            as model parameters.

    Raises:
        ValueError: If flat_params size doesn't match model parameter count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if flat_params.numel() != total_params:
        raise ValueError(
            f"flat_params has {flat_params.numel()} elements, "
            f"but model has {total_params} parameters"
        )

    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].reshape(p.shape))
        offset += numel


# ---------------------------------------------------------------------------
# Convenience: create standard loss function
# ---------------------------------------------------------------------------

def get_loss_fn(benchmark: str) -> nn.Module:
    """Get the standard loss function for a benchmark.

    Args:
        benchmark: Benchmark name (e.g., 'mnist', 'cifar10', 'dvs_gesture').

    Returns:
        Loss module (CrossEntropyLoss for all current benchmarks).
    """
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: The random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Validation-split helper for leakage-free model selection
# ---------------------------------------------------------------------------


def make_train_val_split(
    train_loader: DataLoader,
    val_frac: float = 0.1,
    seed: int = 42,
    val_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Carve a deterministic validation subset out of a train DataLoader.

    Use this to drive ``best_state_dict`` selection without peeking at
    the test set. Returns a ``(new_train_loader, val_loader)`` pair where:

    - ``val_loader`` is a deterministic random subset of size
      ``val_frac * len(train_dataset)`` (seed-controlled), held out
      from training.
    - ``new_train_loader`` is the complement, with the same batch_size,
      shuffle, and num_workers as the input loader.

    Args:
        train_loader: Original DataLoader covering the full training set.
        val_frac: Fraction of the training set held out for validation.
        seed: Random seed for the deterministic split.
        val_batch_size: Batch size for the val_loader. Defaults to the
            train loader's batch size.

    Returns:
        ``(new_train_loader, val_loader)``.
    """
    dataset = train_loader.dataset
    n_total = len(dataset)
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=g,
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
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size if val_batch_size is not None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(train_loader, "pin_memory", False),
    )
    return new_train_loader, val_loader


# ---------------------------------------------------------------------------
# Data loading: MNIST (via benchmarks/utils.py)
# ---------------------------------------------------------------------------

def load_mnist(
    data_dir: str = "data/",
    batch_size: int = 512,
    max_train: int = 0,
    max_test: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST train/test as PyTorch DataLoaders.

    Wraps ``polystep.benchmarks.utils.get_mnist_loaders()`` with a
    default ``data_dir`` pointing to the repo-level ``data/`` directory
    instead of ``/tmp/mnist``.

    Args:
        data_dir: Directory to store/load MNIST data. Defaults to
            ``data/`` relative to the current working directory.
        batch_size: Batch size for the training loader.
        max_train: Maximum training samples (0 = full dataset, ~60K).
        max_test: Maximum test samples (0 = full dataset, ~10K).

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: images (batch, 1, 28, 28), labels (batch,).
    """
    mnist_dir = os.path.join(data_dir, "mnist")
    return _base_get_mnist_loaders(
        data_dir=mnist_dir,
        batch_size=batch_size,
        normalize=True,
        max_train=max_train,
        max_test=max_test,
    )


# ---------------------------------------------------------------------------
# Data loading: Fashion-MNIST
# ---------------------------------------------------------------------------

def load_fashion_mnist(
    data_dir: str = "data/",
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Load Fashion-MNIST train/test as PyTorch DataLoaders.

    Similar to load_mnist but uses Fashion-MNIST dataset (10 classes:
    T-shirt, Trouser, Pullover, etc.). Uses separate normalization stats.

    Args:
        data_dir: Directory to store/load Fashion-MNIST data.
        batch_size: Batch size for the training loader.

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: images (batch, 1, 28, 28), labels (batch,).
    """
    from torchvision import datasets, transforms

    fmnist_dir = os.path.join(data_dir, "fashion_mnist")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(fmnist_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(fmnist_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Data loading: CIFAR-10 (via benchmarks/utils.py)
# ---------------------------------------------------------------------------

def load_cifar10(
    data_dir: str = "data/",
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 train/test as PyTorch DataLoaders.

    Wraps ``polystep.benchmarks.utils.get_cifar10_loaders()`` with a
    default ``data_dir`` pointing to the repo-level ``data/`` directory.

    Args:
        data_dir: Directory to store/load CIFAR-10 data.
        batch_size: Batch size for the training loader.

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: images (batch, 3, 32, 32), labels (batch,).
    """
    cifar_dir = os.path.join(data_dir, "cifar10")
    return _base_get_cifar10_loaders(
        data_dir=cifar_dir,
        batch_size=batch_size,
        normalize=True,
    )


# ---------------------------------------------------------------------------
# Data loading: DVS-Gesture (Tonic, no synthetic fallback)
# ---------------------------------------------------------------------------

def load_dvs_gesture(
    data_dir: str = "data/",
    num_steps: int = 25,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """Load DVS-Gesture dataset via Tonic.

    Uses ``tonic.datasets.DVSGesture`` with Denoise + ToFrame transforms.
    There is **no synthetic fallback** -- if Tonic is not installed or the
    dataset cannot be downloaded, a ``RuntimeError`` is raised with
    instructions for manual resolution.

    Args:
        data_dir: Root directory for dataset storage. The DVS-Gesture
            data will be placed under ``{data_dir}/dvs_gesture/``.
        num_steps: Number of time bins for spike-to-frame conversion.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: (batch, num_steps, 2, 128, 128), labels (batch,).

    Raises:
        RuntimeError: If Tonic is not installed or dataset loading fails.
    """
    if not _HAS_TONIC:
        raise RuntimeError(
            "DVS-Gesture dataset not available. Install tonic "
            "(pip install tonic) and ensure data directory exists at "
            f"{data_dir}. For manual download, see "
            "https://research.ibm.com/interactive/dvsgesture/"
        )

    dvs_dir = os.path.join(data_dir, "dvs_gesture")
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=num_steps,
        ),
    ])

    try:
        train_ds = tonic.datasets.DVSGesture(
            save_to=dvs_dir, train=True, transform=transform,
        )
        test_ds = tonic.datasets.DVSGesture(
            save_to=dvs_dir, train=False, transform=transform,
        )
    except Exception as e:
        raise RuntimeError(
            f"DVS-Gesture dataset loading failed: {e}. "
            "Install tonic (pip install tonic) and ensure data directory "
            f"exists at {dvs_dir}. For manual download, see "
            "https://research.ibm.com/interactive/dvsgesture/"
        ) from e

    collate_fn = tonic.collation.PadTensors()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Data loading: N-MNIST (Tonic, no synthetic fallback)
# ---------------------------------------------------------------------------

def load_nmnist(
    data_dir: str = "data/",
    num_steps: int = 25,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Load N-MNIST dataset via Tonic.

    Uses ``tonic.datasets.NMNIST`` with Denoise + ToFrame transforms.
    There is **no synthetic fallback** -- raises ``RuntimeError`` if
    Tonic is not installed or the dataset cannot be loaded.

    Args:
        data_dir: Root directory for dataset storage. The N-MNIST data
            will be placed under ``{data_dir}/nmnist/``.
        num_steps: Number of time bins for spike-to-frame conversion.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: (batch, num_steps, 2, 34, 34), labels (batch,).

    Raises:
        RuntimeError: If Tonic is not installed or dataset loading fails.
    """
    if not _HAS_TONIC:
        raise RuntimeError(
            "N-MNIST dataset not available. Install tonic "
            "(pip install tonic) and ensure data directory exists at "
            f"{data_dir}. See https://tonic.readthedocs.io/ for details."
        )

    nmnist_dir = os.path.join(data_dir, "nmnist")
    sensor_size = tonic.datasets.NMNIST.sensor_size

    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=num_steps,
        ),
    ])

    try:
        train_ds = tonic.datasets.NMNIST(
            save_to=nmnist_dir, train=True, transform=transform,
        )
        test_ds = tonic.datasets.NMNIST(
            save_to=nmnist_dir, train=False, transform=transform,
        )
    except Exception as e:
        raise RuntimeError(
            f"N-MNIST dataset loading failed: {e}. "
            "Install tonic (pip install tonic) and ensure data directory "
            f"exists at {nmnist_dir}."
        ) from e

    collate_fn = tonic.collation.PadTensors()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Data loading: SHD (Tonic, no synthetic fallback)
# ---------------------------------------------------------------------------

class _ClampedToFrame:
    """Wrapper around tonic.transforms.ToFrame that clamps event indices.

    SHD dataset sometimes contains events with indices outside sensor_size
    bounds (e.g., x=934 when sensor_size=(700,1,1)). This wrapper clamps
    event coordinates before applying ToFrame to avoid IndexError.
    """

    def __init__(self, to_frame, sensor_size):
        self.to_frame = to_frame
        self.sensor_size = sensor_size

    def __call__(self, events):
        import numpy as np
        if isinstance(events, np.ndarray) and 'x' in events.dtype.names:
            events = events.copy()
            events['x'] = np.clip(events['x'], 0, self.sensor_size[0] - 1)
            if 'y' in events.dtype.names and len(self.sensor_size) > 1:
                events['y'] = np.clip(events['y'], 0, self.sensor_size[1] - 1)
        return self.to_frame(events)


def _shd_collate_fn(batch):
    """Custom collate for SHD: (batch, time, 700), squeeze extra channel dim."""
    data_list, label_list = [], []
    for frames, label in batch:
        t = torch.tensor(frames, dtype=torch.float32)
        # SHD ToFrame produces (time, 1, 700) -- squeeze channel dim
        if t.dim() == 3:
            t = t.squeeze(1)
        data_list.append(t)
        label_list.append(label)
    data = torch.stack(data_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)
    return data, labels


def load_shd(
    data_dir: str = "data/",
    num_steps: int = 100,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Load SHD (Spiking Heidelberg Digits) dataset via Tonic.

    Uses ``tonic.datasets.SHD`` with ToFrame transform. There is **no
    synthetic fallback** -- raises ``RuntimeError`` if Tonic/h5py is
    not installed or the dataset cannot be loaded.

    Args:
        data_dir: Root directory for dataset storage. The SHD data
            will be placed under ``{data_dir}/shd/``.
        num_steps: Number of time bins for spike-to-frame conversion.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: (batch, num_steps, 700), labels (batch,).

    Raises:
        RuntimeError: If Tonic or h5py is not installed, or dataset
            loading fails.
    """
    if not _HAS_TONIC:
        raise RuntimeError(
            "SHD dataset not available. Install tonic and h5py "
            "(pip install tonic h5py)."
        )

    shd_dir = os.path.join(data_dir, "shd")
    sensor_size = tonic.datasets.SHD.sensor_size

    base_transform = tonic.transforms.ToFrame(
        sensor_size=sensor_size, n_time_bins=num_steps,
    )
    # Clamp event indices to sensor_size bounds (SHD has occasional
    # out-of-bounds events that cause IndexError in ToFrame)
    transform = _ClampedToFrame(base_transform, sensor_size)

    try:
        train_ds = tonic.datasets.SHD(
            save_to=shd_dir, train=True, transform=transform,
        )
        test_ds = tonic.datasets.SHD(
            save_to=shd_dir, train=False, transform=transform,
        )
    except Exception as e:
        raise RuntimeError(
            f"SHD dataset loading failed: {e}. "
            "Install tonic and h5py (pip install tonic h5py)."
        ) from e

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=_shd_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_shd_collate_fn,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Function evaluation counter
# ---------------------------------------------------------------------------

class FunctionEvalCounter:
    """Wraps a loss function to count forward pass evaluations.

    Each call to the counter increments count by 1 (counting forward passes,
    not individual samples). This provides a fair comparison metric across
    methods: polystep calls the closure once per step (with vmap batching
    parameter perturbations internally), OpenAI-ES calls once per perturbation,
    SPSA calls twice per iteration.

    Usage:
        counter = FunctionEvalCounter(nn.CrossEntropyLoss())
        loss = counter(model_output, targets)
        print(f"Forward passes: {counter.count}")
    """

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.count = 0

    def __call__(self, outputs, targets):
        self.count += 1
        return self.loss_fn(outputs, targets)

    def reset(self):
        self.count = 0

    def to(self, device):
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(device)
        return self


# ---------------------------------------------------------------------------
# Unified experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_fn,
    train_loader,
    test_loader,
    method='polystep',
    benchmark='nondiff',
    seeds=None,
    device='cuda',
    epochs=20,
    method_config=None,
    results_dir=None,
    loss_fn=None,
):
    """Run experiment across all seeds, saving JSON results.

    Unified runner for all methods. Creates a fresh model per seed via
    model_fn(), trains with the specified method, evaluates accuracy,
    tracks wall-clock time and function evaluations, saves JSON results.

    Args:
        model_fn: Callable returning a fresh nn.Module instance.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        method: One of 'polystep', 'cmaes', 'openai_es', 'spsa', 'adam'.
        benchmark: Benchmark name for JSON filename.
        seeds: List of seeds (defaults to SEEDS = [42, 123, 456, 789, 1337]).
        device: Device string ('cuda' or 'cpu').
        epochs: Number of training epochs (for polystep and adam).
        method_config: Dict of method-specific hyperparameters. Defaults
            provided per method.
        results_dir: Directory for JSON results (defaults to experiments/results/).
        loss_fn: Loss function (defaults to CrossEntropyLoss). Pass custom
            loss for non-standard problems (e.g., MAX-SAT loss).

    Returns:
        List of result file paths (one per seed).
    """
    if seeds is None:
        seeds = SEEDS
    if results_dir is None:
        results_dir = _DEFAULT_RESULTS_DIR
    if method_config is None:
        method_config = {}

    result_paths = []

    for seed in seeds:
        set_seed(seed)
        model = model_fn()
        model = model.to(device)

        # Default loss function (fresh per seed unless user provided one)
        current_loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        start_time = time.time()

        with track_gpu_memory() as mem:
            if method == 'adam':
                result = _run_adam(
                    model, train_loader, test_loader, current_loss_fn,
                    epochs, device, seed, method_config,
                )
            elif method == 'openai_es':
                result = _run_openai_es(
                    model, train_loader, test_loader, current_loss_fn,
                    device, seed, method_config,
                )
            elif method == 'spsa':
                result = _run_spsa(
                    model, train_loader, test_loader, current_loss_fn,
                    device, seed, method_config,
                )
            elif method == 'cmaes':
                result = _run_cmaes(
                    model, train_loader, test_loader, current_loss_fn,
                    device, seed, method_config,
                )
            elif method == 'polystep':
                result = _run_polystep(
                    model, train_loader, test_loader, current_loss_fn,
                    epochs, device, seed, method_config,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        wall_time = time.time() - start_time

        # Extract or compute metrics
        metrics = result.get("metrics", {})
        # Override wall_time with our own measurement (includes overhead)
        metrics["wall_time_seconds"] = wall_time
        # Fill in GPU memory from our tracker
        metrics.setdefault("peak_gpu_memory_mb", mem["peak_gpu_memory_mb"])

        filepath = save_result(
            benchmark=benchmark,
            method=method,
            seed=seed,
            metrics=metrics,
            hyperparameters=result.get("hyperparameters", method_config),
            epoch_logs=result.get("epoch_logs", []),
            results_dir=results_dir,
        )
        result_paths.append(filepath)

    return result_paths


def _run_adam(model, train_loader, test_loader, loss_fn, epochs, device, seed, config):
    """Dispatch to SGD baseline with Adam optimizer."""
    from experiments.baselines.sgd_baseline import train_sgd

    result = train_sgd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer_name='adam',
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0),
        epochs=epochs,
        device=device,
        seed=seed,
    )
    return result


def _run_openai_es(model, train_loader, test_loader, loss_fn, device, seed, config):
    """Dispatch to OpenAI ES baseline."""
    from experiments.baselines.openai_es import train_openai_es

    result = train_openai_es(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        sigma=config.get('sigma', 0.02),
        lr=config.get('lr', 0.01),
        population_size=config.get('population_size', 50),
        generations=config.get('generations', 200),
        lr_decay=config.get('lr_decay', False),
        weight_decay=config.get('weight_decay', 0.0),
        fitness_shaping=config.get('fitness_shaping', 'zscore'),
        device=device,
        seed=seed,
    )
    return result


def _run_spsa(model, train_loader, test_loader, loss_fn, device, seed, config):
    """Dispatch to SPSA baseline."""
    from experiments.baselines.spsa import train_spsa

    result = train_spsa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        a=config.get('a', 0.1),
        c=config.get('c', 0.1),
        A=config.get('A', None),
        alpha=config.get('alpha', 0.602),
        gamma=config.get('gamma', 0.101),
        max_iters=config.get('max_iters', 5000),
        device=device,
        seed=seed,
    )
    return result


def _run_cmaes(model, train_loader, test_loader, loss_fn, device, seed, config):
    """Dispatch to CMA-ES baseline (EvoTorch or pycma)."""
    try:
        from polystep.benchmarks.baselines import train_cmaes, has_evotorch
    except ImportError:
        raise RuntimeError(
            "CMA-ES requires polystep.benchmarks.baselines. "
            "Install evotorch or pycma."
        )

    if not has_evotorch():
        raise RuntimeError(
            "CMA-ES requires EvoTorch. Install with: pip install evotorch"
        )

    # Extract full dataset tensors from loaders
    train_data_list, train_labels_list = [], []
    for data, labels in train_loader:
        train_data_list.append(data)
        train_labels_list.append(labels)
    train_data = torch.cat(train_data_list).to(device)
    train_labels = torch.cat(train_labels_list).to(device)

    test_data_list, test_labels_list = [], []
    for data, labels in test_loader:
        test_data_list.append(data)
        test_labels_list.append(labels)
    test_data = torch.cat(test_data_list).to(device)
    test_labels = torch.cat(test_labels_list).to(device)

    result = train_cmaes(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        generations=config.get('generations', 200),
        popsize=config.get('popsize', 16),
        stdev_init=config.get('stdev_init', 0.5),
        device=device,
        verbose=False,
    )

    return {
        "metrics": {
            "final_accuracy": result.final_accuracy,
            "best_accuracy": result.best_accuracy,
            "wall_time_seconds": 0.0,  # overridden by caller
            "peak_gpu_memory_mb": 0.0,  # overridden by caller
            "function_evals": result.function_evals,
            "total_steps": result.total_steps,
        },
        "hyperparameters": config,
        "epoch_logs": result.epoch_logs,
    }


def _run_polystep(model, train_loader, test_loader, loss_fn, epochs, device, seed, config):
    """Run polystep PolyStepOptimizer inline training loop."""
    from polystep.optimizer import PolyStepOptimizer
    from polystep.cost_nn import NNCostEvaluator

    optimizer = PolyStepOptimizer(
        model,
        compile=config.get('compile', False),
        seed=seed,
        epsilon=config.get('epsilon', 0.5),
        step_radius=config.get('step_radius', 2.0),
        probe_radius=config.get('probe_radius', 1.0),
        num_probe=config.get('num_probe', 3),
        sinkhorn_max_iters=config.get('sinkhorn_max_iters', 50),
        amortize_steps=config.get('amortize_steps', 2),
        amortize_ema=config.get('amortize_ema', 0.7),
        biased_rotation=config.get('biased_rotation', True),
        anderson_depth=config.get('anderson_depth', 5),
        adaptive_omega=config.get('adaptive_omega', True),
        solver=config.get('solver'),
    )

    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    epoch_logs = []
    best_accuracy = 0.0
    step_count = 0
    fwd_pass_count = 0

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
                loss = loss_fn(output, targets)
                if hasattr(loss, 'item'):
                    epoch_loss += loss.item()
                else:
                    epoch_loss += float(loss)
            step_count += 1

        test_acc = evaluate_accuracy(model, test_loader, device=device)
        best_accuracy = max(best_accuracy, test_acc)
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(len(train_loader), 1)

        epoch_logs.append({
            "epoch": epoch + 1,
            "accuracy": test_acc,
            "loss": avg_loss,
            "time": epoch_time,
        })

    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    return {
        "metrics": {
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": 0.0,  # overridden by caller
            "peak_gpu_memory_mb": 0.0,  # overridden by caller
            "function_evals": fwd_pass_count,
            "total_steps": step_count,
        },
        "hyperparameters": config,
        "epoch_logs": epoch_logs,
    }


__all__ = [
    "SEEDS",
    "save_result",
    "get_environment_info",
    "evaluate_accuracy",
    "load_flat_params",
    "set_flat_params",
    "track_gpu_memory",
    "get_loss_fn",
    "set_seed",
    "BenchmarkResult",
    "MNISTNet",
    "CIFAR10Net",
    "load_mnist",
    "load_fashion_mnist",
    "load_cifar10",
    "load_dvs_gesture",
    "load_nmnist",
    "load_shd",
    "FunctionEvalCounter",
    "run_experiment",
]
