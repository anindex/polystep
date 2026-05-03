"""Benchmark utilities: data loaders, model factories, metrics, and output formatting.

Shared utilities for all benchmark scripts:
- Data loading (MNIST, CIFAR-10, N-MNIST, DVS-Gesture) without torchvision dependency
- Model factories for consistent architectures (MLP, SNN with LIF neurons)
- Evaluation and metrics utilities
- Environment info collection
- JSON output and console table formatting

SNN utilities require snnTorch for optimal experience:
    pip install snntorch

If snnTorch is not installed, falls back to:
- Pure PyTorch LIF neurons (truly non-differentiable)
- Synthetic spike data for neuromorphic datasets
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import platform
import struct as pystruct
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Standard validation seeds 
# ---------------------------------------------------------------------------

# Benchmark validation seeds (3 seeds). Paper experiments use 5 seeds
# defined in paper/experiments/common.py instead.
SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# MNIST data loading (no torchvision required)
# ---------------------------------------------------------------------------

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, filepath: str) -> None:
    """Download a file from URL if not already present."""
    if not os.path.exists(filepath):
        print(f"  Downloading {os.path.basename(filepath)}...")
        urlretrieve(url, filepath)


def _download_mnist(data_dir: str) -> None:
    """Download MNIST dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        download_file(MNIST_URL + filename, filepath)


def _load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST images from gzipped IDX file."""
    with gzip.open(filepath, "rb") as f:
        _magic, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def _load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from gzipped IDX file."""
    with gzip.open(filepath, "rb") as f:
        _magic, _num = pystruct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def get_mnist_loaders(
    data_dir: str = "/tmp/mnist",
    batch_size: int = 512,
    normalize: bool = True,
    max_train: int = 0,
    max_test: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST train/test as PyTorch DataLoaders.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for training
        normalize: Whether to normalize with MNIST mean/std
        max_train: Maximum training samples (0=full dataset)
        max_test: Maximum test samples (0=full dataset)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    _download_mnist(data_dir)

    train_images = _load_mnist_images(os.path.join(data_dir, MNIST_FILES["train_images"]))
    train_labels = _load_mnist_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    test_images = _load_mnist_images(os.path.join(data_dir, MNIST_FILES["test_images"]))
    test_labels = _load_mnist_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))

    if normalize:
        mean, std = 0.1307, 0.3081
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std

    if max_train > 0:
        train_images = train_images[:max_train]
        train_labels = train_labels[:max_train]
    if max_test > 0:
        test_images = test_images[:max_test]
        test_labels = test_labels[:max_test]

    train_ds = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
    test_ds = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CIFAR-10 data loading (no torchvision required)
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"


def _download_cifar10(data_dir: str) -> None:
    """Download CIFAR-10 dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, CIFAR10_FILENAME)
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if not os.path.exists(extracted_dir):
        download_file(CIFAR10_URL, tar_path)
        print(f"  Extracting {CIFAR10_FILENAME}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)


def _load_cifar10_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 batch file."""
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    # Data is stored as (num_samples, 3072) where 3072 = 3*32*32
    # Reshape to (num_samples, 3, 32, 32)
    images = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b"labels"], dtype=np.int64)
    return images, labels


def get_cifar10_loaders(
    data_dir: str = "/tmp/cifar10",
    batch_size: int = 512,
    normalize: bool = True,
    max_train: int = 0,
    max_test: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 train/test as PyTorch DataLoaders.

    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for training
        normalize: Whether to normalize with CIFAR-10 mean/std
        max_train: Maximum training samples (0=full dataset)
        max_test: Maximum test samples (0=full dataset)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    _download_cifar10(data_dir)

    batch_dir = os.path.join(data_dir, "cifar-10-batches-py")

    # Load all training batches
    train_images_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch_path = os.path.join(batch_dir, f"data_batch_{i}")
        images, labels = _load_cifar10_batch(batch_path)
        train_images_list.append(images)
        train_labels_list.append(labels)

    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    # Load test batch
    test_path = os.path.join(batch_dir, "test_batch")
    test_images, test_labels = _load_cifar10_batch(test_path)

    if normalize:
        # CIFAR-10 normalization values (per channel)
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std

    if max_train > 0:
        train_images = train_images[:max_train]
        train_labels = train_labels[:max_train]
    if max_test > 0:
        test_images = test_images[:max_test]
        test_labels = test_labels[:max_test]

    train_ds = TensorDataset(
        torch.from_numpy(train_images.astype(np.float32)),
        torch.from_numpy(train_labels),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_images.astype(np.float32)),
        torch.from_numpy(test_labels),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# SST-2 data loading (HuggingFace with simple tokenization)
# ---------------------------------------------------------------------------

# Check for HuggingFace datasets at module load time
try:
    from datasets import load_dataset as _hf_load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def _simple_tokenize(
    texts: List[str],
    vocab: Dict[str, int],
    max_seq_len: int,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize texts using simple word-to-index mapping.

    Args:
        texts: List of text strings to tokenize.
        vocab: Vocabulary mapping word -> index.
        max_seq_len: Maximum sequence length (pad/truncate to this).
        pad_idx: Index for padding token.
        unk_idx: Index for unknown token.

    Returns:
        Tuple of (input_ids, attention_mask) as numpy arrays.
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        # Simple word tokenization: lowercase, split on whitespace/punctuation
        words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()

        # Convert to indices
        ids = [vocab.get(w, unk_idx) for w in words]

        # Truncate if needed
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        # Create attention mask (1 for real tokens)
        mask = [1] * len(ids)

        # Pad if needed
        padding_len = max_seq_len - len(ids)
        if padding_len > 0:
            ids = ids + [pad_idx] * padding_len
            mask = mask + [0] * padding_len

        input_ids.append(ids)
        attention_masks.append(mask)

    return np.array(input_ids, dtype=np.int64), np.array(attention_masks, dtype=np.int64)


def _build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    """Build vocabulary from texts.

    Args:
        texts: List of text strings.
        vocab_size: Maximum vocabulary size (including special tokens).

    Returns:
        Dictionary mapping word -> index.
    """
    # Count word frequencies
    word_counts: Dict[str, int] = defaultdict(int)
    for text in texts:
        words = text.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split()
        for w in words:
            word_counts[w] += 1

    # Sort by frequency and take top vocab_size - 2 (reserve 0=PAD, 1=UNK)
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in sorted_words[:vocab_size - 2]:
        vocab[word] = len(vocab)

    return vocab


def _generate_synthetic_sst2(
    num_samples: int,
    max_seq_len: int,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate synthetic SST-2-like data for testing.

    Args:
        num_samples: Number of samples to generate.
        max_seq_len: Maximum sequence length.
        vocab_size: Vocabulary size.

    Returns:
        Tuple of (input_ids, attention_mask, labels, vocab_size).
    """
    print(f"  Generating {num_samples} synthetic SST-2 samples...")

    # Random token sequences (excluding PAD=0)
    input_ids = np.random.randint(1, vocab_size, (num_samples, max_seq_len), dtype=np.int64)

    # Random sequence lengths between max_seq_len//2 and max_seq_len
    seq_lens = np.random.randint(max_seq_len // 2, max_seq_len + 1, num_samples)

    # Create attention masks and apply padding
    attention_mask = np.zeros((num_samples, max_seq_len), dtype=np.int64)
    for i, seq_len in enumerate(seq_lens):
        attention_mask[i, :seq_len] = 1
        input_ids[i, seq_len:] = 0  # PAD token

    # Random binary labels
    labels = np.random.randint(0, 2, num_samples, dtype=np.int64)

    return input_ids, attention_mask, labels, vocab_size


def get_sst2_loaders(
    max_train: int = 0,
    max_test: int = 0,
    max_seq_len: int = 64,
    batch_size: int = 32,
    vocab_size: int = 10000,
    data_dir: str = "/tmp/sst2",
) -> Tuple[DataLoader, DataLoader, int, str]:
    """Load SST-2 sentiment dataset as PyTorch DataLoaders.

    Uses HuggingFace datasets with simple word-to-index tokenization.
    Raises error if HuggingFace datasets is not installed (no synthetic fallback).

    Args:
        max_train: Maximum training samples (0=all available).
        max_test: Maximum test samples (0=all available).
        max_seq_len: Maximum sequence length for padding/truncation.
        batch_size: Batch size for DataLoaders.
        vocab_size: Maximum vocabulary size for tokenization.
        data_dir: Directory for caching HuggingFace datasets.

    Returns:
        Tuple of (train_loader, test_loader, actual_vocab_size, data_type).
        DataLoaders yield (input_ids, attention_mask, labels) tuples.
        data_type is "real" for HuggingFace data.

    Raises:
        ImportError: If HuggingFace datasets is not installed.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "HuggingFace datasets required for SST-2 benchmark. "
            "Install with: pip install datasets"
        )

    print("  Loading SST-2 from HuggingFace datasets...")
    dataset = _hf_load_dataset("glue", "sst2", cache_dir=data_dir)

    # Get train/validation splits (SST-2 test set has no labels)
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["validation"]["sentence"]
    test_labels = dataset["validation"]["label"]

    # Apply max samples
    if max_train > 0:
        train_texts = train_texts[:max_train]
        train_labels = train_labels[:max_train]
    if max_test > 0:
        test_texts = test_texts[:max_test]
        test_labels = test_labels[:max_test]

    print(f"  Building vocabulary from {len(train_texts)} training samples...")
    vocab = _build_vocab(train_texts, vocab_size)
    actual_vocab_size = len(vocab)
    print(f"  Vocabulary size: {actual_vocab_size}")

    # Tokenize
    print(f"  Tokenizing with max_seq_len={max_seq_len}...")
    train_ids, train_mask = _simple_tokenize(train_texts, vocab, max_seq_len)
    test_ids, test_mask = _simple_tokenize(test_texts, vocab, max_seq_len)
    train_labels_arr = np.array(train_labels, dtype=np.int64)
    test_labels_arr = np.array(test_labels, dtype=np.int64)

    # Convert to tensors and create DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(train_ids),
        torch.from_numpy(train_mask),
        torch.from_numpy(train_labels_arr),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_ids),
        torch.from_numpy(test_mask),
        torch.from_numpy(test_labels_arr),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    data_type = "real"
    print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    print(f"  Data type: {data_type} (HuggingFace SST-2)")
    return train_loader, test_loader, actual_vocab_size, data_type


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """Two-layer MLP for MNIST digit classification.

    Architecture: 784 -> hidden -> 10
    Uses ReLU activation, ~101K params with hidden=128.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR10Net(nn.Module):
    """Standard small CNN for CIFAR-10 classification (vmap-compatible).

    Architecture: 3×Conv2d + MaxPool → FC(128) → FC(10)
    No BatchNorm (incompatible with vmap). ~189K parameters.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CIFAR10NetGroupNorm(nn.Module):
    """CIFAR-10 CNN with GroupNorm (vmap-compatible normalization).

    Same architecture as CIFAR10Net but with GroupNorm after each Conv2d.
    GroupNorm normalizes over channel groups within each sample (no batch
    statistics), making it fully compatible with vmap.
    ~189K parameters + negligible GroupNorm affine params.
    """

    def __init__(self, num_groups: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(min(num_groups, 32), 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(min(num_groups, 64), 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(min(num_groups, 64), 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def create_model(
    dataset_name: str,
    hidden: int = 128,
    **kwargs,
) -> nn.Module:
    """Factory function to create models for different datasets.

    Args:
        dataset_name: One of 'mnist', 'cifar10'
        hidden: Hidden layer size
        **kwargs: Additional model-specific arguments

    Returns:
        nn.Module: The created model

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return MNISTNet(hidden=hidden)
    elif dataset_name == "cifar10":
        return CIFAR10Net(hidden=hidden if hidden else 256)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: mnist, cifar10")


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    """Compute classification accuracy on a DataLoader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with (inputs, labels) or (inputs, attention_mask, labels)

    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    for batch in dataloader:
        if len(batch) == 2:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
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
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single optimizer run result.

    Attributes:
        optimizer: Name of the optimizer (e.g., 'polystep', 'adam', 'cmaes')
        seed: Random seed used
        final_accuracy: Accuracy at end of training
        best_accuracy: Best accuracy achieved during training
        final_loss: Loss at end of training (None for optimizers that don't compute loss)
        wall_time_seconds: Total wall clock time in seconds
        peak_gpu_memory_mb: Peak GPU memory usage in MB
        total_steps: Total optimization steps/iterations
        function_evals: Total function evaluations (steps * popsize for ES)
        convergence_epoch: Epoch when target accuracy first reached (None if never)
        epoch_logs: List of per-epoch metrics dicts
    """
    optimizer: str
    seed: int
    final_accuracy: float
    best_accuracy: float
    final_loss: Optional[float]
    wall_time_seconds: float
    peak_gpu_memory_mb: float
    total_steps: int
    function_evals: int
    convergence_epoch: Optional[int]
    epoch_logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Environment info collection
# ---------------------------------------------------------------------------

def get_environment_info() -> Dict[str, Any]:
    """Collect environment info for reproducibility.

    Returns:
        Dict with torch_version, cuda_version, gpu_model, python_version, platform
    """
    info = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_model"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["cuda_version"] = None
        info["gpu_model"] = None
        info["gpu_count"] = 0

    return info


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def compute_summary_stats(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics grouped by optimizer.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Dict mapping optimizer names to stats dicts with mean_accuracy, std_accuracy, etc.
    """
    by_optimizer: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_optimizer[r.optimizer].append(r)

    summary = {}
    for optimizer, runs in by_optimizer.items():
        accs = [r.best_accuracy for r in runs]
        times = [r.wall_time_seconds for r in runs]
        mems = [r.peak_gpu_memory_mb for r in runs]
        evals = [r.function_evals for r in runs]

        summary[optimizer] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "mean_memory_mb": float(np.mean(mems)),
            "mean_function_evals": float(np.mean(evals)),
            "num_runs": len(runs),
        }

    return summary


def save_results_json(
    results: List[BenchmarkResult],
    output_dir: str,
    benchmark_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Save benchmark results to JSON file.

    Args:
        results: List of BenchmarkResult objects
        output_dir: Directory to save results
        benchmark_name: Name of the benchmark (e.g., 'mnist', 'cifar10')
        config: Optional benchmark configuration dict

    Returns:
        Path to the saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    filename = f"{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "benchmark": benchmark_name,
        "timestamp": timestamp,
        "environment": get_environment_info(),
        "config": config or {},
        "results": [r.to_dict() for r in results],
        "summary": compute_summary_stats(results),
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    return filepath


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format results as a console table using tabulate.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Formatted table string
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback to simple formatting if tabulate not installed
        return _format_results_simple(results)

    headers = ["Optimizer", "Mean Acc", "Std", "Mean Time", "Peak Mem (MB)", "Func Evals"]
    rows = []

    # Group by optimizer
    by_opt: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_opt[r.optimizer].append(r)

    for opt, runs in sorted(by_opt.items()):
        mean_acc = sum(r.best_accuracy for r in runs) / len(runs)
        std_acc = (sum((r.best_accuracy - mean_acc) ** 2 for r in runs) / len(runs)) ** 0.5
        mean_time = sum(r.wall_time_seconds for r in runs) / len(runs)
        mean_mem = sum(r.peak_gpu_memory_mb for r in runs) / len(runs)
        total_evals = sum(r.function_evals for r in runs) // len(runs)  # Average

        rows.append([
            opt,
            f"{mean_acc * 100:.1f}%",
            f"+/-{std_acc * 100:.1f}%",
            f"{mean_time:.1f}s",
            f"{mean_mem:.0f}",
            f"{total_evals:,}",
        ])

    return tabulate(rows, headers=headers, tablefmt="github")


def _format_results_simple(results: List[BenchmarkResult]) -> str:
    """Simple fallback table formatting without tabulate."""
    lines = []
    lines.append("| Optimizer | Mean Acc | Std | Mean Time | Peak Mem (MB) | Func Evals |")
    lines.append("|-----------|----------|-----|-----------|---------------|------------|")

    # Group by optimizer
    by_opt: Dict[str, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_opt[r.optimizer].append(r)

    for opt, runs in sorted(by_opt.items()):
        mean_acc = sum(r.best_accuracy for r in runs) / len(runs)
        std_acc = (sum((r.best_accuracy - mean_acc) ** 2 for r in runs) / len(runs)) ** 0.5
        mean_time = sum(r.wall_time_seconds for r in runs) / len(runs)
        mean_mem = sum(r.peak_gpu_memory_mb for r in runs) / len(runs)
        total_evals = sum(r.function_evals for r in runs) // len(runs)

        lines.append(
            f"| {opt:<9} | {mean_acc * 100:>6.1f}% | +/-{std_acc * 100:.1f}% | "
            f"{mean_time:>8.1f}s | {mean_mem:>13.0f} | {total_evals:>10,} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# snnTorch availability check
# ---------------------------------------------------------------------------

_HAS_SNNTORCH = False
try:
    import snntorch as snn
    _HAS_SNNTORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pure-PyTorch LIF neuron (fallback when snnTorch unavailable)
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with hard threshold spike.

    This is truly non-differentiable: the spike function has zero
    gradient almost everywhere, making backpropagation useless.
    polystep sidesteps this entirely with gradient-free optimization.

    Args:
        beta: Membrane decay factor (0 < beta < 1). Higher values = longer memory.
        threshold: Spike threshold for membrane potential.
    """

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One timestep of LIF dynamics.

        Args:
            x: Input current, shape (batch, features).
            mem: Membrane potential, shape (batch, features).

        Returns:
            (spike, new_mem): Binary spikes and updated membrane.
        """
        mem = self.beta * mem + x
        # This is THE non-differentiable operation: d(spike)/d(mem) = 0
        # almost everywhere. Backpropagation gives zero gradients through
        # this line. polystep never differentiates through it.
        spike = (mem >= self.threshold).float()
        mem = mem * (1.0 - spike)  # Reset after spike
        return spike, mem


# ---------------------------------------------------------------------------
# SNN models
# ---------------------------------------------------------------------------

class SpikingNet(nn.Module):
    """SNN with LIF neurons for classification.

    Uses snnTorch.Leaky if available, falls back to pure PyTorch LIF neurons.

    Architecture: Linear -> LIF -> Linear -> LIF
    Output: mean spike rate over num_steps timesteps.

    Why gradient-free for SNNs?
        SNNs use hard threshold spikes: d(spike)/d(membrane) = 0.
        Backpropagation gives zero gradients through spikes.
        Surrogate gradients are an approximation hack.
        polystep needs NO gradients -- only forward passes!

    Args:
        input_dim: Input dimension (flattened).
        hidden: Number of hidden neurons.
        output: Number of output classes.
        beta: Membrane decay factor (0.9-0.99 typical).
        num_steps: Number of timesteps for spike integration.
        use_snntorch: Use snnTorch if available (default: True).

    Example:
        >>> model = SpikingNet(input_dim=32*32*2, hidden=128, output=10, num_steps=25)
        >>> x = torch.randn(32, 25, 2, 32, 32)  # (batch, time, polarity, H, W)
        >>> out = model(x)  # (batch, output) spike rates
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 128,
        output: int = 10,
        beta: float = 0.95,
        num_steps: int = 25,
        use_snntorch: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output
        self.num_steps = num_steps
        self.beta = beta
        self.use_snntorch = use_snntorch and _HAS_SNNTORCH

        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output)

        if self.use_snntorch:
            self.lif1 = snn.Leaky(beta=beta)
            self.lif2 = snn.Leaky(beta=beta)
        else:
            self.lif1 = LIFNeuron(beta=beta)
            self.lif2 = LIFNeuron(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: process spike data over multiple timesteps.

        Supports two input formats:
        1. Temporal spike data: (num_steps, batch, ...) or (batch, num_steps, ...)
        2. Static input: (batch, ...) - presented at each timestep

        Args:
            x: Input tensor. If temporal, shape is (time, batch, ...) or (batch, time, ...)
               If static, shape is (batch, input_dim) or (batch, channels, H, W).

        Returns:
            Spike rates, shape (batch, output_dim). Values in [0, 1].
        """
        # Detect input format and normalize to (num_steps, batch, features)
        if x.dim() >= 3 and x.shape[0] == self.num_steps:
            # Temporal format: (num_steps, batch, ...)
            num_steps = x.shape[0]
            batch = x.shape[1]
            x_seq = x.reshape(num_steps, batch, -1)  # (T, B, features)
        elif x.dim() >= 3 and x.shape[1] == self.num_steps:
            # Alternate temporal: (batch, num_steps, ...)
            batch = x.shape[0]
            num_steps = x.shape[1]
            x_seq = x.reshape(batch, num_steps, -1).permute(1, 0, 2)  # (T, B, features)
        else:
            # Static input: repeat across timesteps
            batch = x.shape[0]
            num_steps = self.num_steps
            x_flat = x.reshape(batch, -1)
            x_seq = x_flat.unsqueeze(0).expand(num_steps, -1, -1)  # (T, B, features)

        # Initialize membrane potentials
        if self.use_snntorch:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
        else:
            mem1 = torch.zeros(batch, self.hidden, device=x.device, dtype=x.dtype)
            mem2 = torch.zeros(batch, self.output_dim, device=x.device, dtype=x.dtype)

        total_spikes = torch.zeros(batch, self.output_dim, device=x.device, dtype=x.dtype)

        for t in range(num_steps):
            cur1 = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            total_spikes = total_spikes + spk2

        return total_spikes / num_steps  # Spike rate in [0, 1]


# ---------------------------------------------------------------------------
# N-MNIST data loading
# ---------------------------------------------------------------------------

def get_nmnist_loaders(
    data_dir: str = "/tmp/nmnist",
    num_steps: int = 25,
    batch_size: int = 64,
    max_train: int = 0,
    max_test: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load N-MNIST neuromorphic dataset.

    N-MNIST is a neuromorphic version of MNIST created by displaying MNIST
    digits on a monitor and recording with a DVS (Dynamic Vision Sensor) camera.
    It contains ON and OFF polarity events encoding digit patterns.

    Uses snnTorch spikevision if available, falls back to synthetic spike data.

    Args:
        data_dir: Directory to store/load N-MNIST data.
        num_steps: Number of timesteps for spike binning.
        batch_size: Batch size for DataLoader.
        max_train: Maximum training samples (0=full dataset, ~60K).
        max_test: Maximum test samples (0=full dataset, ~10K).

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: (batch, num_steps, 2, 34, 34) for snnTorch
                     (batch, num_steps, 2, 32, 32) for synthetic
        Label format: (batch,) with values 0-9.

    Note:
        With snnTorch, the actual data shape depends on snnTorch version.
        The SpikingNet model handles both formats automatically.

    Example:
        >>> train_loader, test_loader = get_nmnist_loaders(max_train=1000)
        >>> for data, labels in train_loader:
        ...     # data: (batch, time, polarity, H, W)
        ...     pass
    """
    if _HAS_SNNTORCH:
        try:
            from snntorch.spikevision import spikedata
            return _load_nmnist_snntorch(data_dir, num_steps, batch_size, max_train, max_test)
        except Exception as e:
            print(f"  snnTorch N-MNIST loading failed: {e}")
            print("  Falling back to synthetic spike data...")

    return _generate_synthetic_nmnist(num_steps, batch_size, max_train, max_test)


def _load_nmnist_snntorch(
    data_dir: str,
    num_steps: int,
    batch_size: int,
    max_train: int,
    max_test: int,
) -> Tuple[DataLoader, DataLoader]:
    """Load N-MNIST using snnTorch spikevision."""
    from snntorch.spikevision import spikedata

    print(f"  Loading N-MNIST from snnTorch (data_dir={data_dir})...")

    # snnTorch NMNIST returns data in format (time, polarity, H, W)
    train_ds = spikedata.NMNIST(
        root=data_dir,
        train=True,
        num_steps=num_steps,
        dt=1000,  # 1ms time bins
    )
    test_ds = spikedata.NMNIST(
        root=data_dir,
        train=False,
        num_steps=num_steps,
        dt=1000,
    )

    # Apply sample limits using Subset
    if max_train > 0 and len(train_ds) > max_train:
        train_ds = torch.utils.data.Subset(train_ds, range(max_train))
    if max_test > 0 and len(test_ds) > max_test:
        test_ds = torch.utils.data.Subset(test_ds, range(max_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def _generate_synthetic_nmnist(
    num_steps: int,
    batch_size: int,
    max_train: int,
    max_test: int,
) -> Tuple[DataLoader, DataLoader]:
    """Generate synthetic N-MNIST-like spike data.

    Creates synthetic spike patterns for digits 0-9. Each digit has a
    distinct activation pattern simulating rate-coded neural spikes.
    """
    print("  Generating synthetic N-MNIST data...")

    num_train = max_train if max_train > 0 else 5000
    num_test = max_test if max_test > 0 else 1000
    height, width = 32, 32
    num_classes = 10

    def generate_digit_spikes(num_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = torch.Generator().manual_seed(seed)

        # Create templates for each digit - different spatial patterns
        templates = torch.zeros(num_classes, 2, height, width)
        for c in range(num_classes):
            # Each digit activates different regions
            y_start = (c // 2) * (height // 5)
            x_start = (c % 5) * (width // 5)
            y_end = min(y_start + height // 4, height)
            x_end = min(x_start + width // 4, width)
            templates[c, 0, y_start:y_end, x_start:x_end] = 0.8  # ON polarity
            templates[c, 1, y_start:y_end, x_start:x_end] = 0.3  # OFF polarity

        labels = torch.randint(0, num_classes, (num_samples,), generator=rng)

        # Generate spike data: (samples, time, polarity, H, W)
        data = torch.zeros(num_samples, num_steps, 2, height, width)
        for i, label in enumerate(labels):
            template = templates[label.item()]
            # Spike with probability proportional to template
            for t in range(num_steps):
                noise = torch.rand(2, height, width, generator=rng)
                spikes = (noise < template).float()
                data[i, t] = spikes

        return data, labels

    train_data, train_labels = generate_digit_spikes(num_train, seed=42)
    test_data, test_labels = generate_digit_spikes(num_test, seed=123)

    train_ds = TensorDataset(train_data, train_labels)
    test_ds = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Generated {num_train} train, {num_test} test samples (synthetic)")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# DVS-Gesture data loading
# ---------------------------------------------------------------------------

def get_dvs_gesture_loaders(
    data_dir: str = "/tmp/dvs_gesture",
    num_steps: int = 50,
    batch_size: int = 16,
    max_train: int = 0,
    max_test: int = 0,
    synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Load DVS-Gesture neuromorphic dataset.

    DVS-Gesture contains 11 hand gestures recorded with a DVS camera.
    The dataset is ~1.5GB, so fallback to synthetic is supported.

    Uses snnTorch spikevision if available, falls back to synthetic spike data.

    Args:
        data_dir: Directory to store/load DVS-Gesture data.
        num_steps: Number of timesteps for spike binning.
        batch_size: Batch size for DataLoader.
        max_train: Maximum training samples (0=full dataset, ~1077).
        max_test: Maximum test samples (0=full dataset, ~264).
        synthetic: Force synthetic data (skip download attempt).

    Returns:
        Tuple of (train_loader, test_loader).
        Data format: (batch, num_steps, 2, 128, 128) or downsampled.
        Label format: (batch,) with values 0-10 (11 gesture classes).

    Gesture classes:
        0: hand_clapping, 1: right_hand_wave, 2: left_hand_wave,
        3: right_arm_cw, 4: right_arm_ccw, 5: left_arm_cw,
        6: left_arm_ccw, 7: arm_roll, 8: air_drums,
        9: air_guitar, 10: other_gestures

    Example:
        >>> train_loader, test_loader = get_dvs_gesture_loaders(synthetic=True)
        >>> for data, labels in train_loader:
        ...     # data: (batch, time, polarity, H, W)
        ...     pass
    """
    if synthetic:
        return _generate_synthetic_dvs_gesture(num_steps, batch_size, max_train, max_test)

    if _HAS_SNNTORCH:
        try:
            from snntorch.spikevision import spikedata
            return _load_dvs_gesture_snntorch(data_dir, num_steps, batch_size, max_train, max_test)
        except Exception as e:
            print(f"  snnTorch DVS-Gesture loading failed: {e}")
            print("  Falling back to synthetic spike data...")

    return _generate_synthetic_dvs_gesture(num_steps, batch_size, max_train, max_test)


def _load_dvs_gesture_snntorch(
    data_dir: str,
    num_steps: int,
    batch_size: int,
    max_train: int,
    max_test: int,
) -> Tuple[DataLoader, DataLoader]:
    """Load DVS-Gesture using snnTorch spikevision."""
    from snntorch.spikevision import spikedata

    print(f"  Loading DVS-Gesture from snnTorch (data_dir={data_dir})...")
    print("  Note: First download is ~1.5GB and may take several minutes.")

    train_ds = spikedata.DVSGesture(
        root=data_dir,
        train=True,
        num_steps=num_steps,
        dt=1000,
    )
    test_ds = spikedata.DVSGesture(
        root=data_dir,
        train=False,
        num_steps=num_steps,
        dt=1000,
    )

    if max_train > 0 and len(train_ds) > max_train:
        train_ds = torch.utils.data.Subset(train_ds, range(max_train))
    if max_test > 0 and len(test_ds) > max_test:
        test_ds = torch.utils.data.Subset(test_ds, range(max_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def _generate_synthetic_dvs_gesture(
    num_steps: int,
    batch_size: int,
    max_train: int,
    max_test: int,
) -> Tuple[DataLoader, DataLoader]:
    """Generate synthetic DVS-Gesture-like spike data.

    Creates synthetic gesture patterns. Each gesture has a distinct
    spatio-temporal activation pattern.
    """
    print("  Generating synthetic DVS-Gesture data...")

    num_train = max_train if max_train > 0 else 500
    num_test = max_test if max_test > 0 else 100
    height, width = 32, 32  # Downsampled from 128x128 for efficiency
    num_classes = 11

    def generate_gesture_spikes(num_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = torch.Generator().manual_seed(seed)

        labels = torch.randint(0, num_classes, (num_samples,), generator=rng)
        data = torch.zeros(num_samples, num_steps, 2, height, width)

        for i, label in enumerate(labels):
            c = label.item()
            # Each gesture has different temporal pattern
            for t in range(num_steps):
                # Spatial pattern varies by gesture class
                phase = (t / num_steps + c / num_classes) * 2 * 3.14159
                center_y = int((height // 2) + (height // 4) * np.sin(phase * (c % 3 + 1)))
                center_x = int((width // 2) + (width // 4) * np.cos(phase * (c % 4 + 1)))

                # Create activation region around center
                y_start = max(0, center_y - height // 8)
                y_end = min(height, center_y + height // 8)
                x_start = max(0, center_x - width // 8)
                x_end = min(width, center_x + width // 8)

                # Spike probability in active region
                prob = 0.3 + 0.3 * np.sin(phase)
                noise = torch.rand(y_end - y_start, x_end - x_start, generator=rng)
                spikes = (noise < prob).float()

                data[i, t, c % 2, y_start:y_end, x_start:x_end] = spikes

        return data, labels

    train_data, train_labels = generate_gesture_spikes(num_train, seed=42)
    test_data, test_labels = generate_gesture_spikes(num_test, seed=123)

    train_ds = TensorDataset(train_data, train_labels)
    test_ds = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Generated {num_train} train, {num_test} test samples (synthetic gestures)")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# SNN evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_snn_accuracy(model: nn.Module, dataloader: DataLoader, scale_output: float = 10.0) -> float:
    """Compute classification accuracy for SNN models.

    Args:
        model: SNN model that outputs spike rates.
        dataloader: DataLoader with (spike_data, labels).
        scale_output: Scale factor for spike rates before argmax.
            SNNs output rates in [0,1], scaling helps with CrossEntropyLoss.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    for batch in dataloader:
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)

        # SNN outputs spike rates
        outputs = model(data) * scale_output
        preds = outputs.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    model.train()
    return correct / total if total > 0 else 0.0
