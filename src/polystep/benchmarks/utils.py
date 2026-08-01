"""Benchmark utilities shared by the experiment runners.

MNIST and CIFAR-10 loaders that read the raw archives instead of pulling in
torchvision, the MLP and SNN architectures the paper uses, accuracy evaluation, and
environment capture for the result JSON.

The SNN path uses snnTorch when installed (``pip install snntorch``) and otherwise
falls back to pure-PyTorch LIF neurons, which are non-differentiable either way.
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
import platform
import struct as pystruct
import tempfile
import tarfile
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _default_data_dir(name: str) -> str:
    """Default download location for a dataset, under the platform temp dir.

    Hardcoding "/tmp/..." makes these defaults unusable on Windows, which the package
    does not otherwise exclude.
    """
    return os.path.join(tempfile.gettempdir(), name)


# Benchmark validation seeds. The published experiments use 5 seeds, set in
# experiments/runners/common.py.


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
    data_dir: Optional[str] = None,
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
    data_dir = data_dir or _default_data_dir("mnist")
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
    data_dir: Optional[str] = None,
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
    data_dir = data_dir or _default_data_dir("cifar10")
    _download_cifar10(data_dir)

    batch_dir = os.path.join(data_dir, "cifar-10-batches-py")

    train_images_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch_path = os.path.join(batch_dir, f"data_batch_{i}")
        images, labels = _load_cifar10_batch(batch_path)
        train_images_list.append(images)
        train_labels_list.append(labels)

    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

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


class MNISTNet(nn.Sequential):
    """Two-layer MLP for MNIST: 784 -> hidden -> 10, ~101K params at hidden=128.

    An ``nn.Sequential`` subclass, not a plain ``nn.Module``: every batched
    evaluator checks ``type(model).forward is nn.Sequential.forward`` before it will
    build a plan, so an identical hand-written ``forward`` silently opts the model out
    of the bmm and subspace-delta paths. The ``OrderedDict`` keeps the ``fc1``/``fc2``
    state_dict keys.
    """

    def __init__(self, hidden: int = 128):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(784, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, 10)),
                ]
            )
        )


class CIFAR10Net(nn.Module):
    """Standard small CNN for CIFAR-10 classification (vmap-compatible).

    Architecture: 3xConv2d + MaxPool -> FC(128) -> FC(10)
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


_HAS_SNNTORCH = False
try:
    import snntorch as snn

    _HAS_SNNTORCH = True
except ImportError:
    pass


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


class SpikingNet(nn.Module):
    """SNN with LIF neurons for classification.

    Uses snnTorch.Leaky if available, falls back to pure PyTorch LIF neurons.

    Architecture: Linear -> LIF -> Linear -> LIF
    Output: mean spike rate over num_steps timesteps.

    Why gradient-free for SNNs?
        SNNs use hard threshold spikes: d(spike)/d(membrane) = 0.
        Backpropagation gives zero gradients through spikes.
        Surrogate gradients are an approximation hack.
        polystep needs NO gradients: only forward passes!

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
        # Static input is constant in time, so fc1 runs once and broadcasts, which is
        # bit-identical to the T calls it replaces. Temporal input keeps its per-timestep
        # call: one (T*B, F) GEMM blocks differently from T (B, F) ones, and the LIF
        # threshold turns that ULP into a whole spike.
        # Feature count first: on shape alone a static batch of num_steps images looks
        # like a T x 1 x F sequence, and the size test read it as temporal.
        _static_features = x.dim() >= 2 and math.prod(x.shape[1:]) == self.input_dim
        if _static_features:
            batch = x.shape[0]
            num_steps = self.num_steps
            x_seq = None
            cur_seq = self.fc1(x.reshape(batch, -1)).unsqueeze(0).expand(num_steps, -1, -1)
        elif x.dim() >= 3 and x.shape[0] == self.num_steps:
            # Temporal format: (num_steps, batch, ...)
            num_steps = x.shape[0]
            batch = x.shape[1]
            x_seq = x.reshape(num_steps, batch, -1)
            cur_seq = None
        elif x.dim() >= 3 and x.shape[1] == self.num_steps:
            # Alternate temporal: (batch, num_steps, ...)
            batch = x.shape[0]
            num_steps = x.shape[1]
            x_seq = x.reshape(batch, num_steps, -1).permute(1, 0, 2)
            cur_seq = None
        else:
            batch = x.shape[0]
            num_steps = self.num_steps
            cur_seq = self.fc1(x.reshape(batch, -1)).unsqueeze(0).expand(num_steps, -1, -1)

        if self.use_snntorch:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
        else:
            mem1 = torch.zeros(batch, self.hidden, device=x.device, dtype=x.dtype)
            mem2 = torch.zeros(batch, self.output_dim, device=x.device, dtype=x.dtype)

        total_spikes = torch.zeros(batch, self.output_dim, device=x.device, dtype=x.dtype)

        for t in range(num_steps):
            spk1, mem1 = self.lif1(cur_seq[t] if cur_seq is not None else self.fc1(x_seq[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total_spikes = total_spikes + spk2

        return total_spikes / num_steps  # Spike rate in [0, 1]
