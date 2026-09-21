"""Shared benchmark utilities: MNIST loaders, MLP/SNN models, accuracy evaluation, environment capture."""

from __future__ import annotations

import gzip
import math
import os
import platform
import random
import struct as pystruct
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _seed_worker(worker_id: int) -> None:
    """Re-seed numpy/random inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_loader_kwargs(seed: Optional[int] = None) -> Dict[str, Any]:
    """DataLoader kwargs that pin shuffle order and worker RNG to a seed."""
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() if seed is None else seed)
    return {"generator": generator, "worker_init_fn": _seed_worker}


def _default_data_dir(name: str) -> str:
    """Default download location under the platform temp dir."""
    return os.path.join(tempfile.gettempdir(), name)


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


def _load_mnist_images(filepath: str, limit: int = 0) -> np.ndarray:
    """Load MNIST images from gzipped IDX file."""
    with gzip.open(filepath, "rb") as f:
        _magic, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        num = min(num, limit) if limit > 0 else num
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
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
    """Load MNIST train/test as PyTorch DataLoaders."""
    data_dir = data_dir or _default_data_dir("mnist")
    _download_mnist(data_dir)

    train_images = _load_mnist_images(os.path.join(data_dir, MNIST_FILES["train_images"]), max_train)
    train_labels = _load_mnist_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    test_images = _load_mnist_images(os.path.join(data_dir, MNIST_FILES["test_images"]), max_test)
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, **seeded_loader_kwargs())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, **seeded_loader_kwargs())
    return train_loader, test_loader


class MNISTNet(nn.Sequential):
    """Two-layer MLP for MNIST: 784 -> hidden -> 10. An nn.Sequential subclass, as the batched evaluators require."""

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


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    """Compute classification accuracy on a DataLoader."""
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
    """Single optimizer run result."""

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


def get_environment_info() -> Dict[str, Any]:
    """Collect environment info for reproducibility."""
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
    """Leaky integrate-and-fire neuron with a hard threshold spike; non-differentiable."""

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One timestep of LIF dynamics."""
        mem = self.beta * mem + x
        # Non-differentiable: d(spike)/d(mem) = 0 almost everywhere.
        spike = (mem >= self.threshold).float()
        mem = mem * (1.0 - spike)
        return spike, mem


class SpikingNet(nn.Module):
    """SNN with LIF neurons; uses snnTorch.Leaky when available, else pure PyTorch LIF."""

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
        """Forward pass over timesteps; supports temporal and static inputs."""
        # Static input broadcasts one fc1 call (bit-identical to per-timestep); temporal input keeps per-timestep calls. Check feature count first: a static batch of num_steps images looks temporal on shape alone.
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
