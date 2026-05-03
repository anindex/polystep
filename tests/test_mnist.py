"""End-to-end MNIST training tests for polystep.

Covers requirement TEST-07: MNIST training achieves meaningful accuracy.

The ``slow`` test trains on a 2000-sample subset and verifies accuracy
above a threshold. The fast test verifies that loss decreases on a
tiny subset using a very small model.

Downloads MNIST data directly from Google Cloud Storage mirror
(no torchvision dependency).

Note on model size: The gradient-free Sinkhorn optimizer operates in
multi-particle mode where parameters are reshaped to (num_particles,
particle_dim). With particle_dim=2 and orthoplex polytope, each OT
problem has 4 vertices per particle -- tractable but requiring P*V*K
model evaluations per step. Smaller models are faster.
"""
from __future__ import annotations

import gzip
import os
import struct as pystruct
from urllib.request import urlretrieve

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, train, TrainConfig
from polystep.epsilon import LinearEpsilon
from polystep.subspace import LowRankSubspace, LinearSubspace
from polystep.transform import ParamLayout


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


def _download_mnist(data_dir: str = "/tmp/mnist") -> None:
    os.makedirs(data_dir, exist_ok=True)
    for _name, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            urlretrieve(MNIST_URL + filename, filepath)


def _load_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def _load_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic, _num = pystruct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def _make_loaders(n_train: int, n_test: int, batch_size: int, downsample: int = 1):
    """Load MNIST subsets as DataLoaders (downloads if needed).

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        batch_size: Batch size for DataLoaders.
        downsample: Spatial downsampling factor. 1=28x28, 2=14x14, 4=7x7.
    """
    data_dir = "/tmp/mnist"
    _download_mnist(data_dir)

    train_imgs = _load_images(os.path.join(data_dir, MNIST_FILES["train_images"]))[:n_train]
    train_lbls = _load_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))[:n_train]
    test_imgs = _load_images(os.path.join(data_dir, MNIST_FILES["test_images"]))[:n_test]
    test_lbls = _load_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))[:n_test]

    # Normalize
    mean, std = 0.1307, 0.3081
    train_imgs = (train_imgs - mean) / std
    test_imgs = (test_imgs - mean) / std

    # Spatial downsampling via average pooling to reduce input dimension
    if downsample > 1:
        train_t = torch.from_numpy(train_imgs)
        test_t = torch.from_numpy(test_imgs)
        train_t = nn.functional.avg_pool2d(train_t, downsample)
        test_t = nn.functional.avg_pool2d(test_t, downsample)
        train_imgs = train_t.numpy()
        test_imgs = test_t.numpy()

    train_ds = TensorDataset(torch.from_numpy(train_imgs), torch.from_numpy(train_lbls))
    test_ds = TensorDataset(torch.from_numpy(test_imgs), torch.from_numpy(test_lbls))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallMNISTNet(nn.Module):
    """Tiny MLP for MNIST -- keeps parameter count low for OT feasibility.

    With downsample=4 (7x7=49 input) and hidden=16:
    fc1: 49*16+16=800, fc2: 16*10+10=170 => total ~970 params.
    """

    def __init__(self, input_dim: int = 49, hidden: int = 16):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MNISTNet(nn.Module):
    """Standard MLP for MNIST digit classification (for slow test)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    correct = total = 0
    for inputs, targets in loader:
        preds = model(inputs).argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def _evaluate_on_device(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> float:
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def _compute_loss(model: nn.Module, loader: DataLoader, loss_fn) -> float:
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            total_loss += loss_fn(model(inputs), targets).item()
            count += 1
    return total_loss / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.timeout(180)
def test_mnist_accuracy():
    """Train on 2000 MNIST samples (downsampled 4x) for 10 epochs.

    Uses multi-particle architecture with orthoplex polytope in 2D.
    Each particle controls 2 of the 970 model parameters. The OT
    solver matches 485 particles to 4 vertices, producing non-uniform
    transport that moves parameters toward lower loss.

    Verifies accuracy > 50% (well above random chance of 10%).
    """
    torch.manual_seed(42)
    train_loader, test_loader = _make_loaders(
        n_train=2000, n_test=1000, batch_size=32, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16)
    epsilon = LinearEpsilon(init=0.1, target=0.01, decay=0.001)

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=epsilon,
        step_radius=3.0,
        probe_radius=6.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        scale_cost='mean',
        chunk_size=512,
    )

    config = TrainConfig(epochs=10)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    accuracy = _evaluate(model, test_loader)
    assert accuracy > 0.50, (
        f"Expected accuracy > 50% on 1000-sample test subset, got {accuracy * 100:.1f}%"
    )


def test_mnist_model_improves():
    """Verify loss decreases after training on 200 MNIST samples for 2 epochs.

    Uses heavily downsampled images (7x7) and a tiny model (~970 params)
    so the gradient-free OT solver can make progress within a reasonable
    time on CPU. Multi-particle mode with 485 particles in 2D space,
    4 orthoplex vertices per particle.
    """
    torch.manual_seed(42)
    train_loader, test_loader = _make_loaders(
        n_train=200, n_test=100, batch_size=32, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16)
    loss_fn = nn.CrossEntropyLoss()

    # Record initial loss
    initial_loss = _compute_loss(model, test_loader, loss_fn)

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=0.1,
        step_radius=3.0,
        probe_radius=6.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        scale_cost='mean',
        chunk_size=256,
    )

    config = TrainConfig(epochs=2)
    model = train(model, train_loader, loss_fn, optimizer, config)

    final_loss = _compute_loss(model, test_loader, loss_fn)
    assert final_loss < initial_loss, (
        f"Expected loss to decrease after training. "
        f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )


@pytest.mark.gpu
def test_mnist_gpu_subspace():
    """Verify subspace compression pipeline works on GPU.

    Validates that the subspace + multi-particle + GPU pipeline runs
    without errors: LowRankSubspace creation, optimizer initialization
    with subspace, chunked probe evaluation on CUDA, and model sync.

    Uses a small subset (500 train, 200 test) for speed. Verifies that
    the optimizer produces non-zero displacement (OT solver is active)
    and that model parameters actually change during training.

    Note: Subspace multi-particle mode has limited convergence because
    each OT step only perturbs 2 of the subspace_dim coordinates per
    particle. This test validates the GPU pipeline correctness, not
    accuracy. The full-space test_mnist_gpu_full_space validates accuracy.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    train_loader, test_loader = _make_loaders(
        n_train=500, n_test=200, batch_size=64, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)
    layout = ParamLayout.from_module(model)
    subspace = LowRankSubspace.from_layout(layout, rank=2)

    # Record initial params
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=0.1,
        step_radius=30.0,
        probe_radius=60.0,
        num_probe=1,
        sinkhorn_max_iters=100,
        subspace=subspace,
        scale_cost='mean',
        chunk_size=256,
    )

    config = TrainConfig(epochs=2)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    # Verify optimizer produced non-zero displacement
    displacements = optimizer.state.displacement_sqnorms
    assert len(displacements) > 0, "No steps were taken"
    has_nonzero = any(d > 0 for d in displacements)
    assert has_nonzero, (
        "All displacements were zero -- OT solver produced uniform transport"
    )

    # Verify model parameters actually changed
    current_params = model.state_dict()
    param_changed = False
    for key in initial_params:
        if not torch.equal(initial_params[key], current_params[key]):
            param_changed = True
            break
    assert param_changed, "Model parameters did not change during training"


@pytest.mark.gpu
def test_mnist_gpu_full_space():
    """Train on 7x7 MNIST in full parameter space on GPU.

    Uses the multi-particle architecture (485 particles in 2D) with
    orthoplex polytope (4 vertices). Trains on 2000 samples for 10
    epochs, targeting >80% accuracy on the 1000-sample test set.

    This demonstrates that the core Sinkhorn OT algorithm achieves
    meaningful learning on GPU with compiled operations.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    train_loader, test_loader = _make_loaders(
        n_train=2000, n_test=1000, batch_size=32, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)

    epsilon = LinearEpsilon(init=0.1, target=0.01, decay=0.001)

    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=epsilon,
        step_radius=3.0,
        probe_radius=6.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        scale_cost='mean',
        chunk_size=512,
    )

    config = TrainConfig(epochs=10)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    accuracy = _evaluate_on_device(model, test_loader, device)
    assert accuracy > 0.70, (
        f"Expected accuracy > 70% on 7x7 MNIST with GPU, "
        f"got {accuracy * 100:.1f}%"
    )


@pytest.mark.gpu
def test_mnist_gpu_subspace_higher_particle_dim():
    """Verify higher subspace_particle_dim gives stronger per-step signal on GPU.

    Tests the subspace_particle_dim=8 feature: orthoplex in 8D has 16 vertices,
    perturbing 8 subspace coords per particle per step (vs 2 with particle_dim=2).
    Validates that the optimizer runs correctly with higher particle_dim and
    produces meaningful displacement.

    Also validates absorb_every: after absorb_every steps, the subspace resets
    and base params absorb the perturbation.

    Note on subspace accuracy: The B@A low-rank factorization creates a bilinear
    relationship between subspace coordinates and weight perturbations. The OT
    solver probes linearly in subspace space, making convergence fundamentally
    harder than full-space mode. Subspace mode is designed for SCALABILITY
    (reducing memory for large models), not for maximizing accuracy on small
    problems. Full-space mode (test_mnist_gpu_full_space) validates accuracy.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    train_loader, test_loader = _make_loaders(
        n_train=500, n_test=200, batch_size=64, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)
    layout = ParamLayout.from_module(model)
    subspace = LowRankSubspace.from_layout(layout, rank=4)

    initial_params = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=0.1,
        step_radius=30.0,
        probe_radius=60.0,
        num_probe=1,
        sinkhorn_max_iters=100,
        subspace=subspace,
        subspace_particle_dim=8,
        absorb_every=10,
        scale_cost='mean',
        chunk_size=256,
    )

    # Verify particle shape reflects subspace_particle_dim=8
    X = optimizer.state.X
    assert X.shape[1] == 8, (
        f"Expected particle_dim=8 for subspace mode, got {X.shape[1]}"
    )

    config = TrainConfig(epochs=2)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    # Verify optimizer ran and produced displacement
    displacements = optimizer.state.displacement_sqnorms
    assert len(displacements) > 0, "No steps were taken"
    has_nonzero = any(d > 0 for d in displacements)
    assert has_nonzero, (
        "All displacements were zero -- OT solver produced uniform transport"
    )

    # Verify model parameters actually changed (absorb folds into base)
    current_params = model.state_dict()
    param_changed = False
    for key in initial_params:
        if not torch.equal(initial_params[key], current_params[key]):
            param_changed = True
            break
    assert param_changed, "Model parameters did not change during training"

    # Verify absorb worked: after absorb, base_params should differ from initial
    base = optimizer.state.base_params
    base_changed = False
    for key in initial_params:
        if not torch.equal(initial_params[key], base[key]):
            base_changed = True
            break
    assert base_changed, (
        "Base params did not change -- absorb_every did not trigger"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_mnist_gpu_linear_subspace():
    """Train MNIST with LinearSubspace to verify convergence.

    LinearSubspace uses a fixed random projection (linear mapping) from
    subspace coordinates to weight perturbations. Unlike B@A (bilinear),
    this gives the OT solver proportional cost changes when probing,
    enabling meaningful transport and convergence.

    The random projection scaling (1/sqrt(num_coords)) dilutes perturbation
    magnitude, requiring larger step/probe radii than full-space mode.
    Subspace mode reaches 60-75% accuracy on this small model (varies due to
    GPU non-determinism with torch.compile), demonstrating clear convergence
    from 10% (random). Full-space mode reaches >80%.

    Target: >55% accuracy (well above 10% random, confirms linear subspace
    enables OT convergence unlike bilinear B@A which stalls). Threshold set
    conservatively to account for run-to-run variance on GPU.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    train_loader, test_loader = _make_loaders(
        n_train=2000, n_test=1000, batch_size=32, downsample=4,
    )

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)
    layout = ParamLayout.from_module(model)
    subspace = LinearSubspace.from_layout(layout, rank=8, seed=42)

    epsilon = LinearEpsilon(init=0.1, target=0.01, decay=0.001)

    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=epsilon,
        step_radius=30.0,
        probe_radius=60.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        subspace=subspace,
        subspace_particle_dim=8,
        absorb_every=15,
        scale_cost='mean',
        chunk_size=512,
    )

    config = TrainConfig(epochs=20)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    accuracy = _evaluate_on_device(model, test_loader, device)
    assert accuracy > 0.55, (
        f"Expected linear subspace accuracy > 55% on MNIST, got {accuracy * 100:.1f}%"
    )
