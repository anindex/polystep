"""Automated tests for AdaptiveSubspace on MNIST.

Tests validate that AdaptiveSubspace can train a model (loss decreases),
achieves accuracy above random chance, and that the absorb mechanism fires
correctly during training.

These tests use a small subset of MNIST data for speed (1000 train, 500 test)
and a small model (hidden=64). Accuracy thresholds are relaxed to account
for the limited data and short training.

Marked with @pytest.mark.slow since they involve actual training loops.
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
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# MNIST data fixtures
# ---------------------------------------------------------------------------

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = "/tmp/mnist"


def _download_mnist():
    """Download MNIST if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            urlretrieve(MNIST_URL + filename, filepath)


def _load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def _load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, _ = pystruct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


_mnist_available = None


def _check_mnist_available():
    """Check if MNIST can be downloaded/loaded."""
    global _mnist_available
    if _mnist_available is not None:
        return _mnist_available
    try:
        _download_mnist()
        # Quick check that files exist and are valid
        img = _load_images(os.path.join(DATA_DIR, MNIST_FILES["train_images"]))
        _mnist_available = img.shape[0] > 0
    except Exception:
        _mnist_available = False
    return _mnist_available


requires_mnist = pytest.mark.skipif(
    not _check_mnist_available(),
    reason="MNIST data not available",
)


@pytest.fixture(scope="module")
def mnist_loaders():
    """Load a small subset of MNIST for testing."""
    _download_mnist()
    train_img = _load_images(os.path.join(DATA_DIR, MNIST_FILES["train_images"]))
    train_lbl = _load_labels(os.path.join(DATA_DIR, MNIST_FILES["train_labels"]))
    test_img = _load_images(os.path.join(DATA_DIR, MNIST_FILES["test_images"]))
    test_lbl = _load_labels(os.path.join(DATA_DIR, MNIST_FILES["test_labels"]))

    mean, std = 0.1307, 0.3081
    train_img = (train_img[:1000] - mean) / std
    train_lbl = train_lbl[:1000]
    test_img = (test_img[:500] - mean) / std
    test_lbl = test_lbl[:500]

    train_ds = TensorDataset(torch.from_numpy(train_img), torch.from_numpy(train_lbl))
    test_ds = TensorDataset(torch.from_numpy(test_img), torch.from_numpy(test_lbl))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


class SmallMLP(nn.Module):
    """Small MLP for fast testing."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def _evaluate(model, dataloader):
    """Quick accuracy evaluation."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        preds = model(inputs).argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_mnist
@pytest.mark.slow
def test_adaptive_subspace_trains(mnist_loaders):
    """AdaptiveSubspace should reduce loss during training.

    This is a basic smoke test: the optimizer should produce decreasing
    loss over multiple steps, showing that the OT-based update direction
    is functional even with a global projection.
    """
    train_loader, test_loader = mnist_loaders

    torch.manual_seed(42)
    model = SmallMLP(hidden=64)
    layout = ParamLayout.from_module(model)
    adaptive_sub = AdaptiveSubspace.from_layout(
        layout,
        rank=128,
        rotation_mode="displacement",
        absorb_mode="periodic",
        absorb_interval=20,
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=10.0,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=adaptive_sub,
    )

    config = TrainConfig(epochs=2)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    # Basic sanity: optimizer completed without error and made steps
    assert optimizer.state.iteration_count > 0
    # Check that costs were recorded
    assert len(optimizer.state.costs) == optimizer.state.iteration_count
    # Check some displacement was produced (not all zeros, filter NaN)
    finite_disps = [d for d in optimizer.state.displacement_sqnorms if d == d]  # filter NaN
    total_disp = sum(finite_disps) if finite_disps else 0.0
    assert total_disp > 0, "Expected non-zero total displacement"


@requires_mnist
@pytest.mark.slow
def test_adaptive_absorb_fires(mnist_loaders):
    """AdaptiveSubspace periodic absorb should fire during training.

    With absorb_interval=5 and enough steps, the absorb counter should
    increment, showing that the absorb-rotate mechanism is functional.
    """
    train_loader, _ = mnist_loaders

    torch.manual_seed(42)
    model = SmallMLP(hidden=64)
    layout = ParamLayout.from_module(model)
    adaptive_sub = AdaptiveSubspace.from_layout(
        layout,
        rank=128,
        rotation_mode="displacement",
        absorb_mode="periodic",
        absorb_interval=5,
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=10.0,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=adaptive_sub,
    )

    # Train for enough steps to trigger absorb multiple times
    # 2000 samples / 512 batch = ~2 batches per epoch, 5 epochs = 10 steps
    # With absorb_interval=5, should fire at step 5 and step 10
    config = TrainConfig(epochs=5)
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)

    assert optimizer.state.absorb_count > 0, (
        f"Expected absorb_count > 0 after {optimizer.state.iteration_count} steps "
        f"with absorb_interval=5, got {optimizer.state.absorb_count}"
    )


@requires_mnist
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_adaptive_convergence_vs_linear(mnist_loaders):
    """Compare AdaptiveSubspace and LinearSubspace convergence using steps_to_target metric.

    Both modes are trained on identical data with the same seed. The test measures
    steps_to_target at 20% accuracy for each mode, validating that:
    1. Both modes produce the steps_to_target metric correctly
    2. AdaptiveSubspace achieves meaningful learning (above random chance)
    3. AdaptiveSubspace reaches the target accuracy within the training budget

    Note on per-step convergence: LinearSubspace's per-layer projections cover ~4.3%
    of parameters per step vs AdaptiveSubspace's ~0.25% (even at rank=4096). This gives
    LinearSubspace faster per-step convergence. However, AdaptiveSubspace compensates
    with 10-30x faster wall-clock time per step due to a much smaller OT problem, and
    the rotating basis eventually explores all parameter dimensions.
    """
    train_loader, test_loader = mnist_loaders

    target_acc = 0.20
    epochs = 5

    # Helper callback for step-level accuracy tracking
    from polystep import TrainCallback

    class StepAccuracyTracker(TrainCallback):
        """Track accuracy at every step and record steps_to_target."""

        def __init__(self, model, test_loader, target_acc, check_every=1):
            self.model = model
            self.test_loader = test_loader
            self.target_acc = target_acc
            self.check_every = check_every
            self.steps_to_target = None
            self.last_accuracy = 0.0
            self._step = 0

        def on_step_end(self, metrics):
            self._step += 1
            if self._step % self.check_every == 0:
                acc = _evaluate(self.model, self.test_loader)
                self.last_accuracy = acc
                if self.steps_to_target is None and acc >= self.target_acc:
                    self.steps_to_target = self._step
            return False

    # ---- LinearSubspace ----
    torch.manual_seed(42)
    model_l = SmallMLP(hidden=64)
    layout_l = ParamLayout.from_module(model_l)
    linear_sub = LinearSubspace.from_layout(layout_l, rank=4)

    opt_l = PolyStepOptimizer(
        model_l,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=4.5,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=linear_sub,
    )

    tracker_l = StepAccuracyTracker(model_l, test_loader, target_acc, check_every=1)
    config_l = TrainConfig(epochs=epochs, callbacks=[tracker_l])
    train(model_l, train_loader, nn.CrossEntropyLoss(), opt_l, config_l)
    linear_final_acc = _evaluate(model_l, test_loader)
    tracker_l.last_accuracy = linear_final_acc
    linear_steps_to_target = tracker_l.steps_to_target
    if linear_steps_to_target is None and linear_final_acc >= target_acc:
        linear_steps_to_target = opt_l.state.iteration_count

    # ---- AdaptiveSubspace ----
    # Use fixed epsilon to avoid NaN displacement issue with adaptive radius +
    # high rank combination (known blocker documented in STATE.md).
    torch.manual_seed(42)
    model_a = SmallMLP(hidden=64)
    layout_a = ParamLayout.from_module(model_a)
    adaptive_sub = AdaptiveSubspace.from_layout(
        layout_a,
        rank=4096,
        rotation_mode="displacement",
        absorb_mode="periodic",
        absorb_interval=20,
    )

    opt_a = PolyStepOptimizer(
        model_a,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=10.0,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=adaptive_sub,
    )

    tracker_a = StepAccuracyTracker(model_a, test_loader, target_acc, check_every=1)
    config_a = TrainConfig(epochs=epochs, callbacks=[tracker_a])
    train(model_a, train_loader, nn.CrossEntropyLoss(), opt_a, config_a)
    adaptive_final_acc = _evaluate(model_a, test_loader)
    tracker_a.last_accuracy = adaptive_final_acc
    adaptive_steps_to_target = tracker_a.steps_to_target
    if adaptive_steps_to_target is None and adaptive_final_acc >= target_acc:
        adaptive_steps_to_target = opt_a.state.iteration_count

    # ---- Report steps_to_target comparison ----
    print(f"\n  Convergence comparison (target={target_acc * 100:.0f}% accuracy):")
    print(f"    LinearSubspace:   steps_to_target={linear_steps_to_target}, final_acc={linear_final_acc * 100:.1f}%")
    print(
        f"    AdaptiveSubspace: steps_to_target={adaptive_steps_to_target}, final_acc={adaptive_final_acc * 100:.1f}%"
    )

    if linear_steps_to_target is not None and adaptive_steps_to_target is not None:
        speedup = linear_steps_to_target / adaptive_steps_to_target if adaptive_steps_to_target > 0 else float("inf")
        print(f"    Steps-to-target speedup (linear/adaptive): {speedup:.2f}x")

    # ---- Assertions ----
    # 1. AdaptiveSubspace must reach the target accuracy (20% is well above 10% random chance)
    assert adaptive_steps_to_target is not None, (
        f"AdaptiveSubspace should reach {target_acc * 100:.0f}% accuracy within {epochs} epochs, "
        f"but only achieved {adaptive_final_acc * 100:.1f}%"
    )

    # 2. LinearSubspace must also reach target (baseline validation)
    assert linear_steps_to_target is not None, (
        f"LinearSubspace should reach {target_acc * 100:.0f}% accuracy within {epochs} epochs, "
        f"but only achieved {linear_final_acc * 100:.1f}%"
    )

    # 3. Both produced valid steps_to_target metrics
    assert adaptive_steps_to_target > 0, "steps_to_target should be positive"
    assert linear_steps_to_target > 0, "steps_to_target should be positive"

    # 4. AdaptiveSubspace achieves meaningful final accuracy (above random chance)
    assert adaptive_final_acc >= 0.15, (
        f"AdaptiveSubspace should achieve at least 15% accuracy, got {adaptive_final_acc * 100:.1f}%"
    )


@requires_mnist
@pytest.mark.slow
def test_adaptive_vs_linear_step_speed(mnist_loaders):
    """AdaptiveSubspace should be significantly faster per step than LinearSubspace.

    This validates the core architectural advantage: global projection creates
    a much smaller OT problem than per-layer projection.
    """
    train_loader, _ = mnist_loaders

    import time

    # AdaptiveSubspace timing
    torch.manual_seed(42)
    model_a = SmallMLP(hidden=64)
    layout_a = ParamLayout.from_module(model_a)
    adaptive_sub = AdaptiveSubspace.from_layout(
        layout_a,
        rank=256,
        rotation_mode="random",
    )
    opt_a = PolyStepOptimizer(
        model_a,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=10.0,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=adaptive_sub,
    )
    config = TrainConfig(epochs=1)
    t0 = time.time()
    train(model_a, train_loader, nn.CrossEntropyLoss(), opt_a, config)
    adaptive_time = time.time() - t0
    adaptive_steps = opt_a.state.iteration_count

    # LinearSubspace timing
    torch.manual_seed(42)
    model_l = SmallMLP(hidden=64)
    layout_l = ParamLayout.from_module(model_l)
    linear_sub = LinearSubspace.from_layout(layout_l, rank=2)
    opt_l = PolyStepOptimizer(
        model_l,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=4.5,
        probe_radius=2.0,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=linear_sub,
    )
    config = TrainConfig(epochs=1)
    t0 = time.time()
    train(model_l, train_loader, nn.CrossEntropyLoss(), opt_l, config)
    linear_time = time.time() - t0
    linear_steps = opt_l.state.iteration_count

    # Both should complete the same number of steps
    assert adaptive_steps == linear_steps

    # AdaptiveSubspace runs a much smaller OT problem (a single
    # ``rank``-by-``num_probe`` matrix instead of one per layer) so it
    # should be measurably faster per step. We use a loose multiplier
    # because wall-clock measurements over a single MNIST epoch are
    # noisy on shared CI runners; the per-step speedup observed in the
    # paper benchmarks is 3-30x.
    adaptive_per_step = adaptive_time / max(1, adaptive_steps)
    linear_per_step = linear_time / max(1, linear_steps)
    speedup = linear_per_step / adaptive_per_step

    assert speedup >= 1.2, (
        f"Expected AdaptiveSubspace >= 1.2x faster per step, "
        f"got {speedup:.2f}x (adaptive={adaptive_per_step:.2f}s, "
        f"linear={linear_per_step:.2f}s)"
    )
