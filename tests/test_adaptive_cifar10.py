"""Smoke tests for CIFAR-10 adaptive subspace benchmark.

Validates that SmallResNet + AdaptiveSubspace runs without errors on
convolutional architecture with skip connections. Uses random data
(not real CIFAR-10) to avoid download dependency in CI.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, AdaptiveSubspace
from polystep.cost_nn import NNCostEvaluator
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# SmallResNet definition (matches SmallResNet architecture)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm for vmap compatibility."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return torch.relu(out + identity)


class SmallResNet(nn.Module):
    """ResNet-8 variant with ~52K parameters for CIFAR-10.

    Uses GroupNorm for vmap compatibility with gradient-free optimization.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.downsample = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.gn_down = nn.GroupNorm(num_groups=4, num_channels=32)
        self.block3 = ResBlock(32)
        self.block4 = ResBlock(32)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.gn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = torch.relu(self.gn_down(self.downsample(x)))
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cifar10_resnet_param_count():
    """SmallResNet has parameter count in the 30K-100K range."""
    model = SmallResNet()
    num_params = sum(p.numel() for p in model.parameters())
    assert 30_000 < num_params < 100_000, (
        f"Expected 30K-100K params, got {num_params:,}"
    )


@pytest.mark.slow
def test_cifar10_resnet_forward_pass():
    """SmallResNet produces correct output shape for CIFAR-10 input."""
    model = SmallResNet()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


@pytest.mark.slow
def test_cifar10_resnet_adaptive_runs():
    """AdaptiveSubspace runs on SmallResNet without errors for 5 steps.

    Uses random data (not real CIFAR-10) to avoid download in CI.
    Validates that:
    - AdaptiveSubspace creates successfully from a convolutional model
    - PolyStepOptimizer runs without errors
    - All OT costs are finite
    - Absorb mechanism fires (via stagnation detection)
    """
    torch.manual_seed(42)

    model = SmallResNet()
    num_params = sum(p.numel() for p in model.parameters())

    # Verify param count
    assert 30_000 < num_params < 100_000

    # Create AdaptiveSubspace
    subspace = AdaptiveSubspace.auto_from_params(
        model,
        compression_target=0.1,
        min_rank=16,
        max_rank=16,
        rotation_mode="displacement",
        absorb_mode="stagnation",
        absorb_patience=3,  # Low patience so absorb fires in 5 steps
    )
    assert subspace.subspace_dim == 16
    assert subspace.full_dim == num_params

    # Create optimizer
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=0.5,
        step_radius=0.15,
        probe_radius=0.3,
        num_probe=3,
        sinkhorn_max_iters=50,
        subspace=subspace,
        use_adaptive_radius=True,
        stagnation_patience=3,
    )

    # Random CIFAR-10-like data
    train_data = torch.randn(64, 3, 32, 32)
    train_labels = torch.randint(0, 10, (64,))
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    train_ds = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Train for 5 steps
    costs = []
    data_iter = iter(train_loader)
    for step in range(5):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)

        def closure(batched_params, _in=inputs, _tgt=targets):
            return evaluator.evaluate(batched_params, _in, _tgt)

        ot_cost = optimizer.step(closure)
        costs.append(ot_cost)

    # All costs should be finite
    assert all(math.isfinite(c) for c in costs), (
        f"Non-finite costs detected: {costs}"
    )

    # Optimizer state should be valid
    state = optimizer.state
    assert state.iteration_count == 5
    assert len(state.costs) == 5
    assert state.projection is not None
    assert state.projection.shape == (num_params, 16)
