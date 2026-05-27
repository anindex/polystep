"""Shared tiny-SNN demo helpers.

Single source of truth for the small spiking network used by the
``examples/02_snn_starter.py`` runnable demo. Kept tiny (~1-3K
parameters) so the example finishes in under a couple of minutes on
CPU and can also drive small 2D loss-landscape visualizations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer


__all__ = [
    "LIFNeuron",
    "TinySNN",
    "SNNDemoConfig",
    "OUTPUT_SCALE",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_HIDDEN",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_NUM_STEPS",
    "DEFAULT_NUM_TRAIN",
    "DEFAULT_NUM_TEST",
    "DEFAULT_BATCH_SIZE",
    "make_dataset",
    "make_loaders",
    "make_optimizer",
    "evaluate_accuracy",
]


DEFAULT_INPUT_DIM = 16
DEFAULT_HIDDEN = 24
DEFAULT_NUM_CLASSES = 4
DEFAULT_NUM_STEPS = 20
DEFAULT_NUM_TRAIN = 256
DEFAULT_NUM_TEST = 64
DEFAULT_BATCH_SIZE = 32

OUTPUT_SCALE = 10.0


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with hard threshold spike.

    The spike function ``(mem >= threshold).float()`` is genuinely
    non-differentiable: ``d(spike)/d(mem) == 0`` almost everywhere, so
    backpropagation through this layer gives zero gradient. PolyStep
    only evaluates the forward pass, so the spike stays as-is.
    """

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        spike = (mem >= self.threshold).float()
        mem = mem * (1.0 - spike)
        return spike, mem


class TinySNN(nn.Module):
    """A 1-hidden-layer SNN with hard LIF spikes.

    Architecture: ``Linear -> LIF -> Linear -> LIF``. Output is the mean
    spike rate over ``num_steps`` simulated timesteps.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden: int = DEFAULT_HIDDEN,
        num_classes: int = DEFAULT_NUM_CLASSES,
        num_steps: int = DEFAULT_NUM_STEPS,
        beta: float = 0.95,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.lif1 = LIFNeuron(beta=beta)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.lif2 = LIFNeuron(beta=beta)
        self.num_steps = num_steps
        self.hidden = hidden
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        mem1 = torch.zeros(batch, self.hidden, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, self.num_classes, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, self.num_classes, device=x.device, dtype=x.dtype)
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2
        # Mean spike rate per output class in [0, 1]; multiply by OUTPUT_SCALE
        # at loss / argmax time to sharpen the cross-entropy signal.
        return total / self.num_steps


def make_dataset(num_samples: int, *, input_dim: int = DEFAULT_INPUT_DIM,
                 num_classes: int = DEFAULT_NUM_CLASSES, seed: int = 42):
    """Synthetic class-conditional rate-coded dataset.

    Each class is a distinct subset of "on" features; samples are noisy
    versions of the per-class template. Easy enough for the tiny network to
    learn in tens of steps yet rugged enough that the LIF non-smoothness
    shows up in the loss landscape (which is the whole point of the GIF).
    """
    rng = torch.Generator().manual_seed(seed)
    neurons_per_class = max(1, input_dim // num_classes)
    templates = torch.zeros(num_classes, input_dim)
    for c in range(num_classes):
        start = c * neurons_per_class
        end = min(start + neurons_per_class, input_dim)
        templates[c, start:end] = 1.0
    templates = (templates + 0.2 * torch.randn(num_classes, input_dim, generator=rng)).clamp(0, 1)
    labels = torch.randint(0, num_classes, (num_samples,), generator=rng)
    noise = 0.15 * torch.randn(num_samples, input_dim, generator=rng)
    data = (templates[labels] + noise).clamp(0, 1)
    return data, labels


def make_loaders(*, num_train: int = DEFAULT_NUM_TRAIN, num_test: int = DEFAULT_NUM_TEST,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 input_dim: int = DEFAULT_INPUT_DIM,
                 num_classes: int = DEFAULT_NUM_CLASSES, seed: int = 42):
    train_x, train_y = make_dataset(
        num_train, input_dim=input_dim, num_classes=num_classes, seed=seed,
    )
    test_x, test_y = make_dataset(
        num_test, input_dim=input_dim, num_classes=num_classes, seed=seed + 1,
    )
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


@dataclass
class SNNDemoConfig:
    """Hyperparameters for the tiny SNN demo. Tuned for fast CPU runs."""
    epsilon: float = 0.5
    step_radius: float = 2.0
    probe_radius: float = 4.0
    num_probe: int = 1
    sinkhorn_max_iters: int = 100


def make_optimizer(model: nn.Module, *, seed: int = 42,
                   config: SNNDemoConfig | None = None) -> PolyStepOptimizer:
    """Build the PolyStepOptimizer with SNN-tuned hyperparameters.

    Larger radii than typical NN settings (0.15 / 0.3) because the LIF
    temporal dynamics make the loss landscape rugged on the scale of a few
    weight units. Values come from the historical ``examples/spiking_nn.py``.
    """
    cfg = config or SNNDemoConfig()
    return PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=cfg.epsilon,
        step_radius=cfg.step_radius,
        probe_radius=cfg.probe_radius,
        num_probe=cfg.num_probe,
        sinkhorn_max_iters=cfg.sinkhorn_max_iters,
    )


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Classification accuracy using the ``OUTPUT_SCALE`` convention."""
    correct = total = 0
    for inputs, targets in loader:
        logits = model(inputs) * OUTPUT_SCALE
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0
