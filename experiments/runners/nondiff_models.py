"""Non-differentiable model definitions for paper experiments (Phases 36-40).

All models contain at least one non-differentiable operation in their forward pass
where backpropagation gives zero gradients. polystep trains them using only forward
passes via entropic optimal transport.

Building blocks:
  - LIFNeuron: Leaky Integrate-and-Fire with hard threshold spike
  - QuantizedLinear: Int8 weight rounding (round)
  - BinaryLinear: Binary weights via sign()
  - TernaryLinear: Ternary quantization via sign() * (abs >= threshold)
  - BinaryConv2d: Conv2d with binary weights via sign()
  - DiscreteAttention: Argmax routing over attention slots
  - StaircaseActivation: Piecewise-constant floor-based activation
  - HardMoELayer: Hard top-1 expert gating via argmax

STE autograd functions (for gradient-based baselines):
  - STESign: sign() forward, identity backward (clamped at |w| <= 1)
  - STETernary: ternary forward, identity backward (clamped at |w| <= 1)

STE-enabled layers (for gradient-based baselines):
  - BinaryLinearSTE, TernaryLinearSTE, BinaryConv2dSTE

Full models (sized for MNIST 28x28):
  - SpikingMNISTNet, QuantizedMLP, BinaryMNISTNet, TernaryMNISTNet
  - BinaryMNISTNetSTE, TernaryMNISTNetSTE (STE baselines)
  - DiscreteAttentionNet, StaircaseNet, HardMoENet

Full models (sized for CIFAR-10 32x32):
  - BinaryCIFAR10Net (non-differentiable), BinaryCIFAR10NetSTE (STE baseline)

MAX-SAT utilities (direct parameter optimization, no hidden layers):
  - MaxSATModel, cra_penalty, evaluate_sat_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "LIFNeuron",
    "SpikingMNISTNet",
    "QuantizedLinear",
    "QuantizedMLP",
    "BinaryLinear",
    "BinaryMNISTNet",
    "TernaryLinear",
    "TernaryMNISTNet",
    "STESign",
    "STETernary",
    "BinaryLinearSTE",
    "TernaryLinearSTE",
    "BinaryConv2d",
    "BinaryConv2dSTE",
    "BinaryMNISTNetSTE",
    "TernaryMNISTNetSTE",
    "BinaryCIFAR10Net",
    "BinaryCIFAR10NetSTE",
    "DiscreteAttention",
    "DiscreteAttentionNet",
    "StaircaseActivation",
    "StaircaseNet",
    "HardMoELayer",
    "HardMoENet",
    "MaxSATModel",
    "evaluate_sat_loss",
    "cra_penalty",
    "SmoothLIFNeuron",
    "SmoothSpikingMNISTNet",
    "SmoothQuantizedMLP",
    "SmoothDiscreteAttentionNet",
    "SmoothStaircaseNet",
    "SoftMoELayer",
    "SoftMoENet",
    "compute_expert_utilization",
    "HardPermutationNet",
    "SoftPermutationNet",
    "PermutationLoss",
]


# ---------------------------------------------------------------------------
# Building blocks (non-differentiable operations)
# ---------------------------------------------------------------------------


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with hard threshold spike.

    d(spike)/d(membrane) = 0 everywhere except at the discontinuity.
    """

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        spike = (mem >= self.threshold).float()  # NON-DIFFERENTIABLE
        mem = mem * (1.0 - spike)
        return spike, mem


class QuantizedLinear(nn.Module):
    """Linear layer with int8 weight quantization in the forward pass.

    d(round)/dx = 0 almost everywhere.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = torch.clamp(torch.round(self.weight / self.scale), -128, 127) * self.scale
        b_q = torch.clamp(torch.round(self.bias / self.scale), -128, 127) * self.scale
        return x @ w_q.t() + b_q


class BinaryLinear(nn.Module):
    """Linear layer with binary weights via sign().

    Effective weights are in {-1, +1}. sign(0) = 0 but randn rarely produces
    exact zero.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = torch.sign(self.weight)  # NON-DIFFERENTIABLE
        return x @ w_b.t() + self.bias


class TernaryLinear(nn.Module):
    """Linear layer with ternary weight quantization.

    Weights below threshold become 0, above become +/-1.
    Effective weights are in {-1, 0, +1}.
    """

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_t = torch.sign(self.weight) * (self.weight.abs() >= self.threshold).float()  # NON-DIFFERENTIABLE
        return x @ w_t.t() + self.bias


# ---------------------------------------------------------------------------
# STE autograd functions (for gradient-based baselines)
# ---------------------------------------------------------------------------


class STESign(torch.autograd.Function):
    """Straight-Through Estimator for sign() binarization.

    Forward: sign(x) -> {-1, +1}
    Backward: gradient passed through where |x| <= 1 (saturated STE per Bengio 2013)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # Saturated STE: pass gradient where |input| <= 1
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


class STETernary(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization.

    Forward: sign(x) * (|x| >= threshold) -> {-1, 0, +1}
    Backward: gradient passed through where |x| <= 1 (saturated STE)
    """

    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return torch.sign(input) * (input.abs() >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input, None  # None for threshold (not trainable)


# ---------------------------------------------------------------------------
# STE-enabled layers (for gradient-based baselines)
# ---------------------------------------------------------------------------


class BinaryLinearSTE(nn.Module):
    """Binary linear layer with STE for gradient-based training."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = STESign.apply(self.weight)  # STE in backward pass
        return x @ w_b.t() + self.bias


class TernaryLinearSTE(nn.Module):
    """Ternary linear layer with STE for gradient-based training."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_t = STETernary.apply(self.weight, self.threshold)  # STE in backward pass
        return x @ w_t.t() + self.bias


class BinaryConv2dSTE(nn.Module):
    """Conv2d with binary weights via STE for gradient-based training."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = STESign.apply(self.weight)  # STE in backward pass
        return F.conv2d(x, w_b, self.bias, padding=self.padding)


# ---------------------------------------------------------------------------
# Non-STE conv layer (for polystep / ES methods)
# ---------------------------------------------------------------------------


class BinaryConv2d(nn.Module):
    """Conv2d with binary weights via sign(). NON-DIFFERENTIABLE."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = torch.sign(self.weight)  # NON-DIFFERENTIABLE
        return F.conv2d(x, w_b, self.bias, padding=self.padding)


class DiscreteAttention(nn.Module):
    """Attention-like layer that uses argmax routing (non-differentiable).

    Routes each input to the most similar key via hard argmax, then
    applies the corresponding value transform.
    """

    def __init__(self, dim: int, num_slots: int = 8):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, dim) * 0.5)
        self.values = nn.Linear(dim, dim)
        self.num_slots = num_slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        sim = x @ self.keys.t()  # (batch, num_slots)
        # Hard routing via argmax -- NON-DIFFERENTIABLE
        slot_idx = sim.argmax(dim=-1)  # (batch,)
        # Gather the selected key for each sample
        selected_keys = self.keys[slot_idx]  # (batch, dim)
        # Gate the value transform by the selected key
        return self.values(x) * torch.sigmoid(selected_keys)


class StaircaseActivation(nn.Module):
    """Piecewise-constant staircase activation function.

    Clamps to [0,1] via sigmoid then quantizes: floor(sigmoid(x) * levels) / levels.
    Gradient is zero everywhere (piecewise constant).
    """

    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(torch.sigmoid(x) * self.levels) / self.levels  # NON-DIFFERENTIABLE


class HardMoELayer(nn.Module):
    """Hard Mixture-of-Experts layer with top-1 argmax gating.

    ALL experts are evaluated on every forward pass (vmap-safe).
    Selection is done via one_hot * stacked outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)  # (batch, num_experts)
        expert_idx = gate_logits.argmax(dim=-1)  # NON-DIFFERENTIABLE (batch,)

        # Evaluate ALL experts (vmap-safe: no conditional branching)
        all_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, num_experts, hidden_dim)

        # One-hot selection
        one_hot = F.one_hot(expert_idx, self.num_experts).float()  # (batch, num_experts)
        return (all_outputs * one_hot.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)


# ---------------------------------------------------------------------------
# Full models (sized for real datasets -- MNIST 28x28)
# ---------------------------------------------------------------------------


class SpikingMNISTNet(nn.Module):
    """SNN with hard-threshold LIF neurons for MNIST.

    Accumulates spikes over num_steps timesteps. Returns total spike counts
    (NOT divided by num_steps -- scale in loss if needed).
    """

    def __init__(self, num_steps: int = 15):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = LIFNeuron()
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = LIFNeuron()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, -1)  # Flatten to (batch, 784)

        mem1 = torch.zeros(batch, 128, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2

        return total  # (batch, 10) -- raw spike counts


class QuantizedMLP(nn.Module):
    """MLP where hidden layer uses int8 quantized weights."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.quant = QuantizedLinear(hidden, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.relu(self.quant(x))  # NON-DIFFERENTIABLE quantized layer
        return self.fc2(x)


class BinaryMNISTNet(nn.Module):
    """MNIST classifier with binary weights via sign()."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = BinaryLinear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = BinaryLinear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TernaryMNISTNet(nn.Module):
    """MNIST classifier with ternary weights via sign() * threshold."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = TernaryLinear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class BinaryMNISTNetSTE(nn.Module):
    """MNIST classifier with binary weights via STE for gradient-based training."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = BinaryLinearSTE(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = BinaryLinearSTE(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TernaryMNISTNetSTE(nn.Module):
    """MNIST classifier with ternary weights via STE for gradient-based training."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = TernaryLinearSTE(input_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = TernaryLinearSTE(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class BinaryCIFAR10Net(nn.Module):
    """CIFAR-10 CNN with all binary weights via sign(). NON-DIFFERENTIABLE.

    Matches CIFAR10Net architecture: Conv32->Conv64->Conv64->FC128->FC10.
    All conv and linear layers use sign() binarization.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BinaryConv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BinaryConv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BinaryConv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(64 * 4 * 4, 128),
            nn.ReLU(),
            BinaryLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class BinaryCIFAR10NetSTE(nn.Module):
    """CIFAR-10 CNN with all binary weights via STE for gradient-based training.

    Matches CIFAR10Net architecture: Conv32->Conv64->Conv64->FC128->FC10.
    All conv and linear layers use STE binarization.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BinaryConv2dSTE(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BinaryConv2dSTE(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BinaryConv2dSTE(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            BinaryLinearSTE(64 * 4 * 4, 128),
            nn.ReLU(),
            BinaryLinearSTE(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DiscreteAttentionNet(nn.Module):
    """MLP with discrete argmax attention routing."""

    def __init__(
        self, input_dim: int = 784, hidden: int = 128, output: int = 10, num_slots: int = 8,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.attn = DiscreteAttention(hidden, num_slots)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.attn(x)  # NON-DIFFERENTIABLE argmax routing
        return self.fc2(x)


class StaircaseNet(nn.Module):
    """MLP with piecewise-constant staircase activation."""

    def __init__(
        self, input_dim: int = 784, hidden: int = 128, output: int = 10, levels: int = 5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.staircase = StaircaseActivation(levels)
        self.fc2 = nn.Linear(hidden, hidden)
        self.staircase2 = StaircaseActivation(levels)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.staircase(self.fc1(x))    # NON-DIFFERENTIABLE
        x = self.staircase2(self.fc2(x))   # NON-DIFFERENTIABLE
        return self.fc3(x)


class HardMoENet(nn.Module):
    """Classifier with hard Mixture-of-Experts layer."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 20,
        num_experts: int = 4,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.moe = HardMoELayer(hidden_dim, hidden_dim, num_experts)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.moe(x)  # NON-DIFFERENTIABLE argmax gating
        return self.fc_out(x)


# ---------------------------------------------------------------------------
# Smooth variants (for Adam baseline -- differentiable analogs)
# ---------------------------------------------------------------------------


class SmoothLIFNeuron(nn.Module):
    """Smooth (differentiable) analog of LIFNeuron.

    Replaces the hard threshold spike (mem >= threshold).float() with a
    sigmoid approximation: sigmoid((mem - threshold) * temperature).
    At high temperature, this closely approximates the hard threshold
    but provides non-zero gradients everywhere.
    """

    def __init__(self, beta: float = 0.95, threshold: float = 1.0, temperature: float = 10.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        spike = torch.sigmoid((mem - self.threshold) * self.temperature)  # DIFFERENTIABLE
        mem = mem * (1.0 - spike)
        return spike, mem


class SmoothSpikingMNISTNet(nn.Module):
    """Smooth (differentiable) analog of SpikingMNISTNet.

    Same architecture as SpikingMNISTNet but uses SmoothLIFNeuron instead of
    LIFNeuron. Identical parameter count (101,770). Serves as Adam accuracy
    ceiling showing 'what if gradients were available'.
    """

    def __init__(self, num_steps: int = 15):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = SmoothLIFNeuron()
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = SmoothLIFNeuron()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, -1)  # Flatten to (batch, 784)

        mem1 = torch.zeros(batch, 128, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2

        return total  # (batch, 10) -- raw spike counts


class SmoothQuantizedMLP(nn.Module):
    """Smooth (differentiable) analog of QuantizedMLP.

    Same architecture as QuantizedMLP but replaces QuantizedLinear with a
    standard nn.Linear. Identical parameter count (118,282). Serves as Adam
    accuracy ceiling showing 'what if no quantization noise'.
    """

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_hidden = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.relu(self.fc_hidden(x))  # DIFFERENTIABLE (no quantization)
        return self.fc2(x)


class SmoothAttention(nn.Module):
    """Smooth (differentiable) analog of DiscreteAttention.

    Replaces the argmax hard routing with softmax-weighted combination of keys.
    Same parameters: keys (num_slots, dim) and values Linear(dim, dim).
    """

    def __init__(self, dim: int, num_slots: int = 8):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, dim) * 0.5)
        self.values = nn.Linear(dim, dim)
        self.num_slots = num_slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        sim = x @ self.keys.t()  # (batch, num_slots)
        weights = F.softmax(sim, dim=-1)  # DIFFERENTIABLE (no argmax)
        selected = weights @ self.keys  # (batch, dim)
        return self.values(x) * torch.sigmoid(selected)


class SmoothDiscreteAttentionNet(nn.Module):
    """Smooth (differentiable) analog of DiscreteAttentionNet.

    Same architecture as DiscreteAttentionNet but uses SmoothAttention instead
    of DiscreteAttention. Identical parameter count (119,306). Serves as Adam
    accuracy ceiling showing 'what if soft attention instead of hard routing'.
    """

    def __init__(
        self, input_dim: int = 784, hidden: int = 128, output: int = 10, num_slots: int = 8,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.attn = SmoothAttention(hidden, num_slots)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.attn(x)  # DIFFERENTIABLE soft attention
        return self.fc2(x)


class SmoothStaircaseNet(nn.Module):
    """Smooth (differentiable) analog of StaircaseNet.

    Same architecture as StaircaseNet but replaces StaircaseActivation with
    plain torch.sigmoid. Identical parameter count (118,282). Serves as Adam
    accuracy ceiling showing 'what if smooth activation instead of staircase'.
    """

    def __init__(
        self, input_dim: int = 784, hidden: int = 128, output: int = 10, levels: int = 5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = torch.sigmoid(self.fc1(x))   # DIFFERENTIABLE (no staircase)
        x = torch.sigmoid(self.fc2(x))   # DIFFERENTIABLE (no staircase)
        return self.fc3(x)


class SoftMoELayer(nn.Module):
    """Soft Mixture-of-Experts layer with softmax gating (differentiable).

    Same architecture as HardMoELayer but replaces argmax with softmax:
    output = sum(softmax(gate) * expert_outputs). Standard differentiable MoE.
    Identical parameter count to HardMoELayer (verified: 235,672 for full SoftMoENet).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)  # (batch, num_experts)
        weights = F.softmax(gate_logits, dim=-1)  # DIFFERENTIABLE soft routing
        all_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, num_experts, hidden_dim)
        return (all_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)


class SoftMoENet(nn.Module):
    """Classifier with soft Mixture-of-Experts layer (Adam baseline).

    Same architecture as HardMoENet but uses SoftMoELayer instead of HardMoELayer.
    Identical parameter count (235,672). Serves as Adam accuracy ceiling showing
    'what if soft routing instead of hard argmax gating'.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 20,
        num_experts: int = 4,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.moe = SoftMoELayer(hidden_dim, hidden_dim, num_experts)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.moe(x)  # DIFFERENTIABLE softmax gating
        return self.fc_out(x)


@torch.no_grad()
def compute_expert_utilization(model, test_loader, device):
    """Compute per-expert routing statistics on test set.

    Records which expert each test sample routes to via the hard argmax gate,
    computes per-expert share of total inputs, routing entropy, and collapse detection.

    Args:
        model: HardMoENet or any model with .fc1, .moe.gate attributes.
        test_loader: Test DataLoader yielding (inputs, labels).
        device: Device for computation.

    Returns:
        Dict with keys:
            - expert_utilization: {expert_0: float, expert_1: float, ...} shares summing to ~1.0
            - max_expert_share: float (highest single expert share)
            - collapsed: bool (True if max_expert_share > 0.40)
            - routing_entropy: float (-sum(p*log(p)))
            - normalized_entropy: float (entropy / max_entropy, 1.0 = perfectly uniform)
    """
    import math

    model.eval()
    expert_counts = {}
    total = 0

    for data, _ in test_loader:
        data = data.to(device)
        x = data.reshape(data.shape[0], -1)
        x = torch.relu(model.fc1(x))
        gate_logits = model.moe.gate(x)
        expert_idx = gate_logits.argmax(dim=-1)  # (batch,)
        for idx in expert_idx.tolist():
            expert_counts[idx] = expert_counts.get(idx, 0) + 1
        total += data.shape[0]

    utilization = {f"expert_{k}": v / total for k, v in sorted(expert_counts.items())}
    max_share = max(utilization.values()) if utilization else 0.0
    collapsed = max_share > 0.40

    # Routing entropy: -sum(p * log(p)), clamped to 0 for numerical stability
    entropy = max(0.0, -sum(p * math.log(p + 1e-10) for p in utilization.values()))
    num_experts = len(utilization) if utilization else 1
    max_entropy = math.log(num_experts) if num_experts > 1 else 1.0

    model.train()

    return {
        "expert_utilization": utilization,
        "max_expert_share": max_share,
        "collapsed": collapsed,
        "routing_entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Permutation learning models
# ---------------------------------------------------------------------------


class HardPermutationNet(nn.Module):
    """Maps input sequence to sorting permutation via shared MLP + argmax.

    Each number in the input sequence is independently processed by a shared MLP
    to produce N scores. The resulting (batch, N, N) score matrix uses row-wise
    argmax to produce hard permutation indices. NON-DIFFERENTIABLE.

    Args:
        N: Sequence length (also determines score matrix size N x N).
        hidden_dim: Hidden dimension of the shared MLP (default 64).
    """

    def __init__(self, N, hidden_dim=64):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N),
        )

    def forward(self, x):
        # x: (batch, N) sequences of numbers
        batch = x.shape[0]
        x_flat = x.reshape(-1, 1)              # (batch*N, 1)
        scores = self.mlp(x_flat)               # (batch*N, N)
        score_matrix = scores.reshape(batch, self.N, self.N)  # (batch, N, N)
        # Transpose: MLP produces scores[input_pos][output_pos], but we need
        # score_matrix[output_pos][input_pos] so argmax gives "which input goes
        # to this output position" (matching argsort target convention).
        score_matrix = score_matrix.transpose(-1, -2)
        perm = score_matrix.argmax(dim=-1)      # NON-DIFFERENTIABLE
        return perm                              # (batch, N) long


class SoftPermutationNet(nn.Module):
    """Maps input sequence to soft permutation via Sinkhorn normalization.

    Identical shared MLP architecture as HardPermutationNet, but replaces argmax
    with log-domain Sinkhorn normalization to produce a doubly-stochastic matrix.
    Used as Gumbel-Sinkhorn baseline (Adam optimizer). Temperature-annealed.

    At evaluation, convert to hard permutation via Hungarian algorithm
    (scipy.optimize.linear_sum_assignment).

    Args:
        N: Sequence length.
        hidden_dim: Hidden dimension of shared MLP (default 64).
        n_sinkhorn_iters: Number of Sinkhorn normalization iterations (default 20).
        tau: Temperature parameter for Sinkhorn (default 1.0). Lower = sharper.
    """

    def __init__(self, N, hidden_dim=64, n_sinkhorn_iters=20, tau=1.0):
        super().__init__()
        self.N = N
        self.n_iters = n_sinkhorn_iters
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N),
        )

    def sinkhorn_normalize(self, log_alpha):
        """Log-domain Sinkhorn: alternating row/col logsumexp normalization."""
        for _ in range(self.n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def forward(self, x):
        batch = x.shape[0]
        x_flat = x.reshape(-1, 1)
        scores = self.mlp(x_flat)
        score_matrix = scores.reshape(batch, self.N, self.N)
        # Transpose: MLP produces scores[input_pos][output_pos], but Sinkhorn
        # normalization + bmm(P, x) needs P[output_pos][input_pos] so that
        # (P @ x)[i] = weighted sum of inputs for output position i.
        soft_perm = self.sinkhorn_normalize(score_matrix.transpose(-1, -2) / self.tau)
        return soft_perm  # (batch, N, N) doubly-stochastic


class PermutationLoss(nn.Module):
    """Fraction of incorrect position assignments.

    For use with HardPermutationNet and gradient-free methods (polystep, CMA-ES, etc.).
    Non-differentiable: compares discrete permutation indices.
    """

    def forward(self, pred_perm, target_perm):
        # pred_perm, target_perm: (batch, N) long tensors
        return (pred_perm != target_perm).float().mean()


# ---------------------------------------------------------------------------
# MAX-SAT utilities (direct parameter optimization, no hidden layers)
# ---------------------------------------------------------------------------


class MaxSATModel(nn.Module):
    """MAX-SAT solver via continuous relaxation.

    No hidden layers -- only self.assignments parameter. Uses sigmoid for [0,1]
    relaxation, round() for {0,1} hard evaluation.
    """

    def __init__(self, num_vars: int):
        super().__init__()
        self.assignments = nn.Parameter(torch.randn(num_vars) * 0.1)

    def forward(
        self,
        clause_vars: torch.Tensor,
        clause_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate unsatisfied clause ratio.

        Args:
            clause_vars: (num_clauses, vars_per_clause) int tensor of variable indices.
            clause_signs: (num_clauses, vars_per_clause) float tensor. 1.0 = positive
                literal, 0.0 = negated literal.

        Returns:
            Scalar tensor: fraction of unsatisfied clauses.
        """
        soft = torch.sigmoid(self.assignments)
        hard = torch.round(soft)  # NON-DIFFERENTIABLE: {0, 1}

        # Gather variable assignments for each clause
        gathered = hard[clause_vars]  # (num_clauses, vars_per_clause)

        # Compute literal satisfaction
        # positive literal: gathered == 1 means satisfied
        # negated literal: gathered == 0 means satisfied
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)

        # A clause is satisfied if ANY literal is true (> 0.5)
        satisfied = (literals > 0.5).any(dim=-1).float()  # (num_clauses,)

        unsat_ratio = 1.0 - satisfied.mean()
        return unsat_ratio


# ---------------------------------------------------------------------------
# Program Synthesizer (Section 7 killer-app)
# ---------------------------------------------------------------------------


def cra_penalty(
    soft_assignments: torch.Tensor,
    alpha: int = 2,
) -> torch.Tensor:
    """Continuous Relaxation Annealing (CRA) penalty.

    Encourages assignments toward {0, 1} by penalizing intermediate values.
    Returns (1 - (2*x - 1)^alpha).sum().

    For x in {0, 1}: penalty = 0. For x = 0.5: penalty = 1 (per element).
    """
    return (1.0 - (2.0 * soft_assignments - 1.0) ** alpha).sum()


def evaluate_sat_loss(
    assignments_soft: torch.Tensor,
    clause_vars: torch.Tensor,
    clause_signs: torch.Tensor,
    cra_lambda: float = 0.1,
    cra_alpha: int = 2,
) -> torch.Tensor:
    """Combined MAX-SAT loss: unsat_ratio + CRA penalty.

    Args:
        assignments_soft: Soft variable assignments in [0, 1].
        clause_vars: (num_clauses, vars_per_clause) variable indices.
        clause_signs: (num_clauses, vars_per_clause) literal signs.
        cra_lambda: Weight for CRA penalty.
        cra_alpha: Exponent for CRA penalty.

    Returns:
        Scalar loss tensor.
    """
    hard = torch.round(assignments_soft)  # {0, 1}

    gathered = hard[clause_vars]
    literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
    satisfied = (literals > 0.5).any(dim=-1).float()
    unsat_ratio = 1.0 - satisfied.mean()

    penalty = cra_penalty(assignments_soft, alpha=cra_alpha)

    return unsat_ratio + cra_lambda * penalty
