"""Non-differentiable model definitions for the paper experiments.

Every model has at least one forward-pass operation whose gradient is zero
almost everywhere (hard thresholds, rounding, sign, argmax). STE variants serve
the gradient-based baselines and smooth variants serve as Adam ceilings.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

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


INIT_SCALE = 0.1  # every quantizing layer below initializes at randn * INIT_SCALE


class QuantizedLinear(nn.Module):
    """Linear layer with int8 weight quantization in the forward pass.

    d(round)/dx = 0 almost everywhere. ``forward`` calls the declared transforms rather
    than repeating them, so the batched evaluators cannot read a stale rule.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * INIT_SCALE)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = scale

    def polystep_weight_transform(self, w: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.round(w / self.scale), -128, 127) * self.scale

    polystep_bias_transform = polystep_weight_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.polystep_weight_transform(self.weight).t() + self.polystep_bias_transform(self.bias)


class BinaryLinear(nn.Module):
    """Linear layer with binary weights via sign(); effective weights in {-1, +1}.

    The bias is not quantized.
    """

    polystep_weight_transform = staticmethod(torch.sign)
    polystep_bias_transform = None

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * INIT_SCALE)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.polystep_weight_transform(self.weight).t() + self.bias  # NON-DIFFERENTIABLE


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights: below threshold -> 0, above -> +/-1.

    The threshold must move with the initialization scale: several sigma out
    zeroes every weight and the layer returns its bias for any input.
    """

    polystep_bias_transform = None

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5 * INIT_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * INIT_SCALE)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.threshold = threshold
        if not bool((self.weight.detach().abs() >= threshold).any()):
            warnings.warn(
                f"TernaryLinear({in_features}, {out_features}) with threshold={threshold} zeroes every "
                f"initial weight, so the layer outputs its bias for any input. Lower the threshold "
                f"or widen the initialization.",
                stacklevel=2,
            )

    def polystep_weight_transform(self, w: torch.Tensor) -> torch.Tensor:
        return torch.sign(w) * (w.abs() >= self.threshold).to(w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.polystep_weight_transform(self.weight).t() + self.bias  # NON-DIFFERENTIABLE


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


class BinaryLinearSTE(nn.Module):
    """Binary linear layer with STE for gradient-based training."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * INIT_SCALE)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = STESign.apply(self.weight)  # STE in backward pass
        return x @ w_b.t() + self.bias


class TernaryLinearSTE(nn.Module):
    """Ternary linear layer with STE for gradient-based training."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5 * INIT_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * INIT_SCALE)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_t = STETernary.apply(self.weight, self.threshold)  # STE in backward pass
        return x @ w_t.t() + self.bias


class BinaryConv2dSTE(nn.Module):
    """Conv2d with binary weights via STE for gradient-based training."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = STESign.apply(self.weight)  # STE in backward pass
        return F.conv2d(x, w_b, self.bias, padding=self.padding)


class BinaryConv2d(nn.Module):
    """Conv2d with binary weights via sign(). NON-DIFFERENTIABLE."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_b = torch.sign(self.weight)  # NON-DIFFERENTIABLE
        return F.conv2d(x, w_b, self.bias, padding=self.padding)


class DiscreteAttention(nn.Module):
    """Attention-like layer that routes each input to its most similar key via hard argmax."""

    def __init__(self, dim: int, num_slots: int = 8):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, dim) * 0.5)
        self.values = nn.Linear(dim, dim)
        self.num_slots = num_slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        sim = x @ self.keys.t()  # (batch, num_slots)
        # Hard routing via argmax, NON-DIFFERENTIABLE
        slot_idx = sim.argmax(dim=-1)  # (batch,)
        selected_keys = self.keys[slot_idx]  # (batch, dim)
        return self.values(x) * torch.sigmoid(selected_keys)


class StaircaseActivation(nn.Module):
    """Piecewise-constant staircase: floor(sigmoid(x) * levels) / levels. Gradient zero everywhere."""

    polystep_elementwise = True  # coordinatewise, no state: the batched paths accept it

    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(torch.sigmoid(x) * self.levels) / self.levels  # NON-DIFFERENTIABLE


class HardMoELayer(nn.Module):
    """Hard Mixture-of-Experts layer with top-1 argmax gating.

    ALL experts are evaluated on every forward pass (vmap-safe: no conditional
    branching); selection is one_hot * stacked outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)  # (batch, num_experts)
        expert_idx = gate_logits.argmax(dim=-1)  # NON-DIFFERENTIABLE (batch,)

        # Evaluate ALL experts (vmap-safe: no conditional branching)
        all_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, num_experts, hidden_dim)

        one_hot = F.one_hot(expert_idx, self.num_experts).float()  # (batch, num_experts)
        return (all_outputs * one_hot.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)


class SpikingMNISTNet(nn.Module):
    """SNN with hard-threshold LIF neurons for MNIST.

    Returns total spike counts over num_steps timesteps (NOT divided by
    num_steps: scale in the loss if needed).
    """

    neuron_cls = LIFNeuron

    def __init__(self, num_steps: int = 15):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = self.neuron_cls()
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = self.neuron_cls()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, -1)  # Flatten to (batch, 784)

        mem1 = torch.zeros(batch, 128, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)

        cur1 = self.fc1(x)  # static input: same injected current at every timestep
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(cur1, mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2

        return total  # (batch, 10): raw spike counts


class QuantizedMLP(nn.Sequential):
    """MLP where hidden layer uses int8 quantized weights."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(input_dim, hidden)),
                    ("act1", nn.ReLU()),
                    ("quant", QuantizedLinear(hidden, hidden)),  # NON-DIFFERENTIABLE
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, output)),
                ]
            )
        )


class BinaryMNISTNet(nn.Sequential):
    """MNIST classifier with binary weights via sign()."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", BinaryLinear(input_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", BinaryLinear(hidden, output)),
                ]
            )
        )


class TernaryMNISTNet(nn.Sequential):
    """MNIST classifier with ternary weights via sign() * threshold."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", TernaryLinear(input_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", TernaryLinear(hidden, output)),
                ]
            )
        )


class BinaryMNISTNetSTE(nn.Sequential):
    """MNIST classifier with binary weights via STE for gradient-based training."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", BinaryLinearSTE(input_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", BinaryLinearSTE(hidden, output)),
                ]
            )
        )


class TernaryMNISTNetSTE(nn.Sequential):
    """MNIST classifier with ternary weights via STE for gradient-based training."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", TernaryLinearSTE(input_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", TernaryLinearSTE(hidden, output)),
                ]
            )
        )


class DiscreteAttentionNet(nn.Module):
    """MLP with discrete argmax attention routing."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 128,
        output: int = 10,
        num_slots: int = 8,
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


class StaircaseNet(nn.Sequential):
    """MLP with piecewise-constant staircase activation."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 128,
        output: int = 10,
        levels: int = 5,
    ):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(input_dim, hidden)),
                    ("staircase", StaircaseActivation(levels)),
                    ("fc2", nn.Linear(hidden, hidden)),
                    ("staircase2", StaircaseActivation(levels)),
                    ("fc3", nn.Linear(hidden, output)),
                ]
            )
        )


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


class SmoothLIFNeuron(nn.Module):
    """Differentiable analog of LIFNeuron: sigmoid((mem - threshold) * temperature) instead of the hard spike."""

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


class SmoothSpikingMNISTNet(SpikingMNISTNet):
    """SpikingMNISTNet with SmoothLIFNeuron, same parameter count; Adam ceiling."""

    neuron_cls = SmoothLIFNeuron


class SmoothQuantizedMLP(nn.Sequential):
    """QuantizedMLP with a plain nn.Linear in place of QuantizedLinear; Adam ceiling."""

    def __init__(self, input_dim: int = 784, hidden: int = 128, output: int = 10):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(input_dim, hidden)),
                    ("act1", nn.ReLU()),
                    ("fc_hidden", nn.Linear(hidden, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, output)),
                ]
            )
        )


class SmoothAttention(nn.Module):
    """Differentiable analog of DiscreteAttention: softmax-weighted keys instead of argmax routing."""

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
    """DiscreteAttentionNet with SmoothAttention, same parameter count; Adam ceiling."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 128,
        output: int = 10,
        num_slots: int = 8,
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


class SmoothStaircaseNet(nn.Sequential):
    """StaircaseNet with plain sigmoid instead of the staircase activation; Adam ceiling."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 128,
        output: int = 10,
        levels: int = 5,
    ):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(input_dim, hidden)),
                    ("act1", nn.Sigmoid()),
                    ("fc2", nn.Linear(hidden, hidden)),
                    ("act2", nn.Sigmoid()),
                    ("fc3", nn.Linear(hidden, output)),
                ]
            )
        )


class SoftMoELayer(nn.Module):
    """Differentiable analog of HardMoELayer: softmax gating instead of top-1 argmax."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)  # (batch, num_experts)
        weights = F.softmax(gate_logits, dim=-1)  # DIFFERENTIABLE soft routing
        all_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, num_experts, hidden_dim)
        return (all_outputs * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)


class SoftMoENet(nn.Module):
    """HardMoENet with SoftMoELayer, same parameter count; Adam ceiling."""

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
    """Per-expert routing statistics for a HardMoENet-style model on the test set.

    Returns per-expert shares, the max share, a collapse flag (max share > 0.40),
    and routing entropy (raw and normalized).
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


class HardPermutationNet(nn.Module):
    """Maps an input sequence to a sorting permutation via a shared MLP + row-wise argmax.

    NON-DIFFERENTIABLE.
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
        x_flat = x.reshape(-1, 1)  # (batch*N, 1)
        scores = self.mlp(x_flat)  # (batch*N, N)
        score_matrix = scores.reshape(batch, self.N, self.N)  # (batch, N, N)
        # Transpose: we need score_matrix[output_pos][input_pos] so argmax gives
        # "which input goes to this output position" (argsort convention).
        score_matrix = score_matrix.transpose(-1, -2)
        perm = score_matrix.argmax(dim=-1)  # NON-DIFFERENTIABLE
        return perm  # (batch, N) long


class SoftPermutationNet(nn.Module):
    """Soft permutation via log-domain Sinkhorn normalization (Gumbel-Sinkhorn Adam baseline).

    Same MLP as HardPermutationNet, but produces a doubly-stochastic matrix,
    converted to a hard permutation at evaluation via the Hungarian algorithm.
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
        # Transpose: Sinkhorn + bmm(P, x) needs P[output_pos][input_pos].
        soft_perm = self.sinkhorn_normalize(score_matrix.transpose(-1, -2) / self.tau)
        return soft_perm  # (batch, N, N) doubly-stochastic


class PermutationLoss(nn.Module):
    """Fraction of incorrect position assignments; non-differentiable."""

    def forward(self, pred_perm, target_perm):
        # pred_perm, target_perm: (batch, N) long tensors
        return (pred_perm != target_perm).float().mean()


class MaxSATModel(nn.Module):
    """MAX-SAT via continuous relaxation: sigmoid for [0,1], round() for hard {0,1} evaluation."""

    def __init__(self, num_vars: int):
        super().__init__()
        self.assignments = nn.Parameter(torch.randn(num_vars) * 0.1)

    def forward(
        self,
        clause_vars: torch.Tensor,
        clause_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Fraction of unsatisfied clauses.

        clause_signs: 1.0 = positive literal, 0.0 = negated literal.
        """
        soft = torch.sigmoid(self.assignments)
        hard = torch.round(soft)  # NON-DIFFERENTIABLE: {0, 1}

        gathered = hard[clause_vars]  # (num_clauses, vars_per_clause)

        # Literal satisfaction: a positive literal is satisfied when gathered == 1,
        # a negated one when gathered == 0.
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)

        # A clause is satisfied if ANY literal is true (> 0.5)
        satisfied = (literals > 0.5).any(dim=-1).float()  # (num_clauses,)

        unsat_ratio = 1.0 - satisfied.mean()
        return unsat_ratio


def cra_penalty(
    soft_assignments: torch.Tensor,
    alpha: int = 2,
) -> torch.Tensor:
    """Continuous Relaxation Annealing penalty: pushes assignments toward {0, 1}.

    (1 - (2*x - 1)^alpha).sum(): 0 at x in {0, 1}, 1 per element at x = 0.5.
    """
    return (1.0 - (2.0 * soft_assignments - 1.0) ** alpha).sum()


def evaluate_sat_loss(
    assignments_soft: torch.Tensor,
    clause_vars: torch.Tensor,
    clause_signs: torch.Tensor,
    cra_lambda: float = 0.1,
    cra_alpha: int = 2,
) -> torch.Tensor:
    """Combined MAX-SAT loss: unsat_ratio + cra_lambda * CRA penalty."""
    hard = torch.round(assignments_soft)  # {0, 1}

    gathered = hard[clause_vars]
    literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
    satisfied = (literals > 0.5).any(dim=-1).float()
    unsat_ratio = 1.0 - satisfied.mean()

    penalty = cra_penalty(assignments_soft, alpha=cra_alpha)

    return unsat_ratio + cra_lambda * penalty
