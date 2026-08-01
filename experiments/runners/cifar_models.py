"""CIFAR-10 architectures at ~5-10M parameters whose forward pass is discontinuous.

The reviewer's objection was scale, not novelty, so nothing here is a new
primitive: the trunk is the usual vmap-safe conv stack (no BatchNorm) and the
discontinuity is one of the three blocks already used at MNIST scale in
``nondiff_models.py`` -- hard-threshold LIF, int8 weight rounding, argmax top-1
MoE routing. Only the width changed.

Each net takes ``smooth=True`` to swap the discontinuous block for its
differentiable twin at *identical parameter count*, which is what makes the
Adam-on-surrogate run an accuracy ceiling rather than a different experiment.

Parameter counts (default widths)::

    SpikingCIFARNet    5,625,994    hard LIF threshold, 8 timesteps
    QuantizedCIFARNet  5,625,994    int8 round(), no straight-through estimator
    HardMoECIFARNet    6,680,210    argmax top-1 gating over 8 experts

No BatchNorm anywhere: running stats are not vmap-safe, and the gradient-free
methods score a population of candidates under vmap.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.runners.nondiff_models import (  # noqa: E402
    HardMoELayer,
    LIFNeuron,
    QuantizedLinear,
    SmoothLIFNeuron,
    SoftMoELayer,
)

__all__ = [
    "ConvTrunk",
    "SpikingCIFARNet",
    "QuantizedCIFARNet",
    "HardMoECIFARNet",
    "TRUNK_FEATURES",
]

#: Flat feature width of :class:`ConvTrunk` on a 32x32 input.
TRUNK_FEATURES = 256 * 4 * 4


def init_relu_(module: nn.Module) -> nn.Module:
    """Kaiming-normal every plain ``Conv2d``/``Linear``, biases at zero.

    Not cosmetic. With PyTorch's default uniform init the three-conv trunk comes
    out at std 0.095, the spiking head's injected current never reaches the LIF
    threshold, and :class:`SpikingCIFARNet` emits exactly zero spikes for every
    input -- a loss that is *constant* under any perturbation below ~0.05, on
    which no method can make progress and CMA-ES declares convergence at
    generation 1. Kaiming puts the trunk at std ~1.5 and about 45% of the output
    neurons spiking.

    ``QuantizedLinear`` and ``BinaryLinear`` are deliberately skipped: they are
    not ``nn.Linear`` subclasses and they set their own initialization scale to
    match their quantization grid.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class ConvTrunk(nn.Sequential):
    """3 conv blocks, 32x32 -> 4x4, 370,816 parameters.

    Shared by all three showcases so the only difference between them is the
    discontinuity. ReLU + MaxPool only; see the module docstring on BatchNorm.
    """

    def __init__(self, widths=(64, 128, 256)):
        w1, w2, w3 = widths
        super().__init__(
            nn.Conv2d(3, w1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w1, w2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w2, w3, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )


class SpikingCIFARNet(nn.Module):
    """Conv trunk into a 3-layer spiking MLP with hard-threshold LIF neurons.

    ``(mem >= threshold).float()`` has zero derivative everywhere it is
    defined. The trunk output is a static injected current, as in
    ``SpikingMNISTNet``; the readout is the summed output spike count.

    Args:
        hidden: Width of the two spiking hidden layers.
        num_steps: Timesteps the membrane is integrated over.
        smooth: Use :class:`SmoothLIFNeuron` (sigmoid surrogate) instead.
            Same parameters, differentiable: the Adam ceiling.
    """

    def __init__(self, hidden: int = 1024, num_steps: int = 8, smooth: bool = False):
        super().__init__()
        neuron = SmoothLIFNeuron if smooth else LIFNeuron
        self.trunk = ConvTrunk()
        self.fc1 = nn.Linear(TRUNK_FEATURES, hidden)
        self.lif1 = neuron()
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = neuron()
        self.fc3 = nn.Linear(hidden, 10)
        self.lif3 = neuron()
        self.hidden = hidden
        self.num_steps = num_steps
        init_relu_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = self.fc1(self.trunk(x))
        batch = cur.shape[0]
        z = torch.zeros(batch, self.hidden, device=cur.device, dtype=cur.dtype)
        mem1, mem2 = z, z
        mem3 = torch.zeros(batch, 10, device=cur.device, dtype=cur.dtype)
        total = torch.zeros(batch, 10, device=cur.device, dtype=cur.dtype)
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(cur, mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            total = total + spk3
        return total  # (batch, 10) raw spike counts


class QuantizedCIFARNet(nn.Module):
    """Conv trunk into an int8-quantized MLP head. 5.2M of the 5.6M params are quantized.

    ``QuantizedLinear`` rounds the weight to an int8 grid inside ``forward``,
    with no straight-through estimator, so ``d(loss)/d(weight)`` is zero almost
    everywhere and exactly undefined on the bin boundaries.

    Args:
        hidden: Head width.
        bins_per_std: Quantization bins per unit of weight standard deviation.
            ``QuantizedLinear``'s own default (init std 0.1, step 0.01) is 10 at
            MNIST widths; keeping the *ratio* rather than the absolute step is
            what makes the grid equally coarse at 4096 inputs instead of 128.
        smooth: Replace the quantized layers with plain ``nn.Linear``.
    """

    def __init__(self, hidden: int = 1024, bins_per_std: float = 10.0, smooth: bool = False):
        super().__init__()

        def linear(i, o):
            return nn.Linear(i, o) if smooth else QuantizedLinear(i, o)

        self.net = nn.Sequential(
            ConvTrunk(),
            linear(TRUNK_FEATURES, hidden),
            nn.ReLU(),
            linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )
        init_relu_(self)
        # QuantizedLinear initializes at a fixed std 0.1, which was unit-gain at
        # fan_in 128 and is a 6x amplification at 4096: stacked, the head blew the
        # logits (and the initial loss) up by 20x. Renormalize to Kaiming and move
        # the grid with it, so the layer is scaled like every other one here.
        for m in self.net:
            if isinstance(m, QuantizedLinear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
                m.scale = float(m.weight.detach().std()) / bins_per_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardMoECIFARNet(nn.Module):
    """Conv trunk into an argmax top-1 mixture of experts. 4.2M params live in the experts.

    The router picks one expert with ``argmax``; moving a gate logit changes the
    output in a jump the moment the argmax flips, and not at all before that.

    Args:
        hidden: Router input width and expert width.
        num_experts: Experts to route between.
        smooth: Use :class:`SoftMoELayer` (softmax gating) instead.
    """

    def __init__(self, hidden: int = 512, num_experts: int = 8, smooth: bool = False):
        super().__init__()
        moe = SoftMoELayer if smooth else HardMoELayer
        self.trunk = ConvTrunk()
        self.fc1 = nn.Linear(TRUNK_FEATURES, hidden)
        self.relu = nn.ReLU()
        self.moe = moe(hidden, hidden, num_experts)
        self.fc_out = nn.Linear(hidden, 10)
        init_relu_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.trunk(x)))
        return self.fc_out(self.moe(x))


def _demo() -> None:
    """Parameter counts, twin parity, and the property the paper is about.

    The check that matters: in each hard net there is a weight backprop cannot
    see -- its gradient is *exactly* zero -- and in the smooth twin the same
    weight has a usable gradient. That is the whole reason a black-box cost
    oracle is needed, so it is worth one assert.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    # (class, expected params, the weight backprop loses)
    cases = [
        (SpikingCIFARNet, 5_625_994, "fc2.weight"),  # every spike blocks the path
        (QuantizedCIFARNet, 5_625_994, "net.1.weight"),  # round() has zero derivative
        (HardMoECIFARNet, 6_680_210, "moe.gate.weight"),  # argmax router
    ]
    for cls, expected, key in cases:
        hard, soft = cls(), cls(smooth=True)
        n = sum(p.numel() for p in hard.parameters())
        assert hard(x).shape == (4, 10), cls.__name__
        assert n == sum(p.numel() for p in soft.parameters()), f"{cls.__name__}: twin differs in size"
        assert n == expected, f"{cls.__name__}: {n} != {expected}"
        assert 5e6 <= n <= 10e6, f"{cls.__name__}: {n} outside the 5-10M band"

        grads = []
        for net in (hard, soft):
            net.zero_grad()
            out = net(x).sum()
            # SpikingCIFARNet is so dead that the output carries no grad_fn at all:
            # every path to a parameter runs through a spike indicator.
            if out.requires_grad:
                out.backward()
            g = dict(net.named_parameters())[key].grad
            grads.append(0.0 if g is None else float(g.abs().max()))
        assert grads[0] == 0.0, f"{cls.__name__}.{key} has a nonzero gradient ({grads[0]}): not discontinuous"
        assert grads[1] > 0.0, f"{cls.__name__}.{key} twin is also gradient-free: not a usable ceiling"
        print(f"{cls.__name__:20s} {n:>10,} params  |grad {key}| hard={grads[0]:g} smooth={grads[1]:.3g}")


if __name__ == "__main__":
    _demo()
