"""Small policy networks and parameter helpers for RL benchmarks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn


class DiscreteMLPPolicy(nn.Module):
    """MLP policy for discrete-action direct policy search."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _quantize_int8_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric INT8 quantize/dequantize (no STE).

    Forward: ``q = round(x / scale) * scale`` where ``scale = max|x| / 127``.
    The ``round`` op has zero gradient (PyTorch returns 0), so backprop through
    this layer produces a degenerate signal - exactly what we want to expose as
    a failure mode for PPO/DQN.
    """

    if x.numel() == 0:
        return x
    max_abs = x.detach().abs().amax()
    if not torch.isfinite(max_abs) or max_abs.item() == 0.0:
        return x
    scale = (max_abs / 127.0).clamp(min=1e-8)
    return torch.round(x / scale) * scale


class NonDiffActivation(nn.Module):
    """Non-differentiable activation layer.

    Variants
    --------
    ``"int8"``  : Per-tensor symmetric INT8 quantize/dequantize via ``round``.
    ``"binary"``: ``sign(x)`` activation (collapses to {-1, +1}).
    ``"float32"``: Identity (sanity-check baseline).

    None of the variants implement straight-through estimation (STE). Backprop
    through ``round`` yields zero gradient and through ``sign`` yields zero
    almost everywhere; PPO/DQN trained on a policy containing this layer
    therefore receive no useful gradient past the non-diff op and stagnate at
    random performance.
    """

    def __init__(self, mode: str = "binary"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"float32", "int8", "binary"}:
            raise ValueError(f"NonDiffActivation mode must be float32/int8/binary; got {mode!r}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "float32":
            return x
        if self.mode == "int8":
            return _quantize_int8_per_tensor(x)
        # binary
        return torch.sign(x)


class NonDiffMLPPolicy(nn.Module):
    """Discrete-action MLP policy with a non-differentiable activation layer.

    Topology: ``Linear(obs_dim, hidden) → NonDiffActivation(mode) → Linear(hidden, action_dim)``.

    Compared to :class:`DiscreteMLPPolicy` the only change is the inner Tanh
    being replaced by a non-diff op. PolyStep treats the policy as a black box
    and is unaffected; gradient methods (PPO, DQN) collapse because no useful
    gradient flows through the non-diff op (no STE).
    """

    def __init__(self, obs_dim: int, hidden: int, action_dim: int, *, mode: str = "binary"):
        super().__init__()
        self.mode = str(mode).lower()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            NonDiffActivation(self.mode),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ContinuousMLPPolicy(nn.Module):
    """MLP policy for continuous-control direct policy search.

    Args:
        obs_dim: Observation space size.
        hidden_sizes: Sequence of hidden layer widths.
        action_dim: Action space size.
        activation: Hidden layer activation (``'tanh'`` or ``'elu'``).
        output_tanh: If True, apply tanh to the final output for [-1, 1] action clipping.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int],
        action_dim: int,
        activation: str = "tanh",
        output_tanh: bool = True,
    ):
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "elu": nn.ELU}.get(activation.lower(), nn.Tanh)
        layers: list[nn.Module] = []
        prev = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(act_cls())
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        if output_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def make_taxi_policy(hidden: int = 32) -> DiscreteMLPPolicy:
    """Create the tiny one-hot Taxi policy used by the RL runner."""

    return DiscreteMLPPolicy(obs_dim=500, hidden=hidden, action_dim=6)


def stack_module_params(
    module: nn.Module,
    num_candidates: int,
    *,
    noise_scale: float = 0.0,
    seed: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Repeat a module's parameters along a candidate dimension.

    The returned dictionary matches the closure contract expected by
    ``PolyStepOptimizer.step``: each parameter has shape ``(N, *param.shape)``.
    """

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    stacked: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, param in module.named_parameters():
        values = param.detach().unsqueeze(0).repeat(num_candidates, *([1] * param.ndim))
        if noise_scale > 0:
            noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
            values = values + noise.to(values.device) * noise_scale
        stacked[name] = values
    return stacked


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply per-candidate linear layer to tensors shaped ``(N, R, D)``."""

    return torch.einsum("nrd,nod->nro", x, weight) + bias.unsqueeze(1)


def taxi_logits_from_stacked_params(
    stacked_params: Dict[str, torch.Tensor],
    states: torch.Tensor,
    *,
    num_states: int = 500,
) -> torch.Tensor:
    """Evaluate a stacked Taxi policy for candidate/state batches.

    Args:
        stacked_params: parameters from ``DiscreteMLPPolicy`` with leading
            candidate dimension.
        states: integer Taxi states with shape ``(N, R)``.
        num_states: one-hot state count.

    Returns:
        Logits with shape ``(N, R, 6)``.
    """

    obs = torch.nn.functional.one_hot(states.long(), num_classes=num_states).to(dtype=torch.float32)
    first = torch.tanh(_linear(obs, stacked_params["net.0.weight"], stacked_params["net.0.bias"]))
    return _linear(first, stacked_params["net.2.weight"], stacked_params["net.2.bias"])


def count_stacked_candidates(stacked_params: Dict[str, torch.Tensor]) -> int:
    """Return the leading candidate dimension for a stacked parameter dict."""

    first = next(iter(stacked_params.values()))
    return int(first.shape[0])


def named_trainable_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Small wrapper to keep runner code independent of policy internals."""

    return module.named_parameters()
