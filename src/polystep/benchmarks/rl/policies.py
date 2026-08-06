"""Small policy networks and parameter helpers for RL benchmarks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


class DiscreteMLPPolicy(nn.Sequential):
    """MLP policy for discrete-action direct policy search. An nn.Sequential subclass, as the batched evaluators require."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int):
        super().__init__(
            OrderedDict(
                [
                    (
                        "net",
                        nn.Sequential(
                            nn.Linear(obs_dim, hidden),
                            nn.Tanh(),
                            nn.Linear(hidden, action_dim),
                        ),
                    )
                ]
            )
        )


def _quantize_int8_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric INT8 quantize/dequantize (no STE)."""

    if x.numel() == 0:
        return x
    max_abs = x.detach().abs().amax()
    if not torch.isfinite(max_abs) or max_abs.item() == 0.0:
        return x
    scale = (max_abs / 127.0).clamp(min=1e-8)
    return torch.round(x / scale) * scale


class NonDiffActivation(nn.Module):
    """Non-differentiable activation layer: "int8", "binary", or "float32"; no straight-through estimation."""

    def __init__(self, mode: str = "binary"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"float32", "int8", "binary"}:
            raise ValueError(f"NonDiffActivation mode must be float32/int8/binary; got {mode!r}")
        self.mode = mode
        # sign() and identity are coordinatewise; int8 couples every output to every input and stays on the vmap path.
        self.polystep_elementwise = mode != "int8"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "float32":
            return x
        if self.mode == "int8":
            return _quantize_int8_per_tensor(x)
        # binary
        return torch.sign(x)


class NonDiffMLPPolicy(nn.Sequential):
    """Discrete-action MLP policy with a non-differentiable activation layer instead of Tanh."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int, *, mode: str = "binary"):
        mode = str(mode).lower()
        super().__init__(
            OrderedDict(
                [
                    (
                        "net",
                        nn.Sequential(
                            nn.Linear(obs_dim, hidden),
                            NonDiffActivation(mode),
                            nn.Linear(hidden, action_dim),
                        ),
                    )
                ]
            )
        )
        self.mode = mode


def stack_module_params(
    module: nn.Module,
    num_candidates: int,
    *,
    noise_scale: float = 0.0,
    seed: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Repeat a module's parameters along a candidate dimension."""

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


def count_stacked_candidates(stacked_params: Dict[str, torch.Tensor]) -> int:
    """Return the leading candidate dimension for a stacked parameter dict."""

    first = next(iter(stacked_params.values()))
    return int(first.shape[0])
