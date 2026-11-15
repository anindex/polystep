"""Gradient estimates from forward passes, by probing layer outputs instead of weights."""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from .cost_nn import _first_input_tensor, _forward_args


def _tile(tree, batch: int, copies: int):
    """``tree`` with every tensor of leading size ``batch`` repeated ``copies`` times along it."""
    if copies == 1:
        return tree
    return tree_map(
        lambda t: (
            t.repeat(copies, *[1] * (t.dim() - 1))
            if isinstance(t, torch.Tensor) and t.dim() and t.shape[0] == batch
            else t
        ),
        tree,
    )


def _working(t: torch.Tensor) -> torch.Tensor:
    """``t`` at the precision the differences are taken in: its own for FP64, else FP32."""
    return t if t.dtype == torch.float64 else t.float()


def _rms(tensors) -> torch.Tensor:
    """Root mean square over all entries of ``tensors``, or 1 when they are all zero."""
    square = sum(_working(t).pow(2).sum() for t in tensors) / sum(t.numel() for t in tensors)
    return torch.where(square > 0, square.sqrt(), 1.0)


def _trains(param) -> bool:
    """Whether ``param`` is a parameter the optimizer updates, not a frozen, missing or derived tensor."""
    return isinstance(param, nn.Parameter) and param.requires_grad


class _Pass(NamedTuple):
    """One recorded forward: what went in, the site that produced the outputs and the loss of each sample."""

    inputs: object
    targets: object
    loss_fn: Callable
    head: Optional[nn.Module]
    losses: torch.Tensor
    batch: int


class ProbeGradient:
    """Fill ``.grad`` of a model's ``nn.Linear`` and ``nn.Conv2d`` layers without a backward pass.

    Each sample probes an orthoplex around its own layer output: ``probes`` output units are moved by
    ``+radius`` and ``-radius``, the model is rerun from that layer on, and the central difference of the
    per-sample loss is the gradient with respect to those units. The weight gradient is that estimate times
    the layer input. A convolution is probed through single weight entries instead, which is the smaller
    space when weights are shared over positions. With ``probes`` at least the number of output units the
    frame is complete and the result is the finite-radius gradient; with fewer it is an unbiased estimate
    of it, scaled by ``units / probes``.

    The probed layer's output is expanded to one copy per vertex inside the forward, so the layers ahead of
    it run once, and the last layer is probed on the recorded outputs with no model forward. The first call
    checks both shortcuts against the model run on repeated inputs. A layer where they differ, such as one
    under a residual connection or a reshape to the input's batch size, keeps the repeated inputs. The same
    call warns when samples it left unmoved change their loss, because the estimate needs every sample
    scored on its own.

    ``loss_fn`` must return one loss per sample. Parameters outside these layers get no gradient. The model
    is scored in eval mode. A finite ``probe_radius`` is what lets a probe cross a hard threshold, so raise
    it for spiking, sign or quantized activations.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        probes: int = 64,
        probe_radius: float = 0.1,
        seed: Optional[int] = None,
        max_probe_batch: int = 32768,
    ):
        if probes < 1:
            raise ValueError(f"probes must be >= 1, got {probes}.")
        if not probe_radius > 0:
            raise ValueError(f"probe_radius must be positive, got {probe_radius}.")
        if max_probe_batch < 2:
            raise ValueError(f"max_probe_batch must be >= 2, got {max_probe_batch}.")
        self.model = model
        self.probes = probes
        self.probe_radius = probe_radius
        self.max_probe_batch = max_probe_batch
        self.sites = [
            m
            for m in model.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d)) and (_trains(m.weight) or _trains(m.bias))
        ]
        if not self.sites:
            raise ValueError("ProbeGradient found no trainable nn.Linear or nn.Conv2d layer in the model.")
        self._seed = int(torch.randint(2**31 - 1, ()).item()) if seed is None else seed
        self._generator = None
        self._verified = set()
        self._repeat = set()
        self._warned = set()

    @torch.inference_mode(False)
    @torch.no_grad()
    def __call__(self, inputs, targets=None, *, loss_fn: Callable) -> torch.Tensor:
        """Write the estimated gradient of the mean loss into ``.grad`` and return that mean loss."""
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            calls, outputs, head = self._record(inputs)
            batch = _first_input_tensor(inputs).shape[0]
            losses = self._losses(loss_fn, outputs, targets, batch, 1)
            run = _Pass(inputs, targets, loss_fn, head, losses, batch)
            for module in self.sites:
                for param in (module.weight, module.bias):
                    if _trains(param):
                        param.grad = None
            for module in self.sites:
                recorded = calls.get(module, [])
                conv = isinstance(module, nn.Conv2d)
                if any(z.shape[0] != batch for _, z in recorded):
                    self._warn(module, "ran on a batch size other than the input's; it gets no gradient.")
                elif conv and (
                    len(recorded) > 1
                    or module.groups != 1
                    or module.padding_mode != "zeros"
                    or isinstance(module.padding, str)
                ):
                    self._warn(module, "is grouped, padded by mode or name, or called twice; it gets no gradient.")
                elif recorded:
                    probe = self._probe_weights if conv else self._probe_outputs
                    for param, grad in zip((module.weight, module.bias), probe(module, recorded, run)):
                        if _trains(param):
                            grad = grad.to(param.dtype)
                            param.grad = grad if param.grad is None else param.grad + grad
        finally:
            for module, mode in modes:
                module.training = mode
        return losses.mean()

    def _warn(self, module: nn.Module, message: str) -> None:
        if module not in self._warned:
            self._warned.add(module)
            warnings.warn(f"ProbeGradient: {type(module).__name__} {message}", stacklevel=3)

    def _forward(self, inputs):
        args, kwargs = _forward_args(inputs)
        return self.model(*args, **kwargs)

    def _record(self, inputs):
        """Copies of each site's ``(input, output)`` per call, the model outputs and the site that produced them.

        That site is the one whose single returned tensor is the model's output and was not written in place.
        """
        calls, returned = {}, {}

        def keep(mod, args, out):
            calls.setdefault(mod, []).append((args[0].clone(), out.clone()))
            returned[mod] = (out, out._version)

        hooks = [m.register_forward_hook(keep, prepend=True) for m in self.sites]
        try:
            outputs = self._forward(inputs)
        finally:
            for hook in hooks:
                hook.remove()
        head = next(
            (
                mod
                for mod, (out, version) in returned.items()
                if out is outputs and len(calls[mod]) == 1 and out._version == version
            ),
            None,
        )
        return calls, outputs, head

    @staticmethod
    def _losses(loss_fn, outputs, targets, batch: int, copies: int) -> torch.Tensor:
        loss = loss_fn(outputs) if targets is None else loss_fn(outputs, _tile(targets, batch, copies))
        if loss.dim() == 0 or loss.shape[0] != batch * copies:
            raise ValueError(
                "ProbeGradient needs one loss per sample, but loss_fn returned shape "
                f"{tuple(loss.shape)} for a batch of {batch * copies}. Use reduction='none'."
            )
        loss = _working(loss)
        return loss.flatten(1).mean(1) if loss.dim() > 1 else loss

    def _rng(self, device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device).manual_seed(self._seed)
        return self._generator

    def _units(self, batch: int, dim: int, count: int, device) -> torch.Tensor:
        if count >= dim:
            return torch.arange(dim, device=device).expand(batch, dim)
        return torch.rand(batch, dim, generator=self._rng(device), device=device).topk(count, dim=1).indices

    def _probe_losses(self, module, offsets, zs, run, repeat: bool = False) -> torch.Tensor:
        """``(V, B)`` losses for ``(V, B, D)`` offsets added to the outputs of ``module``, one call after another.

        The layer that produced the model's output needs the loss alone. Any other layer reruns the model
        with its output expanded to ``V`` copies. Both shortcuts give way to the model run on repeated
        inputs when ``repeat`` is set, when the first call found the shortcut wrong, or when the expanded
        forward fails. The layer's first call in a forward restarts from the recorded output, so every
        kind of forward probes the same values.
        """
        copies, batch = offsets.shape[0], run.batch
        repeat = repeat or module in self._repeat
        if module is run.head and not repeat:
            probed = (zs[0].unsqueeze(0) + offsets.view(copies, *zs[0].shape).to(zs[0].dtype)).flatten(0, 1)
            return self._losses(run.loss_fn, probed, run.targets, batch, copies).view(copies, batch)

        def score(inputs):
            cursor = 0

            def shift(mod, args, out):
                nonlocal cursor
                width = out[0].numel()
                delta = offsets[:, :, cursor : cursor + width].reshape(copies, batch, *out.shape[1:]).to(out.dtype)
                out = out if cursor else zs[0]
                cursor += width
                if out.shape[0] == batch:
                    return (out.unsqueeze(0) + delta).flatten(0, 1)
                return out + delta.flatten(0, 1)

            hook = module.register_forward_hook(shift, prepend=True)
            try:
                outputs = self._forward(inputs)
            finally:
                hook.remove()
            return self._losses(run.loss_fn, outputs, run.targets, batch, copies).view(copies, batch)

        if not repeat:
            try:
                return score(run.inputs)
            except (RuntimeError, ValueError):
                self._repeat.add(module)
        return score(_tile(run.inputs, batch, copies))

    def _verify(self, module, zs, run) -> None:
        """Check the shortcuts that score ``module`` and that its samples are scored on their own.

        A random half of the samples moves every output by the probe radius along one sign pattern in one
        copy of the batch, and a second copy moves nothing. The shortcut is kept when it reproduces the
        model run on repeated inputs, with both copies in one batch or one copy at a time: the two differ
        by rounding only, which a hard threshold can amplify in either. Every unmoved sample must then keep
        the loss it has when nothing moves.
        """
        batch, device = run.batch, zs[0].device
        rng = self._rng(device)
        radius = self.probe_radius * _rms(zs)
        moved = torch.randperm(batch, generator=rng, device=device)[: (batch + 1) // 2]
        offsets = radius.new_zeros(2, batch, sum(z[0].numel() for z in zs))
        offsets[0, moved] = radius * (2 * torch.randint(2, offsets.shape[2:], generator=rng, device=device) - 1)
        tolerance = torch.finfo(zs[0].dtype).eps ** 0.5
        floor = tolerance * run.losses.abs().mean().item()

        def same(a, b):
            return torch.allclose(a, b, rtol=tolerance, atol=floor, equal_nan=True)

        plain = self._probe_losses(module, offsets, zs, run, repeat=True)
        fast = self._probe_losses(module, offsets, zs, run)
        if not same(fast, plain):
            single = [self._probe_losses(module, copy[None], zs, run, repeat=True) for copy in offsets]
            if not same(fast, torch.cat(single)):
                self._repeat.add(module)
        still = self._probe_losses(module, torch.zeros_like(offsets), zs, run, repeat=True)
        unmoved = torch.ones_like(plain, dtype=torch.bool)
        unmoved[0, moved] = False
        if not same(plain[unmoved], still[unmoved]):
            self._warn(
                module,
                "changed the loss of samples it left unmoved: the forward or the loss mixes samples or is not "
                "deterministic, so its gradient is unreliable.",
            )
        self._verified.add(module)

    def _central_differences(self, module, recorded, units, radius, build, run) -> torch.Tensor:
        """``(B, p)`` loss slopes along the probed ``units``; ``build`` turns a slice of them into offsets."""
        zs = [z for _, z in recorded]
        if module not in self._verified:
            self._verify(module, zs, run)
        step = max(1, self.max_probe_batch // (2 * run.batch))
        slopes = radius.new_zeros(units.shape)
        for lo in range(0, units.shape[1], step):
            moved = build(units[:, lo : lo + step])
            losses = self._probe_losses(module, torch.cat([moved, -moved]), zs, run)
            half = moved.shape[0]
            slopes[:, lo : lo + step] = (losses[:half] - losses[half:]).T / (2 * radius)
        return slopes

    def _probe_outputs(self, module, recorded, run):
        zs = [z for _, z in recorded]
        widths = [z[0].numel() for z in zs]
        batch, dim = run.batch, sum(widths)
        count = min(self.probes, dim)
        units = self._units(batch, dim, count, zs[0].device)
        radius = self.probe_radius * _rms(zs)

        def build(chosen):
            moved = radius.new_zeros(chosen.shape[1], batch, dim)
            return moved.scatter_(2, chosen.t().unsqueeze(-1), 1.0) * radius

        slopes = self._central_differences(module, recorded, units, radius, build, run) * (dim / count)
        grad_z = slopes.new_zeros(batch, dim).scatter_add_(1, units, slopes)
        grad_w = torch.zeros_like(module.weight, dtype=slopes.dtype)
        grad_b = slopes.new_zeros(module.out_features)
        for (x, _), part in zip(recorded, grad_z.split(widths, 1)):
            part = part.reshape(-1, module.out_features)
            grad_w += part.T @ _working(x).reshape(-1, module.in_features)
            grad_b += part.sum(0)
        return grad_w / batch, grad_b / batch

    def _probe_weights(self, module, recorded, run):
        x, z = recorded[0]
        batch, channels = run.batch, z.shape[1]
        patches = F.unfold(_working(x), module.kernel_size, module.dilation, module.padding, module.stride)
        kernel = patches.shape[1] if _trains(module.weight) else 0
        has_bias = _trains(module.bias)
        patches = patches[:, :kernel]
        if has_bias:
            patches = torch.cat([patches, patches.new_ones(batch, 1, patches.shape[2])], 1)
        features, positions = patches.shape[1:]
        dim = channels * features
        count = min(self.probes, dim)
        units = self._units(batch, dim, count, z.device)
        radius = self.probe_radius * _rms([module.weight])

        def build(chosen):
            picked = chosen.shape[1]
            moved = patches.gather(1, (chosen % features).unsqueeze(-1).expand(batch, picked, positions))
            offsets = patches.new_zeros(picked, batch, channels, positions)
            index = (chosen // features).t().reshape(picked, batch, 1, 1).expand(picked, batch, 1, positions)
            offsets.scatter_(2, index, (moved.transpose(0, 1) * radius).unsqueeze(2))
            return offsets.flatten(2)

        slopes = self._central_differences(module, recorded, units, radius, build, run) * (dim / count)
        grid = slopes.new_zeros(dim).scatter_add_(0, units.reshape(-1), slopes.reshape(-1)).view(channels, features)
        grad_w = grid[:, :kernel].reshape(module.weight.shape) / batch if kernel else None
        grad_b = grid[:, -1] / batch if has_bias else None
        return grad_w, grad_b
