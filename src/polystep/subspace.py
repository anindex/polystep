"""Shared pieces of the subspace compressors."""

from __future__ import annotations

import contextlib
import math
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _stable_entry_seed(*parts: object) -> int:
    """Hash a tuple of seed components to a 31-bit integer, deterministically
    across processes (unlike Python's salted ``hash``).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


def absorb_due(
    absorb_mode: str,
    absorb_patience: int,
    absorb_interval: int,
    stagnation_count: int,
    iteration: int,
) -> bool:
    """Whether an absorb-and-rotate is due this step, shared by every subspace class."""
    if absorb_mode == "stagnation":
        return stagnation_count >= absorb_patience
    if absorb_mode == "periodic" and absorb_interval > 0:
        return iteration > 0 and iteration % absorb_interval == 0
    return False


def history_unusable(rotation_mode: str, history: Optional[torch.Tensor]) -> bool:
    """Whether a rotation redraws at random: random mode, or a missing, empty, non-finite or near-zero history."""
    if rotation_mode == "random" or history is None or history.shape[0] == 0:
        return True
    disp_norm = torch.linalg.vector_norm(history).item()
    return not math.isfinite(disp_norm) or disp_norm < 1e-10


def top_directions(D: torch.Tensor, k: int) -> torch.Tensor:
    """Unit-norm top-``k`` left singular directions of a tall ``D``, dropping numerically null ones."""
    device_type = D.device.type
    autocast_available = torch.amp.is_autocast_available(device_type)
    with torch.amp.autocast(device_type, enabled=False) if autocast_available else contextlib.nullcontext():
        h = D.shape[1]
        rows = D.shape[0] if D.dtype == torch.float64 else (1 << 22) // h
        chunks = map(torch.Tensor.double, D.split(max(1, rows)))
        w, V = torch.linalg.eigh(sum(c.mT @ c for c in chunks))
        tol = max(h * torch.finfo(w.dtype).eps, (h * torch.finfo(D.dtype).eps) ** 2)
        keep = min(k, int((w > w[-1:] * tol).sum()))
        return D @ (V[:, h - keep :].flip(1) * w[h - keep :].flip(0).rsqrt()).to(D.dtype)


class ProjectedAbsorbMixin:
    """``absorb`` and ``should_absorb`` for subspaces whose ``apply_perturbation`` takes a projection."""

    def absorb(self, projection, base_sd, flat_subspace, residual=None):
        """Fold the perturbation into the base weights, returning ``(new_base_sd, zeroed_subspace_vector)``.

        ``residual`` carries each half-precision entry's rounding error into the next absorb.
        """
        half = {k for k, v in base_sd.items() if residual is not None and v.dtype in (torch.float16, torch.bfloat16)}
        updated = self.apply_perturbation(
            projection, {k: torch.zeros_like(v) if k in half else v for k, v in base_sd.items()}, flat_subspace
        )
        for k in half & updated.keys():
            exact = base_sd[k].float() + updated[k].float() + residual.get(k, 0)
            updated[k] = exact.to(base_sd[k].dtype)
            residual[k] = (exact - updated[k].float()).to(updated[k].dtype)
        return {**base_sd, **updated}, torch.zeros_like(flat_subspace)

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Whether to fold the perturbation into the base weights this step."""
        return absorb_due(
            self.absorb_mode,
            self.absorb_patience,
            self.absorb_interval,
            stagnation_count,
            iteration,
        )


class SvdRatioMixin:
    """Linear ramp of the SVD-derived fraction of a rotated basis."""

    def get_svd_ratio(self, step: int, total_steps: int) -> float:
        """Linear interpolation from ``svd_ratio_init`` at step 0 to ``svd_ratio_final``."""
        progress = min(1.0, step / max(1, total_steps or 1))
        return self.svd_ratio_init + progress * (self.svd_ratio_final - self.svd_ratio_init)

    def check_svd_ratios(self) -> None:
        """Require ``svd_ratio_init`` and ``svd_ratio_final`` in [0, 1]."""
        for name in ("svd_ratio_init", "svd_ratio_final"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}.")


@dataclass(frozen=True)
class ProjectionSpec:
    """How one parameter entry maps to its slice of the subspace vector."""

    entry_key: str
    original_shape: Tuple[int, ...]
    num_params: int
    num_coords: int
    flat_start: int
    flat_end: int
    is_projected: bool = True
