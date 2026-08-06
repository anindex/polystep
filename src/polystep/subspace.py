"""Shared pieces of the subspace compressors."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Tuple

import torch


def _stable_entry_seed(*parts: object) -> int:
    """Hash a tuple of seed components to a 31-bit integer, deterministically
    across processes (unlike Python's salted ``hash``).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.adler32(key)) & 0x7FFFFFFF


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


class ProjectedAbsorbMixin:
    """``absorb`` for subspaces whose ``apply_perturbation`` takes a projection."""

    def absorb(self, projection, base_sd, flat_subspace):
        """Fold the perturbation into the base weights, returning ``(new_base_sd, zeroed_subspace_vector)``."""
        updated = self.apply_perturbation(projection, base_sd, flat_subspace)
        return {**base_sd, **updated}, torch.zeros_like(flat_subspace)


class SvdRatioMixin:
    """Linear ramp of the SVD-derived fraction of a rotated basis."""

    def get_svd_ratio(self, step: int, total_steps: int) -> float:
        """Linear interpolation from ``svd_ratio_init`` at step 0 to ``svd_ratio_final``."""
        progress = min(1.0, step / max(1, total_steps or 1))
        return self.svd_ratio_init + progress * (self.svd_ratio_final - self.svd_ratio_init)


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
