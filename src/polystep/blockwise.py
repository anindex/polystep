"""Block-wise Sinkhorn decomposition for scaling to large parameter spaces."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .transform import ParamLayout


@dataclass(frozen=True)
class BlockConfig:
    """Configuration for one block in block-wise Sinkhorn."""

    name: str
    leaf_indices: Tuple[int, ...]
    flat_start: int
    flat_end: int
    num_particles: int
    particle_dim: int


def create_per_layer_blocks(
    layout: ParamLayout,
    particle_dim: int = 2,
) -> List[BlockConfig]:
    """Create one block per parameter entry in the layout."""
    blocks: List[BlockConfig] = []
    offset = 0

    for i, entry in enumerate(layout.entries):
        num_params = entry.numel
        padded = num_params + (-num_params % particle_dim)
        num_particles = padded // particle_dim
        if num_particles == 0:
            # A zero-width block divides by zero in the uniform marginal.
            continue
        blocks.append(
            BlockConfig(
                name=entry.key,
                leaf_indices=(i,),
                flat_start=offset,
                flat_end=offset + padded,
                num_particles=num_particles,
                particle_dim=particle_dim,
            )
        )
        offset += padded

    return blocks


def create_grouped_blocks(
    layout: ParamLayout,
    group_size: int = 2,
    particle_dim: int = 2,
) -> List[BlockConfig]:
    """Create blocks by grouping consecutive layout entries."""
    blocks: List[BlockConfig] = []
    offset = 0
    entries = layout.entries

    for g_start in range(0, len(entries), group_size):
        g_end = min(g_start + group_size, len(entries))
        group_entries = entries[g_start:g_end]
        leaf_indices = tuple(range(g_start, g_end))
        num_params = sum(e.numel for e in group_entries)
        padded = num_params + (-num_params % particle_dim)
        num_particles = padded // particle_dim
        if num_particles == 0:
            # An all-zero-numel group has nothing to search.
            continue
        blocks.append(
            BlockConfig(
                name=f"block_{g_start}_{g_end}",
                leaf_indices=leaf_indices,
                flat_start=offset,
                flat_end=offset + padded,
                num_particles=num_particles,
                particle_dim=particle_dim,
            )
        )
        offset += padded

    return blocks


@functools.lru_cache(maxsize=8)
def _block_layout_spans(
    blocks: Tuple[BlockConfig, ...],
    layout: "ParamLayout",
) -> Tuple[Tuple[int, int, int], ...]:
    """``(layout_offset, block_offset, numel)`` per parameter entry, cached."""
    spans = []
    for block in blocks:
        internal_offset = 0
        for leaf_idx in block.leaf_indices:
            entry = layout.entries[leaf_idx]
            spans.append((entry.offset, block.flat_start + internal_offset, entry.numel))
            internal_offset += entry.numel
    return tuple(spans)


@torch.inference_mode()
def blocks_to_layout_flat(
    block_flat: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """``(total_block_flat_size,)`` to ``(layout.padded_size,)``, ready for
    ``layout.batch_unflatten``. Handles per-layer and grouped blocks alike."""
    layout_flat = block_flat.new_zeros(layout.padded_size + 1)
    layout_flat[block_to_layout_columns(blocks, layout, block_flat.device)] = block_flat
    return layout_flat[:-1]


@torch.inference_mode()
def layout_flat_to_block_flat(
    layout_flat: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """Inverse of :func:`blocks_to_layout_flat`; per-block padding stays zero."""
    padded = torch.cat([layout_flat, layout_flat.new_zeros(1)])
    return padded[block_to_layout_columns(blocks, layout, layout_flat.device)]


@functools.lru_cache(maxsize=8)
def _block_to_layout_columns(
    blocks: Tuple[BlockConfig, ...],
    layout: "ParamLayout",
    device: torch.device,
) -> torch.Tensor:
    total = blocks[-1].flat_end if blocks else 0
    columns = torch.full((total,), layout.padded_size, dtype=torch.long, device=device)
    for lo, bo, numel in _block_layout_spans(blocks, layout):
        columns[bo : bo + numel] = torch.arange(lo, lo + numel, device=device)
    return columns


def block_to_layout_columns(
    blocks: List[BlockConfig],
    layout: "ParamLayout",
    device: torch.device,
) -> torch.Tensor:
    """Layout column per block-order position, cached.

    Per-block padding maps one past ``layout.padded_size``; callers must not write to the result.
    """
    return _block_to_layout_columns(tuple(blocks), layout, device)
