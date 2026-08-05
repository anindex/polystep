"""Block-wise Sinkhorn decomposition for scaling to large parameter spaces."""

from __future__ import annotations

import functools
import warnings
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


def split_particles(
    particles: torch.Tensor,
    blocks: List[BlockConfig],
) -> List[torch.Tensor]:
    """Flatten ``particles`` and slice it into one tensor per block."""
    flat = particles.reshape(-1)
    result: List[torch.Tensor] = []

    for block in blocks:
        block_flat = flat[block.flat_start : block.flat_end]
        result.append(block_flat.reshape(block.num_particles, block.particle_dim))

    return result


@torch.inference_mode()
def reassemble_blocks(
    block_particles: List[torch.Tensor],
    blocks: List[BlockConfig],
    total_flat_size: int,
) -> torch.Tensor:
    """Inverse of :func:`split_particles`, into one ``(total_flat_size,)`` vector."""
    if not block_particles:
        return torch.zeros(total_flat_size)

    full_flat = torch.zeros(total_flat_size, dtype=block_particles[0].dtype, device=block_particles[0].device)

    for block_X, block in zip(block_particles, blocks):
        block_flat = block_X.reshape(-1)
        block_size = block.flat_end - block.flat_start
        full_flat[block.flat_start : block.flat_end] = block_flat[:block_size]

    return full_flat


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
    layout_flat = torch.zeros(layout.padded_size, dtype=block_flat.dtype, device=block_flat.device)
    for lo, bo, numel in _block_layout_spans(tuple(blocks), layout):
        layout_flat[lo : lo + numel] = block_flat[bo : bo + numel]
    return layout_flat


@torch.inference_mode()
def layout_flat_to_block_flat(
    layout_flat: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """Inverse of :func:`blocks_to_layout_flat`; per-block padding stays zero."""
    total_block_flat_size = blocks[-1].flat_end if blocks else 0
    block_flat = torch.zeros(total_block_flat_size, dtype=layout_flat.dtype, device=layout_flat.device)
    for lo, bo, numel in _block_layout_spans(tuple(blocks), layout):
        block_flat[bo : bo + numel] = layout_flat[lo : lo + numel]
    return block_flat


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


def create_subspace_blocks(
    subspace_dim: int,
    num_blocks: int,
    subspace_particle_dim: int = 8,
) -> List[BlockConfig]:
    """Create blocks that divide the subspace coordinate space."""
    padded_subspace = subspace_dim + (-subspace_dim % subspace_particle_dim)
    total_particles = padded_subspace // subspace_particle_dim

    if num_blocks > total_particles:
        warnings.warn(
            f"num_blocks ({num_blocks}) exceeds total_particles ({total_particles}). Clamping to total_particles.",
            stacklevel=2,
        )
        num_blocks = total_particles

    base_particles_per_block = total_particles // num_blocks
    remainder = total_particles % num_blocks

    blocks: List[BlockConfig] = []
    offset = 0

    for i in range(num_blocks):
        num_particles = base_particles_per_block + (1 if i < remainder else 0)
        flat_size = num_particles * subspace_particle_dim
        blocks.append(
            BlockConfig(
                name=f"subspace_block_{i}",
                leaf_indices=(),  # unused for subspace blocks
                flat_start=offset,
                flat_end=offset + flat_size,
                num_particles=num_particles,
                particle_dim=subspace_particle_dim,
            )
        )
        offset += flat_size

    return blocks


def split_subspace_to_blocks(
    subspace_coords: torch.Tensor,
    blocks: List[BlockConfig],
) -> List[torch.Tensor]:
    """``split_particles`` over subspace coordinates rather than full flat parameters."""
    flat = subspace_coords.reshape(-1)

    total_block_size = blocks[-1].flat_end if blocks else 0
    if flat.shape[0] < total_block_size:
        flat = torch.nn.functional.pad(flat, (0, total_block_size - flat.shape[0]))

    result: List[torch.Tensor] = []

    for block in blocks:
        block_flat = flat[block.flat_start : block.flat_end]
        result.append(block_flat.reshape(block.num_particles, block.particle_dim))

    return result


def reassemble_blocks_to_subspace(
    block_particles: List[torch.Tensor],
    blocks: List[BlockConfig],
    subspace_dim: int,
) -> torch.Tensor:
    """Inverse of :func:`split_subspace_to_blocks`, trimmed back to ``subspace_dim``."""
    if not block_particles:
        return torch.zeros(subspace_dim)

    total_padded = sum(b.flat_end - b.flat_start for b in blocks)
    device = block_particles[0].device
    dtype = block_particles[0].dtype

    full_flat = torch.zeros(total_padded, dtype=dtype, device=device)

    for block_X, block in zip(block_particles, blocks):
        block_flat = block_X.reshape(-1)
        full_flat[block.flat_start : block.flat_end] = block_flat

    return full_flat[:subspace_dim]
