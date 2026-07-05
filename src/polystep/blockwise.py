"""Block-wise Sinkhorn decomposition for scaling to large parameter spaces.

Instead of solving a single OT problem over all particles, decomposes into
independent smaller OT problems per block (e.g. per layer). Each block has
its own polytope, cost matrix, and OT solve.

Sinkhorn is O(n^2) in particle count. Splitting M particles into L blocks
of M/L reduces total cost from O(M^2) to O(L * (M/L)^2) = O(M^2/L).

Usage::

    optimizer = PolyStepOptimizer(model, block_strategy='per_layer')

See Also:
    ``PolyStepOptimizer`` for the ``block_strategy`` parameter.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .transform import ParamLayout
    from .cost_nn import NNCostEvaluator


@dataclass(frozen=True)
class BlockConfig:
    """Configuration for a single block in block-wise Sinkhorn.

    Attributes:
        name: Human-readable name for the block (e.g. 'fc1.weight').
        leaf_indices: Indices into ParamLayout.entries belonging to this block.
        flat_start: Start offset in the block-wise flat particle vector.
        flat_end: End offset in the block-wise flat particle vector.
        num_particles: Number of particle rows in this block.
        particle_dim: Dimension of each particle in this block.
    """

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
    """Create one block per parameter entry (layer) in the layout.

    Each block computes its own padding independently so that
    ``num_params`` is divisible by ``particle_dim``.

    Args:
        layout: ParamLayout describing model parameter structure.
        particle_dim: Number of elements per particle row.

    Returns:
        List of BlockConfig, one per layout entry.
    """
    blocks: List[BlockConfig] = []
    offset = 0

    for i, entry in enumerate(layout.entries):
        num_params = entry.numel
        padded = num_params + (-num_params % particle_dim)
        num_particles = padded // particle_dim
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
    """Create blocks by grouping consecutive layout entries.

    Bundles consecutive entries (e.g. weight+bias pairs) into blocks.
    Each group's total element count is padded independently.

    Args:
        layout: ParamLayout describing model parameter structure.
        group_size: Number of consecutive entries per block.
        particle_dim: Number of elements per particle row.

    Returns:
        List of BlockConfig.
    """
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
    """Slice full flat particle vector into per-block particle tensors.

    Args:
        particles: 2D tensor of shape ``(total_particles, particle_dim)`` or
            1D flat tensor. Internally flattened and sliced by block offsets.
        blocks: List of BlockConfig with flat_start/flat_end offsets.

    Returns:
        List of tensors, each ``(block.num_particles, block.particle_dim)``.
    """
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
    """Reconstruct full flat vector from per-block particle tensors.

    Args:
        block_particles: List of tensors, each
            ``(block.num_particles, block.particle_dim)``.
        blocks: List of BlockConfig with flat_start/flat_end offsets.
        total_flat_size: Total size of the reassembled flat vector
            (sum of all block padded sizes).

    Returns:
        1D tensor of shape ``(total_flat_size,)``.
    """
    if not block_particles:
        return torch.zeros(total_flat_size)

    full_flat = torch.zeros(total_flat_size, dtype=block_particles[0].dtype, device=block_particles[0].device)

    for block_X, block in zip(block_particles, blocks):
        block_flat = block_X.reshape(-1)
        block_size = block.flat_end - block.flat_start
        full_flat[block.flat_start : block.flat_end] = block_flat[:block_size]

    return full_flat


@torch.inference_mode()
def blocks_to_layout_flat(
    block_flat: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """Map block-indexed flat vector to layout-indexed flat vector.

    Per-layer (and grouped) blocks pad each block independently, creating a
    different offset scheme than ``ParamLayout`` (which concatenates all
    entries contiguously and pads once at the end).  This function copies
    actual parameter data from block offsets to the correct layout offsets,
    producing a vector suitable for ``layout.batch_unflatten``.

    Handles both per-layer blocks (one entry per block) and grouped blocks
    (multiple entries per block) via ``block.leaf_indices``.

    Args:
        block_flat: 1D tensor in block-indexed layout (from ``reassemble_blocks``).
        blocks: List of BlockConfig with ``leaf_indices`` into ``layout.entries``.
        layout: ParamLayout for the model.

    Returns:
        1D tensor of shape ``(layout.padded_size,)`` in layout-indexed layout.
    """
    layout_flat = torch.zeros(
        layout.padded_size,
        dtype=block_flat.dtype,
        device=block_flat.device,
    )

    for block in blocks:
        internal_offset = 0
        for leaf_idx in block.leaf_indices:
            entry = layout.entries[leaf_idx]
            numel = entry.numel
            layout_flat[entry.offset : entry.offset + numel] = block_flat[
                block.flat_start + internal_offset : block.flat_start + internal_offset + numel
            ]
            internal_offset += numel

    return layout_flat


@torch.inference_mode()
def layout_flat_to_block_flat(
    layout_flat: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """Map layout-indexed flat vector to block-indexed flat vector.

    Inverse of ``blocks_to_layout_flat``.  Extracts actual parameter data
    from layout entry offsets and places it at the correct block offsets,
    with per-block padding zeros.

    Args:
        layout_flat: 1D tensor of shape ``(layout.padded_size,)`` in
            layout-indexed format (e.g. from ``layout.flatten(model)``).
        blocks: List of BlockConfig with ``leaf_indices`` into ``layout.entries``.
        layout: ParamLayout for the model.

    Returns:
        1D tensor of shape ``(total_block_flat_size,)`` in block-indexed layout.
    """
    total_block_flat_size = blocks[-1].flat_end if blocks else 0
    block_flat = torch.zeros(
        total_block_flat_size,
        dtype=layout_flat.dtype,
        device=layout_flat.device,
    )

    for block in blocks:
        internal_offset = 0
        for leaf_idx in block.leaf_indices:
            entry = layout.entries[leaf_idx]
            numel = entry.numel
            block_flat[block.flat_start + internal_offset : block.flat_start + internal_offset + numel] = layout_flat[
                entry.offset : entry.offset + numel
            ]
            internal_offset += numel

    return block_flat


@torch.inference_mode()
def blocks_to_layout_flat_batch(
    block_flat_batch: torch.Tensor,
    blocks: List[BlockConfig],
    layout: "ParamLayout",
) -> torch.Tensor:
    """Batched version of ``blocks_to_layout_flat``.

    Args:
        block_flat_batch: 2D tensor of shape ``(N, total_block_flat_size)``.
        blocks: List of BlockConfig with ``leaf_indices`` into ``layout.entries``.
        layout: ParamLayout for the model.

    Returns:
        2D tensor of shape ``(N, layout.padded_size)``.
    """
    N = block_flat_batch.shape[0]
    layout_batch = torch.zeros(
        N,
        layout.padded_size,
        dtype=block_flat_batch.dtype,
        device=block_flat_batch.device,
    )

    for block in blocks:
        internal_offset = 0
        for leaf_idx in block.leaf_indices:
            entry = layout.entries[leaf_idx]
            numel = entry.numel
            layout_batch[:, entry.offset : entry.offset + numel] = block_flat_batch[
                :, block.flat_start + internal_offset : block.flat_start + internal_offset + numel
            ]
            internal_offset += numel

    return layout_batch


# ------------------------------------------------------------------
# Subspace-aware block functions
# ------------------------------------------------------------------


def create_subspace_blocks(
    subspace_dim: int,
    num_blocks: int,
    subspace_particle_dim: int = 8,
) -> List[BlockConfig]:
    """Create blocks that divide the subspace coordinate space.

    For combined subspace + block-wise mode, the OT decomposition operates
    in the projected subspace coordinates, NOT in full parameter space.
    This function divides the subspace_dim coordinates into num_blocks
    equal-sized blocks, each with subspace_particle_dim as the particle
    dimension.

    Key design insight:
    - Block operation space is PROJECTED SUBSPACE, not full parameter space
    - Blocks slice the subspace_dim coordinates
    - Global projection P is shared across all blocks (applied once at start/end)

    Args:
        subspace_dim: Total subspace dimension (e.g., 256).
        num_blocks: Number of blocks to create.
        subspace_particle_dim: Particle dimension within each block (default 8).
            Higher values give more polytope vertices but fewer particles per block.

    Returns:
        List of BlockConfig for subspace-aware block decomposition.

    Example::

        # 256-dim subspace split into 4 blocks with 8-dim particles
        blocks = create_subspace_blocks(256, num_blocks=4, subspace_particle_dim=8)
        # Each block: 64 subspace coords -> 8 particles of dim 8

    """
    # Pad subspace_dim to be divisible by subspace_particle_dim
    padded_subspace = subspace_dim + (-subspace_dim % subspace_particle_dim)
    total_particles = padded_subspace // subspace_particle_dim

    if num_blocks > total_particles:
        warnings.warn(
            f"num_blocks ({num_blocks}) exceeds total_particles ({total_particles}). Clamping to total_particles.",
            stacklevel=2,
        )
        num_blocks = total_particles

    # Divide particles evenly across blocks
    base_particles_per_block = total_particles // num_blocks
    remainder = total_particles % num_blocks

    blocks: List[BlockConfig] = []
    offset = 0

    for i in range(num_blocks):
        # Distribute remainder: first 'remainder' blocks get one extra particle
        num_particles = base_particles_per_block + (1 if i < remainder else 0)
        flat_size = num_particles * subspace_particle_dim
        blocks.append(
            BlockConfig(
                name=f"subspace_block_{i}",
                leaf_indices=(),  # Not used for subspace blocks
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
    """Split 1D subspace coordinate vector into per-block particle tensors.

    Takes a flattened subspace coordinate vector and slices it according to
    the block configuration. Each slice is reshaped to (num_particles, particle_dim).

    This is the subspace-aware equivalent of split_particles(), operating
    on subspace coordinates rather than full parameter flat vectors.

    Args:
        subspace_coords: 1D tensor of shape (subspace_dim,) or 2D tensor of
            shape (num_particles, particle_dim) that will be flattened.
        blocks: List of BlockConfig from create_subspace_blocks().

    Returns:
        List of tensors, each of shape (block.num_particles, block.particle_dim).

    Example::

        coords = torch.randn(256)  # subspace coordinates
        blocks = create_subspace_blocks(256, 4, 8)
        block_particles = split_subspace_to_blocks(coords, blocks)
        # block_particles[0].shape == (8, 8)  # 64 coords -> 8 particles x 8 dim

    """
    flat = subspace_coords.reshape(-1)

    # Pad to total block size if subspace_dim isn't divisible by particle_dim
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
    """Reconstruct 1D subspace coordinate vector from per-block particle tensors.

    Inverse of split_subspace_to_blocks(). Takes the list of per-block particle
    tensors and reassembles them into a contiguous subspace coordinate vector.

    Args:
        block_particles: List of tensors, each of shape
            (block.num_particles, block.particle_dim).
        blocks: List of BlockConfig from create_subspace_blocks().
        subspace_dim: Original subspace dimension (for trimming padding).

    Returns:
        1D tensor of shape (subspace_dim,).

    Example::

        # After per-block OT updates
        updated_coords = reassemble_blocks_to_subspace(
            block_particles, blocks, subspace_dim=256
        )

    """
    if not block_particles:
        return torch.zeros(subspace_dim)

    # Total padded size from blocks
    total_padded = sum(b.flat_end - b.flat_start for b in blocks)
    device = block_particles[0].device
    dtype = block_particles[0].dtype

    full_flat = torch.zeros(total_padded, dtype=dtype, device=device)

    for block_X, block in zip(block_particles, blocks):
        block_flat = block_X.reshape(-1)
        full_flat[block.flat_start : block.flat_end] = block_flat

    # Trim to actual subspace_dim (remove padding)
    return full_flat[:subspace_dim]


def compute_block_cost_matrix(
    block_idx: int,
    X_probe_block: torch.Tensor,
    all_block_particles: List[torch.Tensor],
    blocks: List[BlockConfig],
    layout: ParamLayout,
    evaluator: NNCostEvaluator,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cost matrix for one block: full forward, perturb only this block.

    For each probe point (p, v, k), constructs full model parameters by
    taking the current particles from all blocks and replacing particle row p
    in block_idx with the probe value. This captures cross-block interactions
    through the full model forward pass.

    Args:
        block_idx: Index of the block being evaluated.
        X_probe_block: Probe points for this block, shape
            ``(P, V, K, block_particle_dim)``.
        all_block_particles: Current particles for all blocks (base values).
        blocks: List of all BlockConfig.
        layout: ParamLayout for converting flat vectors to param dicts.
        evaluator: NNCostEvaluator for model evaluation.
        inputs: Input data batch.
        targets: Optional target labels.

    Returns:
        Cost matrix of shape ``(P, V)``.
    """
    P, V, K, D = X_probe_block.shape
    N = P * V * K
    total_flat_size = sum(b.flat_end - b.flat_start for b in blocks)

    # Build base full flat vector from current block particles
    base_flat = reassemble_blocks(all_block_particles, blocks, total_flat_size)

    # Expand base flat to batch: (N, total_flat_size)
    base_batch = base_flat.unsqueeze(0).expand(N, -1).clone()

    # For each probe point (p, v, k), replace particle row p in block_idx
    block = blocks[block_idx]
    flat_probes = X_probe_block.reshape(P * V * K, D)  # (N, D)

    # Each probe n corresponds to particle p = n // (V*K)
    # The flat offset for particle p in the block is:
    #   block.flat_start + p * D
    # Vectorized scatter: compute all row offsets at once
    indices = torch.arange(N, device=flat_probes.device)
    p_indices = indices // (V * K)
    row_starts = block.flat_start + p_indices * D
    # Build column indices for each of the D elements per probe
    col_offsets = torch.arange(D, device=flat_probes.device)  # (D,)
    # (N, D) matrix of column indices into base_batch
    col_indices = row_starts.unsqueeze(1) + col_offsets.unsqueeze(0)
    # Scatter flat_probes into base_batch using advanced indexing
    row_idx = indices.unsqueeze(1).expand(-1, D)  # (N, D)
    base_batch[row_idx, col_indices] = flat_probes

    # Per-layer blocks pad each entry independently, creating different
    # offsets from ParamLayout (which concatenates entries contiguously).
    # Map from block offsets to layout offsets for correct batch_unflatten.
    batch_for_layout = blocks_to_layout_flat_batch(base_batch, blocks, layout)

    # Convert to stacked param dicts
    stacked_params = layout.batch_unflatten(batch_for_layout)

    # Evaluate all probe points
    losses = evaluator.evaluate(stacked_params, inputs, targets)  # (N,)

    # Reshape and average over probe dimension K
    cost_matrix = losses.reshape(P, V, K).mean(dim=-1)  # (P, V)
    return cost_matrix
