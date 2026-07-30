"""Tests for block-wise Sinkhorn decomposition: block construction and layout mapping."""

import pytest
import torch
import torch.nn as nn

from polystep.blockwise import (
    blocks_to_layout_flat,
    create_grouped_blocks,
    create_per_layer_blocks,
    reassemble_blocks,
    split_particles,
)
from polystep.transform import ParamLayout


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestPerLayerBlocks:
    def test_one_block_per_entry(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        # SimpleMLP has 4 entries: fc1.weight, fc1.bias, fc2.weight, fc2.bias
        assert len(blocks) == len(layout.entries)

    def test_block_names_match_keys(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        for block, entry in zip(blocks, layout.entries):
            assert block.name == entry.key

    def test_flat_offsets_contiguous(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        # First block starts at 0
        assert blocks[0].flat_start == 0
        # Each block starts where the previous ends
        for i in range(1, len(blocks)):
            assert blocks[i].flat_start == blocks[i - 1].flat_end


class TestGroupedBlocks:
    @pytest.mark.parametrize("kwargs", [{}, {"group_size": 2}], ids=["default", "explicit"])
    def test_grouped_pairs(self, kwargs):
        """group_size defaults to 2, so both calls give the same 4-entry-into-2 split."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_grouped_blocks(layout, **kwargs)
        assert len(blocks) == 2
        assert [b.leaf_indices for b in blocks] == [(0, 1), (2, 3)]

    def test_grouped_leaf_indices(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_grouped_blocks(layout, group_size=2)
        assert blocks[0].leaf_indices == (0, 1)
        assert blocks[1].leaf_indices == (2, 3)

    def test_grouped_element_counts(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_grouped_blocks(layout, group_size=2)
        # First group: fc1.weight (4*8=32) + fc1.bias (8) = 40
        entries = layout.entries
        group0_numel = entries[0].numel + entries[1].numel
        padded0 = group0_numel + (-group0_numel % 2)
        assert blocks[0].flat_end - blocks[0].flat_start == padded0


class TestSplitReassemble:
    def test_split_correct_shapes(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        total_flat = sum(b.flat_end - b.flat_start for b in blocks)
        flat_vec = torch.randn(total_flat)
        block_parts = split_particles(flat_vec, blocks)
        assert len(block_parts) == len(blocks)
        for bp, block in zip(block_parts, blocks):
            assert bp.shape == (block.num_particles, block.particle_dim)

    def test_reassemble_roundtrip(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        total_flat = sum(b.flat_end - b.flat_start for b in blocks)
        original = torch.randn(total_flat)
        block_parts = split_particles(original, blocks)
        reconstructed = reassemble_blocks(block_parts, blocks, total_flat)
        torch.testing.assert_close(original, reconstructed)


class TestBlockLayoutConversion:
    """Tests for layout_flat_to_block_flat, blocks_to_layout_flat, and the column map."""

    @pytest.mark.parametrize(
        "make_blocks",
        [
            lambda layout: create_per_layer_blocks(layout, particle_dim=2),
            lambda layout: create_grouped_blocks(layout, group_size=2, particle_dim=2),
        ],
    )
    def test_per_layer_roundtrip(self, make_blocks):
        """Block factories: layout->block->layout is identity for real params."""
        from polystep.blockwise import layout_flat_to_block_flat, blocks_to_layout_flat

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=2)
        blocks = make_blocks(layout)

        flat = layout.flatten(model).reshape(-1)
        block_flat = layout_flat_to_block_flat(flat, blocks, layout)
        roundtrip = blocks_to_layout_flat(block_flat, blocks, layout)

        torch.testing.assert_close(flat, roundtrip)

    def test_per_layer_block_isolation(self):
        """Each per-layer block contains only its own layer's data."""
        from polystep.blockwise import layout_flat_to_block_flat

        # Model with misaligned sizes to stress-test padding
        class MisalignedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(13, 1, bias=False)  # 13 params
                self.fc2 = nn.Linear(5, 1, bias=False)  # 5 params

        model = MisalignedModel()
        with torch.no_grad():
            model.fc1.weight.fill_(1.0)
            model.fc2.weight.fill_(2.0)

        pdim = 8
        layout = ParamLayout.from_module(model, particle_dim=pdim)
        blocks = create_per_layer_blocks(layout, particle_dim=pdim)

        flat = layout.flatten(model).reshape(-1)
        block_flat = layout_flat_to_block_flat(flat, blocks, layout)

        # Block 0: 13 fc1 params + 3 padding zeros
        b0_data = block_flat[blocks[0].flat_start : blocks[0].flat_end]
        assert torch.all(b0_data[:13] == 1.0)
        assert torch.all(b0_data[13:] == 0.0)

        # Block 1: 5 fc2 params + 3 padding zeros
        b1_data = block_flat[blocks[1].flat_start : blocks[1].flat_end]
        assert torch.all(b1_data[:5] == 2.0)
        assert torch.all(b1_data[5:] == 0.0)

    @pytest.mark.parametrize(
        "make_blocks",
        [
            lambda layout: create_per_layer_blocks(layout, particle_dim=2),
            lambda layout: create_grouped_blocks(layout, group_size=2, particle_dim=2),
        ],
        ids=["per_layer", "grouped"],
    )
    def test_column_map_scatters_to_the_same_place_as_the_slice_copy(self, make_blocks):
        """The blockwise step scatters candidates straight into layout order.

        The column map has to agree with ``blocks_to_layout_flat``, which builds the
        same vector by slice copies, or a candidate lands on the wrong parameter and
        the cost matrix scores a configuration nobody asked for.
        """
        from polystep.blockwise import block_to_layout_columns

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=2)
        blocks = make_blocks(layout)
        block_flat = torch.randn(blocks[-1].flat_end)

        columns = block_to_layout_columns(blocks, layout, block_flat.device)
        # One past the end absorbs per-block padding, which has no layout counterpart.
        scattered = torch.zeros(layout.padded_size + 1)
        scattered.scatter_(0, columns, block_flat)

        torch.testing.assert_close(scattered[: layout.padded_size], blocks_to_layout_flat(block_flat, blocks, layout))

    def test_split_after_conversion_gives_correct_data(self):
        """split_particles on block-indexed data gives correct per-entry values."""
        from polystep.blockwise import layout_flat_to_block_flat

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(7, 1, bias=False)  # 7 params
                self.w2 = nn.Linear(3, 1, bias=False)  # 3 params

        model = TwoLayer()
        with torch.no_grad():
            model.w1.weight.copy_(torch.arange(7, dtype=torch.float32).view(1, 7))
            model.w2.weight.copy_(torch.arange(100, 103, dtype=torch.float32).view(1, 3))

        pdim = 2
        layout = ParamLayout.from_module(model, particle_dim=pdim)
        blocks = create_per_layer_blocks(layout, particle_dim=pdim)

        flat = layout.flatten(model).reshape(-1)
        block_flat = layout_flat_to_block_flat(flat, blocks, layout)
        block_2d = block_flat.reshape(-1, pdim)
        block_parts = split_particles(block_2d, blocks)

        # Block 0 should contain [0..6] + 1 padding zero
        assert block_parts[0].shape == (4, 2)
        b0_flat = block_parts[0].reshape(-1)
        torch.testing.assert_close(b0_flat[:7], torch.arange(7, dtype=torch.float32))
        assert b0_flat[7].item() == 0.0

        # Block 1 should contain [100, 101, 102] + 1 padding zero
        b1_flat = block_parts[1].reshape(-1)
        torch.testing.assert_close(
            b1_flat[:3],
            torch.arange(100, 103, dtype=torch.float32),
        )
        assert b1_flat[3].item() == 0.0
