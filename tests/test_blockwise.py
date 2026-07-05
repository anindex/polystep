"""Tests for block-wise Sinkhorn decomposition and solver integration."""

import pytest
import torch
import torch.nn as nn

from polystep.blockwise import (
    BlockConfig,
    create_per_layer_blocks,
    create_grouped_blocks,
    split_particles,
    reassemble_blocks,
    compute_block_cost_matrix,
)
from polystep.transform import ParamLayout
from polystep.cost_nn import NNCostEvaluator
from polystep.solver import PolyStep, SolverState


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# BlockConfig tests
# ---------------------------------------------------------------------------


class TestBlockConfig:
    def test_block_config_frozen(self):
        bc = BlockConfig(
            name="test",
            leaf_indices=(0,),
            flat_start=0,
            flat_end=10,
            num_particles=5,
            particle_dim=2,
        )
        assert bc.name == "test"
        assert bc.num_particles == 5
        with pytest.raises(AttributeError):
            bc.name = "modified"


# ---------------------------------------------------------------------------
# create_per_layer_blocks tests
# ---------------------------------------------------------------------------


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

    def test_padding_correct(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particle_dim = 2
        blocks = create_per_layer_blocks(layout, particle_dim=particle_dim)
        for block, entry in zip(blocks, layout.entries):
            padded = entry.numel + (-entry.numel % particle_dim)
            assert block.flat_end - block.flat_start == padded
            assert block.num_particles == padded // particle_dim


# ---------------------------------------------------------------------------
# create_grouped_blocks tests
# ---------------------------------------------------------------------------


class TestGroupedBlocks:
    def test_grouped_pairs(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_grouped_blocks(layout, group_size=2)
        # 4 entries grouped by 2 -> 2 blocks
        assert len(blocks) == 2

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


# ---------------------------------------------------------------------------
# split_particles / reassemble_blocks tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-block dual potentials tests
# ---------------------------------------------------------------------------


class TestBlockDuals:
    def test_solver_state_block_duals_init(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout)
        total_flat = sum(b.flat_end - b.flat_start for b in blocks)
        X = torch.randn(total_flat // 2, 2)
        state = SolverState(X=X, block_duals=[(None, None) for _ in blocks])
        assert len(state.block_duals) == len(blocks)
        for f, g in state.block_duals:
            assert f is None
            assert g is None


# ---------------------------------------------------------------------------
# compute_block_cost_matrix test
# ---------------------------------------------------------------------------


class TestBlockCost:
    def test_block_cost_shape(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        blocks = create_per_layer_blocks(layout, particle_dim=layout.particle_dim)
        total_flat = sum(b.flat_end - b.flat_start for b in blocks)
        flat_vec = torch.randn(total_flat)
        all_block_parts = split_particles(flat_vec, blocks)

        loss_fn = nn.MSELoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 4)
        targets = torch.randn(4, 2)

        # Create fake probe for block 0
        block = blocks[0]
        P, V, K = block.num_particles, 5, 3  # 5 vertices, 3 probes
        X_probe = torch.randn(P, V, K, block.particle_dim)

        cost = compute_block_cost_matrix(
            block_idx=0,
            X_probe_block=X_probe,
            all_block_particles=all_block_parts,
            blocks=blocks,
            layout=layout,
            evaluator=evaluator,
            inputs=inputs,
            targets=targets,
        )
        assert cost.shape == (P, V)
        assert torch.isfinite(cost).all()


# ---------------------------------------------------------------------------
# PolyStep block-wise integration test
# ---------------------------------------------------------------------------


class TestBlockwiseSolverIntegration:
    def test_blockwise_solver_integration(self):
        """End-to-end test: block-wise mode runs on synthetic NN objective."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        loss_fn = nn.MSELoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(8, 4)
        targets = torch.randn(8, 2)

        # Create solver with block-wise per-layer mode
        # dim = particle_dim since particles are block-level
        # For block-wise, the full flat is split per block, so dim is layout.particle_dim
        solver = PolyStep(
            objective_fn=lambda x: x.sum(),  # placeholder, not used in block mode
            dim=layout.particle_dim,
            block_strategy="per_layer",
            nn_evaluator=evaluator,
            layout=layout,
            train_inputs=inputs,
            train_targets=targets,
            compile=False,
            epsilon=0.5,
            num_probe=2,
            sinkhorn_max_iters=50,
        )

        # Initialize particles from model
        X_init = layout.flatten(model)
        state = solver.init_state(X_init)

        # Verify block_duals initialized
        assert state.block_duals is not None
        assert len(state.block_duals) == len(layout.entries)

        # Run a step
        state = solver.step(state)

        # Verify state updated
        assert state.iteration_count == 1
        assert len(state.costs) == 1
        assert state.block_duals is not None
        # Block duals should now have actual tensors
        for f_b, g_b in state.block_duals:
            assert f_b is not None
            assert g_b is not None

    def test_blockwise_grouped_mode(self):
        """Block-wise grouped mode runs end-to-end."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        loss_fn = nn.MSELoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(8, 4)
        targets = torch.randn(8, 2)

        solver = PolyStep(
            objective_fn=lambda x: x.sum(),
            dim=layout.particle_dim,
            block_strategy="grouped",
            block_group_size=2,
            nn_evaluator=evaluator,
            layout=layout,
            train_inputs=inputs,
            train_targets=targets,
            compile=False,
            epsilon=0.5,
            num_probe=2,
            sinkhorn_max_iters=50,
        )

        X_init = layout.flatten(model)
        state = solver.init_state(X_init)
        state = solver.step(state)

        assert state.iteration_count == 1
        assert len(state.block_duals) == 2  # 4 entries / group_size 2 = 2 blocks

    def test_subspace_plus_blockwise_raises(self):
        """Combined subspace + block-wise should raise NotImplementedError."""
        from polystep.subspace import LowRankSubspace

        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        subspace = LowRankSubspace.from_layout(layout, rank=4)

        with pytest.raises(NotImplementedError):
            PolyStep(
                objective_fn=lambda x: x.sum(),
                dim=layout.particle_dim,
                block_strategy="per_layer",
                subspace=subspace,
                layout=layout,
                compile=False,
            )

    def test_invalid_block_strategy_raises(self):
        """Unknown block_strategy should raise ValueError."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        with pytest.raises(ValueError, match="Unknown block_strategy"):
            PolyStep(
                objective_fn=lambda x: x.sum(),
                dim=layout.particle_dim,
                block_strategy="invalid",
                layout=layout,
                compile=False,
            )


# ---------------------------------------------------------------------------
# Layout ↔ block conversion tests
# ---------------------------------------------------------------------------


class TestBlockLayoutConversion:
    """Tests for layout_flat_to_block_flat, blocks_to_layout_flat, and batch variant."""

    def test_per_layer_roundtrip(self):
        """Per-layer blocks: layout->block->layout is identity for real params."""
        from polystep.blockwise import layout_flat_to_block_flat, blocks_to_layout_flat

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=2)
        blocks = create_per_layer_blocks(layout, particle_dim=2)

        flat = layout.flatten(model).reshape(-1)
        block_flat = layout_flat_to_block_flat(flat, blocks, layout)
        roundtrip = blocks_to_layout_flat(block_flat, blocks, layout)

        torch.testing.assert_close(flat, roundtrip)

    def test_grouped_roundtrip(self):
        """Grouped blocks: layout->block->layout is identity for real params."""
        from polystep.blockwise import layout_flat_to_block_flat, blocks_to_layout_flat

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=2)
        blocks = create_grouped_blocks(layout, group_size=2, particle_dim=2)

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

    def test_batch_conversion_matches_unbatched(self):
        """Batch conversion produces same result as looping over single conversion."""
        from polystep.blockwise import (
            layout_flat_to_block_flat,
            blocks_to_layout_flat,
            blocks_to_layout_flat_batch,
        )

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=2)
        blocks = create_per_layer_blocks(layout, particle_dim=2)

        # Create batch of 4 different block-indexed flat vectors
        batch_size = 4
        total_block_flat = blocks[-1].flat_end
        block_batch = torch.randn(batch_size, total_block_flat)

        layout_batch = blocks_to_layout_flat_batch(block_batch, blocks, layout)

        for i in range(batch_size):
            single = blocks_to_layout_flat(block_batch[i], blocks, layout)
            torch.testing.assert_close(layout_batch[i], single)

    def test_split_after_conversion_gives_correct_data(self):
        """split_particles on block-indexed data gives correct per-entry values."""
        from polystep.blockwise import layout_flat_to_block_flat, blocks_to_layout_flat

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
