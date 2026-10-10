"""Tests for block-wise Sinkhorn decomposition: block construction and layout mapping."""

import pytest
import torch
import torch.nn as nn

from polystep.blockwise import create_grouped_blocks, create_per_layer_blocks
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
        assert blocks[0].flat_start == 0
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
        entries = layout.entries
        group0_numel = entries[0].numel + entries[1].numel
        padded0 = group0_numel + (-group0_numel % 2)
        assert blocks[0].flat_end - blocks[0].flat_start == padded0


class TestBlockLayoutConversion:
    """Tests for layout_flat_to_block_flat and blocks_to_layout_flat."""

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
                self.fc1 = nn.Linear(13, 1, bias=False)
                self.fc2 = nn.Linear(5, 1, bias=False)

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

    def test_grouped_blocks_hold_their_entries_in_order(self):
        """A wrong but consistent column map still round-trips, so check the values."""
        from polystep.blockwise import layout_flat_to_block_flat

        model = SimpleMLP()
        layout = ParamLayout.from_module(model, particle_dim=3)
        blocks = create_grouped_blocks(layout, group_size=2, particle_dim=3)
        block_flat = layout_flat_to_block_flat(layout.flatten(model).reshape(-1), blocks, layout)

        sd = model.state_dict()
        for block in blocks:
            expected = torch.cat([sd[layout.entries[i].key].reshape(-1) for i in block.leaf_indices])
            n = expected.numel()
            assert torch.equal(block_flat[block.flat_start : block.flat_start + n], expected)
            assert torch.all(block_flat[block.flat_start + n : block.flat_end] == 0.0)

    def test_split_after_conversion_gives_correct_data(self):
        """Splitting block-indexed data by block gives correct per-entry values."""
        from polystep.blockwise import layout_flat_to_block_flat

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(7, 1, bias=False)
                self.w2 = nn.Linear(3, 1, bias=False)

        model = TwoLayer()
        with torch.no_grad():
            model.w1.weight.copy_(torch.arange(7, dtype=torch.float32).view(1, 7))
            model.w2.weight.copy_(torch.arange(100, 103, dtype=torch.float32).view(1, 3))

        pdim = 2
        layout = ParamLayout.from_module(model, particle_dim=pdim)
        blocks = create_per_layer_blocks(layout, particle_dim=pdim)

        flat = layout.flatten(model).reshape(-1)
        block_flat = layout_flat_to_block_flat(flat, blocks, layout)
        block_parts = block_flat.view(-1, pdim).split([b.num_particles for b in blocks])

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


def test_blockwise_sanitizes_before_averaging_probes_like_the_monolithic_driver():
    """Averaging first let one non-finite probe carry its whole vertex to the penalty.

    The two drivers then built different cost matrices from identical evaluations, and
    nothing in the suite ran both.
    """
    from types import SimpleNamespace

    from polystep._step_blockwise import _cost_from_losses

    # One particle, two vertices, three probes. Vertex 0's middle probe is +inf.
    losses = torch.tensor([1.0, float("inf"), 1.0, 2.0, 2.0, 2.0])
    cost = _cost_from_losses(SimpleNamespace(_all_nonfinite=None), losses, P=1, V=2, K=3)

    assert torch.isfinite(cost).all()
    # Penalty is 2 * max|finite| + 1 = 5. Vertex 0 must come back as a blend of its two
    # finite probes and that penalty, not as the penalty itself, which is what averaging
    # an inf first would give.
    assert 1.0 < cost[0, 0].item() < 5.0
    assert cost[0, 0].item() == pytest.approx((1.0 + 5.0 + 1.0) / 3)


def test_blockwise_radius_follows_the_best_vertex_mean(monkeypatch):
    """The radius controller reads the best-vertex mean while ``state.costs`` keeps the full mean."""
    from torch.func import functional_call, vmap

    import polystep._step_blockwise as step_blockwise
    from polystep import PolyStepOptimizer

    seen = []
    update = step_blockwise.update_stagnation_and_radius
    monkeypatch.setattr(
        step_blockwise,
        "update_stagnation_and_radius",
        lambda opt, state, loss: seen.append(loss) or update(opt, state, loss),
    )
    model = SimpleMLP()
    opt = PolyStepOptimizer(model, block_strategy="per_layer", solver="softmax", use_adaptive_radius=True, seed=0)
    x, y = torch.randn(16, 4), torch.randn(16, 2)

    def closure(params):
        return vmap(lambda p: (functional_call(model, p, (x,)) - y).pow(2).mean())(params)

    opt.step(closure)
    assert len(seen) == 1
    assert seen[0] < opt._state.costs[-1]
