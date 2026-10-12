"""Tests for block-wise Sinkhorn decomposition: block construction and layout mapping."""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.blockwise import create_grouped_blocks, create_per_layer_blocks
from polystep.cost_nn import NNCostEvaluator
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

    def test_grouped_blocks_hold_entries_in_order(self):
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


def test_blockwise_nonfinite_probe_marks_vertex_infeasible():
    """One non-finite probe makes the whole vertex infeasible, as in the monolithic driver.

    A blend of its finite probes and the penalty would let the solve still pick it.
    """
    from types import SimpleNamespace

    from polystep._step_blockwise import _cost_from_losses

    # One particle, two vertices, three probes. Vertex 0's middle probe is +inf.
    losses = torch.tensor([1.0, float("inf"), 1.0, 2.0, 2.0, 2.0])
    cost = _cost_from_losses(SimpleNamespace(_all_nonfinite=None), losses, P=1, V=2, K=3)

    assert cost.tolist() == [[float("inf"), 2.0]]


def test_blockwise_radius_follows_best_vertex_mean(monkeypatch):
    """``state.costs`` still records the full mean."""
    from torch.func import functional_call, vmap

    import polystep._step_blockwise as step_blockwise

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


class TestBlockwiseStepState:
    """Dual momentum, biased rotation and the amortized EMA must
    work in the blockwise step, not just monolithic."""

    def _make_simple_model(self):
        return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))

    def test_blockwise_transport_direction_ema_populated(self):
        """After a blockwise step with amortize_steps>1, _transport_direction_ema
        must be populated (not None) so momentum steps can fire."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            amortize_steps=2,
            amortize_ema=0.7,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        # First step triggers a full OT.
        optimizer.step(closure)
        ema = optimizer._transport_direction_ema
        assert ema is not None, "the amortized momentum step has nothing to coast along"
        # A populated-but-zero EMA makes every cheap step a no-op, which the
        # None check cannot see.
        assert ema.shape == optimizer.state.X.shape
        assert ema.abs().max() > 0, "the EMA is all zeros, so a momentum step would not move anything"

        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step(closure)
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), (
            "the amortized step did not move the model"
        )

    def test_blockwise_biased_rotation_descent_dirs_populated(self):
        """After a blockwise step with biased_rotation=True,
        _prev_block_descent_directions must be populated."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            biased_rotation=True,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        optimizer.step(closure)
        dirs = getattr(optimizer, "_prev_block_descent_directions", None)
        assert dirs is not None, "the next step has no heading to bias its rotation toward"
        assert len(dirs) == len(optimizer._blocks), "one descent direction per block"
        # A list of zero vectors carries no heading, and apply_biased_rotation routes
        # those particles to the unbiased fallback.
        assert any(d is not None and torch.linalg.vector_norm(d) > 0 for d in dirs), (
            "every block descent direction is zero, so the bias is inert"
        )

    def test_blockwise_dual_momentum_prev_duals_populated(self):
        """After 2 blockwise steps with dual_momentum_beta>0,
        _prev_prev_block_duals must be populated for extrapolation."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            dual_momentum_beta=0.3,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        optimizer.step(closure)
        optimizer.step(closure)
        ppbd = getattr(optimizer._state, "_prev_prev_block_duals", None)
        assert ppbd is not None, (
            "After 2 blockwise steps with dual_momentum_beta>0, _prev_prev_block_duals should be populated"
        )
        assert len(ppbd) > 0
        has_duals = any(f is not None for f, g in ppbd)
        assert has_duals, "At least one block should have previous duals"


def test_zero_numel_parameter_gets_no_block():
    """A (0, V) cost matrix would divide by zero in the marginal or give a NaN mean."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 0))
    blocks = create_per_layer_blocks(ParamLayout.from_module(model), particle_dim=2)
    assert blocks, "the non-empty layer must still get a block"
    assert all(b.num_particles > 0 for b in blocks)
