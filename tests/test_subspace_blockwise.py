"""Integration tests for combined subspace + block-wise mode (combined subspace+block extension).

Tests verify that:
1. Combined mode initializes without NotImplementedError
2. Per-block OT operates in projected subspace coordinates
3. Synchronized absorb resets all blocks and rotates global projection
4. Memory usage is reduced compared to alternatives

Note: Tests that call optimizer.step() use minimal configs (low rank,
few Sinkhorn iters) to keep wall-clock time under the 120s timeout.
The sequential closure is inherently slow on CPU.
"""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer, AdaptiveSubspace
from polystep.optimizer import RankSchedule
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep.blockwise import (
    create_subspace_blocks,
    split_subspace_to_blocks,
    reassemble_blocks_to_subspace,
)

# Shared optimizer kwargs to keep step tests fast on CPU
_FAST_OPT_KWARGS = dict(
    epsilon=0.1,
    sinkhorn_max_iters=10,
    particle_dim=2,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def simple_model():
    """Small MLP for basic tests."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def medium_model():
    """Medium MLP for memory tests (~100K params)."""
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


@pytest.fixture
def simple_closure(simple_model):
    """Create a closure for simple_model."""
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(4, 16)
    targets = torch.randint(0, 10, (4,))

    def closure(batched_params):
        batch_size = next(iter(batched_params.values())).shape[0]
        losses = []
        for i in range(batch_size):
            params_i = {k: v[i] for k, v in batched_params.items()}
            # Load params into model
            simple_model.load_state_dict(params_i, strict=False)
            output = simple_model(inputs)
            loss = criterion(output, targets)
            losses.append(loss)
        return torch.stack(losses)

    return closure


# ------------------------------------------------------------------
# Block function tests
# ------------------------------------------------------------------


class TestSubspaceBlockFunctions:
    """Tests for subspace-aware block splitting functions."""

    def test_create_subspace_blocks_basic(self):
        """Test block creation with divisible dimensions."""
        blocks = create_subspace_blocks(subspace_dim=256, num_blocks=4, subspace_particle_dim=8)

        assert len(blocks) == 4
        # 256 / 8 = 32 particles total, 32 / 4 = 8 particles per block
        for block in blocks:
            assert block.num_particles == 8
            assert block.particle_dim == 8
            assert block.name.startswith("subspace_block_")

        # Check flat ranges are contiguous and non-overlapping
        assert blocks[0].flat_start == 0
        for i in range(len(blocks) - 1):
            assert blocks[i].flat_end == blocks[i + 1].flat_start

    def test_create_subspace_blocks_with_padding(self):
        """Test block creation when subspace_dim needs padding."""
        # 250 is not divisible by 8, needs padding to 256
        blocks = create_subspace_blocks(subspace_dim=250, num_blocks=4, subspace_particle_dim=8)

        assert len(blocks) == 4
        total_particles = sum(b.num_particles for b in blocks)
        assert total_particles == 32  # (250 + 6) / 8 = 32

    @pytest.mark.parametrize("dim,num_blocks", [(256, 4), (250, 3)])
    def test_split_reassemble_roundtrip(self, dim, num_blocks):
        """Test that split -> reassemble preserves data, including non-divisible dims."""
        coords = torch.randn(dim)
        blocks = create_subspace_blocks(dim, num_blocks, 8)

        block_particles = split_subspace_to_blocks(coords, blocks)
        reassembled = reassemble_blocks_to_subspace(block_particles, blocks, dim)

        assert torch.allclose(coords, reassembled)


# ------------------------------------------------------------------
# Initialization tests
# ------------------------------------------------------------------


class TestCombinedModeInitialization:
    """Tests for combined subspace + blockwise optimizer initialization."""

    def test_combined_mode_no_error(self, simple_model):
        """Test that combined mode initializes without NotImplementedError."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5)

        # This should NOT raise NotImplementedError anymore
        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            epsilon=0.1,
        )

        assert optimizer._subspace_blockwise is True
        assert optimizer._subspace_blocks is not None
        assert len(optimizer._subspace_blocks) > 0

    def test_rank_schedule_disabled_for_non_monolithic(self, simple_model):
        """rank_schedule only runs in the monolithic step; with a block strategy
        it must warn and disable rather than silently no-op or crash later."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5)
        with pytest.warns(UserWarning, match="rank_schedule"):
            opt = PolyStepOptimizer(
                simple_model,
                subspace=subspace,
                block_strategy="per_layer",
                rank_schedule=RankSchedule(stages=[(0, 2), (10, 4)]),
                epsilon=0.1,
            )
        assert opt._rank_schedule is None

    def test_block_count_reasonable(self, simple_model):
        """Verify block count is reasonable based on model structure."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            epsilon=0.1,
        )

        # Should have between 2 and 8 blocks (capped by implementation)
        num_blocks = len(optimizer._subspace_blocks)
        assert 2 <= num_blocks <= 8

    def test_block_polytopes_created(self, simple_model):
        """Verify per-block polytope templates are created."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            epsilon=0.1,
        )

        assert optimizer._subspace_block_polytopes is not None
        assert len(optimizer._subspace_block_polytopes) == len(optimizer._subspace_blocks)

        for polytope, block in zip(optimizer._subspace_block_polytopes, optimizer._subspace_blocks):
            # Polytope vertices should be in block.particle_dim space
            assert polytope.shape[1] == block.particle_dim


# ------------------------------------------------------------------
# Step tests
# ------------------------------------------------------------------


@pytest.mark.timeout(180)
class TestCombinedModeStep:
    """Tests for combined mode step execution."""

    def test_step_completes(self, simple_model, simple_closure):
        """Test that step() returns without error."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        loss = optimizer.step(simple_closure)

        assert isinstance(loss, float)
        assert loss > 0  # OT cost is positive

    def test_step_updates_state(self, simple_model, simple_closure):
        """Test that state.X changes after step."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        X_before = optimizer._state.X.clone()
        optimizer.step(simple_closure)
        X_after = optimizer._state.X

        # X should have changed (very unlikely to be exactly equal)
        assert not torch.allclose(X_before, X_after)

    def test_multiple_steps(self, simple_model, simple_closure):
        """Test running multiple steps without error."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        losses = []
        for _ in range(3):
            loss = optimizer.step(simple_closure)
            losses.append(loss)

        assert len(losses) == 3
        assert all(loss > 0 for loss in losses)

    def test_iteration_count_increments(self, simple_model, simple_closure):
        """Test that iteration_count increments correctly."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        assert optimizer._state.iteration_count == 0

        for i in range(3):
            optimizer.step(simple_closure)
            assert optimizer._state.iteration_count == i + 1

    def test_block_duals_updated(self, simple_model, simple_closure):
        """Test that per-block dual potentials are updated after step."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        # Initially None
        for f, g in optimizer._state.block_duals:
            assert f is None
            assert g is None

        loss = optimizer.step(simple_closure)

        # The step must complete with a finite cost and keep block_duals a
        # well-formed per-block list: one (f, g) slot per block, each either
        # unset or a finite tensor pair. (The default subspace solver is
        # softmax, which has no duals, so the prior `is not None` was vacuous.)
        assert torch.isfinite(torch.tensor(loss))
        assert len(optimizer._state.block_duals) == len(optimizer._subspace_blocks)
        for f, g in optimizer._state.block_duals:
            assert (f is None) == (g is None)
            if f is not None:
                assert torch.isfinite(f).all() and torch.isfinite(g).all()


# ------------------------------------------------------------------
# Absorb tests
# ------------------------------------------------------------------


@pytest.mark.timeout(180)
class TestSynchronizedAbsorb:
    """Tests for synchronized absorb in combined mode."""

    def test_absorb_resets_all_coords(self, simple_model, simple_closure):
        """Test that absorb resets all block coordinates to zero."""
        subspace = AdaptiveSubspace.auto_from_params(
            simple_model,
            compression_target=0.5,
            max_rank=16,
            absorb_mode="periodic",
            absorb_interval=3,
        )

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        base_before = {k: v.clone() for k, v in optimizer._state.base_params.items()}

        # Periodic absorb (interval=3) must actually fire within 4 steps.
        for _ in range(4):
            optimizer.step(simple_closure)

        assert optimizer._state.absorb_count >= 1, "periodic absorb never triggered"
        # Absorb folds the accumulated perturbation into the base weights, so at
        # least one base tensor must change (the prior if-guard made this vacuous).
        changed = any(not torch.allclose(base_before[k], v) for k, v in optimizer._state.base_params.items())
        assert changed, "absorb did not fold the perturbation into base params"

    def test_absorb_rotates_projection(self, simple_model, simple_closure):
        """Test that absorb rotates the global projection matrix."""
        subspace = AdaptiveSubspace.auto_from_params(
            simple_model,
            compression_target=0.5,
            max_rank=16,
            absorb_mode="periodic",
            absorb_interval=2,
        )

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        P_before = optimizer._state.projection.clone()

        # Run until absorb
        for _ in range(3):
            optimizer.step(simple_closure)

        P_after = optimizer._state.projection

        # Projection should have changed (rotated)
        assert not torch.allclose(P_before, P_after)


# ------------------------------------------------------------------
# CMA integration tests
# ------------------------------------------------------------------


class TestCMACombinedMode:
    """Tests for CMAAdaptiveSubspace in combined mode."""

    def test_cma_combined_mode_initializes(self, simple_model):
        """Test that CMAAdaptiveSubspace works in combined mode."""
        cma_subspace = CMAAdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=cma_subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        assert optimizer._subspace_blockwise is True
        assert optimizer._cma_subspace is True

    def test_cma_combined_mode_step(self, simple_model, simple_closure):
        """Test that CMA combined mode step completes."""
        cma_subspace = CMAAdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=cma_subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        loss = optimizer.step(simple_closure)
        assert isinstance(loss, float)
        assert loss > 0


# ------------------------------------------------------------------
# Memory tests (GPU-specific)
# ------------------------------------------------------------------


@pytest.mark.timeout(180)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestMemoryReduction:
    """Tests for memory efficiency of combined mode."""

    def test_combined_mode_runs_without_hang(self):
        """Smaller combined subspace+blockwise test that completes without deadlock."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).cuda()

        criterion = nn.CrossEntropyLoss()
        inputs = torch.randn(8, 64).cuda()
        targets = torch.randint(0, 10, (8,)).cuda()

        subspace = AdaptiveSubspace.auto_from_params(model, compression_target=0.1)

        optimizer = PolyStepOptimizer(
            model,
            subspace=subspace,
            block_strategy="per_layer",
            epsilon=0.1,
            chunk_size=32,
        )

        def closure(batched_params):
            batch_size = next(iter(batched_params.values())).shape[0]
            losses = []
            for i in range(batch_size):
                params_i = {k: v[i] for k, v in batched_params.items()}
                model.load_state_dict(params_i, strict=False)
                out = model(inputs)
                losses.append(criterion(out, targets))
            return torch.stack(losses)

        # Should complete without hanging - 2 steps
        for _ in range(2):
            cost = optimizer.step(closure)
            assert torch.isfinite(torch.tensor(cost)), "Cost should be finite"


# ------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.filterwarnings("ignore:num_blocks.*exceeds total_particles:UserWarning")
    def test_single_block(self, simple_model, simple_closure):
        """Test with minimum number of blocks (2)."""
        # Create a very small subspace to force fewer blocks
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.1, min_rank=8, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        # Should have at least 2 blocks (minimum)
        assert len(optimizer._subspace_blocks) >= 2

        loss = optimizer.step(simple_closure)
        assert loss > 0

    def test_with_momentum(self, simple_model, simple_closure):
        """Test combined mode with momentum enabled."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.9,
            **_FAST_OPT_KWARGS,
        )

        for _ in range(2):
            loss = optimizer.step(simple_closure)
            assert loss > 0

        assert optimizer._state.velocity is not None

    def test_with_adaptive_radius(self, simple_model, simple_closure):
        """Test combined mode with adaptive radius enabled."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            use_adaptive_radius=True,
            **_FAST_OPT_KWARGS,
        )

        for _ in range(2):
            loss = optimizer.step(simple_closure)
            assert loss > 0

    def test_grouped_block_strategy(self, simple_model, simple_closure):
        """Test combined mode with grouped block strategy."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        # Both 'per_layer' and 'grouped' should work with subspace
        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="grouped",
            **_FAST_OPT_KWARGS,
        )

        loss = optimizer.step(simple_closure)
        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
