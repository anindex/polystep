"""Integration tests for combined subspace + block-wise mode (combined subspace+block extension).

Tests verify that:
1. Combined mode initializes without NotImplementedError
2. Per-block OT operates in projected subspace coordinates
3. Synchronized absorb resets all blocks and rotates global projection
4. Memory usage is reduced compared to alternatives

Tests that call optimizer.step() use minimal configs (low rank,
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


class TestSubspaceBlockFunctions:
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


class TestCombinedModeInitialization:
    def test_combined_mode_builds_subspace_blocks(self, simple_model):
        """Combined subspace + blockwise splits the subspace into usable blocks."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5)

        # This should NOT raise NotImplementedError anymore
        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            epsilon=0.1,
        )

        # The blocks must partition the subspace coordinates exactly once, or a step
        # would leave some coordinates unoptimized or update others twice.
        blocks = optimizer._subspace_blocks
        assert [b.flat_start for b in blocks] == [0] + [b.flat_end for b in blocks[:-1]]
        assert blocks[-1].flat_end >= subspace.subspace_dim

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


@pytest.mark.timeout(180)
class TestCombinedModeStep:
    def test_step_updates_state(self, simple_model, simple_closure):
        """Test that the represented point moves after a step.

        Checks model parameters rather than ``state.X``: rotation re-anchors the
        coordinate origin, folding coords into ``base_params`` and zeroing ``X``, so
        ``X`` is 0 both before and after while the weights do move.
        """
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )

        params_before = [p.detach().clone() for p in simple_model.parameters()]
        optimizer.step(simple_closure)

        assert any(not torch.allclose(a, b) for a, b in zip(params_before, simple_model.parameters()))

    def test_block_duals_updated(self, simple_model, simple_closure):
        """Per-block dual potentials are populated after a step by a solver that has them.

        Pinned to sinkhorn: under the default softmax there are no duals at all, so
        every slot stays None and the assertions below never run. rotation_interval=0
        for the same reason: a basis change invalidates them right after the solve.
        """
        subspace = AdaptiveSubspace.auto_from_params(
            simple_model, compression_target=0.5, max_rank=16, rotation_interval=0
        )

        kwargs = {**_FAST_OPT_KWARGS, "solver": "sinkhorn"}
        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **kwargs,
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
        # Sinkhorn has duals, so at least one block must carry a pair.
        assert any(f is not None for f, _ in optimizer._state.block_duals)


@pytest.mark.timeout(180)
class TestSynchronizedAbsorb:
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


def test_cma_combined_mode_steps(simple_model, simple_closure):
    """CMAAdaptiveSubspace runs a combined subspace + blockwise step."""
    cma_subspace = CMAAdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

    optimizer = PolyStepOptimizer(
        simple_model,
        subspace=cma_subspace,
        block_strategy="per_layer",
        **_FAST_OPT_KWARGS,
    )

    # Construction alone proves nothing; the combined mode has to take a step that
    # moves the model. The coordinates themselves can come back to zero, because an
    # absorb folds them into the base and re-anchors the origin.
    before = torch.cat([p.detach().reshape(-1) for p in simple_model.parameters()]).clone()
    optimizer.step(simple_closure)
    assert optimizer.state.iteration_count == 1
    assert not torch.equal(before, torch.cat([p.detach().reshape(-1) for p in simple_model.parameters()]))


@pytest.mark.parametrize("interval, expected_rotations", [(1, 6), (3, 2), (0, 0)])
def test_global_subspace_honours_rotation_interval(simple_model, simple_closure, interval, expected_rotations):
    """The global-subspace branch rotated every step regardless of the interval."""
    sub = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, min_rank=8, max_rank=16)
    sub.rotation_interval = interval
    sub.absorb_mode = "interval"
    sub.absorb_interval = 0

    opt = PolyStepOptimizer(simple_model, subspace=sub, block_strategy="per_layer", **_FAST_OPT_KWARGS)

    seen = [opt.state.projection.clone()]
    for _ in range(6):
        opt.step(simple_closure)
        seen.append(opt.state.projection.clone())

    changed = sum(not torch.equal(a, b) for a, b in zip(seen, seen[1:]))
    assert changed == expected_rotations


@pytest.mark.timeout(180)
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_combined_mode_runs_without_hang():
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


class TestEdgeCases:
    def test_with_momentum(self, simple_model, simple_closure):
        """Test combined mode with momentum enabled."""
        # A basis change zeroes the velocity, which is a displacement in the basis it
        # replaces. AdaptiveSubspace rotates every step by default, so hold the basis
        # still or there is never a second step for momentum to accumulate over.
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)
        subspace.absorb_mode = "none"
        subspace.rotation_interval = 0

        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.9,
            **_FAST_OPT_KWARGS,
        )

        params_before = [p.detach().clone() for p in simple_model.parameters()]
        for _ in range(2):
            optimizer.step(simple_closure)

        assert optimizer._state.velocity is not None
        assert torch.linalg.vector_norm(optimizer._state.velocity) > 0, "momentum never accumulated"
        assert any(not torch.allclose(a, b) for a, b in zip(params_before, simple_model.parameters()))

    def test_grouped_block_strategy_warns_and_behaves_like_per_layer(self, simple_model, simple_closure):
        """In subspace mode the blocks slice coordinates, so 'grouped' cannot group anything.

        It produces exactly the per_layer blocks, so accepting it silently would hide
        that the requested grouping did not happen.
        """
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        with pytest.warns(UserWarning, match="no effect in subspace mode"):
            grouped = PolyStepOptimizer(
                simple_model,
                subspace=subspace,
                block_strategy="grouped",
                **_FAST_OPT_KWARGS,
            )
        per_layer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            **_FAST_OPT_KWARGS,
        )
        assert len(grouped._subspace_blocks) == len(per_layer._subspace_blocks)

        params_before = [p.detach().clone() for p in simple_model.parameters()]
        grouped.step(simple_closure)
        assert any(not torch.allclose(a, b) for a, b in zip(params_before, simple_model.parameters()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_blockwise_keeps_the_displacement_history_in_full_space():
    """The basis rotates every step, so subspace-coordinate history mixes frames.

    Both drivers must store the full-space image under the basis in use at measurement
    time, or the rotation SVD reads old displacements along directions never measured.
    """
    from torch.func import functional_call

    from polystep.adaptive_subspace import AdaptiveSubspace
    from polystep.optimizer import PolyStepOptimizer

    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
    full_dim = sum(p.numel() for p in model.parameters())
    sub = AdaptiveSubspace.auto_from_params(model, compression_target=0.15)
    sub.rotation_mode = "displacement"
    opt = PolyStepOptimizer(model, subspace=sub, block_strategy="per_layer", epsilon=0.1, seed=0, compile=False)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))

    def closure(batched_params):
        n = next(iter(batched_params.values())).shape[0]
        return torch.stack(
            [functional_call(model, {k: v[i] for k, v in batched_params.items()}, (x,)).pow(2).mean() for i in range(n)]
        )

    for _ in range(3):
        opt.step(closure)

    hist_full = opt.state.displacement_history_full
    assert hist_full is not None, "block-wise step kept no full-space history"
    assert hist_full.shape[1] == full_dim
    assert hist_full[: opt.state.displacement_history_count].abs().sum() > 0


def test_blockwise_sanitizes_before_averaging_probes_like_the_monolithic_driver():
    """Averaging first let one non-finite probe carry its whole vertex to the penalty.

    The two drivers then built different cost matrices from identical evaluations, and
    nothing in the suite ran both.
    """
    from polystep._step_blockwise import _cost_from_losses

    # One particle, two vertices, three probes. Vertex 0's middle probe is +inf.
    losses = torch.tensor([1.0, float("inf"), 1.0, 2.0, 2.0, 2.0])
    cost = _cost_from_losses(losses, P=1, V=2, K=3)

    assert torch.isfinite(cost).all()
    # Penalty is 2 * max|finite| + 1 = 5. Vertex 0 must come back as a blend of its two
    # finite probes and that penalty, not as the penalty itself, which is what averaging
    # an inf first would give.
    assert 1.0 < cost[0, 0].item() < 5.0
    assert cost[0, 0].item() == pytest.approx((1.0 + 5.0 + 1.0) / 3)
