"""Tests for HybridSubspace: per-layer projections with synchronized rotation."""

import pytest
import torch
import torch.nn as nn

from polystep.hybrid_subspace import HybridSubspace, LayerProjectionSpec, create_hybrid_blocks
from polystep.optimizer import RankSchedule
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# Helper model fixture
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SimpleMLP()


@pytest.fixture
def layout(model):
    return ParamLayout.from_module(model)


@pytest.fixture
def hybrid_sub(layout):
    return HybridSubspace.from_layout(layout, rank=4)


# ---------------------------------------------------------------------------
# from_layout factory method
# ---------------------------------------------------------------------------


class TestHybridFromLayout:
    def test_from_layout_creates_correct_specs(self, layout):
        """from_layout creates LayerProjectionSpecs matching layout entries."""
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        # Should have one spec per layout entry
        assert len(hybrid.specs) == len(layout.entries)

        # Each spec should have correct entry_key
        spec_keys = [s.entry_key for s in hybrid.specs]
        entry_keys = [e.key for e in layout.entries]
        assert spec_keys == entry_keys

        # Specs should be contiguous
        prev_end = 0
        for spec in hybrid.specs:
            assert spec.flat_start == prev_end
            assert spec.flat_end > spec.flat_start
            prev_end = spec.flat_end

        # Total subspace dim should equal sum of num_coords
        total = sum(s.num_coords for s in hybrid.specs)
        assert hybrid.subspace_dim == total

    def test_from_layout_2d_params_are_projected(self, layout):
        """2D+ params (weights) have is_projected=True."""
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        for spec, entry in zip(hybrid.specs, layout.entries):
            if len(entry.shape) >= 2:
                assert spec.is_projected is True
            else:
                assert spec.is_projected is False

    def test_from_layout_1d_params_have_identity_coords(self, layout):
        """1D params (biases) have num_params == num_coords."""
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        for spec, entry in zip(hybrid.specs, layout.entries):
            if len(entry.shape) == 1:
                assert spec.num_params == spec.num_coords
                assert spec.num_params == entry.numel


# ---------------------------------------------------------------------------
# auto_from_layout factory method
# ---------------------------------------------------------------------------


class TestHybridAutoFromLayout:
    def test_auto_from_layout_creates_reasonable_ranks(self, layout):
        """auto_from_layout selects ranks within min/max bounds."""
        hybrid = HybridSubspace.auto_from_layout(
            layout,
            min_rank=2,
            max_rank=16,
        )

        # Should have one spec per layout entry
        assert len(hybrid.specs) == len(layout.entries)

        # Compression ratio should be > 0 and <= 1
        assert hybrid.compression_ratio > 0
        assert hybrid.compression_ratio <= 1.0

    def test_auto_from_layout_smaller_than_fixed_rank(self, layout):
        """auto_from_layout with small max_rank gives smaller subspace."""
        hybrid_fixed = HybridSubspace.from_layout(layout, rank=16)
        hybrid_auto = HybridSubspace.auto_from_layout(layout, max_rank=4)

        # Auto with max_rank=4 should be smaller than fixed rank=16
        assert hybrid_auto.subspace_dim <= hybrid_fixed.subspace_dim


# ---------------------------------------------------------------------------
# init_projections
# ---------------------------------------------------------------------------


class TestHybridInitProjections:
    def test_init_projections_creates_correct_shapes(self, hybrid_sub):
        """init_projections creates one projection per projected spec; 1D params
        add coords directly and get no (O(n^2)) projection matrix."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)

        projected = [s for s in hybrid_sub.specs if s.is_projected]
        assert len(projections) == len(projected)

        for spec in projected:
            assert projections[spec.entry_key].shape == (spec.num_params, spec.num_coords)
        for spec in hybrid_sub.specs:
            if not spec.is_projected:
                assert spec.entry_key not in projections

    def test_init_projections_has_correct_scaling(self, hybrid_sub):
        """Projection columns have unit norm (QR-orthogonal) or 1/sqrt(N) scaling."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)

        for spec in hybrid_sub.specs:
            if spec.is_projected:
                P = projections[spec.entry_key]
                if spec.num_params >= spec.num_coords:
                    # QR path: columns should have unit norm
                    col_norms = torch.norm(P, dim=0)
                    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), (
                        f"QR columns should have unit norm, got {col_norms}"
                    )
                else:
                    # Scaled Gaussian fallback
                    expected_std = 1.0 / (spec.num_coords**0.5)
                    actual_std = P.std().item()
                    assert abs(actual_std - expected_std) < 0.1 * expected_std


def test_apply_inplace_updates_noncontiguous_param(model, hybrid_sub):
    """A non-contiguous param.data makes reshape(1,-1) a copy, so an out= write
    would be lost. apply_perturbation_inplace must still modify the parameter."""
    p = dict(model.named_parameters())["fc1.weight"]
    p.data = p.data.t().contiguous().t()  # (10, 20) non-contiguous view
    assert not p.data.is_contiguous()

    base_sd = {k: v.detach().clone() for k, v in model.named_parameters()}
    projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
    coords = torch.ones(hybrid_sub.subspace_dim) * 0.5
    before = p.data.clone()
    hybrid_sub.apply_perturbation_inplace(projections, model, base_sd, coords)
    assert not torch.allclose(p.data, before)


# ---------------------------------------------------------------------------
# apply_perturbation
# ---------------------------------------------------------------------------


class TestHybridApplyPerturbation:
    def test_apply_perturbation_matches_manual(self, model, hybrid_sub):
        """apply_perturbation matches manual P @ coords computation."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        coords = torch.randn(hybrid_sub.subspace_dim) * 0.01

        # Method result
        result = hybrid_sub.apply_perturbation(projections, base_sd, coords)

        # Manual computation
        for spec in hybrid_sub.specs:
            chunk = coords[spec.flat_start : spec.flat_end]
            base = base_sd[spec.entry_key]
            if spec.is_projected:
                P = projections[spec.entry_key]
                expected = base + (P @ chunk).reshape(spec.original_shape)
            else:
                expected = base + chunk.reshape(spec.original_shape)
            assert torch.allclose(result[spec.entry_key], expected, atol=1e-6)

    def test_apply_perturbation_zero_is_identity(self, model, hybrid_sub):
        """Zero perturbation returns base params unchanged."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()
        coords = torch.zeros(hybrid_sub.subspace_dim)

        result = hybrid_sub.apply_perturbation(projections, base_sd, coords)

        for key in base_sd:
            assert torch.allclose(result[key], base_sd[key], atol=1e-6)


# ---------------------------------------------------------------------------
# reconstruct_batch
# ---------------------------------------------------------------------------


class TestHybridReconstructBatch:
    def test_reconstruct_batch_matches_loop(self, model, hybrid_sub):
        """reconstruct_batch gives same result as looping apply_perturbation."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        N = 4
        torch.manual_seed(42)
        batch = torch.randn(N, hybrid_sub.subspace_dim) * 0.01

        batch_result = hybrid_sub.reconstruct_batch(projections, base_sd, batch)

        for i in range(N):
            single_result = hybrid_sub.apply_perturbation(projections, base_sd, batch[i])
            for key in single_result:
                assert torch.allclose(batch_result[key][i], single_result[key], atol=1e-5), (
                    f"Row {i}, key {key}: batch vs single mismatch"
                )

    def test_reconstruct_batch_shapes(self, model, hybrid_sub):
        """reconstruct_batch produces (N, *shape) tensors."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        N = 3
        batch = torch.randn(N, hybrid_sub.subspace_dim) * 0.01
        result = hybrid_sub.reconstruct_batch(projections, base_sd, batch)

        for spec in hybrid_sub.specs:
            expected_shape = (N, *spec.original_shape)
            assert result[spec.entry_key].shape == expected_shape


# ---------------------------------------------------------------------------
# absorb
# ---------------------------------------------------------------------------


class TestHybridAbsorb:
    def test_absorb_zeros_subspace(self, model, hybrid_sub):
        """After absorb, coords are zero and base_sd is updated."""
        projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        coords = torch.randn(hybrid_sub.subspace_dim) * 0.01

        # Expected base after absorb
        expected_sd = hybrid_sub.apply_perturbation(projections, base_sd, coords)

        new_base, zeroed = hybrid_sub.absorb(projections, base_sd, coords)

        # Zeroed coords
        assert torch.all(zeroed == 0)
        assert zeroed.shape == coords.shape

        # New base matches expected
        for key in expected_sd:
            assert torch.allclose(new_base[key], expected_sd[key], atol=1e-6)


# ---------------------------------------------------------------------------
# rotate_all random mode
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:HybridSubspace works best:UserWarning")
class TestHybridRotateRandom:
    def test_rotate_random_produces_different_projections(self, hybrid_sub):
        """Random rotation produces different projections."""
        hybrid = HybridSubspace(
            specs=hybrid_sub.specs,
            subspace_dim=hybrid_sub.subspace_dim,
            compression_ratio=hybrid_sub.compression_ratio,
            rotation_mode="random",
            rotation_interval=1,
            _total_params=hybrid_sub._total_params,
        )

        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        new_projections = hybrid.rotate_all(projections, step=1, total_steps=100)

        # At least one projection should be different
        any_different = False
        for key in projections:
            if not torch.allclose(projections[key], new_projections[key], atol=1e-3):
                any_different = True
                break
        assert any_different, "Rotated projections should differ from original"


# ---------------------------------------------------------------------------
# rotate_all displacement mode
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:HybridSubspace works best:UserWarning")
class TestHybridRotateDisplacement:
    def test_rotate_displacement_with_history(self, hybrid_sub):
        """Displacement rotation with non-zero history produces new projections."""
        # Need rotation_interval=1 to actually trigger rotation
        hybrid = HybridSubspace(
            specs=hybrid_sub.specs,
            subspace_dim=hybrid_sub.subspace_dim,
            compression_ratio=hybrid_sub.compression_ratio,
            rotation_mode="displacement",
            rotation_interval=1,
            _total_params=hybrid_sub._total_params,
        )
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)

        torch.manual_seed(77)
        disp_history = torch.randn(3, hybrid.subspace_dim) * 0.1

        new_projections = hybrid.rotate_all(
            projections,
            step=5,
            total_steps=100,
            displacement_history=disp_history,
        )

        # Should produce different projections
        any_different = False
        for key in projections:
            if hybrid.specs[list(projections.keys()).index(key)].is_projected:
                if not torch.allclose(projections[key], new_projections[key], atol=1e-3):
                    any_different = True
                    break
        assert any_different

    def test_rotate_displacement_wide_layer_keeps_shape(self, hybrid_sub):
        """When num_coords > num_params, reduced QR would drop columns; the
        rotated projection must keep shape (num_params, num_coords)."""
        spec = LayerProjectionSpec(
            entry_key="w",
            original_shape=(3, 7),
            num_params=21,
            num_coords=30,
            flat_start=0,
            flat_end=30,
            is_projected=True,
        )
        P_old = torch.randn(21, 30)
        disp = torch.randn(4, 30) * 0.1
        new_P = hybrid_sub._rotate_layer_displacement(
            P_old,
            spec,
            disp,
            svd_ratio=0.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
            step=1,
        )
        assert new_P.shape == (21, 30)

    def test_rotate_displacement_tall_layer_unit_norm_columns(self, hybrid_sub):
        """Tall layers keep unit-norm columns after rotation, matching init, so
        there is no sqrt(N) magnitude jump on the first rotation."""
        spec = LayerProjectionSpec(
            entry_key="w",
            original_shape=(20, 10),
            num_params=200,
            num_coords=8,
            flat_start=0,
            flat_end=8,
            is_projected=True,
        )
        P_old = torch.randn(200, 8)
        disp = torch.randn(4, 8) * 0.1
        new_P = hybrid_sub._rotate_layer_displacement(
            P_old,
            spec,
            disp,
            svd_ratio=0.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
            step=1,
        )
        col_norms = torch.norm(new_P, dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4)

    def test_rotate_displacement_zero_history_falls_back(self, hybrid_sub):
        """Zero displacement history falls back to random rotation."""
        # Need rotation_interval=1 to actually trigger rotation
        hybrid = HybridSubspace(
            specs=hybrid_sub.specs,
            subspace_dim=hybrid_sub.subspace_dim,
            compression_ratio=hybrid_sub.compression_ratio,
            rotation_mode="displacement",
            rotation_interval=1,
            _total_params=hybrid_sub._total_params,
        )
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        disp_history = torch.zeros(3, hybrid.subspace_dim)

        new_projections = hybrid.rotate_all(
            projections,
            step=5,
            total_steps=100,
            displacement_history=disp_history,
        )

        # Should still produce different projections (random fallback)
        any_different = False
        for key in projections:
            if hybrid.specs[list(projections.keys()).index(key)].is_projected:
                if not torch.allclose(projections[key], new_projections[key], atol=1e-3):
                    any_different = True
                    break
        assert any_different


# ---------------------------------------------------------------------------
# should_absorb
# ---------------------------------------------------------------------------


class TestHybridShouldAbsorb:
    def test_should_absorb_stagnation(self):
        """stagnation_count >= absorb_patience returns True."""
        hybrid = HybridSubspace(
            specs=(),
            subspace_dim=10,
            compression_ratio=0.1,
            absorb_mode="stagnation",
            absorb_patience=20,
        )
        assert not hybrid.should_absorb(stagnation_count=19, iteration=50)
        assert hybrid.should_absorb(stagnation_count=20, iteration=50)
        assert hybrid.should_absorb(stagnation_count=25, iteration=50)

    def test_should_absorb_periodic(self):
        """iteration % absorb_interval == 0 returns True (for iteration > 0)."""
        hybrid = HybridSubspace(
            specs=(),
            subspace_dim=10,
            compression_ratio=0.1,
            absorb_mode="periodic",
            absorb_interval=10,
        )
        assert not hybrid.should_absorb(stagnation_count=0, iteration=0)
        assert not hybrid.should_absorb(stagnation_count=0, iteration=5)
        assert hybrid.should_absorb(stagnation_count=0, iteration=10)
        assert hybrid.should_absorb(stagnation_count=0, iteration=20)

    def test_should_absorb_periodic_disabled(self):
        """absorb_interval=0 with periodic mode never triggers."""
        hybrid = HybridSubspace(
            specs=(),
            subspace_dim=10,
            compression_ratio=0.1,
            absorb_mode="periodic",
            absorb_interval=0,
        )
        assert not hybrid.should_absorb(stagnation_count=100, iteration=100)


# ---------------------------------------------------------------------------
# create_hybrid_blocks
# ---------------------------------------------------------------------------


class TestCreateHybridBlocks:
    def test_create_hybrid_blocks_correct_count(self, hybrid_sub):
        """create_hybrid_blocks creates one block per spec."""
        blocks = create_hybrid_blocks(hybrid_sub, particle_dim=8)
        assert len(blocks) == len(hybrid_sub.specs)

    def test_create_hybrid_blocks_names_match_specs(self, hybrid_sub):
        """Block names match spec entry_keys."""
        blocks = create_hybrid_blocks(hybrid_sub, particle_dim=8)
        for block, spec in zip(blocks, hybrid_sub.specs):
            assert block.name == spec.entry_key

    def test_create_hybrid_blocks_contiguous(self, hybrid_sub):
        """Block flat ranges are contiguous."""
        blocks = create_hybrid_blocks(hybrid_sub, particle_dim=8)
        prev_end = 0
        for block in blocks:
            assert block.flat_start == prev_end
            assert block.flat_end > block.flat_start
            prev_end = block.flat_end

    def test_create_hybrid_blocks_particle_dim(self, hybrid_sub):
        """All blocks have the specified particle_dim."""
        particle_dim = 8
        blocks = create_hybrid_blocks(hybrid_sub, particle_dim=particle_dim)
        for block in blocks:
            assert block.particle_dim == particle_dim
            assert block.num_particles > 0
            # Each block's flat size should be num_particles * particle_dim
            assert block.flat_end - block.flat_start == block.num_particles * particle_dim


# ---------------------------------------------------------------------------
# Test: Optimizer integration (quick smoke test)
# ---------------------------------------------------------------------------


class TestHybridOptimizerIntegration:
    def test_optimizer_detects_hybrid_mode(self, model):
        """PolyStepOptimizer correctly detects HybridSubspace."""
        from polystep import PolyStepOptimizer

        layout = ParamLayout.from_module(model)
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        optimizer = PolyStepOptimizer(model, subspace=hybrid, epsilon=0.5)

        assert optimizer._hybrid is True
        assert optimizer._hybrid_subspace is hybrid
        assert optimizer._state.hybrid_projections is not None
        num_projected = sum(1 for s in hybrid.specs if s.is_projected)
        assert len(optimizer._state.hybrid_projections) == num_projected


# ---------------------------------------------------------------------------
# Test: RankSchedule (rank schedule extension)
# ---------------------------------------------------------------------------


class TestRankSchedule:
    def test_rank_schedule_at(self):
        """RankSchedule.at() returns correct rank at various steps."""
        schedule = RankSchedule(stages=[(0, 2), (100, 4), (300, 8)])
        assert schedule.at(0) == 2
        assert schedule.at(50) == 2
        assert schedule.at(99) == 2
        assert schedule.at(100) == 4
        assert schedule.at(200) == 4
        assert schedule.at(299) == 4
        assert schedule.at(300) == 8
        assert schedule.at(1000) == 8

    def test_rank_schedule_transitions(self):
        """transitions() returns correct step numbers."""
        schedule = RankSchedule(stages=[(0, 2), (100, 4), (300, 8)])
        assert schedule.transitions() == [100, 300]

    def test_rank_schedule_validation_empty(self):
        """ValueError for empty stages."""
        with pytest.raises(ValueError, match="at least one stage"):
            RankSchedule(stages=[])

    def test_rank_schedule_validation_no_step_zero(self):
        """ValueError for missing step 0."""
        with pytest.raises(ValueError, match="First stage must start at step 0"):
            RankSchedule(stages=[(10, 4)])

    def test_rank_schedule_validation_negative_rank(self):
        """ValueError for rank < 1."""
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            RankSchedule(stages=[(0, 0)])

    def test_rank_schedule_single_stage(self):
        """Single stage (0, 4) always returns 4."""
        schedule = RankSchedule(stages=[(0, 4)])
        assert schedule.at(0) == 4
        assert schedule.at(100) == 4
        assert schedule.at(999) == 4
        assert schedule.transitions() == []

    def test_rank_schedule_unsorted_stages(self):
        """Unsorted stages are sorted by start_step."""
        schedule = RankSchedule(stages=[(300, 8), (0, 2), (100, 4)])
        assert schedule.at(0) == 2
        assert schedule.at(100) == 4
        assert schedule.at(300) == 8


# ---------------------------------------------------------------------------
# Test: Rank transition integration (rank schedule extension)
# ---------------------------------------------------------------------------


class TestRankTransition:
    def test_rank_transition_e2e(self):
        """Test full rank transition: optimizer starts at rank=2, transitions to rank=4."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)

        schedule = RankSchedule(stages=[(0, 2), (3, 4)])

        from polystep import PolyStepOptimizer

        optimizer = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=schedule,
            epsilon=0.1,
            max_iterations=5,
            compile=False,
        )

        # Simple dummy closure
        target = torch.randn(5)

        def closure(batched_params):
            # batched_params: {key: (N, *shape)}
            first_key = list(batched_params.keys())[0]
            N = batched_params[first_key].shape[0]
            losses = torch.zeros(N)
            for i in range(N):
                x = torch.randn(1, 10)
                # Simple forward using first linear layer weight
                w1 = batched_params["0.weight"][i]
                b1 = batched_params["0.bias"][i]
                w2 = batched_params["2.weight"][i]
                b2 = batched_params["2.bias"][i]
                h = torch.relu(x @ w1.t() + b1)
                out = h @ w2.t() + b2
                losses[i] = ((out - target) ** 2).mean()
            return losses

        # Steps 1-2: rank=2 (subspace_dim unchanged)
        initial_subspace_dim = optimizer.subspace.subspace_dim
        for _ in range(2):
            optimizer.step(closure)
        assert optimizer.subspace.subspace_dim == initial_subspace_dim

        # Step 3: triggers rank transition to 4
        optimizer.step(closure)
        # After transition, subspace_dim should increase (rank=4 > rank=2)
        assert optimizer.subspace.subspace_dim > initial_subspace_dim

        # Step 4: still rank=4, verify it runs without error
        optimizer.step(closure)

    def test_rank_schedule_none_default(self):
        """PolyStepOptimizer with rank_schedule=None works as before."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)

        from polystep import PolyStepOptimizer

        optimizer = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=None,
            epsilon=0.1,
            compile=False,
        )
        assert optimizer._rank_schedule is None

    def test_rank_schedule_requires_subspace(self):
        """ValueError when rank_schedule is provided without subspace."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 5))
        schedule = RankSchedule(stages=[(0, 2), (10, 4)])

        from polystep import PolyStepOptimizer

        with pytest.raises(ValueError, match="rank_schedule requires a subspace"):
            PolyStepOptimizer(
                model,
                subspace=None,
                rank_schedule=schedule,
                epsilon=0.1,
                compile=False,
            )

    def test_rank_transition_resets_duals(self):
        """After transition, state.f and state.g are None."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)

        # Transition at step 2
        schedule = RankSchedule(stages=[(0, 2), (2, 4)])

        from polystep import PolyStepOptimizer

        optimizer = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=schedule,
            epsilon=0.1,
            max_iterations=5,
            compile=False,
        )

        target = torch.randn(5)

        def closure(batched_params):
            first_key = list(batched_params.keys())[0]
            N = batched_params[first_key].shape[0]
            losses = torch.zeros(N)
            for i in range(N):
                x = torch.randn(1, 10)
                w1 = batched_params["0.weight"][i]
                b1 = batched_params["0.bias"][i]
                w2 = batched_params["2.weight"][i]
                b2 = batched_params["2.bias"][i]
                h = torch.relu(x @ w1.t() + b1)
                out = h @ w2.t() + b2
                losses[i] = ((out - target) ** 2).mean()
            return losses

        # Step 1: iteration_count goes from 0 to 1 (still rank=2)
        optimizer.step(closure)

        # Step 2: iteration_count goes from 1 to 2, triggers transition
        optimizer.step(closure)

        # After transition, duals should be reset
        assert optimizer.state.f is None
        assert optimizer.state.g is None


# ---------------------------------------------------------------------------
# Tests for structured projection mode (structured projection)
# ---------------------------------------------------------------------------


class TestDefaultRotationInterval:
    def test_default_rotation_interval_no_warning(self, model):
        """Default rotation_interval=0 should NOT trigger a warning."""
        import warnings as _warnings
        from polystep.hybrid_subspace import HybridSubspace
        from polystep.transform import ParamLayout

        layout = ParamLayout.from_module(model)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            hs = HybridSubspace.from_layout(layout, rank=4)
        assert hs.rotation_interval == 0


class TestStructuredProjection:
    def test_random_mode_backward_compat(self, layout):
        """projection_mode='random' (default) produces same projections as before."""
        hybrid_default = HybridSubspace.from_layout(layout, rank=4)
        hybrid_random = HybridSubspace.from_layout(layout, rank=4, projection_mode="random")

        proj_default = hybrid_default.init_projections(torch.device("cpu"), torch.float32)
        proj_random = hybrid_random.init_projections(torch.device("cpu"), torch.float32)

        for key in proj_default:
            torch.testing.assert_close(proj_default[key], proj_random[key])

    def test_structured_produces_block_diagonal(self, layout):
        """projection_mode='structured' produces block-diagonal projections."""
        hybrid = HybridSubspace.from_layout(layout, rank=4, projection_mode="structured")
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)

        for spec in hybrid.specs:
            if not spec.is_projected:
                continue
            P = projections[spec.entry_key]
            d_out = spec.original_shape[0]
            d_in = spec.num_params // d_out
            coords_per_block = max(1, spec.num_coords // d_out)

            # Check off-diagonal blocks are zero
            for i in range(d_out):
                row_start = i * d_in
                row_end = row_start + d_in
                col_start = i * coords_per_block
                col_end = col_start + coords_per_block
                # Diagonal block should have non-zero entries
                block = P[row_start:row_end, col_start:col_end]
                assert block.abs().sum() > 0, f"Diagonal block ({i}) is all zeros"
                # Off-diagonal: all columns outside this block's range for these rows should be zero
                off_diag_cols = torch.cat(
                    [
                        P[row_start:row_end, :col_start],
                        P[row_start:row_end, col_end:],
                    ],
                    dim=1,
                )
                assert (off_diag_cols == 0).all(), f"Off-diagonal block ({i}) has non-zero entries"

    def test_structured_correct_shape(self, layout):
        """Structured projections have same shape (num_params, num_coords) as random."""
        hybrid_random = HybridSubspace.from_layout(layout, rank=4, projection_mode="random")
        hybrid_struct = HybridSubspace.from_layout(layout, rank=4, projection_mode="structured")

        proj_random = hybrid_random.init_projections(torch.device("cpu"), torch.float32)
        proj_struct = hybrid_struct.init_projections(torch.device("cpu"), torch.float32)

        for key in proj_random:
            assert proj_random[key].shape == proj_struct[key].shape, (
                f"Shape mismatch for {key}: {proj_random[key].shape} vs {proj_struct[key].shape}"
            )

    def test_reconstruct_works_with_structured(self, model, layout):
        """reconstruct (apply_perturbation) works with structured projections."""
        hybrid = HybridSubspace.from_layout(layout, rank=4, projection_mode="structured")
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        coords = torch.randn(hybrid.subspace_dim) * 0.01

        result = hybrid.apply_perturbation(projections, base_sd, coords)

        # Result should have correct keys and shapes
        for spec in hybrid.specs:
            assert spec.entry_key in result
            assert result[spec.entry_key].shape == spec.original_shape

    def test_reconstruct_batch_works_with_structured(self, model, layout):
        """reconstruct_batch works identically with structured projections."""
        hybrid = HybridSubspace.from_layout(layout, rank=4, projection_mode="structured")
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        N = 4
        torch.manual_seed(42)
        batch = torch.randn(N, hybrid.subspace_dim) * 0.01

        batch_result = hybrid.reconstruct_batch(projections, base_sd, batch)

        # Check shapes
        for spec in hybrid.specs:
            expected_shape = (N, *spec.original_shape)
            assert batch_result[spec.entry_key].shape == expected_shape

        # Verify batch matches single reconstructions
        for i in range(N):
            single_result = hybrid.apply_perturbation(projections, base_sd, batch[i])
            for key in single_result:
                torch.testing.assert_close(batch_result[key][i], single_result[key], atol=1e-5, rtol=1e-5)

    def test_structured_on_real_model(self):
        """Structured projections on a real model produce valid parameter reconstructions."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        layout = ParamLayout.from_module(model)
        hybrid = HybridSubspace.from_layout(layout, rank=2, projection_mode="structured")
        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()

        coords = torch.randn(hybrid.subspace_dim) * 0.01
        result = hybrid.apply_perturbation(projections, base_sd, coords)

        # All params should be finite
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite values in {key}"

        # Result should be different from base (non-zero perturbation)
        any_different = False
        for key in base_sd:
            if not torch.allclose(result[key], base_sd[key], atol=1e-8):
                any_different = True
                break
        assert any_different, "Structured projection perturbation should change parameters"


# ---------------------------------------------------------------------------
# Tests for max_subspace_dim parameter
# ---------------------------------------------------------------------------


class TestMaxSubspaceDim:
    """Tests for the max_subspace_dim budget cap."""

    def _make_conv_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        )

    def test_none_is_noop(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        layout = ParamLayout.from_module(model)
        h_default = HybridSubspace.from_layout(layout, rank=4)
        h_none = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=None)
        assert h_default.subspace_dim == h_none.subspace_dim
        for s1, s2 in zip(h_default.specs, h_none.specs):
            assert s1.num_coords == s2.num_coords

    def test_caps_total_dim(self):
        model = self._make_conv_model()
        layout = ParamLayout.from_module(model)
        h = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=500)
        # Allow small overshoot from rounding (max 1 per spec from the max(1,...) floor)
        n_specs = len(h.specs)
        assert h.subspace_dim <= 500 + n_specs, f"dim {h.subspace_dim} > 500 + {n_specs}"

    def test_preserves_proportions(self):
        model = self._make_conv_model()
        layout = ParamLayout.from_module(model)
        h_full = HybridSubspace.from_layout(layout, rank=4)
        h_cap = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=h_full.subspace_dim // 2)
        # Each projected layer's fraction of total should be approximately preserved
        for s_full, s_cap in zip(h_full.specs, h_cap.specs):
            if s_full.num_coords > 1:
                frac_full = s_full.num_coords / h_full.subspace_dim
                frac_cap = s_cap.num_coords / h_cap.subspace_dim
                assert abs(frac_full - frac_cap) < 0.1, f"Proportions diverged for {s_full.entry_key}"

    def test_min_one_coord_per_spec(self):
        model = self._make_conv_model()
        layout = ParamLayout.from_module(model)
        h = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=5)
        for spec in h.specs:
            assert spec.num_coords >= 1

    def test_flips_is_projected_for_1d(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        layout = ParamLayout.from_module(model)
        # Very aggressive cap should force 1D biases to become projected
        h_full = HybridSubspace.from_layout(layout, rank=4)
        h_cap = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=h_full.subspace_dim // 10)
        for spec in h_cap.specs:
            if spec.num_coords < spec.num_params:
                assert spec.is_projected, f"{spec.entry_key} should be projected when coords < params"

    def test_larger_than_natural_is_noop(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        layout = ParamLayout.from_module(model)
        h_full = HybridSubspace.from_layout(layout, rank=4)
        h_big = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=999999)
        assert h_full.subspace_dim == h_big.subspace_dim

    def test_reconstruction_roundtrip(self):
        model = self._make_conv_model()
        layout = ParamLayout.from_module(model)
        h = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=200)
        projections = h.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()
        coords = torch.randn(h.subspace_dim) * 0.01
        result = h.apply_perturbation(projections, base_sd, coords)
        for key, val in result.items():
            assert torch.isfinite(val).all(), f"Non-finite in {key}"
            assert val.shape == base_sd[key].shape, f"Shape mismatch for {key}"

    def test_absorb_with_cap(self):
        model = self._make_conv_model()
        layout = ParamLayout.from_module(model)
        h = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=200)
        projections = h.init_projections(torch.device("cpu"), torch.float32)
        base_sd = model.state_dict()
        coords = torch.randn(h.subspace_dim) * 0.01
        new_base, new_coords = h.absorb(projections, base_sd, coords)
        assert new_coords.shape == (h.subspace_dim,)
        assert torch.allclose(new_coords, torch.zeros_like(new_coords))


class TestHybridReconstructionProperties:
    """Reconstruction-side properties: surjectivity at saturation, the
    1D-pass-through identity, and tied-weight deduplication."""

    def test_exact_reconstruction_at_saturation(self):
        """When ``r >= min(d_in, d_out)`` the layer projection is
        surjective onto the parameter delta space, so any target delta
        is reachable via a least-norm coordinate solution.
        """
        model = nn.Linear(4, 4, bias=False)
        layout = ParamLayout.from_module(model, particle_dim=2)
        hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

        assert len(hybrid.specs) == 1
        spec = hybrid.specs[0]
        assert spec.is_projected
        # num_coords = d_out*r + r*d_in = 4*4 + 4*4 = 32 (>= num_params=16)
        assert spec.num_coords == 32
        assert spec.num_params == 16

        projections = hybrid.init_projections(
            torch.device("cpu"),
            torch.float32,
        )
        P = projections[spec.entry_key]
        assert P.shape == (16, 32)

        rank = torch.linalg.matrix_rank(P).item()
        assert rank == 16, f"Hybrid projection at saturation should span the full 16-dim param space; got rank {rank}"

        target_delta = torch.randn(4, 4)
        target_flat = target_delta.reshape(-1)
        coords = torch.linalg.lstsq(P, target_flat).solution

        base_sd = {spec.entry_key: torch.zeros(4, 4)}
        perturbed = hybrid.apply_perturbation(projections, base_sd, coords)
        recovered = perturbed[spec.entry_key]
        assert torch.allclose(recovered, target_delta, atol=1e-4), (
            "saturated HybridSubspace failed to reconstruct target delta; "
            f"max diff {(recovered - target_delta).abs().max().item():.3e}"
        )

    def test_bias_pass_through_is_identity(self):
        """Biases (1D params) carry ``is_projected=False`` and one coord
        per element, so a per-element coord write must appear verbatim
        in the perturbed bias.
        """
        model = nn.Linear(4, 8, bias=True)
        layout = ParamLayout.from_module(model, particle_dim=2)
        hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

        bias_specs = [s for s in hybrid.specs if s.entry_key == "bias"]
        assert len(bias_specs) == 1
        spec = bias_specs[0]

        assert not spec.is_projected
        assert spec.num_params == 8
        assert spec.num_coords == 8

        projections = hybrid.init_projections(
            torch.device("cpu"),
            torch.float32,
        )
        coords = torch.zeros(hybrid.subspace_dim)
        delta_bias = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        coords[spec.flat_start : spec.flat_end] = delta_bias

        base_sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        perturbed = hybrid.apply_perturbation(projections, base_sd, coords)
        assert torch.equal(perturbed["bias"], delta_bias)

    def test_tied_weights_are_projected_once(self):
        """A tied embedding/lm_head pair must produce exactly one
        :class:`LayerProjectionSpec` instead of one per state_dict key.
        """

        class _TiedHead(nn.Module):
            def __init__(self, vocab: int = 8, dim: int = 4) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab, dim)
                self.lm_head = nn.Linear(dim, vocab, bias=False)
                self.lm_head.weight = self.embedding.weight

        model = _TiedHead(vocab=8, dim=4)
        layout = ParamLayout.from_module(model, particle_dim=2)

        canonical_keys = [e.key for e in layout.entries]
        assert "embedding.weight" in canonical_keys
        assert "lm_head.weight" not in canonical_keys

        hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)
        assert len(hybrid.specs) == len(layout.entries)
        spec_keys = [s.entry_key for s in hybrid.specs]
        assert spec_keys.count("embedding.weight") == 1
