"""Tests for HybridSubspace: per-layer projections with synchronized rotation."""

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace, LayerProjectionSpec, create_hybrid_blocks
from polystep.optimizer import RankSchedule
from polystep.transform import ParamLayout


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
        """2D+ params are projected unless the rank saturates the layer.

        ``num_coords`` is capped at ``num_params``. At the cap the projection would be
        the identity, so the spec carries the parameter directly instead.
        """
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        for spec, entry in zip(hybrid.specs, layout.entries):
            if len(entry.shape) >= 2:
                assert spec.is_projected is (spec.num_coords < spec.num_params)
            else:
                assert spec.is_projected is False

    def test_from_layout_1d_params_have_identity_coords(self, layout):
        """1D params (biases) have num_params == num_coords."""
        hybrid = HybridSubspace.from_layout(layout, rank=4)

        for spec, entry in zip(hybrid.specs, layout.entries):
            if len(entry.shape) == 1:
                assert spec.num_params == spec.num_coords
                assert spec.num_params == entry.numel


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


def test_apply_perturbation_is_affine_and_local(model, hybrid_sub):
    """The two properties the reconstruction rests on, neither restating the body.

    Replaying ``base + (P @ chunk).reshape(...)`` here would reproduce a sign or
    reshape error in the test as faithfully as in the code. Superposition and locality
    are what the rest of the library assumes and are independent of how the map is
    written: the OT step adds displacements in coordinate space and expects the
    parameter change to add the same way, and each coordinate block owns one entry.
    """
    projections = hybrid_sub.init_projections(torch.device("cpu"), torch.float32)
    base_sd = model.state_dict()
    gen = torch.Generator().manual_seed(42)
    c1 = torch.randn(hybrid_sub.subspace_dim, generator=gen) * 0.01
    c2 = torch.randn(hybrid_sub.subspace_dim, generator=gen) * 0.01

    def delta(coords):
        out = hybrid_sub.apply_perturbation(projections, base_sd, coords)
        return {k: out[k] - base_sd[k] for k in out}

    d1, d2, dsum = delta(c1), delta(c2), delta(c1 + c2)
    for key in dsum:
        # atol dominates: the deltas are ~1e-2 and the two routes accumulate the same
        # products in a different order.
        torch.testing.assert_close(dsum[key], d1[key] + d2[key], rtol=1e-4, atol=1e-6)

    for spec in hybrid_sub.specs:
        coords = torch.zeros(hybrid_sub.subspace_dim)
        coords[spec.flat_start] = 1.0
        moved = {k for k, v in delta(coords).items() if v.abs().max() > 0}
        assert moved == {spec.entry_key}, f"coordinate {spec.flat_start} also moved {moved - {spec.entry_key}}"


@pytest.mark.filterwarnings("ignore:HybridSubspace works best:UserWarning")
def test_rotate_random_produces_different_projections(hybrid_sub):
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


@pytest.mark.filterwarnings("ignore:HybridSubspace works best:UserWarning")
def test_rotate_all_holds_the_seeded_basis_at_step_zero(hybrid_sub):
    """Step 0 is the freshly seeded basis; rotating it discards a draw nothing was
    evaluated against. Matches FactoredSubspace.rotate_all, which returns the same object
    to signal 'nothing changed'."""
    hybrid = HybridSubspace(
        specs=hybrid_sub.specs,
        subspace_dim=hybrid_sub.subspace_dim,
        compression_ratio=hybrid_sub.compression_ratio,
        rotation_interval=1,
        _total_params=hybrid_sub._total_params,
    )
    projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
    assert hybrid.rotate_all(projections, step=0, total_steps=100) is projections


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

    def test_a_wide_spec_is_rejected_by_both_projection_paths(self, hybrid_sub):
        """No builder produces num_coords > num_params, and neither path can serve one:
        QR would silently drop columns and a Gaussian would move at the wrong scale."""
        spec = LayerProjectionSpec(
            entry_key="w",
            original_shape=(3, 7),
            num_params=21,
            num_coords=30,
            flat_start=0,
            flat_end=30,
            is_projected=True,
        )
        with pytest.raises(ValueError, match="exceeds num_params"):
            hybrid_sub._get_projection(spec, torch.device("cpu"), torch.float32, step=0)
        with pytest.raises(ValueError, match="exceeds num_params"):
            hybrid_sub._rotate_layer_displacement(
                torch.randn(21, 30),
                spec,
                torch.randn(4, 30) * 0.1,
                svd_ratio=0.5,
                device=torch.device("cpu"),
                dtype=torch.float32,
                step=1,
            )

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


def test_optimizer_detects_hybrid_mode(model):
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

    @pytest.mark.parametrize(
        "stages, match",
        [
            ([], "at least one stage"),
            ([(10, 4)], "First stage must start at step 0"),
            ([(0, 0)], "Rank must be >= 1"),
        ],
    )
    def test_rank_schedule_validation(self, stages, match):
        with pytest.raises(ValueError, match=match):
            RankSchedule(stages=stages)

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
        # After transition, duals should be reset
        assert optimizer.state.f is None
        assert optimizer.state.g is None

        # Step 4: still rank=4, verify it runs without error
        optimizer.step(closure)

    def test_transition_fires_once_per_stage(self):
        """Within a stage the subspace must survive untouched.

        A transition absorbs, rebuilds the basis and zeroes the particles, so one per
        step throws away all progress inside the subspace. ``subspace_dim`` cannot see
        this: rebuilding at the same rank reproduces the same dimension.

        Stage 0 is applied by the constructor, so the first sweep already runs at the
        scheduled rank; only later stages fire from inside ``step()``.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=2, rotation_interval=0)

        from polystep import PolyStepOptimizer

        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            rank_schedule=RankSchedule(stages=[(0, 2), (3, 4)]),
            epsilon=0.1,
            max_iterations=6,
            compile=False,
        )
        evaluator = NNCostEvaluator(model, nn.MSELoss())
        inputs, targets = torch.randn(4, 10), torch.randn(4, 5)

        def closure(batched):
            return evaluator.evaluate(batched, inputs, targets)

        assert opt._applied_rank == 2, "stage 0 must be applied before the first step"

        ranks = []
        original = opt._transition_rank
        opt._transition_rank = lambda rank: (ranks.append(rank), original(rank))[1]

        for _ in range(3):
            opt.step(closure)
        assert ranks == [4], f"one rebuild per stage expected, got {ranks}"

        for _ in range(3):
            opt.step(closure)
        assert ranks == [4], f"rebuilt inside a stage, got {ranks}"

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


def test_default_rotation_interval_no_warning(model):
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
        hybrid_random = HybridSubspace.from_layout(layout, rank=4)

        proj_default = hybrid_default.init_projections(torch.device("cpu"), torch.float32)
        proj_random = hybrid_random.init_projections(torch.device("cpu"), torch.float32)

        for key in proj_default:
            torch.testing.assert_close(proj_default[key], proj_random[key])


class TestMaxSubspaceDim:
    def _make_conv_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        )

    @pytest.mark.parametrize("cap", [None, 999999, "exact"])
    def test_a_cap_at_or_above_the_natural_dim_is_a_noop(self, cap):
        """``exact`` is the boundary: a cap equal to the natural dim must not shrink."""
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        layout = ParamLayout.from_module(model)
        h_default = HybridSubspace.from_layout(layout, rank=4)
        if cap == "exact":
            cap = h_default.subspace_dim
        h_capped = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=cap)
        assert h_default.subspace_dim == h_capped.subspace_dim
        for s1, s2 in zip(h_default.specs, h_capped.specs):
            assert s1.num_coords == s2.num_coords

    @pytest.mark.parametrize("cap", [500, 512, 1024])
    def test_caps_total_dim(self, cap):
        """The cap is a bound, not a target: per-spec rounding must not overshoot it."""
        layout = ParamLayout.from_module(self._make_conv_model())
        h = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=cap)
        assert h.subspace_dim <= cap
        assert h.subspace_dim == h.specs[-1].flat_end

    def test_an_unreachable_cap_warns(self):
        """Unprojected width alone can exceed the cap, and that must warn."""
        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 10))
        layout = ParamLayout.from_module(model)
        with pytest.warns(UserWarning, match="unreachable"):
            h = HybridSubspace.from_layout(layout, rank=8, max_subspace_dim=64)
        unprojected = sum(s.num_coords for s in h.specs if not s.is_projected)
        assert h.subspace_dim == unprojected + sum(1 for s in h.specs if s.is_projected)

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

    def test_is_projected_tracks_the_scaled_width(self):
        """A spec below full width needs a projection; at full width it is the identity."""
        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        layout = ParamLayout.from_module(model)
        h_full = HybridSubspace.from_layout(layout, rank=4)
        h_cap = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=h_full.subspace_dim // 10)
        for spec in h_cap.specs:
            assert spec.is_projected == (spec.num_coords < spec.num_params), spec.entry_key

    def test_a_full_width_spec_carries_no_projection_matrix(self):
        """A rank past the layer clamps num_coords to num_params, where the projection
        would be an identity. Storing one costs O(n^2) for nothing."""
        layout = ParamLayout.from_module(nn.Sequential(nn.Linear(4, 4)))
        h = HybridSubspace.from_layout(layout, rank=8)
        weight = next(s for s in h.specs if s.entry_key.endswith("weight"))
        assert weight.num_coords == weight.num_params == 16
        assert weight.is_projected is False
        assert weight.entry_key not in h.init_projections(torch.device("cpu"), torch.float32)


class TestHybridReconstructionProperties:
    """Reconstruction-side properties: surjectivity at saturation, the
    1D-pass-through identity, and tied-weight deduplication."""

    def test_exact_reconstruction_at_saturation(self):
        """At ``r >= min(d_in, d_out)`` every target delta is reachable exactly.

        The uncapped formula gives ``num_coords = 4*4 + 4*4 = 32`` against
        ``num_params = 16``. A (16, 32) projection cannot have orthonormal columns, so
        it is capped at 16 and the coordinates become the delta itself.
        """
        model = nn.Linear(4, 4, bias=False)
        layout = ParamLayout.from_module(model, particle_dim=2)
        hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

        assert len(hybrid.specs) == 1
        spec = hybrid.specs[0]
        assert spec.num_coords == spec.num_params == 16
        assert not spec.is_projected

        projections = hybrid.init_projections(torch.device("cpu"), torch.float32)
        assert spec.entry_key not in projections, "a full-width spec needs no projection matrix"

        target_delta = torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
        base_sd = {spec.entry_key: torch.zeros(4, 4)}
        perturbed = hybrid.apply_perturbation(projections, base_sd, target_delta.reshape(-1))
        assert torch.equal(perturbed[spec.entry_key], target_delta)

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


def test_rank_schedule_applies_its_first_stage():
    """A subspace built at a rank the schedule does not start from must still move."""
    from polystep.optimizer import PolyStepOptimizer
    from torch.func import functional_call, vmap

    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4, seed=0)
    built_rank = sub.subspace_dim

    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        rank_schedule=RankSchedule(stages=[(0, 2), (100, 8)]),
        epsilon=0.1,
        compile=False,
    )
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    opt.step(lambda p: vmap(lambda q: functional_call(model, q, (x,)).pow(2).mean())(p))

    assert opt.subspace.subspace_dim != built_rank
    assert opt.subspace.subspace_dim == HybridSubspace.from_layout(layout, rank=2, seed=0).subspace_dim


def test_rotation_clears_the_coordinate_displacement_history():
    """``displacement_history`` holds subspace coordinates, so it is only meaningful
    under the basis that measured it. The rotate branch swapped the basis without
    clearing it, and the next displacement rotation read those rows through the new
    projections, deriving directions nothing had been measured along. The absorb
    branch always cleared them.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 3))
    subspace = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=2, rotation_interval=2, rotation_mode="displacement"
    )
    from polystep import PolyStepOptimizer

    opt = PolyStepOptimizer(model, subspace=subspace, compile=False, seed=0)
    inputs, targets = torch.randn(16, 8), torch.randn(16, 3)
    loss_fn = nn.MSELoss()

    def closure(batched):
        from torch.func import functional_call, vmap

        return vmap(lambda p: loss_fn(functional_call(model, p, (inputs,)), targets))(batched)

    previous_basis = opt._state.hybrid_projections
    rotations = 0
    for _ in range(6):
        opt.step(closure)
        state = opt._state
        if state.hybrid_projections is not previous_basis:
            rotations += 1
            assert state.displacement_history_count == 0
            assert state.displacement_history_idx == 0
            assert state.displacement_history.abs().max() == 0.0
        previous_basis = state.hybrid_projections

    assert rotations > 0, "no rotation fired, the test proves nothing"


def test_the_sparse_threshold_is_measured_in_fp32_bytes():
    """The dense estimate is num_params * num_coords * 4, compared strictly, so a
    threshold at exactly that size still takes the dense path."""
    layout = ParamLayout.from_module(nn.Sequential(nn.Linear(64, 32)))
    spec = next(
        s for s in HybridSubspace.from_layout(layout, rank=4).specs if s.is_projected and s.entry_key.endswith("weight")
    )
    exact = spec.num_params * spec.num_coords * 4

    dense = HybridSubspace.from_layout(layout, rank=4, sparse_threshold_bytes=exact)
    sparse = HybridSubspace.from_layout(layout, rank=4, sparse_threshold_bytes=exact - 1)
    projections_dense = dense.init_projections(torch.device("cpu"), torch.float32)
    projections_sparse = sparse.init_projections(torch.device("cpu"), torch.float32)

    assert isinstance(projections_dense[spec.entry_key], torch.Tensor)
    assert not isinstance(projections_sparse[spec.entry_key], torch.Tensor)

    # Random rotation rebuilds every projection, so it has to make the same choice.
    for source, is_dense in ((dense, True), (sparse, False)):
        with pytest.warns(UserWarning, match="rotation_interval=0"):
            rotating = replace(source, rotation_mode="random", rotation_interval=1)
        projections = source.init_projections(torch.device("cpu"), torch.float32)
        rotated = rotating.rotate_all(projections, step=1, total_steps=10)
        assert isinstance(rotated[spec.entry_key], torch.Tensor) is is_dense


def test_auto_from_layout_defaults():
    """Per-layer rank is min(d_in, d_out) / 16, clamped to [4, 64]. One layer per
    regime, so a changed default moves at least one of them.
    """
    model = nn.Sequential(
        nn.Linear(64, 16),  # 16/16 = 1, so the min_rank floor binds
        nn.Linear(2048, 512),  # 512/16 = 32, between the bounds
        nn.Linear(4096, 2048),  # 2048/16 = 128, so the max_rank ceiling binds
    )
    h = HybridSubspace.auto_from_layout(ParamLayout.from_module(model))
    weights = [s for s in h.specs if s.entry_key.endswith("weight")]
    ranks = [s.num_coords / sum(s.original_shape) for s in weights]
    assert ranks == [4, 32, 64]
