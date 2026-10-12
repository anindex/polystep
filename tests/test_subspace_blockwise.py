"""Subspace runs under a block strategy: row segments of the Sinkhorn solve, their
warm starts, and absorb.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import HybridSubspace, PolyStepOptimizer, AdaptiveSubspace
from polystep.optimizer import RankSchedule
from polystep.transform import ParamLayout

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
            simple_model.load_state_dict(params_i, strict=False)
            output = simple_model(inputs)
            loss = criterion(output, targets)
            losses.append(loss)
        return torch.stack(losses)

    return closure


class TestCombinedModeInitialization:
    def test_row_segments_split_only_two_sided_solves(self, simple_model):
        """Sinkhorn splits the particles into 2 to 8 row segments; a one-sided solver
        keeps the monolithic step and says so."""
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5)

        sinkhorn = PolyStepOptimizer(
            simple_model, subspace=subspace, block_strategy="per_layer", solver="sinkhorn", epsilon=0.1
        )
        assert 2 <= sinkhorn._row_segments <= 8

        with pytest.warns(UserWarning, match="no effect in subspace mode"):
            softmax = PolyStepOptimizer(simple_model, subspace=subspace, block_strategy="per_layer", epsilon=0.1)
        assert softmax._row_segments is None


@pytest.mark.timeout(180)
class TestCombinedModeStep:
    def test_step_updates_state(self, simple_model, simple_closure):
        """The represented point must move after a step; check model params, since
        rotation re-anchors the origin and zeroes ``X``.
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

    def test_segment_duals_warm_start(self, simple_model, simple_closure):
        """Segment duals live in ``state.f`` (P,) and ``state.g`` (segments, V) and warm-start
        the next solve. rotation_interval=0 keeps the basis from invalidating them, and
        warnings are errors so a mismatched warm start cannot fall back to zeros silently.
        """
        subspace = AdaptiveSubspace.auto_from_params(
            simple_model, compression_target=0.5, max_rank=16, rotation_interval=0
        )
        optimizer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            solver="sinkhorn",
            **_FAST_OPT_KWARGS,
        )
        P, V = optimizer.state.X.shape[0], optimizer._polytope_vertices.shape[0]
        segments = optimizer._row_segments
        seen = []
        solve = optimizer.solver.solve

        def spy(**kwargs):
            seen.append(kwargs["init_g"])
            return solve(**kwargs)

        optimizer.solver.solve = spy
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(2):
                optimizer.step(simple_closure)
                state = optimizer.state
                assert state.f.shape == (P,)
                assert state.g.shape == (segments, V)
                assert torch.isfinite(state.f).all() and torch.isfinite(state.g).all()

        assert state.block_duals is None
        assert seen[:segments] == [None] * segments
        assert [g.shape for g in seen[segments:]] == [(V,)] * segments

    def test_all_nonfinite_step_reports_no_loss(self, simple_model):
        """The segments solved a flat penalty matrix, so the step reports no loss, no
        convergence and no progress, and keeps the previous warm start."""
        sub = HybridSubspace.from_layout(ParamLayout.from_module(simple_model), rank=2, absorb_mode="none")
        opt = PolyStepOptimizer(
            simple_model,
            subspace=sub,
            block_strategy="per_layer",
            solver="sinkhorn",
            use_adaptive_radius=True,
            **_FAST_OPT_KWARGS,
        )
        values = iter([10.0, float("nan")])

        def closure(bp):
            return torch.full((next(iter(bp.values())).shape[0],), next(values))

        opt.step(closure)
        g = opt.state.g
        with pytest.warns(RuntimeWarning, match="carries no ranking information"):
            opt.step(closure)
        assert opt.state.costs == [10.0, float("inf")]
        assert opt.state.linear_convergence == [True, False]
        assert opt.state.radius_multiplier == 1.0
        assert opt.state.g is g


@pytest.mark.timeout(180)
class TestSynchronizedAbsorb:
    def test_absorb_resets_all_coords(self, simple_model, simple_closure):
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
            solver="sinkhorn",
            **_FAST_OPT_KWARGS,
        )

        base_before = {k: v.clone() for k, v in optimizer._state.base_params.items()}

        # Periodic absorb (interval=3) must actually fire within 4 steps.
        for _ in range(4):
            optimizer.step(simple_closure)

        assert optimizer._state.absorb_count >= 1, "periodic absorb never triggered"
        # Absorb folds the perturbation into the base weights, so at least one
        # base tensor must change.
        changed = any(not torch.allclose(base_before[k], v) for k, v in optimizer._state.base_params.items())
        assert changed, "absorb did not fold the perturbation into base params"

    def test_absorb_rotates_projection(self, simple_model, simple_closure):
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
            solver="sinkhorn",
            **_FAST_OPT_KWARGS,
        )

        P_before = optimizer._state.projection.clone()

        # Run until absorb
        for _ in range(3):
            optimizer.step(simple_closure)

        P_after = optimizer._state.projection

        assert not torch.allclose(P_before, P_after)


@pytest.mark.parametrize("interval, expected_rotations", [(1, 6), (3, 2), (0, 0)])
def test_global_subspace_honours_rotation_interval(simple_model, simple_closure, interval, expected_rotations):
    """The global-subspace branch rotates only every rotation_interval steps."""
    sub = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, min_rank=8, max_rank=16)
    sub.rotation_interval = interval
    sub.absorb_mode = "interval"
    sub.absorb_interval = 0

    opt = PolyStepOptimizer(
        simple_model, subspace=sub, block_strategy="per_layer", solver="sinkhorn", **_FAST_OPT_KWARGS
    )

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
            solver="sinkhorn",
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

    def test_grouped_strategy_warns_matches_per_layer(self, simple_model, simple_closure):
        """In subspace mode the segments slice particles, so 'grouped' cannot group anything.

        It produces exactly the per_layer segments, so accepting it silently would hide
        that the requested grouping did not happen.
        """
        subspace = AdaptiveSubspace.auto_from_params(simple_model, compression_target=0.5, max_rank=16)

        with pytest.warns(UserWarning, match="no effect in subspace mode"):
            grouped = PolyStepOptimizer(
                simple_model,
                subspace=subspace,
                block_strategy="grouped",
                solver="sinkhorn",
                **_FAST_OPT_KWARGS,
            )
        per_layer = PolyStepOptimizer(
            simple_model,
            subspace=subspace,
            block_strategy="per_layer",
            solver="sinkhorn",
            **_FAST_OPT_KWARGS,
        )
        assert grouped._row_segments == per_layer._row_segments is not None

        params_before = [p.detach().clone() for p in simple_model.parameters()]
        grouped.step(simple_closure)
        assert any(not torch.allclose(a, b) for a, b in zip(params_before, simple_model.parameters()))


def test_blockwise_displacement_history_in_full_space():
    """The basis rotates every step, so subspace-coordinate history mixes frames.

    The step must store the full-space image under the basis in use at measurement
    time, or the rotation SVD reads old displacements along directions never measured.
    """
    from torch.func import functional_call

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


def test_rank_schedule_runs_under_row_segments(simple_model, simple_closure):
    """A rank transition changes the particle count; the stored segment count survives it."""
    sub = HybridSubspace.from_layout(ParamLayout.from_module(simple_model), rank=2)
    opt = PolyStepOptimizer(
        simple_model,
        subspace=sub,
        block_strategy="per_layer",
        solver="sinkhorn",
        rank_schedule=RankSchedule(stages=[(0, 2), (1, 4)]),
        **_FAST_OPT_KWARGS,
    )
    P0 = opt.state.X.shape[0]
    for _ in range(3):
        opt.step(simple_closure)

    P, V = opt.state.X.shape[0], opt._polytope_vertices.shape[0]
    assert opt._applied_rank == 4 and P > P0
    assert opt.state.g.shape == (min(opt._row_segments, P), V)


@pytest.mark.parametrize("rank", [1, 4])
def test_row_segments_follow_scheduled_particle_count(rank):
    """The segment count and the frozen-segment warning follow the particle count each
    scheduled rank produces, whatever rank the subspace was built at."""
    from torch.func import functional_call

    model = nn.Sequential(
        nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)
    )
    x = torch.randn(16, 4, generator=torch.Generator().manual_seed(0))

    def closure(batched_params):
        n = next(iter(batched_params.values())).shape[0]
        return torch.stack(
            [functional_call(model, {k: v[i] for k, v in batched_params.items()}, (x,)).pow(2).mean() for i in range(n)]
        )

    with pytest.warns(UserWarning, match="6 of 6 subspace row segment"):
        opt = PolyStepOptimizer(
            model,
            subspace=HybridSubspace.from_layout(ParamLayout.from_module(model), rank=rank),
            block_strategy="per_layer",
            solver="sinkhorn",
            rank_schedule=RankSchedule(stages=[(0, 1), (1, 4)]),
            epsilon=0.1,
            seed=0,
            compile=False,
        )
    with pytest.warns(UserWarning, match="7 of 8 subspace row segment"):
        for _ in range(3):
            opt.step(closure)

    assert opt.state.X.shape[0] == 9
    assert opt.state.g.shape == (8, opt._polytope_vertices.shape[0])
