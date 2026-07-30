"""Lifecycle of the cached fused block-diagonal projection.

``reconstruct_batch`` scores candidates through the cached ``_fused_P`` while
``_sync_model`` writes the barycenter through ``state.hybrid_projections``. If a basis
change refreshes one and not the other, the optimizer evaluates in one basis and steps
in another, with no error raised. The rebuild is a full dense reconstruction, so it must
also not run on the steps where the basis held still.
"""

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.optimizer import RankSchedule
from polystep.transform import ParamLayout


def _fused_disagrees_with_projections(sub, projections, base_sd, coords):
    """Max deviation between the fused reconstruction and the per-layer one."""
    fused = sub.reconstruct_batch(projections, base_sd, coords.unsqueeze(0))
    worst = 0.0
    for spec, _ in sub._fused_dense_specs:
        P = projections[spec.entry_key]
        chunk = coords[spec.flat_start : spec.flat_end]
        want = (base_sd[spec.entry_key].reshape(-1) + P @ chunk).reshape(spec.original_shape)
        worst = max(worst, (fused[spec.entry_key][0] - want).abs().max().item())
    return worst


def test_fused_matrix_tracks_the_basis_across_an_aligned_absorb():
    """absorb_aligned_active draws a different basis; the fused cache must follow."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(
        layout,
        rank=2,
        seed=0,
        absorb_mode="periodic",
        absorb_interval=2,
        absorb_aligned_active=True,
    )
    opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.1, step_radius=0.05, adaptive_probes=False)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), layout)

    def closure(params):
        return ev.evaluate(params, x, y)

    for i in range(5):
        opt.step(closure, objective_token=i)
        state = opt.state
        # A unit probe, not state.X: after an absorb X is zero, and base + P @ 0 agrees
        # for any P, so checking at X would pass against a completely stale basis.
        coords = torch.ones(sub.subspace_dim)
        assert _fused_disagrees_with_projections(sub, state.hybrid_projections, state.base_params, coords) < 1e-4, (
            f"step {i}: fused matrix describes a different basis than state.hybrid_projections"
        )
    assert opt.state.absorb_count > 0, "no absorb fired, the test proved nothing"


def test_blockwise_mode_still_rotates_a_per_layer_subspace():
    """absorb_interval must not be silently ignored under a block strategy.

    The blockwise path tracked only state.projection, which a HybridSubspace does not
    use, so its basis stayed frozen for the whole run with no error raised.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(
        layout,
        rank=2,
        seed=0,
        absorb_mode="periodic",
        absorb_interval=2,
        absorb_aligned_active=True,
    )
    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        epsilon=0.1,
        step_radius=0.05,
        block_strategy="per_layer",
        adaptive_probes=False,
    )
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), layout)
    before = {k: v.clone() for k, v in opt.state.hybrid_projections.items()}

    for i in range(4):
        opt.step(lambda params: ev.evaluate(params, x, y), objective_token=i)

    assert opt.state.absorb_count > 0, "absorb_interval was ignored in blockwise mode"
    moved = max((opt.state.hybrid_projections[k] - before[k]).abs().max().item() for k in before)
    assert moved > 1e-3, "the basis never changed, so absorb_aligned_active did nothing"


def test_fused_projection_not_rebuilt_when_static():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4)
    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        solver="softmax",
        epsilon=0.5,
        step_radius=0.3,
        probe_radius=1.0,
    )

    # Count rebuilds after construction (the init build has already happened).
    calls = {"n": 0}
    original = sub.build_fused_projection

    def counting(projections):
        calls["n"] += 1
        return original(projections)

    sub.build_fused_projection = counting

    x = torch.randn(24, 32)
    y = torch.randint(0, 4, (24,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    for _ in range(5):
        opt.step(lambda s: ev.evaluate(s, x, y))

    assert calls["n"] == 0, f"expected no rebuilds with static projections, got {calls['n']}"


def test_absorb_reuses_the_seeded_basis_without_redrawing_it():
    """An absorb re-anchors the origin; the basis is the same seeded draw either way.

    ``init_projections`` memoizes so the QR does not rerun, and the fused matrix, being a
    pure function of that basis, is not rebuilt. A cache that returned a *different* basis
    would move the weights with nothing evaluated behind it.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4, absorb_mode="periodic", absorb_interval=1)

    first = sub.init_projections(torch.device("cpu"), torch.float32)
    assert sub.init_projections(torch.device("cpu"), torch.float32) is first

    fresh = HybridSubspace.from_layout(layout, rank=4).init_projections(torch.device("cpu"), torch.float32)
    assert fresh.keys() == first.keys()
    assert all(torch.equal(first[k], v) for k, v in fresh.items()), "cache must not change the basis"

    opt = PolyStepOptimizer(model, subspace=sub, solver="softmax", epsilon=0.5, step_radius=0.3, probe_radius=1.0)
    calls = {"n": 0}
    original = sub.build_fused_projection
    sub.build_fused_projection = lambda projections: (calls.__setitem__("n", calls["n"] + 1), original(projections))[1]

    # Handing the dict out by identity is sound only while nothing writes into it.
    snapshot = {k: v.clone() for k, v in first.items() if isinstance(v, torch.Tensor)}
    x = torch.randn(24, 32)
    y = torch.randint(0, 4, (24,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    for _ in range(5):
        opt.step(lambda s: ev.evaluate(s, x, y))

    assert opt._state.absorb_count > 0, "absorb_interval=1 must absorb"
    assert calls["n"] == 0, f"absorb must not rebuild the fused matrix, got {calls['n']}"
    mutated = [k for k, v in snapshot.items() if not torch.equal(first[k], v)]
    assert not mutated, f"cached projections were written in place: {mutated}"


def test_rank_transition_rebuilds_the_fused_projection():
    """A transition swaps in a new-rank basis, so the fused matrix must follow it.

    The step only rebuilds on a basis-object change, and nothing changes the basis again
    after a transition, so a transition that does not build its own leaves ``_fused_P``
    at the old rank's shape for the rest of the run. Silent: reconstruction still works
    through the per-layer path, only slower.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    sub = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=4, absorb_mode="periodic", absorb_interval=1, rotation_interval=0
    )
    opt = PolyStepOptimizer(
        model, subspace=sub, seed=0, rank_schedule=RankSchedule([(0, 4), (3, 2)]), max_iterations=10
    )
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    x = torch.randn(16, 64)
    y = torch.randint(0, 16, (16,))
    before = opt.subspace._fused_P.shape[1]
    for i in range(8):
        opt.register_evaluator(ev, x, y)
        opt.step(lambda p: ev.evaluate(p, x, y), objective_token=i)

    fused = opt.subspace._fused_P
    assert fused is not None, "fused projection lost after the rank transition"
    assert fused.shape[1] < before, f"fused matrix kept the old rank's width: {fused.shape[1]} vs {before}"
    assert fused.shape[1] == sum(s.num_coords for s, _ in opt.subspace._fused_dense_specs)


if __name__ == "__main__":
    test_fused_projection_not_rebuilt_when_static()
    test_absorb_reuses_the_seeded_basis_without_redrawing_it()
    test_rank_transition_rebuilds_the_fused_projection()
    print("ok")
