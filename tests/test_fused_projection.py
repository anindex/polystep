"""Lifecycle of the cached fused block-diagonal projection.

If a basis change refreshes the fused cache but not the projections (or vice versa),
the optimizer evaluates in one basis and steps in another, with no error raised. The
rebuild is a full dense reconstruction, so it must not run when the basis held still.
"""

import pytest
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


def test_fused_matrix_tracks_aligned_absorb_basis():
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
    opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.1, step_radius=0.05)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())

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
        assert sub._fused_src is state.hybrid_projections
    assert opt.state.absorb_count > 0, "no absorb fired, the test proved nothing"


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 5e-2)])
def test_fused_paths_use_passed_projections(dtype, tol):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 3)).to(dtype)
    sub = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=1, rotation_interval=1, rotation_mode="random"
    )
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    old = sub.init_projections(torch.device("cpu"), dtype)
    sub.build_fused_projection(old)
    assert sub._fused_P is not None
    new = sub.rotate_all(old, step=1, total_steps=4)
    coords = torch.randn(sub.subspace_dim, generator=torch.Generator().manual_seed(3)).to(dtype)
    want = {
        s.entry_key: base_sd[s.entry_key].float().reshape(-1)
        + new[s.entry_key].float() @ coords[s.flat_start : s.flat_end].float()
        for s in sub.specs
        if s.is_projected
    }
    sub.apply_perturbation_inplace(new, model, base_sd, coords)
    got = {
        "batch": {k: v[0] for k, v in sub.reconstruct_batch(new, base_sd, coords.unsqueeze(0)).items()},
        "single": sub.apply_perturbation(new, base_sd, coords),
        "inplace": dict(model.named_parameters()),
    }
    for path, sd in got.items():
        for k, w in want.items():
            assert (sd[k].detach().float().reshape(-1) - w).abs().max() < tol, (path, k)


def test_blockwise_mode_rotates_per_layer_subspace():
    """absorb_interval must not be silently ignored under a block strategy."""
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
    )
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
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


def test_absorb_reuses_seeded_basis():
    """Without rotation, an absorb re-anchors the origin and keeps the current basis.

    The fused matrix is a pure function of that basis, so it is never rebuilt. A basis
    swapped at absorb would move the weights with nothing evaluated behind it.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4, absorb_mode="periodic", absorb_interval=1)

    opt = PolyStepOptimizer(model, subspace=sub, solver="softmax", epsilon=0.5, step_radius=0.3, probe_radius=1.0)
    first = opt._state.hybrid_projections
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
    assert opt._state.hybrid_projections is first, "absorb must keep the current basis"
    assert calls["n"] == 0, f"absorb must not rebuild the fused matrix, got {calls['n']}"
    mutated = [k for k, v in snapshot.items() if not torch.equal(first[k], v)]
    assert not mutated, f"cached projections were written in place: {mutated}"
    fresh = HybridSubspace.from_layout(layout, rank=4).init_projections(torch.device("cpu"), torch.float32)
    assert fresh.keys() == first.keys()
    assert all(torch.equal(first[k], v) for k, v in fresh.items()), "absorb must not redraw the basis"


def test_rank_transition_rebuilds_fused_projection():
    """A transition swaps in a new-rank basis, so the fused matrix must follow it.

    The step only rebuilds on a basis-object change, so a transition that does not
    build its own leaves ``_fused_P`` at the old rank's shape for the rest of the run.
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
