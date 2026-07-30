"""Regressions for defects that returned a plausible but wrong answer.

None of these ever raised. A reported OT cost off by a factor of the cost scale, a
resume that drifted, an absorb that re-fired every step, a cost row reused across
minibatches, an fp64 objective truncated to fp32: each looked like a working run.
Every test here fails on the code as it stood before its fix.

Defects that raise, or that produce a malformed result, live in
test_regressions.py.
"""

import math
import copy
import warnings

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator
from polystep.costs import resolve_cost_scale
from polystep.geometry import apply_biased_rotation, get_random_rotation_matrices
from polystep.hybrid_subspace import HybridSubspace
from polystep.optimizer import PolyStepOptimizer
from polystep.solvers import MinCostGreedySolver, SinkhornSolver, SoftmaxSolver
from polystep.solvers._shared import recenter_cost
from polystep._compiled import _fused_softmax_project
from polystep.transform import ParamLayout


def _quadratic_closure(params):
    v = next(iter(params.values()))
    return v.reshape(v.shape[0], -1).pow(2).sum(1)


def _small_model():
    return nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))


@pytest.mark.parametrize("scale_cost", [None, "mean", "max_cost", 2.0])
def test_sinkhorn_ent_reg_cost_is_in_the_caller_frame(scale_cost):
    """``cost_scale`` was rebound to the warm-start clamp before the frame was undone."""
    C = torch.tensor([[0.0, 3.0, 1.0, 2.0], [2.0, 0.0, 4.0, 1.0], [1.0, 2.0, 0.0, 3.0]])
    solver = SinkhornSolver(epsilon=0.5, max_iterations=5000, threshold=1e-13)
    result = solver.solve(cost_matrix=C.clone(), scale_cost=scale_cost)

    # Solving the scaled problem at eps is the raw problem at eps * divisor.
    divisor = float(resolve_cost_scale(recenter_cost(C.clone())[0], scale_cost))
    P = result.matrix
    entropy = -(P * (P.clamp(min=1e-30).log() - 1)).sum()
    expected = (C * P).sum() - 0.5 * divisor * entropy

    assert result.ent_reg_cost == pytest.approx(expected.item(), rel=1e-5)


def test_greedy_reports_cost_in_the_caller_frame():
    """Greedy scaled the cost but never multiplied the divisor back."""
    C = torch.tensor([[0.0, 4.0], [6.0, 2.0]])
    plain = MinCostGreedySolver().solve(cost_matrix=C.clone())
    scaled = MinCostGreedySolver().solve(cost_matrix=C.clone(), scale_cost="max_cost")

    assert torch.equal(plain.matrix, scaled.matrix)  # argmin is scale-invariant
    assert scaled.ent_reg_cost == pytest.approx(plain.ent_reg_cost)


def test_softmax_row_above_global_minimum_does_not_underflow_to_nan():
    """Global recentring left row 2's logits at -inf, so the row came back NaN."""
    C = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transport = SoftmaxSolver(epsilon=1e-45).solve(cost_matrix=C.clone()).matrix

    assert torch.isfinite(transport).all()
    # Each row still puts its mass on that row's cheaper vertex.
    assert transport[1, 0] > transport[1, 1]


def test_fused_softmax_row_above_global_minimum_does_not_underflow_to_nan():
    C = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    verts = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    rot = torch.eye(2).expand(2, 2, 2).contiguous()
    X_new, transport = _fused_softmax_project(C, 1e-45, torch.full((2,), 0.5), verts, rot, 0.1, torch.zeros(2, 2), True)
    assert torch.isfinite(transport).all()
    assert torch.isfinite(X_new).all()


def _run(cfg, n_pre, n_post, resume):
    torch.manual_seed(0)
    model = _small_model()
    opt = PolyStepOptimizer(model, seed=0, **cfg)
    for _ in range(n_pre):
        opt.step(_quadratic_closure)
    if resume:
        sd = copy.deepcopy(opt.state_dict())
        weights = copy.deepcopy(model.state_dict())
        torch.manual_seed(0)
        model = _small_model()
        model.load_state_dict(weights)
        opt = PolyStepOptimizer(model, seed=0, **cfg)
        opt.load_state_dict(sd)
    for _ in range(n_post):
        opt.step(_quadratic_closure)
    return opt.state.X.clone()


@pytest.mark.parametrize(
    "cfg",
    [
        {"biased_rotation": True},
        {"amortize_steps": 4, "use_quadratic_model": True, "num_probe": 3},
        {"adaptive_num_probe": True, "num_probe": 3},
    ],
    ids=["biased_rotation", "amortize_quadratic", "adaptive_num_probe"],
)
def test_resume_is_bit_exact(cfg):
    uninterrupted = _run(cfg, 5, 5, resume=False)
    resumed = _run(cfg, 5, 5, resume=True)
    assert torch.equal(uninterrupted, resumed)


def test_format_2_checkpoint_still_loads_but_warns():
    torch.manual_seed(0)
    model = _small_model()
    opt = PolyStepOptimizer(model, seed=0, biased_rotation=True)
    for _ in range(3):
        opt.step(_quadratic_closure)

    legacy = copy.deepcopy(opt.state_dict())
    legacy["format"] = 2
    for key in ("_prev_descent_direction", "_newton_direction", "_loss_decreasing_count"):
        legacy["control"].pop(key, None)

    with pytest.warns(UserWarning, match="pre-format-3"):
        opt.load_state_dict(legacy)


def test_stagnation_absorb_does_not_fire_every_step():
    """The counter kept climbing past absorb_patience, so the basis was redrawn forever."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    subspace = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=2, absorb_mode="stagnation", absorb_patience=2
    )
    opt = PolyStepOptimizer(model, subspace=subspace, seed=0)

    def flat(params):  # a plateau: stagnation is permanent
        return torch.ones(next(iter(params.values())).shape[0])

    for _ in range(10):
        opt.step(flat)

    # With patience 2 the counter has to climb again after each absorb.
    assert opt.state.absorb_count <= 4


def test_resync_from_model_forces_the_next_step_to_evaluate():
    """An amortized step coasted along the stale EMA direction, moving fresh weights."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PolyStepOptimizer(model, seed=0, amortize_steps=2)
    calls = [0]

    def counting(params):
        calls[0] += 1
        return _quadratic_closure(params)

    opt.step(counting)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)
    opt.resync_from_model()
    # Both have to clear, or the next step coasts along a direction measured at the
    # old anchor and writes over the weights the caller just set.
    assert opt._transport_direction_ema is None
    assert opt._amortize_counter == 0

    calls[0] = 0
    opt.step(counting)
    assert calls[0] > 0


def _constant_closure(value, counter):
    def closure(params):
        counter[0] += 1
        return torch.full((next(iter(params.values())).shape[0],), float(value))

    return closure


def test_adaptive_probes_do_not_reuse_costs_across_objectives():
    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, adaptive_probes=True, adaptive_probes_threshold=1e9)
    calls = [0]

    opt.step(_constant_closure(1.0, calls), objective_token=0)
    calls[0] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cost = opt.step(_constant_closure(9.0, calls), objective_token=1)

    assert calls[0] > 0
    assert cost == pytest.approx(9.0)


def test_adaptive_probes_still_reuse_for_a_stationary_objective():
    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, adaptive_probes=True, adaptive_probes_threshold=1e9)
    calls = [0]

    opt.step(_constant_closure(1.0, calls))
    calls[0] = 0
    opt.step(_constant_closure(1.0, calls))
    assert calls[0] == 0


def test_adaptive_probes_reuse_cannot_latch_on_a_drifting_particle():
    """Drift is measured from the cached matrix, not from the previous step.

    A step small enough to pass the threshold used to re-anchor the comparison at the
    new position, re-arming the gate every step: the particles marched indefinitely on
    one cost matrix, reporting a frozen loss and never evaluating the closure again.
    """
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = PolyStepOptimizer(model, seed=0, adaptive_probes=True, step_radius=1e-3, epsilon=0.3)

    start = model.weight.detach().clone()
    losses = [opt.step(_quadratic_closure) for _ in range(8)]

    # The particle drifts: each step is under the threshold, the total is well over it.
    assert (model.weight.detach() - start).pow(2).sum() > opt._adaptive_probes_threshold
    # So the costs must be re-measured rather than reused for the whole run.
    assert len(set(losses)) > 1


def test_amortized_step_invalidates_the_reuse_cache():
    """A cheap momentum step moves the particles; the cached rows describe the old spot."""
    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, amortize_steps=4, adaptive_probes=True)
    for _ in range(2):  # one OT step, then one momentum step
        opt.step(_quadratic_closure)
    assert opt._prev_cost_matrix is None


def test_blockwise_biased_rotation_runs_under_bfloat16():
    """torch.det on a bf16 matrix raised NotImplementedError on the second step."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    opt = PolyStepOptimizer(model, seed=0, mixed_precision=True, block_strategy="per_layer", biased_rotation=True)

    def closure(params):
        v = next(iter(params.values()))
        return v.reshape(v.shape[0], -1).sum(1).float()

    losses = [opt.step(closure) for _ in range(3)]

    assert all(math.isfinite(loss) for loss in losses)
    # The raise landed on step 2, so reaching step 3 with the particles intact is the
    # thing under test, not just the absence of an exception.
    assert opt.state.iteration_count == 3
    assert torch.isfinite(opt.state.X).all()


def test_zero_bias_direction_keeps_a_proper_rotation():
    """A zero bias left column 0 zero, so +e0 and -e0 probes coincided."""
    rot = get_random_rotation_matrices(
        4, 3, device="cpu", dtype=torch.float32, generator=torch.Generator().manual_seed(0)
    )
    bias = torch.zeros(4, 3)
    bias[0, 0] = 1.0  # one real direction, three degenerate

    out = apply_biased_rotation(rot, bias)
    assert torch.allclose(torch.det(out), torch.ones(4), atol=1e-4)
    gram = out.transpose(-1, -2) @ out
    assert torch.allclose(gram, torch.eye(3).expand(4, -1, -1), atol=1e-4)


def test_double_precision_objective_still_moves_particles():
    torch.manual_seed(0)
    model = nn.Linear(4, 2).double()
    opt = PolyStepOptimizer(model, seed=0, step_radius=0.5, probe_radius=0.5)

    def tiny_differences(params):
        v = next(iter(params.values()))
        return 1.0 + v.reshape(v.shape[0], -1)[:, 0] * 1e-11

    start = opt.state.X.clone()
    for _ in range(3):
        opt.step(tiny_differences)

    assert not torch.equal(opt.state.X, start)


def _screen_step(num_probe, keep_ratio, screen_fidelity=0.25):
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = PolyStepOptimizer(
        model,
        seed=0,
        particle_dim=4,
        multifidelity_screen=True,
        polytope_type="orthoplex",
        screen_keep_ratio=keep_ratio,
        screen_fidelity=screen_fidelity,
        num_probe=num_probe,
    )
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(16, 8), torch.randn(16, 4)

    def closure(batched, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched, _in, _tgt)

    return opt, closure, inputs, targets


def test_multifidelity_screen_refuses_configurations_that_cost_more_than_they_save():
    """screen_fidelity/num_probe + keep_ratio >= 1: the screen would buy work, not save it."""
    opt, closure, inputs, targets = _screen_step(num_probe=1, keep_ratio=0.9, screen_fidelity=0.5)
    with pytest.warns(UserWarning, match="costs more work than it saves"):
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    assert opt._last_screen_savings == 0.0


@pytest.mark.parametrize("num_probe", [1, 5])
def test_multifidelity_screen_saves_work_when_it_qualifies(num_probe):
    """A cheap stage one pays for itself even at a single probe scale."""
    opt, closure, inputs, targets = _screen_step(num_probe=num_probe, keep_ratio=0.5)
    opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    assert opt._last_screen_savings > 0.0


@pytest.mark.parametrize("chunk_size", [3, 7, None])
def test_chunking_does_not_move_an_infeasible_vertex_up_the_ranking(chunk_size):
    """Chunking splits the candidate loop; it must not change a single cost.

    ``sanitize_cost`` maps a hard-constraint ``+inf`` to ``2 * max|finite| + 1`` of
    whatever matrix it is handed. Per chunk, that penalty is relative to the costs
    sharing the chunk, so an infeasible vertex among cheap candidates scores below a
    legitimate expensive one elsewhere and the plan transports mass onto it.
    """

    torch.manual_seed(0)
    reference = nn.Linear(4, 2)
    # A fixed threshold, not a batch statistic: infeasibility must be a function of the
    # candidate's own parameters so the same candidates are masked however the loop is
    # split. The 1000x scale gives the finite costs a wide enough spread that a
    # chunk-local penalty lands below them.
    threshold = float(sum(p.detach().pow(2).sum() for p in reference.parameters()) * 1000.0)

    def closure(batched):
        cost = sum(v.flatten(1).pow(2).sum(dim=1) for v in batched.values()) * 1000.0
        return torch.where(cost > threshold, torch.full_like(cost, float("inf")), cost)

    model = copy.deepcopy(reference)
    opt = PolyStepOptimizer(model, compile=False, chunk_size=chunk_size, seed=0)
    loss = opt.step(closure)

    baseline_model = copy.deepcopy(reference)
    baseline = PolyStepOptimizer(baseline_model, compile=False, chunk_size=None, seed=0)
    expected = baseline.step(closure)

    assert loss == pytest.approx(expected)
    assert torch.allclose(model.weight, baseline_model.weight)


def test_probe_count_drops_on_a_descending_negative_objective():
    """K reduction keys off descent, not sign.

    RL returns and margin losses are routinely negative. Gating the trigger on
    ``all(c > 0)`` meant those objectives paid the full ``K`` forward passes forever.
    """
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PolyStepOptimizer(
        model,
        compile=False,
        num_probe=3,
        adaptive_num_probe=True,
        adaptive_probe_warmup=0,
        seed=0,
    )

    # A quadratic offset below zero: minimizable, and always negative.
    def closure(batched):
        return sum(v.flatten(1).pow(2).sum(dim=1) for v in batched.values()) - 1000.0

    for _ in range(8):
        opt.step(closure)

    assert all(c < 0 for c in opt._ot_step_costs), "objective was meant to stay negative"
    assert opt._loss_decreasing_count >= 3, "descent on a negative objective never registered"


def test_inplace_evaluator_rejects_unknown_parameter_keys():
    model = nn.Linear(4, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)
    stacked = {
        "weight": model.weight.detach().unsqueeze(0).repeat(2, 1, 1),
        "ghost": torch.zeros(2, 3),
    }
    with pytest.raises(ValueError, match="ghost"):
        evaluator.evaluate(stacked, torch.randn(5, 4), torch.randn(5, 2))


def test_the_screen_ranks_directions_per_particle():
    """The kept set is per particle, not one index list shared by all of them.

    Each particle carries its own rotation, so vertex index j points somewhere
    different in every row. Averaging the contrast down the rows made the topk a
    draw between exchangeable values, and the screen spent its full-fidelity budget
    on directions picked at random.
    """
    from polystep import _step_monolithic

    torch.manual_seed(0)
    inputs, targets = torch.randn(64, 12), torch.randn(64, 3)
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(batched, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched, _in, _tgt)

    seen = {}
    original = _step_monolithic._fill_screened_losses

    def capture(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs):
        seen["mask"] = keep_mask.clone()
        seen["screen"] = screen_cost.clone()
        return original(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs)

    opt = PolyStepOptimizer(
        model,
        epsilon=0.1,
        max_iterations=20,
        seed=3,
        multifidelity_screen=True,
        polytope_type="orthoplex",
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )
    _step_monolithic._fill_screened_losses = capture
    try:
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    finally:
        _step_monolithic._fill_screened_losses = original

    mask, screen = seen["mask"], seen["screen"]
    assert mask.dim() == 2, "the kept set is per particle, not one shared index list"
    assert not bool((mask == mask[0]).all()), "every particle kept the same directions"

    # Each row must keep its own highest-contrast pairs.
    pdim = mask.shape[1] // 2
    contrast = (screen[:, :pdim] - screen[:, pdim:]).abs()
    kept = mask[:, :pdim]
    assert bool(
        (
            contrast.masked_fill(~kept, float("inf")).amin(dim=1)
            >= contrast.masked_fill(kept, -float("inf")).amax(dim=1)
        ).all()
    )


def test_es_records_best_when_one_fitness_is_nan():
    """A single NaN made ``torch.min`` return NaN, so ``fmin < best_fitness`` was
    False and no finite candidate was ever recorded, even with valid evaluations
    in the same population.
    """
    from polystep import PolyStepES

    es = PolyStepES(dim=3, num_particles=2, seed=0)
    asked = es.ask()
    fitness = torch.arange(asked.shape[0], dtype=torch.float32)
    fitness[3] = float("nan")

    es.tell(fitness)

    assert es.best_solution is not None, "no finite best recorded despite valid candidates"
    assert es.best_fitness == 0.0


def test_train_releases_evaluator_when_a_callback_raises():
    """``release_evaluator()`` ran only on the normal return, so an exception left
    the batch alive on the optimizer for the rest of its lifetime (device memory
    on CUDA).
    """
    from torch.utils.data import DataLoader, TensorDataset
    from polystep import PolyStepOptimizer, TrainConfig, train
    from polystep.api import TrainCallback

    class Boom(TrainCallback):
        def on_step_end(self, metrics):
            raise RuntimeError("boom")

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = PolyStepOptimizer(model)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4)

    with pytest.raises(RuntimeError, match="boom"):
        train(model, loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=1, callbacks=[Boom()]))

    assert optimizer._cost_evaluator is None
    assert optimizer._fused_inputs is None
    assert optimizer._fused_targets is None


def test_cost_batch_size_zero_is_rejected():
    """0 slices an empty batch: every candidate scores NaN, sanitization flattens
    the cost matrix to a constant, and the run reports finite losses while
    learning nothing.
    """
    from polystep import PolyStepOptimizer

    with pytest.raises(ValueError, match="cost_batch_size"):
        PolyStepOptimizer(nn.Linear(4, 2), cost_batch_size=0)


def test_adaptive_subspace_rejects_rank_above_full_dim():
    """Reduced QR of a wide matrix returns ``full_dim`` columns, not the requested
    ``subspace_dim``, so the optimizer allocated coordinates the projection could
    not consume and the mismatch only surfaced later at reconstruction.
    """
    from polystep.adaptive_subspace import AdaptiveSubspace

    with pytest.raises(ValueError, match="subspace_dim"):
        AdaptiveSubspace(full_dim=2, subspace_dim=3)


def test_factored_subspace_rotates_model_of_only_vectors():
    """Every parameter unprojected leaves ``init_projections()`` empty, and the
    rotation reached for ``next(iter(...))`` on it.
    """
    from polystep.factored_subspace import FactoredSubspace
    from polystep.transform import ParamLayout

    model = nn.Module()
    model.bias_only = nn.Parameter(torch.zeros(5))
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2, rotation_interval=1)
    projections = subspace.init_projections(device=torch.device("cpu"), dtype=torch.float32)

    assert subspace.rotate_all(projections, step=1) is projections


def test_sparse_projection_ignores_default_device():
    """The index/sign factories drew from a CPU generator without ``device="cpu"``,
    so under ``torch.set_default_device`` they targeted the default device and the
    generator mismatched.
    """
    from polystep.projection.sparse import SparseRandomProjection

    target = torch.device("cuda" if torch.cuda.is_available() else "meta")
    torch.set_default_device(target)
    try:
        projection = SparseRandomProjection(full_dim=64, subspace_dim=8, seed=0)
        projection._init_sparse_matrix(torch.device("cpu"), torch.float32)
    finally:
        torch.set_default_device("cpu")

    assert projection._indices.shape == (2, projection.nnz)
