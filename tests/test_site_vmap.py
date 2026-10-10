"""Parity and reach of the site-aware evaluators.

A candidate perturbs one contiguous run of the flat parameter vector, so it differs
from the base inside a single parameter tensor; the vmap evaluator batches only that
tensor and the in-place one writes only it, and each must agree with the path it
replaces.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator, SiteInplaceEvaluator, SiteVmapEvaluator, SparseDeltaEvaluator
from polystep.transform import ParamLayout

PDIM = 2


class RepackedLinear(nn.Module):
    """A layer whose forward reads state ``functional_call`` cannot substitute:
    only an in-place write reaches the packed storage.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self._packed = self.phase.data

    def forward(self, x):
        return x @ torch.cos(self._packed).t()


class NoisyMLP(nn.Module):
    """Draws its own randomness per forward, like thermal phase variation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h + 0.01 * torch.randn_like(h))


class ItemScaledMLP(nn.Module):
    """Reads an activation back as a Python number, which vmap cannot batch."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h) / (1.0 + h.mean().item())


def _small_batch(seed=1, d_in=6, classes=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(7, d_in, generator=gen), torch.randint(0, classes, (7,), generator=gen)


class ConvNet(nn.Module):
    """Outside the Sequential-of-Linear set every other fast path requires."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 4)

    def forward(self, x):
        x = torch.relu(self.norm(self.conv(x)))
        return self.fc(nn.functional.avg_pool2d(x, 2).flatten(1))


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(6, 1, 8, 8, generator=gen), torch.randint(0, 4, (6,), generator=gen)


def test_matches_the_plain_vmap_path_at_every_site():
    """Every entry the evaluator accepts must score exactly like batching everything."""
    torch.manual_seed(0)
    model = ConvNet()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    loss_fn = nn.CrossEntropyLoss()
    dense = NNCostEvaluator(model, loss_fn)
    evaluator = SiteVmapEvaluator.try_build(dense, layout)
    assert evaluator is not None, "a traceable model must be supported"

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    inputs, targets = _batch()

    checked = 0
    for start in range(0, layout.total_params - PDIM, PDIM):
        entry = evaluator.resolve_site(torch.tensor([start]), PDIM, span=(start, start + PDIM))
        if entry is None:
            continue  # straddles two entries, which is the documented fallback
        values = flat[start : start + PDIM] + torch.tensor([0.31, -0.44])
        got = evaluator.evaluate(
            base_sd,
            entry,
            torch.arange(start - entry.offset, start - entry.offset + PDIM).reshape(1, PDIM),
            values.reshape(1, 1, PDIM),
            inputs,
            targets,
        )
        candidate = flat.clone()
        candidate[start : start + PDIM] = values
        reference = dense.evaluate(layout.batch_unflatten(candidate.unsqueeze(0)), inputs, targets)
        torch.testing.assert_close(got.reshape(-1), reference.reshape(-1), rtol=1e-5, atol=1e-6)
        checked += 1

    assert checked > 0, "no site resolved, the test proved nothing"


def test_a_step_on_a_conv_model_skips_the_closure_and_keeps_the_same_losses():
    """The path has to reach a model no other fast path accepts, and not change it."""

    def build():
        torch.manual_seed(0)
        model = ConvNet()
        inputs, targets = _batch()
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        calls = []

        def closure(batched_params):
            calls.append(1)
            return evaluator.evaluate(batched_params, inputs, targets)

        torch.manual_seed(7)
        opt = PolyStepOptimizer(model, epsilon=0.3, step_radius=0.05, seed=3, particle_dim=PDIM)
        return opt, evaluator, inputs, targets, closure, calls

    opt_dense, _, _, _, dense_closure, dense_calls = build()
    dense = [opt_dense.step(dense_closure) for _ in range(2)]

    opt_fast, evaluator, inputs, targets, fast_closure, fast_calls = build()
    opt_fast.register_evaluator(evaluator, inputs, targets)
    assert opt_fast._sparse_delta_evaluator is None, "a conv model must be outside the sparse-delta set"
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    fast = [opt_fast.step(fast_closure) for _ in range(2)]

    assert dense_calls, "the dense run must actually call the closure"
    assert not fast_calls, "every chunk should have resolved a site"
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-6)


def test_try_build_declines_tied_weights():
    """One flat position under two module paths would get half its perturbation."""
    shared = nn.Linear(6, 6)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    dense = NNCostEvaluator(model, nn.MSELoss())
    assert layout.shared_groups, "this model must actually tie its weights"
    assert SiteVmapEvaluator.try_build(dense, layout) is None


@pytest.mark.parametrize("forced", [True, False])
def test_the_in_place_site_path_accepts_tied_weights(forced):
    """Writing the shared storage moves every alias, so in place a tied weight scores
    exactly like the full swap, whether in-place was forced or picked for memory;
    a per-sample evaluator is declined.
    """
    torch.manual_seed(0)
    shared = nn.Linear(6, 6)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True if forced else None)
    evaluator._use_inplace = True
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    assert isinstance(site, SiteInplaceEvaluator)
    per_sample = NNCostEvaluator(model, nn.MSELoss(reduction="none"), use_inplace=True, per_sample=True)
    assert SiteVmapEvaluator.try_build(per_sample, layout) is None

    flat = layout.flatten(model).reshape(-1)
    inputs, targets = torch.randn(7, 6), torch.randn(7, 6)
    values = flat[0:PDIM] + torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
    got = site.evaluate(
        layout.unflatten(flat),
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    candidates = flat.expand(2, -1).clone()
    candidates[:, 0:PDIM] = values
    assert torch.equal(got, evaluator.evaluate(layout.batch_unflatten(candidates), inputs, targets))


def test_sparse_delta_still_wins_where_it_applies():
    """Both paths accept a plain MLP, so they must agree on it."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    loss_fn = nn.CrossEntropyLoss()
    sparse = SparseDeltaEvaluator.try_build(model, loss_fn, layout)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, loss_fn), layout)
    assert sparse is not None and site is not None

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    gen = torch.Generator().manual_seed(1)
    inputs = torch.randn(7, 6, generator=gen)
    targets = torch.randint(0, 3, (7,), generator=gen)

    entry = site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM))
    values = flat[0:PDIM] + torch.tensor([0.2, -0.3])
    local = torch.arange(PDIM).reshape(1, PDIM)
    from_sparse = sparse.evaluate(base_sd, entry.key, local, values.reshape(1, 1, PDIM), inputs, targets)
    from_site = site.evaluate(base_sd, entry, local, values.reshape(1, 1, PDIM), inputs, targets)
    torch.testing.assert_close(from_sparse.reshape(-1), from_site.reshape(-1), rtol=1e-5, atol=1e-6)


def test_subspace_step_on_a_conv_model_matches_the_materializing_path():
    """Per-layer blocks map to one parameter, so the site argument holds in
    coordinate space; the offset is relative to the barycentre the weights carry.
    """
    from polystep import HybridSubspace, ParamLayout

    def build(register):
        torch.manual_seed(0)
        model = ConvNet()
        inputs, targets = _batch()
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        torch.manual_seed(7)
        opt = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            seed=3,
            subspace=HybridSubspace.auto_from_layout(ParamLayout.from_module(model)),
        )
        if register:
            opt.register_evaluator(evaluator, inputs, targets)
        return opt, [opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets)) for _ in range(4)]

    _, dense = build(register=False)
    opt_fast, fast = build(register=True)
    assert opt_fast._subspace_delta_evaluator is None, "a conv model must be outside the delta set"
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    # Later steps are what catch a double-counted barycentre; step 0 starts at zero.
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-5)


def test_the_stateless_paths_score_a_repacked_model_blind():
    """On a repacked model the stateless paths score every candidate blind: one
    loss for all candidates, and no error. The in-place site path writes the packed
    storage, so it tells them apart.
    """
    torch.manual_seed(0)
    model = nn.Sequential(RepackedLinear(6, 5), nn.ReLU(), RepackedLinear(5, 3))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert site is not None, "the model must reach the site path, or this proves nothing"

    flat = layout.flatten(model).reshape(-1)
    entry = site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM))
    values = torch.stack([flat[0:PDIM], flat[0:PDIM] + 5.0])  # wildly different candidates
    blind = site.evaluate(
        {k: v.detach() for k, v in model.state_dict().items()},
        entry,
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    torch.testing.assert_close(blind[0], blind[1])

    inplace = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
    seeing = inplace.evaluate(
        layout.unflatten(flat),
        entry,
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    assert seeing[0] != seeing[1]


@pytest.mark.parametrize(
    "subspace, block_strategy", [(False, "monolithic"), (True, "monolithic"), (False, "per_layer")]
)
def test_forced_inplace_keeps_only_the_in_place_site_path(subspace, block_strategy):
    """``use_inplace=True`` is a correctness contract: the stateless paths score
    through ``functional_call`` and must step aside, in full space and subspace,
    leaving only the site path that writes the real weights. It is built once, and
    a sweep, monolithic or block-wise, lets go of the incumbent it wrote.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    inputs, targets = _small_batch()
    kwargs = (
        {"subspace": HybridSubspace.auto_from_layout(ParamLayout.from_module(model))}
        if subspace
        else {"particle_dim": PDIM, "block_strategy": block_strategy}
    )

    free = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    free.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), inputs, targets)
    armed = (free._subspace_delta_evaluator or free._site_vmap_evaluator) if subspace else free._sparse_delta_evaluator
    assert armed is not None, "a stateless path must arm here, or this proves nothing"

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    forced = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    opt.register_evaluator(forced, inputs, targets)
    site = opt._site_vmap_evaluator
    assert isinstance(site, SiteInplaceEvaluator)
    assert opt._sparse_delta_evaluator is None
    assert opt._subspace_delta_evaluator is None
    assert opt._factored_evaluator is None
    opt.register_evaluator(forced, inputs, targets)
    assert opt._site_vmap_evaluator is site
    opt.step(lambda bp: forced.evaluate(bp, inputs, targets))
    assert site._primed is None


def test_swapping_back_to_an_unforced_evaluator_rearms():
    """Swapping back to an unforced evaluator must rebuild the fast paths."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    loss_fn = nn.CrossEntropyLoss()  # one instance, so the cache key would not move
    inputs, targets = _small_batch()

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, particle_dim=PDIM)
    opt.register_evaluator(NNCostEvaluator(model, loss_fn, use_inplace=True), inputs, targets)
    assert opt._sparse_delta_evaluator is None
    opt.register_evaluator(NNCostEvaluator(model, loss_fn), inputs, targets)
    assert opt._sparse_delta_evaluator is not None or opt._site_vmap_evaluator is not None


def test_forced_inplace_disarms_the_factored_low_rank_path():
    """The factored low-rank path is stateless too; forced in-place must disarm it."""
    from polystep import FactoredSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    inputs, targets = _small_batch()
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)

    free = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, subspace=subspace)
    free.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), inputs, targets)
    assert free._factored_evaluator is not None, "the low-rank path must arm, or this proves nothing"

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, subspace=subspace)
    with pytest.warns(UserWarning, match="low-rank"):
        opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), inputs, targets)
    assert opt._factored_evaluator is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_in_place_site_matches_the_full_swap(dtype):
    """Writing only a candidate's entries scores exactly what swapping in all of its
    weights scores, and the model ends holding the incumbent.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3)).to(dtype)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    flat = layout.flatten(model).reshape(-1)
    base_sd = layout.unflatten(flat.clone())
    inputs, targets = _small_batch()

    gen = torch.Generator().manual_seed(2)
    checked = 0
    for start in range(0, layout.total_params - 2 * PDIM, 2 * PDIM):
        entry = site.resolve_site(torch.tensor([start]), 2 * PDIM, span=(start, start + 2 * PDIM))
        if entry is None:
            continue
        values = (flat[start : start + 2 * PDIM].view(2, 1, PDIM) + 0.3 * torch.randn(2, 3, PDIM, generator=gen)).to(
            dtype
        )
        local = torch.arange(start, start + 2 * PDIM).view(2, PDIM) - entry.offset
        got = site.evaluate(base_sd, entry, local, values, inputs, targets)
        candidates = flat.expand(6, -1).clone()
        for g in range(2):
            candidates[3 * g : 3 * g + 3, start + g * PDIM : start + (g + 1) * PDIM] = values[g]
        assert torch.equal(got, evaluator.evaluate(layout.batch_unflatten(candidates), inputs, targets))
        checked += 1
    assert checked > 0, "no site resolved, the test proved nothing"
    torch.testing.assert_close(layout.flatten(model).reshape(-1), flat, rtol=0, atol=0)


def test_the_in_place_site_keeps_float64_resolution():
    """FP32 loss buffers collapse candidates that differ below FP32 resolution."""
    model = nn.Sequential(nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    layout = ParamLayout.from_module(model, particle_dim=1)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.MSELoss(), use_inplace=True), layout)
    bias = torch.tensor([1.0, 1.0 + 1e-9, 1.0 + 2e-9], dtype=torch.float64)

    losses = site.evaluate(
        layout.unflatten(layout.flatten(model).reshape(-1)),
        site.resolve_site(torch.tensor([2]), 1, span=(2, 3)),
        torch.zeros(1, 1, dtype=torch.long),
        bias.view(1, 3, 1),
        torch.ones(1, 2, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
    )
    assert losses.dtype == torch.float64
    assert len(torch.unique(losses)) == 3


@pytest.mark.parametrize("projection", ["dense", "sparse", "unprojected"])
def test_the_in_place_subspace_site_matches_the_full_swap(projection):
    """Rewriting one spec's parameter from the incumbent, at a column past the spec's
    first, matches reconstructing every weight from the base. Dropout stays off while
    scoring, and the model ends holding the incumbent in train mode.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 4))
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=3, sparse_threshold_bytes=0 if projection == "sparse" else 10**9)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    model.train()
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(1)
    coords = 0.1 * torch.randn(subspace.subspace_dim, generator=gen)
    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    inputs, targets = torch.randn(5, 24, generator=gen), torch.randint(0, 4, (5,), generator=gen)

    if projection == "unprojected":
        spec = next(s for s in subspace.specs if not s.is_projected)
    else:
        spec = next(
            s
            for s in subspace.specs
            if s.is_projected and isinstance(projections[s.entry_key], torch.Tensor) is (projection == "dense")
        )
    col = 2 * PDIM
    start = spec.flat_start + col
    assert site.resolve_spec(subspace, start, start + 2 * PDIM, subspace.subspace_dim) is spec
    values = coords[start : start + 2 * PDIM].view(2, 1, PDIM) + 0.3 * torch.randn(2, 3, PDIM, generator=gen)
    got = site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, col, values, inputs, targets)
    for key, weight in bary_sd.items():
        torch.testing.assert_close(model.state_dict()[key], weight, rtol=0, atol=0)
    assert model.training

    full = coords.expand(6, -1).clone()
    for g in range(2):
        full[3 * g : 3 * g + 3, start + g * PDIM : start + (g + 1) * PDIM] = values[g]
    with torch.no_grad():
        expected = evaluator.evaluate_subspace_inplace(subspace, projections, base_sd, full, inputs, targets)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_the_in_place_site_declines_what_it_cannot_write():
    """A non-contiguous parameter has no flat view to write into, and an FP16/BF16
    subspace weight must be rebuilt from the base, so those sites take the dense path
    while the others stay in place.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    model[0].weight.data = model[0].weight.data.t().contiguous().t()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
    assert site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)) is None
    later = layout.entries[2].offset
    assert site.resolve_site(torch.tensor([later]), PDIM, span=(later, later + PDIM)) is not None
    subspace = HybridSubspace.from_layout(layout, rank=1)
    bias = subspace.specs[1].flat_start
    assert site.resolve_spec(subspace, 0, 1, subspace.subspace_dim) is None
    assert site.resolve_spec(subspace, bias, bias + 1, subspace.subspace_dim) is not None

    for dtype, accepted in ((torch.float32, True), (torch.bfloat16, False)):
        model = nn.Sequential(nn.Linear(6, 5)).to(dtype)
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=1)
        site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
        assert (site.resolve_spec(subspace, 0, 1, subspace.subspace_dim) is not None) is accepted


@pytest.mark.parametrize("back", [torch.bfloat16, torch.float32])
def test_a_half_in_place_hybrid_step_builds_no_incumbent(monkeypatch, back):
    """The in-place site declines every FP16/BF16 spec, so a step whose specs are all
    BF16 must not build an incumbent weight set that nothing reads, whatever dtype the
    buffers carry. An FP32 spec is still written in place and reads the incumbent."""
    from polystep import HybridSubspace

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Linear(6, 5), nn.BatchNorm1d(5), nn.Tanh()).bfloat16()
            self.back = nn.Linear(5, 3).to(back)

        def forward(self, x):
            return self.back(self.front(x.bfloat16()).to(back))

    torch.manual_seed(0)
    model = Model()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, seed=0, subspace=subspace)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    inputs, targets = _small_batch()
    with pytest.warns(UserWarning, match="use_inplace=True asks"):
        opt.register_evaluator(evaluator, inputs, targets)
    assert isinstance(opt._site_vmap_evaluator, SiteInplaceEvaluator)
    calls = []
    build = subspace.reconstruct_batch
    monkeypatch.setattr(subspace, "reconstruct_batch", lambda *a: calls.append(a) or build(*a))
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert bool(calls) == (back is torch.float32)


def test_a_forward_vmap_cannot_batch_falls_back_instead_of_crashing():
    """An ``.item()`` on a batched activation makes vmap fail; the site path must
    fall back to a sequential loop, not crash.
    """
    torch.manual_seed(0)
    model = ItemScaledMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss())
    site = SiteVmapEvaluator.try_build(dense, layout)
    args = (
        {k: v.detach() for k, v in model.state_dict().items()},
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        layout.flatten(model).reshape(-1)[0:PDIM].reshape(1, 1, PDIM),
        inputs,
        targets,
    )

    with pytest.warns(RuntimeWarning, match="site-aware"):
        losses = site.evaluate(*args)
    assert losses.shape == (1,) and torch.isfinite(losses).all()

    # Retired, so the step stops offering it chunks rather than warning on every one.
    assert dense._vmap_failed
    assert site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)) is None
    assert site.resolve_spec(None, 0, PDIM, PDIM) is None


@pytest.mark.parametrize("compile_vmap", [False, True])
def test_a_noisy_forward_stays_batched(compile_vmap):
    """vmap refuses a per-forward noise draw under randomness='error'; both paths
    retry with a draw per candidate and stay batched, compiled or not, so identical
    candidates score differently.
    """
    torch.manual_seed(0)
    model = NoisyMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()
    flat = layout.flatten(model).reshape(-1)

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=compile_vmap)
    site = SiteVmapEvaluator.try_build(dense, layout)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="vmap failed|compile_vmap")
        from_site = site.evaluate(
            {k: v.detach() for k, v in model.state_dict().items()},
            site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
            torch.arange(PDIM).reshape(1, PDIM),
            flat[0:PDIM].expand(1, 2, PDIM),
            inputs,
            targets,
        )
        from_dense = dense.evaluate(layout.batch_unflatten(flat.expand(2, -1)), inputs, targets)

    assert not dense._vmap_failed and not dense._compile_failed and not site._compile_failed_sites
    assert site._randomness == dense._randomness == "different"
    assert from_site[0] != from_site[1]
    assert from_dense[0] != from_dense[1]


class CountingNoisyMLP(NoisyMLP):
    """Advances a buffer before its noise draw, so a refused forward has written it."""

    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return super().forward(x) * self.calls


def test_a_noisy_retry_rewinds_buffers_the_refused_forward_wrote():
    torch.manual_seed(0)
    model = CountingNoisyMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()
    flat = layout.flatten(model).reshape(-1)
    params = layout.batch_unflatten(torch.stack([flat, flat + 0.5]))

    def direct(p):
        out = torch.func.functional_call(model, {**p, "calls": model.calls}, (inputs,))
        return nn.functional.cross_entropy(out, targets)

    torch.manual_seed(1)
    expected = torch.func.vmap(direct, randomness="different")(params)
    model.calls.zero_()

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss())
    site = SiteVmapEvaluator.try_build(dense, layout)
    torch.manual_seed(1)
    torch.testing.assert_close(dense.evaluate(params, inputs, targets), expected)
    assert model.calls.item() == 1

    site.evaluate(
        {k: v.detach() for k, v in model.state_dict().items() if k != "calls"},
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        flat[0:PDIM].expand(1, 2, PDIM),
        inputs,
        targets,
    )
    assert model.calls.item() == 2


def _dense_reference(site, projections, bary_sd, spec, col_start, dcoords):
    """The full-width formulation the block-diagonal bmm replaces."""
    n_groups, n_cand, pdim = dcoords.shape
    n = n_groups * n_cand
    wide = dcoords.new_zeros(n, spec.flat_end - spec.flat_start)
    cols = torch.arange(col_start, col_start + n_groups * pdim)
    cols = cols.reshape(n_groups, 1, pdim).expand(n_groups, n_cand, pdim).reshape(n, pdim)
    wide.scatter_(1, cols, dcoords.reshape(n, pdim))
    base = bary_sd[spec.entry_key]
    P = projections[spec.entry_key] if spec.is_projected else None
    if P is None:
        delta = wide
    elif isinstance(P, torch.Tensor):
        delta = wide.to(P.dtype) @ P.t()
    else:
        delta = P.project(wide.to(P.dtype))
    return (base.reshape(1, -1) + delta.to(base.dtype)).reshape(-1, *base.shape)


@pytest.mark.parametrize(("dtype", "sparse_dtype"), [(torch.float32, torch.float64), (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("projection_type", ["dense", "sparse", "unprojected"])
def test_the_block_diagonal_correction_matches_the_dense_one(projection_type, dtype, sparse_dtype):
    """Only pdim columns per group are nonzero, so building the site from those
    columns alone, past the spec's first column and around a nonzero incumbent, must
    match the full-width ``dcoords @ P.t()``: a bmm for a dense projection, a
    scatter-add for a sparse one or an unprojected spec. A BF16 base is rebuilt from
    ``base_sd`` and the full coordinates while the reference offsets the rounded
    incumbent, so the two round differently and match at BF16 tolerance. Neither
    path redraws a sparse projection's cache, including one held at FP64 under FP32
    coordinates.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 4)).to(dtype)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    subspace = HybridSubspace.from_layout(
        layout, rank=3, sparse_threshold_bytes=0 if projection_type == "sparse" else 10**9
    )
    projections = subspace.init_projections(torch.device("cpu"), dtype)

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert site is not None
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if projection_type == "unprojected":
        spec = next(s for s in subspace.specs if not s.is_projected)
    else:
        spec = next(
            s
            for s in subspace.specs
            if s.is_projected and isinstance(projections[s.entry_key], torch.Tensor) is (projection_type == "dense")
        )

    col = 2 * PDIM
    n_groups = (spec.flat_end - spec.flat_start - col) // PDIM
    gen = torch.Generator().manual_seed(1)
    coords = 0.1 * torch.randn(subspace.subspace_dim, generator=gen)
    dcoords = torch.randn(n_groups, 3, PDIM, generator=gen)
    values = dcoords + coords.narrow(0, spec.flat_start + col, n_groups * PDIM).view(n_groups, 1, PDIM)
    inputs = torch.randn(5, 24, generator=gen)
    targets = torch.randint(0, 4, (5,), generator=gen)

    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    P = projections.get(spec.entry_key)
    if projection_type == "sparse":
        P.project(torch.zeros(spec.num_coords, dtype=sparse_dtype))
    expected = site._vmap_over_site(
        spec.entry_key,
        bary_sd,
        _dense_reference(site, projections, bary_sd, spec, col, dcoords),
        inputs,
        targets,
    )
    held = [P._rows, P._values, P._csr_matrix] if projection_type == "sparse" else []
    got = site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, col, values, inputs, targets)
    if held:
        assert all(a is b for a, b in zip(held, [P._rows, P._values, P._csr_matrix]))
    tol = 1e-5 if dtype is torch.float32 else 1.6e-2
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)


def test_a_half_precision_site_holds_the_weights_reconstruct_batch_builds(monkeypatch):
    """A BF16 bias of 1 at coordinate 0.003 stores 1, so the 0.002 step to a candidate
    at 0.005 vanishes when added to the stored incumbent, while the base plus 0.005
    rounds up to 1.0078125. The site must hold what ``reconstruct_batch`` builds from
    the base and the candidate's full coordinates.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8)).bfloat16()
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=1)
    projections = subspace.init_projections(torch.device("cpu"), torch.bfloat16)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_sd["0.bias"].fill_(1.0)
    coords = torch.full((subspace.subspace_dim,), 0.003, dtype=torch.bfloat16)
    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    assert torch.equal(bary_sd["0.bias"], torch.ones(8, dtype=torch.bfloat16))

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    scored = {}
    monkeypatch.setattr(site, "_vmap_over_site", lambda key, sd, weights, *_: scored.__setitem__(key, weights))
    pdim, steps = 4, torch.tensor([0.005, 0.0035], dtype=torch.bfloat16)
    for spec in subspace.specs:
        n_groups = spec.num_coords // pdim
        values = steps.view(1, -1, 1).expand(n_groups, -1, pdim)
        site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, 0, values, None)
        full = coords.repeat(n_groups, len(steps), 1)
        for g in range(n_groups):
            start = spec.flat_start + g * pdim
            full[g, :, start : start + pdim] = values[g]
        expected = subspace.reconstruct_batch(projections, base_sd, full.flatten(0, 1))[spec.entry_key]
        assert torch.equal(scored[spec.entry_key], expected)
    assert scored["0.bias"].unique().tolist() == [1.0, 1.0078125]


def test_a_half_precision_subspace_step_matches_the_materializing_path():
    """The shared BF16 weights must round as the candidates do: a fused ``addmm``
    rounds once where the candidates round twice, so the two paths would score the
    same candidate differently and could swap a ranking.
    """
    from polystep import HybridSubspace

    def build(register):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 4, bias=False)).bfloat16()
        gen = torch.Generator().manual_seed(2)
        inputs, targets = torch.randn(32, 4, generator=gen), torch.randint(0, 4, (32,), generator=gen)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        evaluator._batched_linear = None
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
        opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=0.1, seed=3, subspace=subspace)
        if register:
            opt.register_evaluator(evaluator, inputs, targets)
        return opt, [opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets)) for _ in range(6)]

    _, dense = build(register=False)
    opt_fast, fast = build(register=True)
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    assert fast == dense


def test_release_evaluator_drops_the_cast_batch_the_site_path_reaches():
    """The site path casts through its owner evaluator and keeps it across a release,
    so the release has to drop the owner's cast copy of the batch as well."""
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)).bfloat16()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, seed=0, subspace=subspace)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = _small_batch()
    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert opt._site_vmap_evaluator._owner is evaluator
    assert evaluator._cast_cache is not None
    opt.release_evaluator()
    assert evaluator._cast_cache is None
