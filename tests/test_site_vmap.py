"""Parity and reach of the site-aware vmap evaluator.

A candidate perturbs one contiguous run of the flat parameter vector, so it differs
from the base inside a single parameter tensor. :class:`SiteVmapEvaluator` batches only
that tensor and leaves the rest shared, which makes the graph ahead of it run once.
Unlike the sparse-delta path it assumes nothing about the module set, so it has to
agree with the plain vmap path on models that path is the only alternative for.
"""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator, SiteVmapEvaluator, SparseDeltaEvaluator
from polystep.transform import ParamLayout

PDIM = 2


class RepackedLinear(nn.Module):
    """A layer whose forward reads state ``functional_call`` cannot substitute.

    Stands in for a photonic MZI mesh: the transfer matrix is rebuilt from phase
    parameters through a reference captured when the layer was packed. An in-place
    write reaches that storage; swapping the Parameter attribute does not, so every
    candidate scores at the base weight.
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
    """A per-layer block maps to one parameter, so the site argument holds in
    coordinate space too.

    The offset is relative to the barycentre the shared weights already carry. Adding
    the absolute coordinates instead counts it twice, which is invisible on the first
    step because the coordinates start at zero.
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
    """The failure the in-place contract exists to prevent, stated as a measurement.

    Without this the guard below looks like a performance preference rather than a
    correctness one: the site path does not error on such a model, it returns one
    number for every candidate and the run reports success while measuring nothing.
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

    forced = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    seeing = forced.evaluate(layout.batch_unflatten(torch.stack([flat, flat.clone()])[:, None, :]), inputs, targets)
    assert seeing.shape[0] == 2


@pytest.mark.parametrize("subspace", [False, True])
def test_forced_inplace_disarms_every_stateless_path(subspace):
    """``use_inplace=True`` is a correctness contract, not only a memory one.

    It already outranked the bmm path. The site-aware paths score through
    ``functional_call`` for the same reason bmm does, so they have to honour it too --
    in full space and in a subspace, since each arms a different set.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    inputs, targets = _small_batch()
    kwargs = (
        {"subspace": HybridSubspace.auto_from_layout(ParamLayout.from_module(model))}
        if subspace
        else {"particle_dim": PDIM}
    )

    free = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    free.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), inputs, targets)
    armed = (free._subspace_delta_evaluator or free._site_vmap_evaluator) if subspace else free._sparse_delta_evaluator
    assert armed is not None, "a stateless path must arm here, or this proves nothing"

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), inputs, targets)
    assert opt._site_vmap_evaluator is None
    assert opt._sparse_delta_evaluator is None
    assert opt._subspace_delta_evaluator is None
    assert opt._factored_evaluator is None
    opt.step(lambda bp: NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True).evaluate(bp, inputs, targets))


def test_swapping_back_to_an_unforced_evaluator_rearms():
    """The disarm must not be permanent for the model and loss it happened on.

    The fast paths are rebuilt only when they are absent, so setting them to None left
    a later unforced evaluator on the same model and loss with no fast path at all.
    """
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
    """The low-rank identity is stateless too, and it dispatches on its own branch.

    ``_step_monolithic`` reaches ``elif _factored_eval is not None`` after the site
    branches, so leaving it armed keeps the model's forward unrun on exactly the models
    the guard exists for.
    """
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


def test_a_forward_that_draws_randomness_falls_back_instead_of_crashing():
    """The step calls the site path directly, so it needs its own vmap fallback.

    ``NNCostEvaluator.evaluate`` demotes a vmap failure to a sequential loop; the site
    path had no such guard, so a per-forward noise draw killed the run instead of
    degrading it. vmap's default is ``randomness='error'``.
    """
    torch.manual_seed(0)
    model = NoisyMLP()
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


def _dense_reference(site, projections, bary_sd, spec, col_start, dcoords):
    """The full-width formulation the block-diagonal bmm replaces."""
    n_groups, n_cand, pdim = dcoords.shape
    n = n_groups * n_cand
    wide = dcoords.new_zeros(n, spec.flat_end - spec.flat_start)
    cols = torch.arange(col_start, col_start + n_groups * pdim)
    cols = cols.reshape(n_groups, 1, pdim).expand(n_groups, n_cand, pdim).reshape(n, pdim)
    wide.scatter_(1, cols, dcoords.reshape(n, pdim))
    base = bary_sd[spec.entry_key]
    P = projections[spec.entry_key]
    delta = wide.to(P.dtype) @ P.t() if isinstance(P, torch.Tensor) else P.project(wide)
    return (base.reshape(1, -1) + delta.to(base.dtype)).reshape(-1, *base.shape)


@pytest.mark.parametrize("projection_type", ["dense", "sparse"])
def test_the_block_diagonal_correction_matches_the_dense_one(projection_type):
    """Only pdim columns per group are nonzero, so the correction is a per-group bmm.

    The dense ``dcoords @ P.t()`` cost O(n * num_coords * num_params) where the nonzero
    structure allows O(n * pdim * num_params). Both must give the same losses.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 4))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    subspace = HybridSubspace.from_layout(layout, rank=3)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert site is not None
    bary_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    wanted_tensor = projection_type == "dense"
    specs = [
        s
        for s in subspace.specs
        if s.is_projected and isinstance(projections[s.entry_key], torch.Tensor) is wanted_tensor
    ]
    if not specs:
        pytest.skip(f"this model produces no {projection_type} projection")
    spec = specs[0]

    n_groups = max(1, (spec.flat_end - spec.flat_start) // PDIM)
    gen = torch.Generator().manual_seed(1)
    dcoords = torch.randn(n_groups, 3, PDIM, generator=gen)
    inputs = torch.randn(5, 24, generator=gen)
    targets = torch.randint(0, 4, (5,), generator=gen)

    got = site.evaluate_subspace(projections, bary_sd, spec, 0, dcoords, inputs, targets)
    expected = site._vmap_over_site(
        spec.entry_key,
        bary_sd,
        _dense_reference(site, projections, bary_sd, spec, 0, dcoords),
        inputs,
        targets,
    )
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
