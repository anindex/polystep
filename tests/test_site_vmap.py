"""Parity and reach of the site-aware vmap evaluator.

A candidate perturbs one contiguous run of the flat parameter vector, so it differs
from the base inside a single parameter tensor. :class:`SiteVmapEvaluator` batches only
that tensor and leaves the rest shared, which makes the graph ahead of it run once.
Unlike the sparse-delta path it assumes nothing about the module set, so it has to
agree with the plain vmap path on models that path is the only alternative for.
"""

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator, SiteVmapEvaluator, SparseDeltaEvaluator
from polystep.transform import ParamLayout

PDIM = 2


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
