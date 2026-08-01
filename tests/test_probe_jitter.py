"""Probe-radius jitter: the smooth density the convergence analysis assumes.

The analysis needs the jitter density to be C-infinity and compactly supported.
The mollifier q(t) ~ exp(-1/(1-t^2)) is; a uniform density is not, and the
difference shows up as Dirac terms in the score of the induced probe kernel.
These tests check support, smooth-vs-uniform shape, and reproducibility.
"""

import math

import torch

from polystep.optimizer import PolyStepOptimizer


def _optimizer(**kw):
    model = torch.nn.Linear(4, 2)
    return PolyStepOptimizer(model, **kw)


def test_jitter_respects_support():
    opt = _optimizer(probe_radius_jitter=0.3, seed=0)
    etas = [opt._sample_jitter(0.3) for _ in range(500)]
    assert all(abs(e) < 0.3 for e in etas)
    assert max(abs(e) for e in etas) > 0.05  # not degenerate


def test_smooth_density_concentrates_more_than_uniform():
    """The mollifier puts more mass near zero and vanishes at the boundary."""
    smooth = _optimizer(probe_radius_jitter=0.9, probe_radius_jitter_dist="smooth", seed=1)
    uniform = _optimizer(probe_radius_jitter=0.9, probe_radius_jitter_dist="uniform", seed=1)

    n = 4000
    s = torch.tensor([smooth._sample_jitter(1.0) for _ in range(n)])
    u = torch.tensor([uniform._sample_jitter(1.0) for _ in range(n)])

    # Uniform on [-1,1] has std 1/sqrt(3) ~ 0.577; the mollifier is tighter.
    assert s.std().item() < u.std().item()
    assert abs(u.std().item() - 1 / math.sqrt(3)) < 0.03

    # The mollifier vanishes to infinite order at +-1, so the extreme tail is empty.
    assert (s.abs() > 0.97).sum().item() == 0
    assert (u.abs() > 0.97).sum().item() > 0


def test_zero_jitter_is_a_noop():
    opt = _optimizer(probe_radius_jitter=0.0, seed=2)
    assert opt._apply_probe_radius_jitter(1.5) == 1.5


def test_jitter_is_seed_reproducible():
    a = [_optimizer(probe_radius_jitter=0.2, seed=7)._sample_jitter(0.2) for _ in range(1)]
    b = [_optimizer(probe_radius_jitter=0.2, seed=7)._sample_jitter(0.2) for _ in range(1)]
    assert a == b


def test_invalid_distribution_rejected():
    try:
        _optimizer(probe_radius_jitter_dist="gaussian")
    except ValueError as exc:
        assert "smooth" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --- Jitter is exclusive with the amortization heuristics -------------------------
#
# Jitter makes the per-step cost a noisy estimate of the same quantity. Adaptive
# probes read that noise as a moved particle, cost-row reuse would mix rows measured
# at different radii, and amortized OT reads it as non-monotone progress and coasts
# on a stale direction. The constructor is the one place every caller routes through,
# so it enforces the exclusion there instead of leaving each config to remember.


def test_jitter_disables_amortization_and_adaptive_probes():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = _optimizer(probe_radius_jitter=0.05, amortize_steps=3, adaptive_probes=True, seed=0)
    assert opt.amortize_steps == 1
    assert opt._adaptive_probes is False
    msg = " ".join(str(w.message) for w in caught)
    assert "amortize_steps" in msg and "adaptive_probes" in msg, msg


def test_no_jitter_leaves_the_heuristics_alone():
    opt = _optimizer(probe_radius_jitter=0.0, amortize_steps=3, adaptive_probes=True, seed=0)
    assert opt.amortize_steps == 3
    assert opt._adaptive_probes is True


# --- Step-radius jitter: absolute continuity of the one-step law -------------------
#
# Probe jitter randomizes where the cost row is measured.  Step jitter randomizes how
# far the iterate moves, which is a different property: without it the displacement
# has a deterministic length, so the one-step law sits on a sphere -- a Lebesgue-null
# subset of the probe plane.  Cor. 4.12 needs a volume there, and the argument that
# the iterate never lands on the discontinuity set is cleaner with one.


def test_step_jitter_is_a_noop_at_zero():
    opt = _optimizer(step_radius_jitter=0.0, seed=2)
    X = torch.zeros(3, 4)
    X_bary = torch.ones(3, 4)
    assert torch.equal(opt._apply_particle_step_jitter(X, X_bary), X_bary)


def test_step_jitter_rejects_out_of_range():
    for bad in (-0.1, 1.0, 2.5):
        try:
            _optimizer(step_radius_jitter=bad)
        except ValueError as exc:
            assert "step_radius_jitter" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_step_jitter_spreads_the_displacement_length():
    """Without jitter every step has the same length; with it, a spread of lengths.

    This is the property the analysis uses, so it is the property worth asserting:
    the displacement norms must not collapse onto a single value.
    """
    from polystep.cost_nn import NNCostEvaluator

    def norms(step_jitter):
        torch.manual_seed(0)
        model = torch.nn.Linear(6, 3)
        opt = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            step_radius_jitter=step_jitter,
            seed=11,
        )
        evaluator = NNCostEvaluator(model, loss_fn=torch.nn.MSELoss())
        inputs = torch.randn(8, 6)
        target = torch.randn(8, 3)

        def closure(batched_params, _i=inputs, _t=target):
            return evaluator.evaluate(batched_params, _i, _t)

        out = []
        for _ in range(12):
            before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
            opt.step(closure)
            after = torch.cat([p.detach().flatten() for p in model.parameters()])
            out.append(float((after - before).norm()))
        return torch.tensor(out)

    jittered = norms(0.3)
    # A degenerate sampler would return a constant; the jitter must actually vary.
    assert jittered.std().item() > 0.0
    assert (jittered > 0).all()


def test_step_jitter_is_drawn_per_particle():
    """One shared eta would leave the joint step on a lower-dimensional manifold.

    Independence across particles is exactly what gives the joint one-step law a
    density on the search subspace, so it is the property to assert.
    """
    opt = _optimizer(step_radius_jitter=0.4, seed=5)
    X = torch.zeros(6, 3)
    X_bary = torch.ones(6, 3)
    scales = (opt._apply_particle_step_jitter(X, X_bary) - X)[:, 0]
    assert scales.std().item() > 0.0, scales
    assert len(set(scales.tolist())) > 1, scales
