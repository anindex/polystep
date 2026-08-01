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
