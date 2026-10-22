"""Probe-radius jitter: the smooth density the convergence analysis assumes.

The analysis needs the density C-infinity and compactly supported; the mollifier
q(t) ~ exp(-1/(1-t^2)) qualifies, a uniform density does not.
"""

import math

import pytest
import torch

from polystep.optimizer import PolyStepOptimizer


def _optimizer(**kw):
    model = torch.nn.Linear(4, 2)
    return PolyStepOptimizer(model, **kw)


def test_jitter_respects_support():
    opt = _optimizer(probe_radius_jitter=0.3, seed=0)
    etas = opt._sample_jitter_batch(0.3, 500)
    assert etas.abs().max().item() < 0.3
    assert etas.abs().max().item() > 0.05  # not degenerate


def test_jitter_reproducible_under_seed():
    a = _optimizer(probe_radius_jitter=0.2, seed=7)._sample_jitter_pooled(0.2)
    b = _optimizer(probe_radius_jitter=0.2, seed=7)._sample_jitter_pooled(0.2)
    assert a == b


def test_invalid_distribution_rejected():
    with pytest.raises(ValueError, match="smooth"):
        _optimizer(probe_radius_jitter_dist="gaussian")


class TestProbeRadiusJitter:
    """Probe-radius jitter gives the step density a spread along any wall normal.

    The probe distribution must be absolutely continuous on a positive-measure
    tube around the (d_p-1)-sphere; the 0.0 default keeps runs reproducible.
    """

    def test_default_jitter_noop(self):
        """With jitter=0, _apply_probe_radius_jitter returns the input unchanged
        and consumes NO random state from the optimizer's generator."""
        opt = _optimizer(particle_dim=2, probe_radius=2.0, seed=42)
        # Snapshot the generator state BEFORE calling the helper.
        state_before = opt._generator.get_state().clone()
        result = opt._apply_probe_radius_jitter(2.0)
        state_after = opt._generator.get_state()
        assert result == 2.0
        # No random call should have occurred.
        assert torch.equal(state_before, state_after)

    def test_jitter_perturbs_probe_radius(self):
        """With jitter > 0, _apply_probe_radius_jitter returns a value in
        the bounded multiplicative interval [(1-eta_max)*r, (1+eta_max)*r]
        and consumes random state from the optimizer's generator."""
        eta_max = 0.05
        opt = _optimizer(particle_dim=2, probe_radius=2.0, probe_radius_jitter=eta_max, seed=42)
        base = 2.0
        # Sample many jitter values and check (a) they are all in the interval,
        # (b) the variance is non-zero (jitter is actually being applied).
        samples = [opt._apply_probe_radius_jitter(base) for _ in range(200)]
        lo = base * (1.0 - eta_max)
        hi = base * (1.0 + eta_max)
        assert all(lo <= s <= hi for s in samples), (
            f"jitter samples must lie in [{lo}, {hi}], got min={min(samples)} max={max(samples)}"
        )
        assert max(samples) - min(samples) > 0.01, "jitter samples should span a non-trivial range"

    @pytest.mark.parametrize("bad", [1.0, -0.1])
    def test_jitter_validation_rejects_out_of_range(self, bad):
        """probe_radius_jitter must lie in [0, 1) - values >= 1 risk negative
        effective probe radius and are rejected at init time."""
        with pytest.raises(ValueError, match="probe_radius_jitter"):
            _optimizer(probe_radius_jitter=bad)

    def test_jitter_step_runs_and_changes_iterates(self, regression_closure):
        """End-to-end: optimization with jitter > 0 still produces a valid
        step and updates the iterate, exercising the helper from inside the
        full step path (resolve_radii / _step_monolithic)."""
        torch.manual_seed(0)
        opt = _optimizer(particle_dim=2, probe_radius=2.0, probe_radius_jitter=0.05, seed=42)
        model = opt.model
        closure = regression_closure(model)
        params_before = torch.cat([p.detach().flatten() for p in model.parameters()])
        opt.step(closure)
        params_after = torch.cat([p.detach().flatten() for p in model.parameters()])
        assert opt._state.iteration_count == 1
        # The step must have produced a finite, non-trivial update.
        assert torch.isfinite(params_after).all()
        assert not torch.equal(params_before, params_after)


# --- Jitter is exclusive with the amortization heuristics -------------------------
#
# Jitter noise reads as particle movement to the amortization heuristic, so the
# constructor turns it off instead of leaving each config to remember.


def test_jitter_disables_amortization():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = _optimizer(probe_radius_jitter=0.05, amortize_steps=3, seed=0)
    assert opt.amortize_steps == 1
    msg = " ".join(str(w.message) for w in caught)
    assert "amortize_steps" in msg, msg


def test_no_jitter_leaves_amortization_alone():
    opt = _optimizer(probe_radius_jitter=0.0, amortize_steps=3, seed=0)
    assert opt.amortize_steps == 3


# --- Step-radius jitter: absolute continuity of the one-step law -------------------
#
# Without jitter the displacement has deterministic length, so the one-step law sits
# on a Lebesgue-null sphere; Cor. 4.12 needs a volume there.


def test_step_jitter_noop_at_zero():
    opt = _optimizer(step_radius_jitter=0.0, seed=2)
    X = torch.zeros(3, 4)
    X_bary = torch.ones(3, 4)
    assert torch.equal(opt._apply_particle_step_jitter(X, X_bary), X_bary)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_step_jitter_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="step_radius_jitter"):
        _optimizer(step_radius_jitter=bad)


def test_step_jitter_spreads_displacement_length():
    """Without jitter every step has the same length; with it, a spread of lengths."""
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
    """One shared eta would leave the joint step on a lower-dimensional manifold."""
    opt = _optimizer(step_radius_jitter=0.4, seed=5)
    X = torch.zeros(6, 3)
    X_bary = torch.ones(6, 3)
    scales = (opt._apply_particle_step_jitter(X, X_bary) - X)[:, 0]
    assert scales.std().item() > 0.0, scales
    assert len(set(scales.tolist())) > 1, scales


def test_jitter_sampler_matches_lemma_4_3():
    """Mollifier tighter than uniform, tail empty at +-1, symmetric about zero."""
    n = 20_000
    smooth = _optimizer(probe_radius_jitter=0.9, probe_radius_jitter_dist="smooth", seed=3)
    uniform = _optimizer(probe_radius_jitter=0.9, probe_radius_jitter_dist="uniform", seed=3)

    s = smooth._sample_jitter_batch(1.0, n)
    u = uniform._sample_jitter_batch(1.0, n)

    assert s.shape == (n,) and u.shape == (n,)
    assert s.abs().max().item() < 1.0 and u.abs().max().item() < 1.0
    assert s.abs().max().item() > 0.05
    assert s.std().item() < u.std().item()
    assert abs(u.std().item() - 1 / math.sqrt(3)) < 0.02
    # The mollifier vanishes to infinite order at +-1; uniform does not.
    assert (s.abs() > 0.97).sum().item() == 0
    assert (u.abs() > 0.97).sum().item() > 0
    # Symmetric about zero, so E[1+eta] = 1 and the radius is unbiased.
    assert abs(s.mean().item()) < 0.02
    assert abs(u.mean().item()) < 0.02
