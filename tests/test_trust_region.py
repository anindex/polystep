"""The trust-region ratio test runs wherever the finite-difference model exists.

Its input is ``use_quadratic_model``. Curvature comes from a regression across probe
scales, which needs the orthoplex's antipodal pairs, or from a shared ``f(X)`` at
``num_probe == 1``, which works on any polytope. The multiplier must move in both,
with or without ``biased_rotation``.
"""

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _closure(model):
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(32, 4)
    targets = torch.randn(32, 1)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


def _run(optimizer, closure, steps=12):
    for _ in range(steps):
        optimizer.step(closure)


def test_trust_region_activates_without_biased_rotation():
    """Multiplier list becomes nonempty and nonconstant with trust_region only."""
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        num_probe=2,
        polytope_type="orthoplex",
        trust_region=True,
        biased_rotation=False,
        compile=False,
        seed=0,
    )
    # trust_region auto-enables the quadratic model.
    assert opt.use_quadratic_model is True

    _run(opt, _closure(model))

    mults = opt._state.trust_region_multipliers
    assert len(mults) > 0, "trust_region recorded no ratio test; multiplier stayed frozen"
    assert len(set(mults)) > 1, "trust_region multiplier never changed"
    for m in mults:
        assert 0.1 <= m <= 3.0


def test_trust_region_one_probe_shares_centre():
    """One shared f(X) replaces the second probe scale the regression needs."""
    torch.manual_seed(0)
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            num_probe=1,
            polytope_type="orthoplex",
            trust_region=True,
            biased_rotation=False,
            compile=False,
            seed=0,
        )
    _run(opt, _closure(model))
    assert len(opt._state.trust_region_multipliers) > 0
    assert opt._center_loss is not None and opt._center_loss.shape == (opt._state.X.shape[0],)


def test_trust_region_runs_simplex_one_probe():
    """The gradient is a closed form on any centred tight frame, and the shared f(X)
    gives tr(H)/d, so d+1 vertices carry the model the orthoplex needed 2d for."""
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        num_probe=1,
        polytope_type="simplex",
        trust_region=True,
        compile=False,
        seed=0,
    )
    _run(opt, _closure(model))

    mults = opt._state.trust_region_multipliers
    assert len(mults) > 0, "trust_region recorded no ratio test on the simplex"
    assert len(set(mults)) > 1, "trust_region multiplier never changed"
    assert all(0.1 <= m <= 3.0 for m in mults)


@pytest.mark.parametrize(("polytope", "antipodal"), [("orthoplex", True), ("cube", False)])
def test_step_gradient_antipodal_only_for_orthoplex(monkeypatch, polytope, antipodal):
    """The cube at particle_dim=2 also has 2*pdim vertices, so routing must key on the polytope."""
    import polystep._step_monolithic as step_module

    calls = []
    real = step_module.extract_fd_antipodal_gradient
    monkeypatch.setattr(step_module, "extract_fd_antipodal_gradient", lambda *a: calls.append(1) or real(*a))
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs, targets = torch.randn(8, 2), torch.randn(8, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(
            model,
            max_iterations=5,
            epsilon=0.1,
            num_probe=1,
            polytope_type=polytope,
            trust_region=True,
            compile=False,
            seed=0,
            particle_dim=2,
        )
    opt.step(lambda params: evaluator.evaluate(params, inputs, targets))
    assert bool(calls) == antipodal


def test_amortized_trust_region_warns_inert():
    """Momentum steps drop the pending prediction, so the ratio test never runs."""
    torch.manual_seed(0)
    model = _model()
    with pytest.warns(UserWarning, match="trust_region never updates"):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            num_probe=1,
            trust_region=True,
            amortize_steps=5,
            compile=False,
            seed=0,
        )
    _run(opt, _closure(model))
    assert len(opt._state.trust_region_multipliers) == 0


def test_simplex_model_uses_shared_centre_multiscale():
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        num_probe=2,
        polytope_type="simplex",
        trust_region=True,
        compile=False,
        seed=0,
    )
    _run(opt, _closure(model))
    assert opt._center_loss is not None
    assert len(opt._state.trust_region_multipliers) > 0


def test_trust_region_survives_adaptive_rotation():
    """A rotation or absorb leaves the weights where they were, so the pending comparison
    survives it, and AdaptiveSubspace rotates every step."""
    from polystep import AdaptiveSubspace

    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        subspace=AdaptiveSubspace.auto_from_params(model),
        max_iterations=50,
        epsilon=0.1,
        trust_region=True,
        compile=False,
        seed=0,
    )
    _run(opt, _closure(model), steps=20)
    assert len(opt._state.trust_region_multipliers) > 0


def _second_move(solver, predicted, pre_step_loss):
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        solver=solver,
        max_iterations=50,
        epsilon=0.1,
        num_probe=1,
        polytope_type="simplex",
        trust_region=True,
        compile=False,
        seed=0,
    )
    closure = _closure(model)
    opt.step(closure)
    opt._prev_predicted_improvement, opt._prev_pre_step_loss = predicted, pre_step_loss
    before = opt.state.X.clone()
    opt.step(closure)
    return opt.state.X - before, opt._state.trust_region_multipliers


@pytest.mark.parametrize("solver", ["sinkhorn", "softmax"])
def test_failed_step_shrinks_same_step_radius(solver):
    """A promised gain that came out as a loss quarters the radius."""
    shrunk, mults = _second_move(solver, torch.tensor(-1.0), -1e9)
    held, _ = _second_move(solver, None, None)
    assert mults == [0.25]
    assert held.abs().max() > 0
    torch.testing.assert_close(shrunk, 0.25 * held)


@pytest.mark.parametrize("first, second, recorded", [(0, 0, 1), (None, None, 1), (0, 1, 0)])
def test_objective_token_change_drops_pending_pair(first, second, recorded):
    """A prediction is scored only against the same objective; None or a constant token is stationary."""
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        num_probe=1,
        polytope_type="simplex",
        trust_region=True,
        compile=False,
        seed=0,
    )
    closure = _closure(model)
    opt.step(closure, objective_token=first)
    assert opt._prev_predicted_improvement is not None
    opt.step(closure, objective_token=second)
    assert len(opt._state.trust_region_multipliers) == recorded


@pytest.mark.parametrize(
    "lam, epsilon, expected",
    [(0.0, 0.5, 0.0), (float("inf"), 0.5, 1.0), (1.0, 1.0, 0.5), (3.0, 1.0, 0.75)],
)
def test_kl_alpha_interpolates_softmax_to_sinkhorn(lam, epsilon, expected):
    """alpha = lam/(lam+eps); a lam/(lam+1) typo still passes both endpoints."""
    from polystep.solvers.kl_softmax import KLSoftmaxSolver

    assert KLSoftmaxSolver(epsilon=epsilon, lam=lam).alpha == pytest.approx(expected)


def test_centred_hessian_matches_regression_on_quadratic():
    """Both estimators recover the same curvature where the loss really is quadratic."""
    from polystep.quadratic_model import extract_fd_hessian_diag, extract_fd_hessian_diag_centered

    torch.manual_seed(0)
    pdim, P = 3, 2
    H = torch.rand(P, pdim) * 2.0 + 0.5
    L0 = torch.randn(P)
    g = torch.randn(P, pdim)
    scales = torch.tensor([0.4, 0.8])
    r = 0.1

    offs = (scales * r).reshape(1, 1, -1)  # (1, 1, K)
    fwd = L0.reshape(-1, 1, 1) + g.unsqueeze(-1) * offs + 0.5 * H.unsqueeze(-1) * offs**2
    bwd = L0.reshape(-1, 1, 1) - g.unsqueeze(-1) * offs + 0.5 * H.unsqueeze(-1) * offs**2
    losses = torch.cat([fwd, bwd], dim=1)  # (P, 2*pdim, K)

    regressed = extract_fd_hessian_diag(losses, scales, r, pdim)
    centred = extract_fd_hessian_diag_centered(losses, scales, r, pdim, L0)
    torch.testing.assert_close(regressed, H, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(centred, H, rtol=1e-4, atol=1e-4)

    # One scale is enough for the centred form and not for the regression.
    one = losses[:, :, :1]
    torch.testing.assert_close(extract_fd_hessian_diag_centered(one, scales[:1], r, pdim, L0), H, rtol=1e-4, atol=1e-4)


def test_centre_evaluation_charged_once():
    """f(X) is one forward, however many particles read it.

    A centre candidate is X with one row rewritten to the value already there,
    so every particle's centre is the same point and the same number. One evaluation
    per particle would bill P-1 forwards that compute nothing, and the budget
    column is what the paper matches methods on.
    """
    torch.manual_seed(0)
    model = _model()
    opt = PolyStepOptimizer(
        model,
        max_iterations=50,
        polytope_type="orthoplex",
        use_quadratic_model=True,
        trust_region=True,
        num_probe=1,
        epsilon=0.1,
        seed=7,
        compile=False,
    )
    opt.step(_closure(model))

    P, pdim = opt.state.X.shape
    assert opt._center_loss.shape == (P,)
    assert float(opt._center_loss.max() - opt._center_loss.min()) == 0.0
    assert opt.candidate_evals == P * (2 * pdim) + 1


@pytest.mark.parametrize("num_probe", [1, 2])
def test_nn_trust_prediction_sums_parameter_blocks(num_probe):
    model = nn.Linear(6, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(2)
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        polytope_type="orthoplex",
        solver="softmax",
        trust_region=True,
        num_probe=num_probe,
        epsilon=1.0,
        step_radius=0.1,
        probe_radius=0.2,
        use_momentum=False,
        compile=False,
        seed=3,
    )

    def closure(params):
        return params["weight"].square().flatten(1).sum(1) / 2

    before = model.weight.square().sum().item() / 2
    opt.step(closure)
    actual = model.weight.square().sum().item() / 2 - before
    assert opt._center_loss is not None
    assert opt._prev_pre_step_loss == pytest.approx(before)
    # The stored scalar describes the entire NN, not an average parameter block.
    assert opt._prev_predicted_improvement.numel() == 1
    assert opt._prev_predicted_improvement.item() == pytest.approx(actual, abs=1e-10)
    assert opt.candidate_evals == 3 * 4 * num_probe + 1
