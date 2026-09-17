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
from polystep.quadratic_model import update_trust_region


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


def test_trust_region_runs_at_num_probe_one_via_the_shared_centre():
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


def test_trust_region_runs_on_the_simplex_at_num_probe_one():
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


def test_trust_region_warns_when_amortization_makes_it_inert():
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


def test_the_simplex_model_uses_the_shared_centre_at_multiple_scales():
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


def test_a_probe_reuse_step_does_not_move_the_trust_region():
    """Reuse re-reports the cached loss, so the ratio would be 0 against a live prediction."""
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
            adaptive_probes=True,
            adaptive_probes_threshold=1e9,
            compile=False,
            seed=0,
        )
    closure = _closure(model)
    opt.step(closure, objective_token=0)
    before, evals = opt._trust_region_multiplier, opt.candidate_evals
    opt.step(closure, objective_token=0)

    assert opt.candidate_evals == evals, "expected a reuse step, got fresh evaluations"
    assert opt._trust_region_multiplier == before


@pytest.mark.parametrize(
    "pred, actual, radius, expected, why",
    [
        (-1.0, 1.0, 1.0, 0.25, "predicted a gain and the loss rose: aggressive shrink"),
        (-1.0, 2.0, 0.3, 0.1, "the aggressive shrink clamps at min_radius"),
        (1.0, -0.4, 1.0, 0.5, "predicted a loss and the loss fell: ordinary shrink, not double"),
        (-1.0, -1.0, 2.5, 3.0, "expansion saturates at max_radius"),
        (-1.0, -0.1, 0.15, 0.1, "shrink saturates at min_radius"),
        (-1.0, -0.5, 1.0, 1.0, "between the thresholds the radius holds"),
        (1.0, 1.0, 1.0, 0.5, "predicted a rise and the loss rose: shrink, never hold"),
        (1e-12, 1.0, 1.7, 1.7, "a degenerate prediction leaves the radius alone"),
    ],
)
def test_trust_region_branches_and_clamps(pred, actual, radius, expected, why):
    """Every branch including both saturation points; the existing test only expands."""
    got = update_trust_region(
        predicted_improvement=torch.tensor([pred]),
        actual_improvement=torch.tensor([actual]),
        current_radius=radius,
        min_radius=0.1,
        max_radius=3.0,
    )
    assert got == pytest.approx(expected), why


def test_kl_marginal_violation_matches_the_generalized_kl():
    """last_marginal_violation is reported to callers and has no other check."""
    from polystep.solvers.kl_softmax import KLSoftmaxSolver

    C = torch.rand(5, 7, generator=torch.Generator().manual_seed(0))
    solver = KLSoftmaxSolver(epsilon=0.2, lam=1.0, max_iterations=400)
    res = solver.solve(C)

    q = res.matrix.sum(dim=0)
    b = torch.full((7,), 1.0 / 7)
    expected = (q * (q.log() - b.log()) - q + b).sum().item()
    assert solver.last_marginal_violation == pytest.approx(expected, rel=1e-6)
    # The mass terms are what keep it non-negative when sum(q) != sum(b).
    assert solver.last_marginal_violation >= 0


@pytest.mark.parametrize(
    "lam, epsilon, expected",
    [(0.0, 0.5, 0.0), (float("inf"), 0.5, 1.0), (1.0, 1.0, 0.5), (3.0, 1.0, 0.75)],
)
def test_kl_alpha_interpolates_softmax_to_sinkhorn(lam, epsilon, expected):
    """alpha = lam/(lam+eps); a lam/(lam+1) typo still passes both endpoints."""
    from polystep.solvers.kl_softmax import KLSoftmaxSolver

    assert KLSoftmaxSolver(epsilon=epsilon, lam=lam).alpha == pytest.approx(expected)


def test_centred_hessian_matches_the_regression_on_a_quadratic():
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


def test_the_centre_evaluation_is_charged_once_not_once_per_particle():
    """f(X) is one forward, however many particles read it.

    A centre candidate is X with one row rewritten to the value already there,
    so every particle's centre is the same point and the same number. Evaluating
    one per particle billed P-1 forwards that computed nothing, and the budget
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
