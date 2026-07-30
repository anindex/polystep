"""The trust-region ratio test runs wherever the finite-difference model exists.

Its inputs are ``use_quadratic_model`` and the orthoplex. Curvature comes from the
regression at ``num_probe >= 2`` and from a shared ``f(X)`` at ``num_probe == 1``, so
the multiplier must move in both, with or without ``biased_rotation``.
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
    """One f(X) per particle replaces the second probe scale the regression needs."""
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


def test_trust_region_stays_off_on_the_simplex():
    """The FD extractors read the orthoplex's antithetic ordering."""
    torch.manual_seed(0)
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    assert len(opt._state.trust_region_multipliers) == 0


@pytest.mark.parametrize(
    "pred, actual, radius, expected, why",
    [
        (-1.0, 1.0, 1.0, 0.25, "predicted a gain and the loss rose: aggressive shrink"),
        (-1.0, 2.0, 0.3, 0.1, "the aggressive shrink clamps at min_radius"),
        (1.0, -0.4, 1.0, 0.5, "predicted a loss and the loss fell: ordinary shrink, not double"),
        (-1.0, -1.0, 2.5, 3.0, "expansion saturates at max_radius"),
        (-1.0, -0.1, 0.15, 0.1, "shrink saturates at min_radius"),
        (-1.0, -0.5, 1.0, 1.0, "between the thresholds the radius holds"),
        (1.0, 1.0, 1.0, 1.0, "an accurate but worsening step must not expand"),
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
    """last_marginal_violation feeds Theorem 4.1 reporting and has no other check."""
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
