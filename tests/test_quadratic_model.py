"""Finite-difference quadratic model: gradient, curvature, Newton step and trust region."""

import math

import pytest
import torch


def _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P, c=10.0):
    """Helper: generate losses_3d for a known quadratic f(x) = c + g*x + 0.5*H*x^2."""
    K = len(scales)
    V = 2 * pdim
    losses_3d = torch.zeros(P, V, K)
    for p in range(P):
        for k in range(K):
            t = scales[k] * probe_radius
            for i in range(pdim):
                losses_3d[p, i, k] = c + true_grad[p, i] * t + 0.5 * true_hess[p, i] * t**2
                losses_3d[p, i + pdim, k] = c - true_grad[p, i] * t + 0.5 * true_hess[p, i] * t**2
    return losses_3d


def test_fd_gradient_and_hessian_recover_quadratic():
    pdim = 2
    P = 1
    K = 5

    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    from polystep.geometry import get_orthoplex_vertices
    from polystep.quadratic_model import extract_fd_gradient, extract_fd_hessian_diag

    grad = extract_fd_gradient(losses_3d, scales, probe_radius, pdim, get_orthoplex_vertices(pdim))
    assert grad.shape == (P, pdim)
    torch.testing.assert_close(grad, true_grad, atol=1e-5, rtol=1e-5)

    hess = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)
    assert hess.shape == (P, pdim)
    torch.testing.assert_close(hess, true_hess, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("probe_radius", [1.0, 1e-2, 2e-3, 1e-4])
def test_fd_hessian_survives_small_probe_radius(probe_radius):
    """Curvature must not depend on the probe radius the loss was measured at.

    The regression denominator scales as probe_radius**4, so regressing on the
    absolute offsets loses the curvature entirely once epsilon anneals.
    """
    pdim, P, K = 2, 1, 2
    true_grad = torch.tensor([[0.5, 0.2]])
    true_hess = torch.tensor([[3.0, -1.0]])
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]

    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P, c=0.0)

    from polystep.quadratic_model import extract_fd_hessian_diag

    hess = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)
    torch.testing.assert_close(hess, true_hess, atol=1e-2, rtol=1e-2)


def test_newton_step_recovers_minimum():
    gradient = torch.tensor([[3.0, -2.0]])
    hessian_diag = torch.tensor([[2.0, 5.0]])

    from polystep.quadratic_model import compute_newton_step

    step = compute_newton_step(gradient, hessian_diag)
    expected = torch.tensor([[-1.5, 0.4]])
    torch.testing.assert_close(step, expected, atol=1e-5, rtol=1e-5)


def test_newton_step_clamps_norm():
    """Newton step should be clamped to max_step_norm."""
    gradient = torch.tensor([[100.0, 0.0]])
    hessian_diag = torch.tensor([[1.0, 1.0]])

    from polystep.quadratic_model import compute_newton_step

    step = compute_newton_step(gradient, hessian_diag, max_step_norm=1.0)
    assert torch.norm(step).item() <= 1.0 + 1e-6


def test_newton_step_regularizes_small_hessian():
    """A flat coordinate divides by hessian_reg, so both clips have to bind.

    Left at the default max_step_norm: -1/1e-4 caps per coordinate to -10, then the
    global clip rescales to a norm of exactly 10.
    """
    gradient = torch.tensor([[1.0, 1.0]])
    hessian_diag = torch.tensor([[1e-10, 1e-10]])

    from polystep.quadratic_model import compute_newton_step

    step = compute_newton_step(gradient, hessian_diag, hessian_reg=1e-4)
    torch.testing.assert_close(step, torch.full((1, 2), -10.0 / math.sqrt(2)))
    assert torch.norm(step).item() == pytest.approx(10.0)


def test_trust_region_shrinks_on_predicted_increase():
    """pred>0 means the model called the step harmful, so it shrinks at any ratio."""
    from polystep.quadratic_model import update_trust_region

    # Confirmed harmful, and ratio = 1 clears both thresholds, so only pred>0 shrinks it.
    assert update_trust_region(torch.tensor([1.0]), torch.tensor([1.0]), current_radius=1.0) == pytest.approx(0.5)
    # Predicted harmful, five times worse in reality.
    assert update_trust_region(torch.tensor([2.0]), torch.tensor([10.0]), current_radius=1.0) == pytest.approx(0.5)


# ratio = actual / pred against the default thresholds (shrink 0.25, expand 0.75).
@pytest.mark.parametrize(
    "actual, expected",
    [
        (-1.6, 1.5),  # ratio 0.8, above the expand threshold
        (-1.4, 1.0),  # ratio 0.7, between the two: hold
        (-0.6, 1.0),  # ratio 0.3, still above the shrink threshold
        (-0.4, 0.5),  # ratio 0.2, below it
    ],
)
def test_trust_region_straddles_default_thresholds(actual, expected):
    from polystep.quadratic_model import update_trust_region

    r = update_trust_region(torch.tensor([-2.0]), torch.tensor([actual]), current_radius=1.0)
    assert r == pytest.approx(expected)


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
    """Every branch, including both saturation points."""
    from polystep.quadratic_model import update_trust_region

    got = update_trust_region(
        predicted_improvement=torch.tensor([pred]),
        actual_improvement=torch.tensor([actual]),
        current_radius=radius,
        min_radius=0.1,
        max_radius=3.0,
    )
    assert got == pytest.approx(expected), why


def test_predicted_improvement():
    """Predicted improvement should match quadratic model."""
    gradient = torch.tensor([[3.0, -2.0]])
    hessian_diag = torch.tensor([[2.0, 5.0]])
    step = torch.tensor([[-1.5, 0.4]])

    from polystep.quadratic_model import compute_predicted_improvement

    pred = compute_predicted_improvement(gradient, hessian_diag, step)
    expected = torch.tensor([-2.65])
    torch.testing.assert_close(pred, expected, atol=1e-4, rtol=1e-4)


def test_predicted_improvement_floors_negative_curvature():
    """Scoring must use the floored-curvature model the step was built from."""
    from polystep.quadratic_model import compute_predicted_improvement

    gradient = torch.zeros(1, 1)
    hessian_diag = torch.tensor([[-4.0]])
    near = compute_predicted_improvement(gradient, hessian_diag, torch.tensor([[1.0]]))
    far = compute_predicted_improvement(gradient, hessian_diag, torch.tensor([[10.0]]))

    assert near.item() > 0.0, "flat gradient with floored curvature cannot predict a gain"
    assert far.item() > near.item(), "a longer step must not score better under a floored model"


def test_fd_gradient_batch_particles():
    """FD gradient should work with multiple particles (P > 1)."""
    pdim = 3
    P = 4
    K = 3

    true_grad = torch.randn(P, pdim)
    true_hess = torch.rand(P, pdim) + 0.5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 0.5
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P, c=5.0)

    from polystep.geometry import get_orthoplex_vertices
    from polystep.quadratic_model import extract_fd_gradient, extract_fd_hessian_diag

    grad = extract_fd_gradient(losses_3d, scales, probe_radius, pdim, get_orthoplex_vertices(pdim))
    hess = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)
    torch.testing.assert_close(grad, true_grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hess, true_hess, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "x_bary, alpha, expected, tol",
    [
        (torch.zeros(1, 2), 1.0, torch.tensor([[-1.5, 0.4]]), 1e-3),
        (torch.tensor([[1.0, 2.0]]), 0.0, torch.tensor([[1.0, 2.0]]), 1e-6),
        (torch.zeros(1, 2), 0.3, torch.tensor([[-0.45, 0.12]]), 1e-3),
    ],
)
def test_newton_alpha_one_moves_toward_minimum(x_bary, alpha, expected, tol):
    """apply_newton_refinement blends X_bary with the Newton-corrected minimum by alpha."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0

    # Quadratic: minimum at x* = -g/H = [-1.5, 0.4]
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    X_refined = apply_newton_refinement(
        X_bary=x_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=x_bary,
        alpha=alpha,
        max_step_norm=10.0,
        hessian_reg=1e-4,
    )

    assert X_refined.shape == (P, pdim)
    torch.testing.assert_close(X_refined, expected, atol=tol, rtol=tol)


def test_newton_refinement_anchors_at_probe_center():
    """The Newton step is measured from the probe center X_current, so when
    X_bary differs the refinement does not double-count the OT move."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0

    # Quadratic centered at the probe center (origin); minimum at -g/H = [-1.5, 0.4].
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_current = torch.zeros(P, pdim)  # probe center
    X_bary = torch.tensor([[0.5, 0.5]])  # OT result, offset from the center
    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    # alpha=1.0 must land at X_current + delta = [-1.5, 0.4] (the true minimum),
    # not X_bary + delta = [-1.0, 0.9].
    X_refined = apply_newton_refinement(
        X_bary=X_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=X_current,
        alpha=1.0,
        max_step_norm=10.0,
        hessian_reg=1e-4,
    )
    expected_minimum = torch.tensor([[-1.5, 0.4]])
    torch.testing.assert_close(X_refined, expected_minimum, atol=1e-3, rtol=1e-3)


def test_newton_gate_keeps_bary_when_worse():
    """When a clipped Newton step lands at a worse modeled point than the OT
    step, the descent gate keeps X_bary instead of the ascent blend."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0

    # Steep quadratic with minimum at [-10, -10]; probe center at origin.
    true_grad = torch.tensor([[10.0, 10.0]])
    true_hess = torch.tensor([[1.0, 1.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_current = torch.zeros(P, pdim)
    X_bary = torch.tensor([[-10.0, -10.0]])  # OT already sits near the minimum
    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    X_refined = apply_newton_refinement(
        X_bary=X_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=X_current,
        alpha=1.0,
        max_step_norm=0.5,  # clips Newton far short of the minimum
        hessian_reg=1e-4,
    )
    torch.testing.assert_close(X_refined, X_bary)


def test_newton_refinement_handles_near_zero_hessian():
    """Regularization keeps the step bounded."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0

    true_grad = torch.tensor([[1.0, 1.0]])
    true_hess = torch.tensor([[1e-10, 1e-10]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_bary = torch.zeros(P, pdim)
    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    X_refined = apply_newton_refinement(
        X_bary=X_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=X_bary,
        alpha=1.0,
        max_step_norm=1.0,
        hessian_reg=1e-4,
    )

    assert torch.isfinite(X_refined).all()
    step_norm = torch.norm(X_refined - X_bary).item()
    assert step_norm <= 1.0 + 1e-6, f"Step norm {step_norm} exceeded max_step_norm 1.0"


def test_newton_refinement_respects_max_step_norm():
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1 : K + 1]
    probe_radius = 1.0

    # Large gradient, small Hessian -> large Newton step (will be clamped)
    true_grad = torch.tensor([[100.0, 100.0]])
    true_hess = torch.tensor([[1.0, 1.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_bary = torch.zeros(P, pdim)
    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    X_refined = apply_newton_refinement(
        X_bary=X_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=X_bary,
        alpha=1.0,
        max_step_norm=0.5,
        hessian_reg=1e-4,
    )

    # With alpha=1.0, X_refined = X_bary + clamped_newton_step
    correction_norm = torch.norm(X_refined - X_bary).item()
    assert correction_norm <= 0.5 + 1e-6, f"Correction norm {correction_norm} exceeded max_step_norm 0.5"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_orthoplex_gradient_matches_central_difference(dtype):
    """sum_v v v^T = (V/d) I collapses the least-squares fit to (d/(V s r)) sum_v L_v v.

    On antipodal pairs that is the central difference with the 2d-2 zero terms dropped,
    up to the rounding of the shared centring and of the mean over probe scales.
    """
    from polystep.geometry import get_orthoplex_vertices
    from polystep.quadratic_model import extract_fd_gradient

    gen = torch.Generator().manual_seed(0)
    for pdim in (2, 4, 8):
        verts = get_orthoplex_vertices(pdim, dtype=dtype)
        for K in (1, 3, 6):
            losses = (1.0 + 0.1 * torch.randn(4, 2 * pdim, K, generator=gen, dtype=torch.float64)).to(dtype)
            scales = torch.linspace(0, 1, K + 2, dtype=dtype)[1 : K + 1]
            for radius in (0.1, 0.4, 1.0):
                denom = (2.0 * scales * radius).unsqueeze(0).unsqueeze(0).clamp(min=1e-10)
                reference = ((losses[:, :pdim, :] - losses[:, pdim:, :]) / denom).mean(dim=-1)
                torch.testing.assert_close(extract_fd_gradient(losses, scales, radius, pdim, verts), reference)


def test_simplex_recovers_gradient_and_curvature_trace():
    """The simplex carries an O(r) third-moment bias the orthoplex cancels; the shared
    centre recovers tr(H)/d exactly at any radius."""
    from polystep.geometry import get_simplex_vertices
    from polystep.quadratic_model import extract_fd_gradient, extract_iso_curvature

    torch.manual_seed(0)
    d = 8
    verts = get_simplex_vertices(d, dtype=torch.float64)
    g_true = torch.randn(d, dtype=torch.float64)
    A = torch.randn(d, d, dtype=torch.float64)
    H = A + A.T
    L0 = 1.234

    for radius, tol in ((0.01, 0.02), (0.1, 0.1)):
        losses = torch.stack([L0 + g_true @ (radius * v) + 0.5 * (radius * v) @ H @ (radius * v) for v in verts])
        losses_3d = losses.reshape(1, verts.shape[0], 1)
        scales = torch.ones(1, dtype=torch.float64)

        ghat = extract_fd_gradient(losses_3d, scales, radius, d, verts)[0]
        assert (ghat - g_true).norm() / g_true.norm() < tol

        curv = extract_iso_curvature(losses_3d, torch.tensor([L0], dtype=torch.float64), scales, radius)
        assert curv.shape == (1, 1)
        assert curv.item() == pytest.approx(torch.trace(H).item() / d, rel=1e-9)


def test_fd_gradient_without_verts_raises():
    """A 2D cube has 2*pdim vertices too, but its halves are not antipodal."""
    from polystep.geometry import get_cube_vertices
    from polystep.quadratic_model import extract_fd_gradient

    v = get_cube_vertices(2).double()
    s = torch.tensor([0.5, 1.0], dtype=torch.float64)
    losses = (0.05 * s.view(1, -1, 1) * v.unsqueeze(1) @ torch.tensor([0.7, -1.3], dtype=torch.float64)).unsqueeze(0)
    with pytest.raises(TypeError):
        extract_fd_gradient(losses, s, 0.05, 2)


def test_newton_step_survives_flat_coordinate():
    """Dividing by hessian_reg gives that coordinate a 1e4x step.

    The norm clip is a single global rescale, so without a per-coordinate cap the flat
    coordinate dominates the norm and shrinks every other component by the same factor.
    """
    from polystep.quadratic_model import compute_newton_step

    gradient = torch.full((1, 10), 0.1)
    hessian = torch.ones(1, 10)
    hessian[0, 9] = 0.0  # flat direction

    delta = compute_newton_step(gradient, hessian, max_step_norm=1.0, hessian_reg=1e-4)

    good = delta[0, :9].abs()
    assert (good > 0.01).all(), good
    assert delta[0, 9].abs() <= 1.0 + 1e-6
    assert (delta <= 0).all(), "Newton step must not ascend"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_simplex_losses_give_zero_gradient(dtype):
    from polystep.geometry import get_simplex_vertices
    from polystep.quadratic_model import extract_fd_gradient

    g = extract_fd_gradient(
        torch.full((2, 9, 1), 1e6, dtype=dtype),
        torch.ones(1, dtype=dtype),
        0.01,
        8,
        get_simplex_vertices(8, dtype=dtype),
    )
    assert torch.count_nonzero(g) == 0
