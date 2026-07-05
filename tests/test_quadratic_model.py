import torch
import pytest


def test_fd_gradient_quadratic_function():
    """FD gradient should recover the true gradient of a known quadratic."""
    pdim = 2
    P = 1
    K = 5
    V = 2 * pdim

    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0
    c = 10.0

    losses_3d = torch.zeros(P, V, K)
    for k in range(K):
        t = scales[k] * probe_radius
        for i in range(pdim):
            losses_3d[0, i, k] = c + true_grad[0, i] * t + 0.5 * true_hess[0, i] * t**2
            losses_3d[0, i + pdim, k] = c - true_grad[0, i] * t + 0.5 * true_hess[0, i] * t**2

    from polystep.quadratic_model import extract_fd_gradient

    grad = extract_fd_gradient(losses_3d, scales, probe_radius, pdim)
    assert grad.shape == (P, pdim)
    torch.testing.assert_close(grad, true_grad, atol=1e-5, rtol=1e-5)


def test_fd_hessian_diagonal_quadratic_function():
    """FD Hessian should recover true diagonal Hessian of a known quadratic."""
    pdim = 2
    P = 1
    K = 5
    V = 2 * pdim

    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0
    c = 10.0

    losses_3d = torch.zeros(P, V, K)
    for k in range(K):
        t = scales[k] * probe_radius
        for i in range(pdim):
            losses_3d[0, i, k] = c + true_grad[0, i] * t + 0.5 * true_hess[0, i] * t**2
            losses_3d[0, i + pdim, k] = c - true_grad[0, i] * t + 0.5 * true_hess[0, i] * t**2

    from polystep.quadratic_model import extract_fd_hessian_diag

    hess = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)
    assert hess.shape == (P, pdim)
    torch.testing.assert_close(hess, true_hess, atol=1e-4, rtol=1e-4)


def test_newton_step_recovers_minimum():
    """Newton step from quadratic model should point toward the minimum."""
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
    """Newton step should handle near-zero Hessian via regularization."""
    gradient = torch.tensor([[1.0, 1.0]])
    hessian_diag = torch.tensor([[1e-10, 1e-10]])

    from polystep.quadratic_model import compute_newton_step

    step = compute_newton_step(gradient, hessian_diag, hessian_reg=1e-4)
    assert torch.isfinite(step).all()
    assert torch.norm(step).item() < 1e6


def test_predicted_improvement():
    """Predicted improvement should match quadratic model."""
    gradient = torch.tensor([[3.0, -2.0]])
    hessian_diag = torch.tensor([[2.0, 5.0]])
    step = torch.tensor([[-1.5, 0.4]])

    from polystep.quadratic_model import compute_predicted_improvement

    pred = compute_predicted_improvement(gradient, hessian_diag, step)
    expected = torch.tensor([-2.65])
    torch.testing.assert_close(pred, expected, atol=1e-4, rtol=1e-4)


def test_fd_gradient_batch_particles():
    """FD gradient should work with multiple particles (P > 1)."""
    pdim = 3
    P = 4
    K = 3
    V = 2 * pdim

    true_grad = torch.randn(P, pdim)
    true_hess = torch.rand(P, pdim) + 0.5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 0.5
    c = 5.0

    losses_3d = torch.zeros(P, V, K)
    for p in range(P):
        for k in range(K):
            t = scales[k] * probe_radius
            for i in range(pdim):
                losses_3d[p, i, k] = c + true_grad[p, i] * t + 0.5 * true_hess[p, i] * t**2
                losses_3d[p, i + pdim, k] = c - true_grad[p, i] * t + 0.5 * true_hess[p, i] * t**2

    from polystep.quadratic_model import extract_fd_gradient, extract_fd_hessian_diag

    grad = extract_fd_gradient(losses_3d, scales, probe_radius, pdim)
    hess = extract_fd_hessian_diag(losses_3d, scales, probe_radius, pdim)
    torch.testing.assert_close(grad, true_grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hess, true_hess, atol=1e-3, rtol=1e-3)


def test_trust_region_expands_on_good_step():
    """Trust region should expand when actual closely matches predicted."""
    from polystep.quadratic_model import update_trust_region

    predicted = torch.tensor([-2.0])
    actual = torch.tensor([-1.8])
    new_radius = update_trust_region(predicted, actual, current_radius=1.0)
    assert new_radius > 1.0


def test_trust_region_shrinks_on_bad_step():
    """Trust region should shrink when actual is much worse than predicted."""
    from polystep.quadratic_model import update_trust_region

    predicted = torch.tensor([-2.0])
    actual = torch.tensor([-0.1])
    new_radius = update_trust_region(predicted, actual, current_radius=1.0)
    assert new_radius < 1.0


# ---------------------------------------------------------------------------
# Tests for apply_newton_refinement (Newton refinement)
# ---------------------------------------------------------------------------


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


def test_newton_refinement_alpha_one_moves_toward_minimum():
    """apply_newton_refinement with alpha=1.0 on a perfect quadratic moves X toward the minimum."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0

    # Quadratic: minimum at x* = -g/H = [-1.5, 0.4]
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    # X_bary is at the origin (far from minimum)
    X_bary = torch.zeros(P, pdim)
    # Identity rotation (no rotation)
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
        max_step_norm=10.0,
        hessian_reg=1e-4,
    )

    assert X_refined.shape == (P, pdim)
    # With alpha=1.0, X_refined = X_bary + Newton step
    # Newton step = -g/H = [-1.5, 0.4]
    # So X_refined should be close to [-1.5, 0.4] (the minimum)
    expected_minimum = torch.tensor([[-1.5, 0.4]])
    torch.testing.assert_close(X_refined, expected_minimum, atol=1e-3, rtol=1e-3)


def test_newton_refinement_anchors_at_probe_center_not_x_bary():
    """The Newton step is measured from the probe center X_current, so when
    X_bary differs the refinement does not double-count the OT move."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0

    # Quadratic centered at the probe center (origin); minimum at -g/H = [-1.5, 0.4].
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_current = torch.zeros(P, pdim)      # probe center
    X_bary = torch.tensor([[0.5, 0.5]])   # OT result, offset from the center
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


def test_newton_refinement_alpha_zero_returns_unchanged():
    """apply_newton_refinement with alpha=0.0 returns X unchanged."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
    losses_3d = _make_quadratic_losses_3d(true_grad, true_hess, scales, probe_radius, pdim, P)

    X_bary = torch.tensor([[1.0, 2.0]])
    rot_mats = torch.eye(pdim).unsqueeze(0).expand(P, -1, -1)

    X_refined = apply_newton_refinement(
        X_bary=X_bary,
        losses_3d=losses_3d,
        scales=scales,
        probe_radius=probe_radius,
        pdim=pdim,
        rot_mats=rot_mats,
        X_current=X_bary,
        alpha=0.0,
        max_step_norm=10.0,
        hessian_reg=1e-4,
    )

    torch.testing.assert_close(X_refined, X_bary, atol=1e-6, rtol=1e-6)


def test_newton_refinement_alpha_blending():
    """apply_newton_refinement with alpha=0.3 blends between X_bary and Newton-corrected position."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0
    true_grad = torch.tensor([[3.0, -2.0]])
    true_hess = torch.tensor([[2.0, 5.0]])
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
        alpha=0.3,
        max_step_norm=10.0,
        hessian_reg=1e-4,
    )

    # Newton step = [-1.5, 0.4], X_newton = [0, 0] + [-1.5, 0.4] = [-1.5, 0.4]
    # Blended: (1-0.3)*[0,0] + 0.3*[-1.5, 0.4] = [-0.45, 0.12]
    expected = torch.tensor([[-0.45, 0.12]])
    torch.testing.assert_close(X_refined, expected, atol=1e-3, rtol=1e-3)


def test_newton_refinement_handles_near_zero_hessian():
    """apply_newton_refinement with near-zero Hessian should not explode (regularization prevents it)."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
    probe_radius = 1.0

    # Near-zero Hessian: flat landscape (H ~ 0)
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
    # Step norm should be clamped to max_step_norm
    step_norm = torch.norm(X_refined - X_bary).item()
    assert step_norm <= 1.0 + 1e-6, f"Step norm {step_norm} exceeded max_step_norm 1.0"


def test_newton_refinement_respects_max_step_norm():
    """apply_newton_refinement should clamp Newton correction to max_step_norm."""
    from polystep.quadratic_model import apply_newton_refinement

    pdim = 2
    P = 1
    K = 5
    scales = torch.linspace(0, 1, K + 2)[1:K + 1]
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
