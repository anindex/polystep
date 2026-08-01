#!/usr/bin/env sage -python
"""What randomized smoothing can and cannot deliver on a discontinuous loss.

Let L_eps be the polytope-smoothed surrogate at radius eps.  Two scalings decide
the shape of the convergence theorem:

  (i)  sup_x ||grad L_eps(x)||        -- how large the smoothed gradient can get
  (ii) K(eps) = sup_x ||hess L_eps(x)||  -- the smoothness constant in the
                                            descent inequality

The claim under test is that a jump of size J across the discontinuity set forces

      sup ||grad L_eps|| = Theta(J / eps),     K(eps) = Theta(J / eps^2).

WHAT AN HONEST CHECK OF THIS CAN AND CANNOT SHOW
------------------------------------------------
An earlier version of this script "measured" the exponent -2 by scaling the
sample cloud, the evaluation point and the finite-difference stencil all
proportionally to eps while reseeding the same RNG at each radius.  Under that
construction the second difference is forced to scale as eps^-2 by pure algebra;
the fitted -2 carried no information.  It also read a single off-centre stencil
at |x_1| = 0.35 eps instead of a supremum over x, which for d = 4 captures only
~65% of the true sup.

The exponent is in fact *structural* for any kernel of the form eps * U with U a
fixed law on the unit ball, whenever L is positively homogeneous:

    L a unit step  =>  L_eps(x) = Phi(x/eps)         => hess = Phi''(x/eps)/eps^2
    L = |x|        =>  L_eps(x) = eps Psi(x/eps)     => hess = Psi''(x/eps)/eps

So the -2 (and the -1 for the Lipschitz kink) is algebra, not evidence, and this
script says so.  The parts that *are* evidence, and that this script measures:

  * the CONSTANT.  For a unit step under uniform-ball smoothing in dimension d
    the first coordinate of the displacement has density
    f_d(t) = c_d (1-t^2)^{(d-1)/2}, c_d = Gamma(d/2+1)/(sqrt(pi) Gamma((d+1)/2)),
    so L_eps depends on x only through x_1 and

        sup_x ||grad L_eps|| = c_d / eps          (attained at x_1 = 0)
        K(eps) = sup_x ||hess L_eps|| = c_d (d-1) max_t t (1-t^2)^{(d-3)/2} / eps^2
                                     = 4 / (pi eps^2) for d = 4,
                 attained at |x_1| / eps = 1 / sqrt(d-2) = 0.7071 for d = 4.

    Both are measured here as suprema over a grid in x, with independent
    randomness per radius, and compared against the closed form.

  * the SECOND-ORDER SEPARATION between a discontinuous and a Lipschitz loss:
    -2 versus -1.  Both are scale families, so neither exponent is a surprise,
    but the gap is the mechanism the theorem is about.

  * a NON-HOMOGENEOUS discontinuous loss (step + quadratic).  Here the scaling is
    not forced by homogeneity, and the -2 has to emerge; it does, and the
    measured constant matches 4/(pi eps^2) + 1.

Consequence, checked at the end: a schedule that decays eps while holding the
step radius r_s fixed makes the descent-inequality error term
K(eps) r_s^2 eps^2 = Theta(J r_s^2) -- constant, not vanishing.  Telescoping then
gives a bound that grows like sqrt(T), i.e. no convergence.  Holding eps fixed
and decaying r_s keeps K finite and restores the rate.
"""

import math

import numpy as np

# Grid of evaluation points, in units of eps.  The supremum of both derivatives
# lives inside |x_1| <= eps, so this covers it with room to spare.  It is a grid
# in x/eps because the maximiser genuinely moves with eps; what the old script
# did wrong was to move the *stencil* and the *seed* with eps as well.
T_GRID = np.linspace(-1.5, 1.5, 61)


def marginal(kernel, n, d, rng):
    """First coordinate of a displacement drawn from a kernel supported in B(0,1).

    All three losses below depend on x only through x_1, so L_eps(x) is
    determined by the law of u_1 alone.  Sampling the full d-dimensional cloud
    and keeping one coordinate is what the smoothing actually does; keeping only
    that coordinate is a memory optimisation, not a change of kernel.
    """
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    if kernel == "ball":
        radii = rng.random(n) ** (1.0 / d)
    elif kernel == "annulus":  # PolyStep's own probe law
        radii = rng.uniform(0.7, 1.0, size=n)
    elif kernel == "truncnorm":
        radii = np.clip(np.abs(rng.standard_normal(n)) * 0.4, 0.0, 1.0)
    else:
        raise ValueError(kernel)
    return radii * g[:, 0]


def sup_derivatives(loss1d, eps, u1, h_frac=0.15):
    """(sup |grad L_eps|, argmax_t, sup |hess L_eps|, argmax_t) over the x-grid.

    ``u1`` is a *unit-radius* displacement sample; it is scaled by eps here, so
    the same cloud serves the whole grid and both stencils (common random
    numbers), which is what makes a second difference of an indicator function
    estimable at all.  A fresh cloud is drawn per radius by the caller.
    """
    h = h_frac * eps
    xs = T_GRID * eps
    disp = eps * u1

    def prof(shift):
        return np.array([loss1d(x + shift + disp).mean() for x in xs])

    lo, mid, hi = prof(-h), prof(0.0), prof(h)
    grad = np.abs(hi - lo) / (2 * h)
    hess = np.abs(hi - 2 * mid + lo) / h**2
    return grad.max(), T_GRID[grad.argmax()], hess.max(), T_GRID[hess.argmax()]


def fit_slope(xs, ys):
    """Log-log slope of ys vs xs."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


# --- closed forms for the uniform ball ---------------------------------------


def c_d(d):
    """Peak density of the first coordinate of a uniform point in the unit ball."""
    return math.gamma(d / 2 + 1) / (math.sqrt(math.pi) * math.gamma((d + 1) / 2))


def ball_sup_grad(d):
    """sup_x ||grad L_eps|| * eps for a unit step, uniform-ball smoothing."""
    return c_d(d)


def ball_sup_hess(d):
    """(sup_x ||hess L_eps|| * eps^2, argmax |x_1|/eps) for a unit step."""
    t = 1.0 / math.sqrt(d - 2)
    return c_d(d) * (d - 1) * t * (1 - t * t) ** ((d - 3) / 2), t


# --- the loss families -------------------------------------------------------


def step_loss(z):
    """Discontinuous: a unit jump across {x_1 = 0}.  Homogeneous of degree 0."""
    return (z > 0).astype(float)


def kink_loss(z):
    """Locally Lipschitz, non-differentiable: |x_1|.  Homogeneous of degree 1."""
    return np.abs(z)


def step_plus_quad(z):
    """Discontinuous and NOT homogeneous: the -2 is not forced by scaling here."""
    return (z > 0).astype(float) + 0.5 * z**2


def good_event_hessian(loss1d, eps, u1, margin=1.4, h_frac=0.15, reach=3.0):
    """sup |hess L_eps| over base points whose whole kernel support misses the wall.

    The convergence proof never expands L at an arbitrary point: it conditions on
    the good event that the entire probe annulus lies in one connected component of
    the complement of D.  On that event the smoothing sees only the smooth part, so
    the relevant second-derivative bound is the smooth part's Lambda_2 and NOT the
    global K(eps) = Theta(J/eps^2), which is attained only within O(eps) of the
    wall.  This is what makes the theorem's condition (vii) non-vacuous, and
    eps-independent: the probe-radius correction it controls is r_p eps Lambda_2,
    not r_p eps K(eps).
    """
    h = h_frac * eps
    grid = np.concatenate([np.linspace(-reach, -margin, 30), np.linspace(margin, reach, 30)])
    xs = grid * eps
    disp = eps * u1

    def prof(shift):
        return np.array([loss1d(x + shift + disp).mean() for x in xs])

    lo, mid, hi = prof(-h), prof(0.0), prof(h)
    return float((np.abs(hi - 2 * mid + lo) / h**2).max())


def check_good_event_hessian_is_bounded():
    """K(eps) blows up; the good-event Hessian does not.  Condition (vii) uses the latter."""
    d = 4
    n = 1_500_000
    epss = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    lam2 = 1.0                       # second derivative of the smooth part, 0.5 z^2
    print("\ngood-event Hessian: base points whose kernel support misses the wall")
    print(f"{'eps':>7} {'global K(eps)':>14} {'good-event sup':>16} {'ratio to Lambda_2':>18}")
    goods = []
    for i, e in enumerate(epss):
        cloud = marginal("ball", n, d, np.random.default_rng(2000 + i))
        glob = sup_derivatives(step_plus_quad, e, cloud)[2]
        good = good_event_hessian(step_plus_quad, e, cloud)
        goods.append(good)
        print(f"{e:7.4g} {glob:14.2f} {good:16.4f} {good / lam2:18.4f}")
    goods = np.array(goods)
    slope = fit_slope(epss, goods)
    print(f"  good-event sup ~ eps^{slope:+.2f} (theory 0: it is Lambda_2, "
          f"independent of the smoothing radius)")
    print("  -> condition (vii) reads r_p eps Lambda_2 <= (1/2)||grad||, which does not")
    print("     tighten as eps falls, so the eps-trade in the bias floor is unaffected.")
    assert abs(slope) < 0.06, slope
    assert np.allclose(goods, lam2, rtol=0.05), goods


def demo():
    d = 4
    n = 1_500_000
    epss = np.array([0.4, 0.2, 0.1, 0.05, 0.025])

    # Independent randomness per radius: no shared seed, so nothing links the
    # measurement at one eps to the measurement at another.
    seeds = {e: np.random.default_rng(1000 + i) for i, e in enumerate(epss)}
    clouds = {e: marginal("ball", n, d, seeds[e]) for e in epss}

    g_sup, g_arg, h_sup, h_arg = ([], [], [], [])
    for e in epss:
        gs, ga, hs, ha = sup_derivatives(step_loss, e, clouds[e])
        g_sup.append(gs)
        g_arg.append(ga)
        h_sup.append(hs)
        h_arg.append(ha)
    g_sup, h_sup = np.array(g_sup), np.array(h_sup)

    pred_g = ball_sup_grad(d)
    pred_h, pred_arg = ball_sup_hess(d)
    print(f"d={d} uniform ball, unit step.  Closed forms: "
          f"sup|grad| = {pred_g:.5f}/eps, K(eps) = {pred_h:.5f}/eps^2 at |x1|/eps = {pred_arg:.4f}")
    print(f"{'eps':>7} {'sup|grad|*eps':>14} {'sup|hess|*eps^2':>16} {'argmax |x1|/eps':>16}")
    for e, gs, hs, ha in zip(epss, g_sup, h_sup, h_arg):
        print(f"{e:7.4g} {gs * e:14.5f} {hs * e**2:16.5f} {abs(ha):16.4f}")

    # The measured supremum must match the closed form, not merely scale like it.
    # Tolerance covers the O(h^2) stencil bias -- (h/eps)^2/6 * f''(0)/f(0) = -1.1%
    # for the gradient at h = 0.15 eps, d = 4 -- plus Monte-Carlo noise on a second
    # difference of an indicator (~1.3% at these n, h).
    assert np.allclose(g_sup * epss, pred_g, rtol=0.05), (g_sup * epss, pred_g)
    assert np.allclose(h_sup * epss**2, pred_h, rtol=0.08), (h_sup * epss**2, pred_h)
    assert np.allclose(np.abs(h_arg), pred_arg, atol=0.06), (h_arg, pred_arg)
    assert np.allclose(np.abs(g_arg), 0.0, atol=0.03), g_arg

    # The old single-stencil probe at |x_1| = 0.35 eps: right exponent, wrong
    # constant.  Quantified so the gap is on the record rather than implied.
    t_old = 0.35
    frac = (t_old * (1 - t_old**2) ** ((d - 3) / 2)) / (pred_arg * (1 - pred_arg**2) ** ((d - 3) / 2))
    print(f"a single stencil at |x1| = 0.35 eps sees {frac:.1%} of the supremum")
    assert 0.6 < frac < 0.7, frac

    s_g, s_h = fit_slope(epss, g_sup), fit_slope(epss, h_sup)
    print(f"\nfitted exponents: sup|grad| ~ eps^{s_g:+.2f} (theory -1), "
          f"K(eps) ~ eps^{s_h:+.2f} (theory -2)")
    print("  NOTE: for a homogeneous loss under a scale-family kernel these two")
    print("  exponents are algebraic identities, not measurements.  The content is")
    print("  the constant above and the -2 vs -1 gap below.")
    assert -1.05 < s_g < -0.95, s_g
    assert -2.10 < s_h < -1.90, s_h

    # Second-order separation: a Lipschitz kink has K(eps) = Theta(1/eps), one
    # power better.  Closed form: 2 c_d / eps, attained at x_1 = 0.
    kink = [sup_derivatives(kink_loss, e, clouds[e]) for e in epss]
    kink_g = np.array([k[0] for k in kink])
    kink_h = np.array([k[2] for k in kink])
    k_slope = fit_slope(epss, kink_h)
    print(f"\nLipschitz kink: sup|grad| ~ eps^{fit_slope(epss, kink_g):+.2f} (theory 0, bounded by 1), "
          f"K(eps) ~ eps^{k_slope:+.2f} (theory -1)")
    assert abs(fit_slope(epss, kink_g)) < 0.05
    assert np.all(kink_g <= 1.0 + 1e-6), kink_g
    assert -1.08 < k_slope < -0.92, k_slope
    assert np.allclose(kink_h * epss, 2 * c_d(d), rtol=0.06), (kink_h * epss, 2 * c_d(d))

    # Non-homogeneous discontinuous loss: nothing forces the scaling here, and the
    # closed form picks up an additive +1 from the quadratic part.
    sq_h = np.array([sup_derivatives(step_plus_quad, e, clouds[e])[2] for e in epss])
    sq_pred = pred_h / epss**2 + 1.0
    print(f"step + quadratic (not homogeneous): K(eps) ~ eps^{fit_slope(epss, sq_h):+.2f}, "
          f"K(eps)/(4/(pi eps^2) + 1) = {np.array2string(sq_h / sq_pred, precision=3)}  (theory: 1)")
    assert -2.05 < fit_slope(epss, sq_h) < -1.90, fit_slope(epss, sq_h)
    assert np.allclose(sq_h, sq_pred, rtol=0.08), (sq_h, sq_pred)

    # Error term of the descent inequality, K(eps) r_s^2 eps^2, at fixed r_s.
    # With K = Theta(1/eps^2) this is Theta(1): it does not vanish as eps -> 0, so
    # telescoping over T steps accumulates Theta(T) error against a
    # sum_t eps_t = Theta(sqrt(T)) normaliser.
    err = h_sup * epss**2
    print(f"\nerror term K(eps) eps^2 ~ eps^{fit_slope(epss, err):+.2f}  "
          f"(theory 0: constant, value {pred_h:.4f})")
    assert abs(fit_slope(epss, err)) < 0.1, fit_slope(epss, err)

    # Kernel shape does not matter: unit mass on an interval of length 2 eps forces
    # the projected density above 1/(2 eps) somewhere, so every kernel supported in
    # B(0, eps) clears the J/(4 eps) floor and shows the same -1.
    print()
    for kernel in ("ball", "annulus", "truncnorm"):
        vals = np.array([
            sup_derivatives(step_loss, e, marginal(kernel, n // 2, d, np.random.default_rng(77 + i)))[0]
            for i, e in enumerate(epss)
        ])
        slope = fit_slope(epss, vals)
        floor = 1.0 / (4.0 * epss)  # J = 1 for step_loss
        assert -1.06 < slope < -0.94, (kernel, slope)
        assert np.all(vals >= floor), (kernel, vals, floor)
        print(f"kernel={kernel:9s} sup|grad L_eps| ~ eps^{slope:+.2f}, "
              f"sup|grad| * eps = {np.mean(vals * epss):.4f}, "
              f"min ratio to J/(4 eps) floor = {np.min(vals / floor):.2f}")


if __name__ == "__main__":
    demo()
    check_good_event_hessian_is_bounded()
    print("OK")
