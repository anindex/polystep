#!/usr/bin/env sage -python
"""A3: the smoothing kernel is an annulus, not a ball, so Flaxman's identity does
not transfer -- and the jitter density decides whether L_eps is smooth at all.

PolyStep's probe law is: direction R v uniform on the unit sphere (R Haar on
SO(d_p), v a uniform orthoplex vertex), radius r_p (1 + eta) eps with jitter eta.
The induced smoothing kernel therefore has density, on the annulus,

    p(u) = q(|u|) / (S_{dp-1} |u|^{dp-1}),        q = density of the radius,

and the correct gradient identity is the score-function form

    grad L_eps(theta) = -E_{u ~ p}[ L(theta + u) * grad log p(u) ],
    grad log p(u)     = ( q'(rho)/q(rho) - (dp-1)/rho ) * u/rho,   rho = |u|.

Two consequences the appendix currently gets wrong:

  * The cited identity, Flaxman et al. Lem. 2.1, is the *uniform-ball* statement
    grad E_ball[L] = (d/eps) E_sphere[L u].  It is a divergence-theorem boundary
    term for the ball and does not transfer to the annulus kernel.
  * With eta ~ Uniform, q is an indicator, q' is a pair of deltas, the score is
    singular, and L_eps is only Lipschitz -- not the claimed C^infinity.  With a
    C^infinity compactly supported bump q the score is smooth and bounded, so
    differentiation under the integral gives L_eps in C^infinity for any bounded
    measurable L, which is what the proof actually needs.

WHAT IS AND IS NOT BEING COMPARED
---------------------------------
An earlier version of this script printed "Flaxman ball form: 41% error" against
the annulus finite difference.  That headline was unfair: the Flaxman estimator
targets grad of the BALL-smoothed loss, the finite difference targets grad of the
ANNULUS-smoothed loss, and those are two different objectives.  A 41% gap between
them indicts nothing.  The script now runs three comparisons:

    annulus score      vs  annulus FD   -> agree.  This is the real result.
    Flaxman ball       vs  ball FD      -> agree.  The ball identity is correct.
    Flaxman ball       vs  annulus FD   -> disagree, and is only reported as the
                                           size of the objective mismatch.

Honest statement: the ball identity is right, and it does not transfer to the
annulus kernel; the annulus needs the score form.

THE KERNEL IS A MIXTURE, NOT ONE ANNULUS
----------------------------------------
The probes do not sit at a single radius.  For each vertex the optimizer evaluates
K scales, at radius r_p lambda_k (1 + eta) eps with lambda_k = k/(K+1), so the
radius law is a MIXTURE of K jittered annuli and the score is the mixture's:

    q_mix(rho) = (1/K) sum_k qhat(rho/(r_p lam_k eps) - 1) / (r_p lam_k eps),
    grad log p(u) = ( q_mix'(rho)/q_mix(rho) - (dp-1)/rho ) u/rho.

The single-annulus form the appendix used to display is the K=1, lambda=1 special
case, and it is wrong twice over: it misses the mixture and it misses the factor
lambda_k inside the radius derivative.  ``check_mixture_score`` tests the mixture
form for K in {1, 3, 5}; ``check_single_annulus_form_is_undefined_on_the_mixture``
shows the displayed form is not merely less general -- its density vanishes on
more than half the mixture's support, where q'/q does not exist at all.

WHICH RADIUS THE SURROGATE IS DEFINED AT
----------------------------------------
mean_k lambda_k = 1/2 for every K, so the realised smoothing radius is
delta = r_p lambdabar E[1+eta] eps = r_p eps / 2, not r_p eps.  That factor of two
is not bookkeeping: ``check_radius_sign_flip`` exhibits a bounded C^infinity loss
whose surrogates at delta and at 2 delta have gradients of OPPOSITE SIGN, so a step
that descends the realised surrogate ascends the naive one.
"""

import numpy as np

RNG = np.random.default_rng(0)

DP = 3
R_P = 1.0
EPS = 0.3
ETA_MAX = 0.5


# --- C^infinity compactly supported bump on [-eta_max, eta_max] -------------

def bump(eta, eta_max=ETA_MAX):
    """Standard mollifier: exp(-1/(1-t^2)) on |t|<1, zero outside. Unnormalised."""
    t = np.asarray(eta) / eta_max
    out = np.zeros_like(t, dtype=float)
    inside = np.abs(t) < 1.0
    out[inside] = np.exp(-1.0 / (1.0 - t[inside] ** 2))
    return out


def bump_dlog(eta, eta_max=ETA_MAX):
    """d/d(eta) log bump(eta) = -2 eta / (eta_max^2 (1 - t^2)^2), t = eta/eta_max."""
    t = np.asarray(eta) / eta_max
    return -2.0 * t / (eta_max * (1.0 - t**2) ** 2)


def sample_eta(n, rng, eta_max=ETA_MAX):
    """Rejection sampling from the bump density."""
    peak = bump(np.array([0.0]), eta_max)[0]
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(-eta_max, eta_max, size=2 * n)
        keep = cand[rng.random(2 * n) * peak < bump(cand, eta_max)]
        take = min(n - filled, keep.size)
        out[filled:filled + take] = keep[:take]
        filled += take
    return out


def lambdas(K):
    """The probe fractions the optimizer uses: lambda_k = k/(K+1), k = 1..K."""
    return np.arange(1, K + 1) / (K + 1)


def sample_probes(n, rng, lam=None):
    """u = r_p lambda_k (1 + eta) eps * (unit direction), k uniform over the scales.

    ``lam=None`` is the single-annulus special case lambda = 1 that the appendix
    used to display; pass ``lambdas(K)`` for the kernel the optimizer realises.
    """
    lam = np.array([1.0]) if lam is None else np.asarray(lam)
    g = rng.standard_normal((n, DP))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    eta = sample_eta(n, rng)
    k = rng.integers(0, lam.size, size=n)
    rho = R_P * lam[k] * (1.0 + eta) * EPS
    return rho[:, None] * g, rho, eta


def radius_density(rho, lam):
    """(q_mix(rho), q_mix'(rho)) for the mixture over probe scales.

    Unnormalised: the bump's normalising constant is common to every component and
    cancels in the ratio q'/q that the score uses.
    """
    rho = np.atleast_1d(rho)
    q = np.zeros_like(rho, dtype=float)
    dq = np.zeros_like(rho, dtype=float)
    for lam_k in lam:
        scale = R_P * lam_k * EPS
        eta = rho / scale - 1.0
        b = bump(eta)
        q += b / scale
        # d/drho bump(eta) = bump'(eta)/scale, and bump' = bump * bump_dlog.
        dq += b * bump_dlog(eta) / scale**2
    return q / lam.size, dq / lam.size


def score(u, rho, eta, lam=None):
    """grad log p(u) for the mixture kernel, p(u) = q_mix(rho)/(S_{dp-1} rho^{dp-1})."""
    lam = np.array([1.0]) if lam is None else np.asarray(lam)
    q, dq = radius_density(rho, lam)
    radial = dq / q - (DP - 1) / rho
    return radial[:, None] * (u / rho[:, None])


# --- the test loss: discontinuous, so only the smoothed object is meaningful -

def loss(pts):
    return (pts[..., 0] > 0.05).astype(float) + 0.3 * pts[..., 1] ** 2


def smoothed(theta, n, rng, lam=None):
    u, _, _ = sample_probes(n, rng, lam)
    return float(np.mean(loss(theta + u)))


def grad_via_score(theta, n, rng, lam=None):
    u, rho, eta = sample_probes(n, rng, lam)
    vals = loss(theta + u)
    return -np.mean(vals[:, None] * score(u, rho, eta, lam), axis=0)


def grad_via_ball_formula(theta, n, rng):
    """Flaxman Lem. 2.1: grad E_ball[L] = (d/eps) E_sphere[L(theta + eps u) u]."""
    g = rng.standard_normal((n, DP))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    vals = loss(theta + R_P * EPS * g)
    return (DP / (R_P * EPS)) * np.mean(vals[:, None] * g, axis=0)


def smoothed_ball(theta, n, rng):
    """E_u[L(theta + u)] for u uniform in the BALL of radius r_p eps."""
    g = rng.standard_normal((n, DP))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = R_P * EPS * (rng.random(n) ** (1.0 / DP))[:, None] * g
    return float(np.mean(loss(theta + u)))


def grad_via_fd(smooth_fn, theta, n, h, seed=99):
    """Central difference of a smoothed loss, with common random numbers.

    Same seed on both sides, so the difference is not swamped by Monte Carlo noise.
    """
    out = np.empty(DP)
    for j in range(DP):
        e = np.zeros(DP)
        e[j] = h
        a = smooth_fn(theta + e, n, np.random.default_rng(seed))
        b = smooth_fn(theta - e, n, np.random.default_rng(seed))
        out[j] = (a - b) / (2 * h)
    return out


def check_mixture_score():
    """The score identity holds for the kernel the optimizer actually realises.

    The displayed single-annulus density is the K=1, lambda=1 corner.  For the real
    mixture over lambda_k = k/(K+1) the score is the mixture's, and it is that form
    the identity needs; the printed q'/(q r_p eps) is off by the lambda_k factor.
    """
    theta = np.array([0.02, 0.4, -0.1])
    n = 4_000_000
    print("\n(2) mixture kernel: probes sit at r_p lambda_k (1+eta) eps, not one radius")
    for K in (1, 3, 5):
        lam = lambdas(K)
        # The finite-difference step has to stay small against the SMALLEST probe
        # radius in the mixture, or the FD bias swamps the identity being tested.
        h = 0.04 * R_P * EPS * lam[0]
        fd = grad_via_fd(lambda t, m, r: smoothed(t, m, r, lam), theta, n, h, seed=41 + K)
        sc = grad_via_score(theta, n, np.random.default_rng(7), lam)
        err = np.linalg.norm(sc - fd) / np.linalg.norm(fd)
        print(f"  K={K}  lambda_k={np.array2string(lam, precision=3)}  "
              f"mixture score vs FD rel.err {err:.3f}")
        assert err < 0.05, (K, err)


def check_single_annulus_form_is_undefined_on_the_mixture():
    """The displayed single-annulus score is not a worse estimator; it is undefined.

    Its density is supported on rho in r_p eps [1-eta_max, 1+eta_max], while the
    mixture puts most of its mass below that band (lambda_k <= 1 shrinks every
    radius).  On the part of the support the two share, the formula is off by the
    missing lambda_k in the radius derivative; on the rest, q = 0 and q'/q does not
    exist.  Reporting a single relative error would flatter it.
    """
    lam = lambdas(3)
    _, rho, _ = sample_probes(400_000, np.random.default_rng(7), lam)
    lo, hi = R_P * EPS * (1 - ETA_MAX), R_P * EPS * (1 + ETA_MAX)
    inside = (rho > lo) & (rho < hi)
    frac = float(np.mean(inside))
    # On the shared band, compare the two radial scores probe by probe.
    q, dq = radius_density(rho[inside], lam)
    mix_radial = dq / q
    disp_radial = bump_dlog(rho[inside] / (R_P * EPS) - 1.0) / (R_P * EPS)
    rel = float(np.median(np.abs(disp_radial - mix_radial) / np.abs(mix_radial)))
    print(f"  displayed single-annulus support covers {100 * frac:.1f}% of the mixture's probes;")
    print(f"  on that overlap its radial score is off by a median {100 * rel:.0f}%")
    assert frac < 0.5, frac
    assert rel > 0.2, rel


def check_radius_sign_flip():
    """A bounded C^infinity loss whose surrogates at delta and 2 delta disagree in sign.

    This is the example the smooth-kernel lemma promises.  For a plane wave
    L(theta) = sin(k.theta) the annulus smoothing at radius rho is exactly

        L_rho(theta) = Omega_{dp}(|k| rho) sin(k.theta),

    with Omega_2 = J_0 the Bessel function, so the surrogate gradient carries the
    factor Omega(|k| rho) and flips sign with it.  Choosing |k| delta = 1.5 puts
    the realised surrogate before the first zero of J_0 (at 2.4048) and the naive
    r_p eps surrogate after it: the same step descends one and ascends the other.
    """
    from scipy.special import j0

    print("\n(3) surrogate radius: delta = r_p lambdabar E[1+eta] eps = r_p eps / 2")
    delta = 0.5 * R_P * EPS                      # lambdabar = 1/2, E[1+eta] = 1
    kmag = 1.5 / delta
    for rho, name in ((delta, "realised delta"), (2 * delta, "naive r_p eps")):
        print(f"  |k| rho = {kmag * rho:5.3f}   J_0 = {j0(kmag * rho):+.4f}   ({name})")
    assert j0(kmag * delta) > 0 and j0(kmag * 2 * delta) < 0

    # Monte Carlo on the real kernel, in d_p = 2 where Omega = J_0 exactly.
    rng = np.random.default_rng(3)
    n = 4_000_000
    k = np.array([kmag, 0.0])
    theta = np.array([0.0, 0.0])                 # cos(k.theta) = 1, gradient maximal

    def wave_grad(radius):
        g = rng.standard_normal((n, 2))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        pts = theta + radius * g
        # d/dtheta_0 of E[sin(k.(theta+u))] = |k| E[cos(k.(theta+u))]
        return kmag * float(np.mean(np.cos(pts @ k)))

    g_small, g_big = wave_grad(delta), wave_grad(2 * delta)
    print(f"  d/dtheta_0 of the surrogate at delta   : {g_small:+.4f}  "
          f"(exact {kmag * j0(kmag * delta):+.4f})")
    print(f"  d/dtheta_0 of the surrogate at 2 delta : {g_big:+.4f}  "
          f"(exact {kmag * j0(kmag * 2 * delta):+.4f})")
    print("  -> opposite signs: a step descending the realised surrogate ascends the naive one")
    assert g_small > 0 > g_big
    assert abs(g_small - kmag * j0(kmag * delta)) < 0.02 * kmag
    assert abs(g_big - kmag * j0(kmag * 2 * delta)) < 0.02 * kmag


def demo():
    theta = np.array([0.02, 0.4, -0.1])
    n = 3_000_000

    fd_ann = grad_via_fd(smoothed, theta, n, 0.02)
    fd_ball = grad_via_fd(smoothed_ball, theta, n, 0.02, seed=123)
    sc = grad_via_score(theta, n, np.random.default_rng(7))
    ball = grad_via_ball_formula(theta, n, np.random.default_rng(7))

    ref_ann = np.linalg.norm(fd_ann)
    ref_ball = np.linalg.norm(fd_ball)
    err_score = np.linalg.norm(sc - fd_ann) / ref_ann
    err_ball_fair = np.linalg.norm(ball - fd_ball) / ref_ball
    err_cross = np.linalg.norm(ball - fd_ann) / ref_ann

    print("target: grad of the ANNULUS-smoothed loss (PolyStep's own kernel)")
    print(f"  annulus finite difference : {np.array2string(fd_ann, precision=4)}")
    print(f"  annulus score identity    : {np.array2string(sc, precision=4)}  rel.err {err_score:.3f}")
    print("target: grad of the BALL-smoothed loss (what Flaxman Lem. 2.1 is about)")
    print(f"  ball finite difference    : {np.array2string(fd_ball, precision=4)}")
    print(f"  Flaxman ball estimator    : {np.array2string(ball, precision=4)}  rel.err {err_ball_fair:.3f}")
    print(f"\ncross-objective gap (Flaxman ball estimator vs annulus FD): {err_cross:.3f}")
    print("  -> that gap measures how far the two SURROGATES are apart, not an error")
    print("     in Flaxman's identity.  The ball identity is correct; it just does not")
    print("     transfer to the annulus kernel, which needs the score form.")

    # The score identity is the correct one for the annulus kernel.
    assert err_score < 0.05, err_score
    # Flaxman's identity is correct for the objective it actually names.
    assert err_ball_fair < 0.05, err_ball_fair
    # The two surrogates are genuinely different objectives, by a wide margin --
    # not a constant factor that could be absorbed into c.
    assert err_cross > 0.25, err_cross

    check_mixture_score()
    check_single_annulus_form_is_undefined_on_the_mixture()
    check_radius_sign_flip()


if __name__ == "__main__":
    demo()
    print("OK")
