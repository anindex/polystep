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


def sample_probes(n, rng):
    """u = r_p (1 + eta) eps * (unit direction), returning u and the radii."""
    g = rng.standard_normal((n, DP))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    eta = sample_eta(n, rng)
    rho = R_P * (1.0 + eta) * EPS
    return rho[:, None] * g, rho, eta


def score(u, rho, eta):
    """grad log p(u), with q the bump density of the radius."""
    # rho = r_p (1 + eta) eps  =>  d(eta)/d(rho) = 1/(r_p eps)
    dlog_q = bump_dlog(eta) / (R_P * EPS)
    radial = dlog_q - (DP - 1) / rho
    return radial[:, None] * (u / rho[:, None])


# --- the test loss: discontinuous, so only the smoothed object is meaningful -

def loss(pts):
    return (pts[..., 0] > 0.05).astype(float) + 0.3 * pts[..., 1] ** 2


def smoothed(theta, n, rng):
    u, _, _ = sample_probes(n, rng)
    return float(np.mean(loss(theta + u)))


def grad_via_score(theta, n, rng):
    u, rho, eta = sample_probes(n, rng)
    vals = loss(theta + u)
    return -np.mean(vals[:, None] * score(u, rho, eta), axis=0)


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


if __name__ == "__main__":
    demo()
    print("OK")
