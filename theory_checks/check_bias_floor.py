#!/usr/bin/env sage -python
"""The two probe-side terms of the bias floor in the convergence theorem.

The descent inequality does not drive the smoothed gradient to zero; it drives it
to a floor

    B = O(r_p^3)            softmax linearization remainder
      + O(J * pi_D / delta) probes straddling the discontinuity set
      + O(delta * Lambda)   block-coordinate coupling

This script checks the first two, the ones that involve the probe law.  Both
predictions depend on the step constant c, so both are re-derived here on the
CORRECTED constant established in ``check_stein_constant.py``:

    c = E[s] / (d_p tau) = r_p eps / (2 d_p tau),      s = r_p lambda_k (1+eta) eps,

which is a factor mean_k lambda_k = 1/2 smaller than r_p/d_p and a factor d_p/2
larger than the printed r_p/d_p^2.  The probe law (lambda_k = k/(K+1), jitter eta)
is imported from that script so the two cannot drift apart.

  (1) On the good event -- the whole probe annulus inside one smooth component --
      the expected softmax barycentric step equals -c grad Lmix up to a remainder
      of order r_p^3.  Expanding the softmax to second order contributes a term
      proportional to sum_v z_v^2 (R v_v) with z_v = O(r_p); a naive count makes
      that O(r_p^2), but it vanishes.  For the ORTHOPLEX it vanishes POINTWISE, for
      every rotation, by antipodal symmetry; for a generic polytope such as the
      simplex it vanishes only in expectation, because R v is uniform on the sphere
      and the third moment of the sphere is zero.  Both are checked, together with
      the fact that the leading constant c is the same for both -- the polytopes
      differ in variance, not in the mean step or in the remainder order.

  (2) pi_D, the probability that a particle's probe cloud meets a hyperplane
      discontinuity, is measured from the REAL probe geometry (orthoplex vertices,
      Haar rotation, the lambda_k grid, jitter) rather than asserted.  It is linear
      in the probe reach, hence in r_p eps.  The bad event enters the descent
      inequality as pi_D times the maximal step norm r_s eps, and dividing by c
      gives a floor contribution

          pi_D * r_s eps / c  =  Theta( d_p tau r_s eps / L_D ),

      L_D the typical spacing between discontinuities.  Linear in eps and
      INDEPENDENT of r_p: shrinking the probes does not touch it, and shrinking eps
      re-inflates K(delta) = Theta(J/delta^2).  Both the eps exponent and the r_p
      cancellation are measured below.  The exponents survive the correction to c;
      only the constant changes (it doubles, since c halved).
"""

import numpy as np

from check_stein_constant import ETA_MAX, LAM, haar_so, moments, orthoplex

RNG = np.random.default_rng(0)

DP = 4
TAU = 1.0
EPS = 1.0

# Anisotropic on purpose.  With an isotropic quadratic c*||x||^2 the second-order
# cost contribution is the same for every unit vertex, so it is constant across the
# row and cancels in the softmax -- suppressing the very term being measured.
HESS = np.diag([2.0, 0.5, 1.3, 0.2])
G = np.array([0.6, -0.3, 0.2, 0.1])


def simplex(dp):
    """Regular simplex: dp+1 unit vertices summing to zero, NOT antipodal.

    Built by taking the dp+1 standard basis vectors of R^{dp+1}, centering them
    (which makes them sum to zero and keeps them equidistant), then mapping into
    the dp-dimensional hyperplane they span and normalizing.
    """
    e = np.eye(dp + 1)
    v = e - e.mean(axis=0)
    basis = np.linalg.svd(v, full_matrices=False)[2][:dp]
    v = v @ basis.T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fit_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def smooth_loss(x):
    """Gradient G at the origin plus an anisotropic quadratic term.

    For a quadratic the uniform-ball smoothing shifts the value but not the
    gradient at 0, so grad Lmix(0) = G exactly at every probe radius.  That keeps
    the remainder measurement clean of surrogate-radius effects, which
    check_stein_constant.py measures separately.
    """
    return x @ G + 0.5 * np.einsum("...i,ij,...j->...", x, HESS, x)


# --- (1) the softmax step tracks -c grad Lmix, with an O(r_p^3) remainder ------


def expected_step(verts, r_p, n_rot, rng):
    """(E[step], E[linearised step]) for the real probe law and cost rule."""
    V = verts.shape[0]
    R = haar_so(n_rot, DP, rng)
    directions = np.einsum("nij,vj->nvi", R, verts)          # (n, V, DP)
    eta = rng.uniform(-ETA_MAX, ETA_MAX, size=n_rot)
    s = r_p * EPS * np.outer(1.0 + eta, LAM)                 # (n, K)
    c = smooth_loss(directions[:, :, None, :] * s[:, None, :, None]).mean(axis=-1)
    w = np.exp(-(c - c.min(axis=1, keepdims=True)) / TAU)
    w /= w.sum(axis=1, keepdims=True)
    step = np.einsum("nv,nvi->i", w, directions) / n_rot
    lin = -np.einsum("nv,nvi->i", c, directions) / (V * TAU * n_rot)
    return step, lin


def check_second_order_cancellation():
    """Where the O(r_p^2) softmax term goes, for each polytope."""
    R = haar_so(200_000, DP, RNG)
    for name, verts in (("orthoplex", orthoplex(DP)), ("simplex", simplex(DP))):
        d = np.einsum("nij,vj->nvi", R, verts)               # (n, V, DP)
        # sum_v (g . d_v)^2 d_v, the second-order softmax contribution
        term = np.einsum("nv,nvi->ni", np.einsum("nvi,i->nv", d, G) ** 2, d)
        per_rotation = np.abs(term).max()
        in_expectation = np.linalg.norm(term.mean(axis=0))
        print(f"  {name:9s} sum_v (g.Rv)^2 Rv : max over rotations {per_rotation:.3e}, "
              f"mean over rotations {in_expectation:.3e}")
        if name == "orthoplex":
            assert per_rotation < 1e-12, per_rotation      # vanishes POINTWISE
        else:
            assert per_rotation > 1e-2, per_rotation       # does not vanish pointwise
            assert in_expectation < 2e-3, in_expectation   # but does in expectation


def check_linearization_remainder():
    r_ps = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    n_rot = 300_000

    print("\n(1) softmax linearization remainder")
    check_second_order_cancellation()
    print(f"  {'polytope':9s} {'V':>3} {'c_hat/(r_p/d_p)':>16} {'residual ~ r_p^':>16}")

    for name, verts in (("orthoplex", orthoplex(DP)), ("simplex", simplex(DP))):
        residuals, ratios = [], []
        for i, r_p in enumerate(r_ps):
            step, lin = expected_step(verts, r_p, n_rot, np.random.default_rng(3000 + i))
            e_s, _ = moments(r_p)
            c = e_s / (DP * TAU)                     # = r_p eps / (2 d_p tau)
            pred = -c * G                            # grad Lmix(0) = G for a quadratic
            # The Stein half: same constant c for both polytopes, despite different V.
            assert np.allclose(lin, pred, rtol=0.02, atol=1e-6), (name, r_p, lin, pred)
            residuals.append(np.linalg.norm(step - lin))
            ratios.append(float(-step @ G / (G @ G)) / (r_p / DP))

        residuals = np.array(residuals)
        # The r_p^5 term is still visible at r_p >= 0.2, so fit the asymptotic tail.
        slope = fit_slope(r_ps[-3:], residuals[-3:])
        print(f"  {name:9s} {verts.shape[0]:3d} {np.mean(ratios[-2:]):16.4f} {slope:16.2f}")
        assert abs(np.mean(ratios[-2:]) - 0.5) < 0.03, (name, ratios)
        assert 2.5 < slope < 3.5, (name, slope)

    print("  theory: coefficient ratio 0.5 (= mean_k lambda_k), remainder exponent 3")
    print("  The orthoplex and the simplex share the constant c = r_p eps/(2 d_p tau)")
    print("  and the r_p^3 remainder order; the orthoplex cancels the r_p^2 term for")
    print("  EVERY rotation, the simplex only on average, so they differ in variance.")


# --- (2) pi_D is linear in the probe reach, so the floor term is linear in eps --


def straddle_probability(r_p, eps, n, rng, half_window=0.5):
    """Fraction of particles whose REAL probe cloud crosses the wall {x_0 = 0}.

    Particles are spread uniformly over [-half_window, half_window], a fixed
    surface density 1/L_D with L_D = 2*half_window.  Probes are the actual
    orthoplex vertices under a Haar rotation, at the actual lambda_k (1+eta) radii.
    Nothing here is forced by construction: the constant depends on the direction
    law and on the scale grid, and is checked against E[max_j |R_0j|] below.
    """
    verts = orthoplex(DP)
    R = haar_so(n, DP, rng)
    d0 = np.einsum("nij,vj->nvi", R, verts)[:, :, 0, None]     # first coord of R v
    eta = rng.uniform(-ETA_MAX, ETA_MAX, size=n)
    s = r_p * eps * np.outer(1.0 + eta, LAM)                  # (n, K)
    probe_x0 = rng.uniform(-half_window, half_window, size=n)[:, None, None] + d0 * s[:, None, :]
    flat = probe_x0.reshape(n, -1)
    return float(np.mean((flat.max(axis=1) > 0) & (flat.min(axis=1) < 0)))


def check_straddle_probability():
    n = 200_000
    half_window = 0.5
    epss = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    r_ps = np.array([1.0, 0.5, 0.25])
    r_s = 1.0                                                  # step radius multiplier

    print("\n(2) straddle probability and the floor term it produces")
    print(f"  {'r_p':>6} {'pi_D ~ eps^':>12} {'floor ~ eps^':>13} {'floor / (d_p tau r_s eps / L_D)':>32}")

    consts = []
    for r_p in r_ps:
        pis = np.array([straddle_probability(r_p, e, n, np.random.default_rng(int(4000 + 100 * r_p + i)))
                        for i, e in enumerate(epss)])
        pi_slope = fit_slope(epss, pis)
        # Floor contribution: bad-event probability x maximal step norm / descent constant.
        c = np.array([moments(r_p)[0] for _ in epss]) * epss / (DP * TAU)
        floor = pis * (r_s * epss) / c
        floor_slope = fit_slope(epss, floor)
        norm = DP * TAU * r_s * epss / (2 * half_window)
        consts.append(floor / norm)
        print(f"  {r_p:6.3f} {pi_slope:12.2f} {floor_slope:13.2f} "
              f"{np.array2string(floor / norm, precision=3):>32}")
        assert 0.9 < pi_slope < 1.1, (r_p, pi_slope)
        assert 0.9 < floor_slope < 1.1, (r_p, floor_slope)

    # The pi_D constant is a real geometric quantity, not a tautology: for the
    # orthoplex the cloud's reach along x_0 is max_k s_k * max_v (R v)_0, and
    # max_v (R v)_0 = max_j |R_0j| over the 2 d_p vertices {+-e_j}.
    reach = np.abs(haar_so(200_000, DP, np.random.default_rng(9))[:, 0, :]).max(axis=1).mean()
    r_p, eps = 0.5, 0.1
    l_d = 2 * half_window
    pred = 2 * LAM[-1] * r_p * eps * reach / l_d           # E[1+eta] = 1
    meas = straddle_probability(r_p, eps, 400_000, np.random.default_rng(10), half_window)
    print(f"  pi_D at r_p={r_p}, eps={eps}: measured {meas:.5f}, "
          f"predicted 2 lambda_max r_p eps E[max_j |R_0j|] / L_D = {pred:.5f}")
    assert abs(meas - pred) / pred < 0.05, (meas, pred)

    consts = np.array(consts)
    spread = float(consts.max() / consts.min())
    const_pred = 4 * LAM[-1] * reach            # = pi_D L_D/(r_p eps) * 2 d_p tau / d_p
    print("  theory: pi_D ~ eps^1, floor ~ eps^1, and floor/(d_p tau r_s eps/L_D) constant in r_p")
    print(f"  spread of that ratio across r_p in {r_ps.tolist()}: {spread:.3f}x  (theory 1.0), "
          f"value {consts.mean():.3f} (theory 4 lambda_max E[max_j |R_0j|] = {const_pred:.3f})")
    assert spread < 1.15, spread
    assert abs(consts.mean() - const_pred) / const_pred < 0.05, (consts.mean(), const_pred)


def demo():
    check_linearization_remainder()
    check_straddle_probability()


if __name__ == "__main__":
    demo()
    print("OK")
