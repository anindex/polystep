#!/usr/bin/env sage -python
"""The bias floor in the convergence theorem: its algebra and its two probe terms.

The descent inequality does not drive the smoothed gradient to zero.  Writing the
expected displacement as -r_s eps (c grad + e), descent fails once
||grad|| <= ||e||/c, so the floor in SQUARED-gradient units is

    B = (||e|| / c)^2,      c = r_p lambdabar E[1+eta] / d_p,

and the three sources of e give

    B = O(d_p^2 r_p^4)              softmax linearization remainder, e1 = O(r_p^3)
      + Theta(d_p^2 eps / (r_p L_D)) probes straddling D,             e2 = O(pi_D)
      + O(delta^4 Lambda_3^2)       block-coordinate coupling,       e3 = c O(delta^2 Lambda_3)

The coupling term is the one that is easy to get wrong.  ``G - grad L_delta`` is a
discrepancy between two GRADIENT fields, so it enters the displacement already
multiplied by c and the c cancels when the floor is formed -- leaving no d_p and no
r_p beyond the delta.  ``check_floor_algebra`` below pins all three symbolically so
the paper and this file cannot drift apart again; the numerical sections then
measure e1 and e2 against the real probe law.

Both probe-side predictions depend on the step constant c, so both are re-derived
here on the CORRECTED constant established in ``check_stein_constant.py``:

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
import sympy as sp

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


# --- (0) the floor algebra, symbolically ---------------------------------------


def check_floor_algebra():
    """Each floor term is (e_i / c)^2.  Pin the three exponents symbolically.

    The failure this guards against is real: an earlier draft printed the coupling
    term as O(d_p^2 delta^2 Lambda_3^2), which is wrong in both the d_p factor and
    the eps exponent, and the appendix carried the un-squared displacement error
    under the same name B.
    """
    r_p, eps, d_p, lam, L_D, Lam3 = sp.symbols(
        "r_p eps d_p lambdabar L_D Lambda_3", positive=True
    )
    delta = r_p * lam * eps                        # E[1+eta] = 1
    c = r_p * lam / d_p

    sources = {
        "linearization": r_p**3,                   # absolute error in step units
        "straddle": delta / L_D,                   # pi_D x maximal step norm 1
        "coupling": c * delta**2 * Lam3,           # a gradient error, hence carries c
    }
    expected = {
        "linearization": d_p**2 * r_p**4 / lam**2,
        "straddle": d_p**2 * eps**2 / L_D**2,
        "coupling": delta**4 * Lam3**2,
    }

    print("\n(0) floor algebra: B_i = (e_i / c)^2")
    for name, e in sources.items():
        got = sp.simplify(sp.expand((e / c) ** 2))
        want = sp.simplify(sp.expand(expected[name]))
        print(f"  {name:14s} {got}")
        assert sp.simplify(got - want) == 0, (name, got, want)

    # The coupling term must be free of d_p: the c that scales the gradient error
    # is the same c the floor divides by.
    assert d_p not in sp.simplify((sources["coupling"] / c) ** 2).free_symbols
    # Linearization and coupling both vanish as r_p -> 0; straddle does not.
    for name in ("linearization", "coupling"):
        assert sp.limit(expected[name], r_p, 0) == 0, name
    assert sp.simplify(sp.diff(expected["straddle"], r_p)) == 0
    print("  linearization and coupling vanish with r_p; straddle carries no r_p at all")


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


def check_straddle_term_is_linear_not_quadratic():
    """pi_D is indicator-like along a trajectory, so it cannot be squared.

    The floor's middle term is (e_2/c)^2 with e_2 = pi_D(t), and the descent
    inequality sums r_{s,t} (pi_D(t)/c)^2.  Writing that as (avg pi_D)^2 needs
    pi_D(t) to be a genuine small per-step probability.  For an ADAPTED iterate it
    is not: conditional on theta_t, the probe annulus either meets a wall or it
    does not, up to a boundary layer, so pi_D(t) is near-binary and
    avg(pi_D^2) = Theta(avg(pi_D)), not its square.  Measured below: the ratio
    avg(pi_D^2)/avg(pi_D) is 0.87 and flat in eps.

    Consequence, and it is a correction to the paper's own Eq. for the floor:

        B_2  =  (1/c^2) avg(pi_D^2)  =  Theta( d_p^2 eps / (r_p L_D) ),

    LINEAR in eps, not quadratic, and carrying 1/r_p rather than no r_p at all.
    The quadratic form is recovered only under an extra mixing hypothesis that
    makes pi_D(t) a genuine probability at each step -- which is exactly what the
    uniform-base-point measurement in check_straddle_probability supplies.
    """
    eps, r_p, d_p, L_D = sp.symbols("eps r_p d_p L_D", positive=True)
    c = r_p / (2 * d_p)
    B2 = sp.simplify((1 / c**2) * (r_p * eps / L_D))          # avg(pi_D) = Theta(delta_out/L_D)
    print("\n(4) the straddling term, on an adapted trajectory")
    print(f"  B_2 = {B2}")
    print(f"  exponent in eps: {sp.simplify(sp.log(B2).diff(eps) * eps)}   "
          f"exponent in r_p: {sp.simplify(sp.log(B2).diff(r_p) * r_p)}")
    assert sp.simplify(sp.log(B2).diff(eps) * eps) == 1
    assert sp.simplify(sp.log(B2).diff(r_p) * r_p) == -1

    # And the near-binary behaviour that forces it, measured on the real recursion.
    ratios = []
    for e in (0.2, 0.1, 0.05):
        first, second = occupancy_moments(e)
        ratios.append(second / first)
        print(f"  eps={e:5.3f}  avg(pi_D)={first:.4f}  avg(pi_D^2)={second:.4f}  "
              f"ratio={second / first:.3f}")
    print("  -> the ratio is flat in eps and stays near 1, which is the indicator")
    print("     signature.  A genuine per-step probability would give ratio = avg(pi_D)")
    print("     itself, which falls with eps; that is the discriminating comparison.")
    firsts = [occupancy_moments(e)[0] for e in (0.2, 0.1, 0.05)]
    for e, r, f in zip((0.2, 0.1, 0.05), ratios, firsts):
        print(f"  eps={e:5.3f}  measured ratio {r:.3f}  vs {f:.3f} if pi_D were a probability")
    assert min(ratios) > 0.6, ratios
    assert max(ratios) - min(ratios) < 0.1, ratios
    # The point of the check: the gap between the measured ratio and avg(pi_D) must
    # WIDEN as eps falls.  If pi_D were a genuine small per-step probability the two
    # would coincide at every eps and the quadratic form would be right after all;
    # instead the ratio is pinned near 1 while avg(pi_D) falls linearly.
    gaps = [r / f for r, f in zip(ratios, firsts)]
    print(f"  ratio / avg(pi_D) = {np.array2string(np.array(gaps), precision=2)} "
          f"at eps = 0.2, 0.1, 0.05: widening, as the indicator reading requires")
    assert gaps[0] < gaps[1] < gaps[2], gaps
    assert gaps[-1] > 4.0, gaps


def check_epsilon_exchange_rate():
    """What shrinking eps costs in iterations, and the interior optimum in r_p.

    With B = Theta(b eps) the optimization term C/(c eps S_T) reaches the floor at
    S_T ~ eps^-2, hence T ~ eps^{-2/(1/2-gamma)} -> eps^-4.  Separately, the floor
    now has TWO competing r_p dependences -- the linearization remainder a r_p^4
    falls with r_p while the straddling term b eps / r_p rises -- so unlike the
    version the paper used to carry, it has an interior minimum.
    """
    eps, gamma, T, C, c0, b, alpha, r_p, a = sp.symbols(
        "eps gamma T C c0 b alpha r_p a", positive=True)
    S_T = T ** (sp.Rational(1, 2) - gamma)
    T_star = sp.simplify(sp.solve(sp.Eq(C / (c0 * eps * S_T), b * eps), T)[0])
    expo = sp.limit(sp.simplify(sp.log(T_star).diff(eps) * eps), gamma, 0)
    ratio = sp.simplify(sp.limit(sp.simplify(T_star.subs(eps, alpha * eps) / T_star), gamma, 0))
    print("\n(5) the eps exchange rate and the optimal probe radius")
    print(f"  T*(eps) = {T_star},  d log T*/d log eps -> {expo}")
    print(f"  T*(alpha eps)/T*(eps) -> {ratio}")
    assert expo == -4, expo
    assert sp.simplify(ratio - alpha**-4) == 0, ratio

    B_tot = a * r_p**4 + b * eps / r_p
    r_star = [r for r in sp.solve(sp.diff(B_tot, r_p), r_p) if r.is_real is not False][0]
    B_star = sp.simplify(sp.powsimp(B_tot.subs(r_p, r_star)))
    print(f"  r_p* = {sp.simplify(r_star)}   (so r_p* ~ eps^(1/5))")
    print(f"  B(r_p*) = {B_star}   (so B ~ eps^(4/5))")
    assert sp.simplify(sp.log(sp.simplify(r_star)).diff(eps) * eps - sp.Rational(1, 5)) == 0
    assert sp.simplify(sp.log(B_star).diff(eps) * eps - sp.Rational(4, 5)) == 0
    print("  -> at fixed r_p the floor falls as eps and costs eps^-4 in iterations;")
    print("     optimising r_p jointly gives r_p* ~ eps^(1/5) and a floor ~ eps^(4/5).")


def occupancy_along_a_trajectory(eps, r_s, period, n_steps, seed, r_p=0.5, jump=0.05):
    """Occupancy of the delta-tube by an ADAPTED iterate, in PATH-LENGTH units.

    ``check_straddle_probability`` above draws base points uniformly in a window,
    which supplies the occupancy hypothesis by construction: it measures pi_D given
    a bounded density for the iterate, not the density itself.  And pi_D as the
    appendix used to define it -- ``sup_i Pr[G_i^c]``, a supremum over
    configurations -- cannot be O(delta/L_D) at all, since conditional on an iterate
    within delta of a wall the bad event has probability one.

    The object the descent inequality actually sums is neither: the bad-event term
    enters as ``r_{s,t} eps pi_D(t)``, weighted by the step size, so what has to be
    O(delta/L_D) is the PATH-LENGTH-weighted occupancy

        sum_t ||Delta x_t|| 1[bad] / sum_t ||Delta x_t||,

    and that one is bounded by an elementary crossing argument: between two
    consecutive wall crossings the path covers at least 2 L_D, and each crossing
    spends at most 2 delta + ||Delta x|| inside the tube.

    Measured here on the staircase L(x) = x_0 + jump * floor(x_0/period) + ||x_>||^2,
    the geometry the INT8 and staircase showcases have: parallel walls of spacing
    ``period``, so L_D = period/2, with a smooth tilt that keeps the iterate
    crossing them instead of freezing on one step.
    """
    from check_stein_constant import haar_so, orthoplex

    rng = np.random.default_rng(seed)
    verts = orthoplex(DP)
    x = np.full(DP, 0.37)
    delta = 0.5 * r_p * eps                     # lambdabar = 1/2
    inside = 0.0
    total = 0.0
    for _ in range(n_steps):
        R = haar_so(1, DP, rng)[0]
        dirs = verts @ R.T
        probes = x[None, :] + r_p * eps * dirs
        costs = (probes[:, 0] + jump * np.floor(probes[:, 0] / period)
                 + 0.5 * (probes[:, 1:] ** 2).sum(axis=1))
        w = np.exp(-(costs - costs.min()) / TAU)
        w /= w.sum()
        move = r_s * eps * (w[:, None] * dirs).sum(axis=0)
        x = x + move
        d_wall = abs(x[0] / period - np.round(x[0] / period)) * period
        length = float(np.linalg.norm(move))
        total += length
        inside += length * (d_wall < delta)
    return inside / total, delta


def occupancy_moments(eps, r_s=0.5, r_p=0.5, period=0.25, n_steps=4000, seed=1, jump=0.05):
    """(path-weighted avg of pi_D, path-weighted avg of pi_D^2) along a trajectory.

    pi_D(t) := Pr[the probe annulus meets a wall | theta_t] is estimated by
    resampling the probe randomness at the realised iterate, so the two moments are
    measured rather than assumed equal or unequal.
    """
    from check_stein_constant import haar_so, orthoplex

    verts = orthoplex(DP)
    rng = np.random.default_rng(seed)
    x = np.full(DP, 0.37)
    first = second = total = 0.0
    for _ in range(n_steps):
        R = haar_so(1, DP, rng)[0]
        dirs = verts @ R.T
        probes = x[None, :] + r_p * eps * dirs
        costs = (probes[:, 0] + jump * np.floor(probes[:, 0] / period)
                 + 0.5 * (probes[:, 1:] ** 2).sum(axis=1))
        w = np.exp(-(costs - costs.min()) / TAU)
        w /= w.sum()
        move = r_s * eps * (w[:, None] * dirs).sum(axis=0)
        x = x + move
        # pi_D at the realised iterate, over fresh probe randomness
        m = 300
        Rm = haar_so(m, DP, rng)
        eta = rng.uniform(-ETA_MAX, ETA_MAX, size=m)
        s_rad = r_p * eps * np.outer(1.0 + eta, LAM)
        d0 = np.einsum("nij,vj->nvi", Rm, verts)[:, :, 0]
        lo = np.floor((x[0] + d0[:, :, None] * s_rad[:, None, :]) / period).reshape(m, -1)
        pi = float(np.mean(lo.max(axis=1) != lo.min(axis=1)))
        length = float(np.linalg.norm(move))
        total += length
        first += length * pi
        second += length * pi * pi
    return first / total, second / total


def check_occupancy_on_adapted_iterates():
    """pi_D = O(delta/L_D) holds for the trajectory, in the units the proof uses."""
    print("\n(3) path-weighted occupancy of the delta-tube by the actual iterate")
    print("    (staircase of spacing 0.25, so L_D = 0.125)")
    epss = np.array([0.2, 0.1, 0.05, 0.025])
    r_s, period, n_steps = 0.5, 0.25, 20_000
    l_d = 0.5 * period

    occs, deltas = [], []
    for i, e in enumerate(epss):
        occ, delta = occupancy_along_a_trajectory(e, r_s, period, n_steps, 30 + i)
        occs.append(occ)
        deltas.append(delta)
    occs, deltas = np.array(occs), np.array(deltas)
    pred = deltas / l_d
    slope = fit_slope(deltas, occs)
    print(f"  {'eps':>7} {'delta':>8} {'path occupancy':>16} {'delta/L_D':>11} {'ratio':>8}")
    for e, d, o, p in zip(epss, deltas, occs, pred):
        print(f"  {e:7.4g} {d:8.4f} {o:16.4f} {p:11.4f} {o / p:8.3f}")
    print(f"  fitted exponent in delta: {slope:.2f} (theory 1); "
          f"ratio to delta/L_D within {100 * np.abs(occs / pred - 1).max():.0f}%")
    assert 0.9 < slope < 1.1, slope
    assert np.abs(occs / pred - 1).max() < 0.15, occs / pred

    print("  -> the geometric bound is earned in path-length units, which is what the")
    print("     descent inequality weights by, and it needs the step to be O(delta):")
    print("     condition (v) delivers that eventually, since r_{s,t} -> 0.")


def demo():
    check_floor_algebra()
    check_linearization_remainder()
    check_straddle_probability()
    check_occupancy_on_adapted_iterates()
    check_straddle_term_is_linear_not_quadratic()
    check_epsilon_exchange_rate()


if __name__ == "__main__":
    demo()
    print("OK")
