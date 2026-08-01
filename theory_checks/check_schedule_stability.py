#!/usr/bin/env sage -python
"""A7: the stable-schedules claim, proved from the drift instead of the envelope.

The corollary used to argue from the displacement ENVELOPE of the fragility
proposition,

    (1 - 2 kappa) r_s eps  <=  ||x^{t+1} - x^t||  <=  r_s eps,

and conclude an "oscillation of amplitude Theta(r_s eps) about theta*".  That is a
non-sequitur: an envelope bounds one increment and says nothing about where the
iterate ends up.  With the direction resampled every step, the envelope alone is
equally consistent with a free random walk whose excursion grows like sqrt(t) r_s
eps and never settles.  ``check_envelope_alone_permits_a_random_walk`` exhibits
exactly that, so the objection is real and the old argument cannot stand.

The conclusion survives anyway, for a reason the old argument never gave: a
NEGATIVE DRIFT, and the mechanism is the polytope's antipodal symmetry rather than
a naive "the winning vertex is the one most anti-aligned with x".  Expanding the
cost about the minimum, the v-dependent part is

    r_p eps <H z, R v>                    linear, flips sign under v -> -v
  + (1/2) r_p^2 eps^2 <H R v, R v>        quadratic, IDENTICAL for v and -v,

and in the regime ||z|| << r_p eps the QUADRATIC term is the larger one.  So the
curvature, not z, selects which antipodal PAIR wins.  That does not break the
argument, because the quadratic term is common to +v and -v and cancels in the
comparison within the winning pair: the sign is set by the linear term alone, and
it is always the descending one.  Hence:

  * The step direction depends on x only through x/||x||.  The process is therefore
    scale-covariant in x: rescaling r_s eps rescales the whole trajectory.  The
    stationary excursion is exactly proportional to r_s eps, which is the
    Theta(r_s eps) the corollary claimed, now with a mechanism.
  * E[<step, x/||x||>] < 0 strictly, uniformly in the direction of x.  That is a
    Foster-Lyapunov negative drift for V(x) = ||x||, so the excursion is confined
    rather than growing, and the confinement is what the corollary needs.

Measured below: the amplitude/(r_s eps) ratio is constant to three digits across a
16x range of r_s (orthoplex, d_p = 4), the growth exponent is ~0 under both
schedules, and the drift is negative by a margin.  The random-walk exponent of 1/2
appears exactly when the drift is switched off, which is the plateau case the freeze
proposition already isolates.
"""

import numpy as np

DP = 4
EPS = 0.3            # probe/step radius scale
TAU = 1e-5           # softmax temperature: saturated, the fragility regime
R_P = 1.0
CURV = 1.0


def orthoplex(dp):
    verts = np.zeros((2 * dp, dp))
    for j in range(dp):
        verts[2 * j, j] = 1.0
        verts[2 * j + 1, j] = -1.0
    return verts


def haar_so(n, dp, rng):
    a = rng.standard_normal((n, dp, dp))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.einsum("nii->ni", r))[:, None, :]


def step(x, r_s, eps, verts, rng, hess=None):
    """One PolyStep displacement on L(x) = 1/2 <H x, x>, H = 2*CURV*I by default."""
    R = haar_so(1, DP, rng)[0]
    dirs = verts @ R.T
    probes = x[None, :] + R_P * eps * dirs
    if hess is None:
        costs = CURV * np.einsum("vi,vi->v", probes, probes)
    else:
        costs = 0.5 * np.einsum("vi,ij,vj->v", probes, hess, probes)
    w = np.exp(-(costs - costs.min()) / TAU)
    w /= w.sum()
    return r_s * eps * (w[:, None] * dirs).sum(axis=0)


def run(schedule, T, seed, eps=EPS, x0=None):
    rng = np.random.default_rng(seed)
    verts = orthoplex(DP)
    if x0 is None:
        # Start near the minimum, not on it: at x = 0 the cost row is symmetric
        # under v -> -v and the step is exactly zero.  That is the antipodal
        # symmetry of the freeze proposition, not a stability result.
        g = rng.standard_normal(DP)
        x = 0.5 * R_P * eps * g / np.linalg.norm(g)
    else:
        x = np.array(x0, dtype=float)
    out = np.empty(T)
    for t in range(T):
        x = x + step(x, schedule(t), eps, verts, rng)
        out[t] = np.linalg.norm(x)
    return out


def fit_exponent(ts, ys):
    m = (ts > ts[-1] // 4) & (ys > 0)
    return float(np.polyfit(np.log(ts[m]), np.log(ys[m]), 1)[0])


def amplitude(r_s, eps=EPS, T=2500, n_runs=6, base=200):
    tails = [np.sqrt((run(lambda t: r_s, T, base + i, eps)[-T // 3:] ** 2).mean())
             for i in range(n_runs)]
    return float(np.mean(tails))


def check_growth_exponent_is_zero():
    """Confined, not a random walk -- under both the flat and the decaying schedule."""
    T, n_runs = 2500, 12
    ts = np.arange(1, T + 1)
    print("(1) excursion ||x_t - x*||: a free random walk would fit exponent 1/2")
    print(f"  {'schedule':>34} {'fitted exponent':>16}")
    for name, sched in (("flat r_s = 0.1", lambda t: 0.1),
                        ("decaying r_s (condition (v))", lambda t: (t + 1) ** (-0.6))):
        runs = np.array([run(sched, T, 300 + i) for i in range(n_runs)])
        rms = np.sqrt((runs**2).mean(axis=0))
        slope = fit_exponent(ts, rms)
        print(f"  {name:>34} {slope:16.3f}")
        assert slope < 0.12, (name, slope)


def check_amplitude_is_proportional_to_step():
    """Scale covariance: amplitude / (r_s eps) is a constant of the polytope."""
    print("\n(2) stationary amplitude vs r_s, at eps = %.2f" % EPS)
    r_ss = np.array([0.4, 0.2, 0.1, 0.05, 0.025])
    amps = np.array([amplitude(r) for r in r_ss])
    ratios = amps / (r_ss * EPS)
    for r, a, q in zip(r_ss, amps, ratios):
        print(f"  r_s = {r:6.3f}   amplitude {a:.5f}   amplitude/(r_s eps) = {q:.3f}")
    slope = float(np.polyfit(np.log(r_ss), np.log(amps), 1)[0])
    print(f"  fitted exponent in r_s: {slope:.3f} (theory 1); "
          f"ratio spread {ratios.max() / ratios.min():.4f}x (theory 1)")
    assert abs(slope - 1.0) < 0.05, slope
    assert ratios.max() / ratios.min() < 1.05, ratios


def check_ratio_to_basin_does_not_vanish():
    """Shrinking eps at fixed r_s does not stabilise: the basin shrinks just as fast."""
    r_s = 0.2
    print("\n(3) stationary amplitude vs eps, at fixed r_s = %.2f" % r_s)
    epss = np.array([0.4, 0.2, 0.1, 0.05])
    amps = np.array([amplitude(r_s, eps=e) for e in epss])
    for e, a in zip(epss, amps):
        print(f"  eps = {e:6.3f}   amplitude {a:.5f}   amplitude/(r_s eps) = {a / (r_s * e):.3f}")
    slope = float(np.polyfit(np.log(epss), np.log(amps), 1)[0])
    print(f"  fitted exponent in eps: {slope:.3f} (theory 1), so amplitude/basin = Theta(r_s):")
    print("  eps -> 0 at fixed r_s does not settle the iterate.")
    assert abs(slope - 1.0) < 0.05, slope


def check_drift_is_negative():
    """The Foster-Lyapunov ingredient: E[<step, x/||x||>] < 0, uniformly in direction."""
    verts = orthoplex(DP)
    r_s = 0.1
    print("\n(4) radial drift E[<step, xhat>] / (r_s eps), by offset magnitude")
    for mag in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(11)
        g = rng.standard_normal(DP)
        x = mag * r_s * EPS * g / np.linalg.norm(g)
        xhat = x / np.linalg.norm(x)
        steps = np.array([step(x, r_s, EPS, verts, np.random.default_rng(700 + i))
                          for i in range(4000)])
        drift = float((steps @ xhat).mean()) / (r_s * EPS)
        print(f"  ||x|| = {mag:3.1f} r_s eps   normalised radial drift {drift:+.3f}")
        assert drift < -0.05, (mag, drift)


def check_envelope_alone_permits_a_random_walk():
    """Same envelope, no drift: exponent 1/2.  This is why the old argument failed."""
    T, n_runs = 2500, 12
    r_s = 0.1
    verts = orthoplex(DP)
    ts = np.arange(1, T + 1)

    def undirected(seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(DP)
        out = np.empty(T)
        for t in range(T):
            d = (verts @ haar_so(1, DP, rng)[0].T)[rng.integers(0, 2 * DP)]
            x = x + r_s * EPS * d
            out[t] = np.linalg.norm(x)
        return out

    rms = np.sqrt((np.array([undirected(500 + i) for i in range(n_runs)]) ** 2).mean(axis=0))
    slope = fit_exponent(ts, rms)
    print(f"\n(5) same displacement envelope, drift removed: exponent {slope:.3f} (theory 1/2)")
    print("  -> the envelope alone is consistent with an unbounded excursion, so the")
    print("     corollary cannot be proved from it.  The drift is what rules this out,")
    print("     and the drift is absent exactly on a plateau, where the freeze")
    print("     proposition says the step is zero instead.")
    assert abs(slope - 0.5) < 0.08, slope


def check_drift_survives_anisotropy():
    """The drift does not need an isotropic Hessian, and the reason is antipodal symmetry.

    Expanding the cost about the minimum, the v-dependent part is

        r_p eps <H z, R v>        (linear, sign-flipping under v -> -v)
      + (1/2) r_p^2 eps^2 <H R v, R v>   (quadratic, IDENTICAL for v and -v).

    In the regime ||z|| << r_p eps the quadratic term is the larger of the two, so it
    is what selects which antipodal PAIR wins -- and it is z-independent, which looks
    at first like a refutation of the drift argument.  It is not, because the
    quadratic term is common to +v and -v and therefore cancels in the comparison
    WITHIN the winning pair.  The sign is set by the linear term alone, and it is
    always the descending one.  So the drift stays strictly negative for any
    H > 0; only the CONSTANT degrades with the conditioning of H, because the pair
    the curvature selects need not be well aligned with z.
    """
    verts = orthoplex(DP)
    r_s = 0.1
    print("\n(6) radial drift and amplitude under an anisotropic Hessian")
    print(f"  {'H':>26} {'drift/(r_s eps)':>18} {'amplitude/(r_s eps)':>21}")
    for name, hess in (("2I (isotropic)", 2 * CURV * np.eye(DP)),
                       ("diag(1,3,10,30)", np.diag([1.0, 3.0, 10.0, 30.0])),
                       ("diag(1,1,1,100)", np.diag([1.0, 1.0, 1.0, 100.0]))):
        rng = np.random.default_rng(11)
        g = rng.standard_normal(DP)
        drifts = []
        for mag in (0.5, 1.0, 2.0):
            x = mag * r_s * EPS * g / np.linalg.norm(g)
            xh = x / np.linalg.norm(x)
            steps = np.array([step(x, r_s, EPS, verts, np.random.default_rng(700 + i), hess)
                              for i in range(3000)])
            drifts.append(float((steps @ xh).mean()) / (r_s * EPS))
        amps = []
        for i in range(6):
            rng = np.random.default_rng(300 + i)
            gg = rng.standard_normal(DP)
            x = 0.5 * R_P * EPS * gg / np.linalg.norm(gg)
            tail = []
            for t in range(2500):
                x = x + step(x, r_s, EPS, verts, rng, hess)
                if t > 800:
                    tail.append(np.linalg.norm(x))
            amps.append(np.sqrt(np.mean(np.array(tail) ** 2)))
        amp = float(np.mean(amps)) / (r_s * EPS)
        print(f"  {name:>26} {np.array2string(np.array(drifts), precision=3):>18} {amp:21.3f}")
        assert max(drifts) < -0.05, (name, drifts)
        assert 0.3 < amp < 4.0, (name, amp)
    print("  -> drift strictly negative and amplitude still Theta(r_s eps) at condition")
    print("     number 100; the constant, not the scaling, is what anisotropy costs.")


def demo():
    check_growth_exponent_is_zero()
    check_amplitude_is_proportional_to_step()
    check_ratio_to_basin_does_not_vanish()
    check_drift_is_negative()
    check_envelope_alone_permits_a_random_walk()
    check_drift_survives_anisotropy()


if __name__ == "__main__":
    demo()
    print("OK")
