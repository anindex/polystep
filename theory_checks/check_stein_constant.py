#!/usr/bin/env sage -python
"""A6: the orthoplex Stein constant, and the step constant it is supposed to give.

Appendix (Lemma "Softmax barycentric step approximates the smoothed gradient")
claims the expected barycentric step equals

    -(2 r_p / (V d_p)) grad L_eps  =  -(r_p / d_p^2) grad L_eps      [as printed]

using V = 2 d_p for the orthoplex, and rests it on

    E_R[ sum_v (R v_v)(R v_v)^T ]  =  (2/d_p) I                      [as printed]

Part 1 below checks the matrix identity.  It is wrong as printed: the sum is 2I,
not (2/d_p) I, and in fact it holds POINTWISE for every orthogonal R, not merely
in expectation, because sum_v v v^T = 2I for the orthoplex and R (2I) R^T = 2I.
Only the per-vertex statement E_R[(Ru)(Ru)^T] = I/d_p needs an expectation.

Part 1 is necessary but nowhere near sufficient, and an earlier version of this
script stopped there: it verified the matrix identity and then HARDCODED 2/V as
the step coefficient without ever evaluating a loss or a softmax weight.  Part 2
tests the operational claim instead, and it does not survive intact.

WHAT THE STEP ACTUALLY TRACKS
-----------------------------
The probes are not at radius r_p eps.  They sit at

    s = r_p * lambda_k * (1 + eta) * eps,   lambda_k = k / (K+1), k = 1..K,

with per-step multiplicative jitter eta (``PolyStepOptimizer.probe_radius_jitter``,
``_probes = linspace(0,1,K+2)[1:K+1]``), and the cost of a vertex is the mean over
the K scales.  Note mean_k lambda_k = 1/2 for EVERY K.

Linearising the softmax (weights near uniform, sum_v R v = 0 kills the shift) and
applying Flaxman's uniform-ball identity radius by radius,

    E[step] = -(1/(V tau)) E[ sum_v C_v R v ] + O(...)
            = -(1/(d_p tau)) E_{k,eta}[ s * grad L^ball_s(x) ] + O(...)
            = -(E[s] / (d_p tau)) grad Lmix(x) + O(...),

    Lmix := the E[s]-normalised, s-WEIGHTED mixture of uniform-ball smoothings at
            the realised probe radii s.

So the correct statement has two corrections to the printed one:

  * the coefficient is E[s] / (d_p tau) = r_p eps / (2 d_p tau), not r_p / d_p^2
    and not the corrected-for-the-matrix-identity r_p / d_p.  With tau = eps
    (the code's default, ``scale_cost = 1.0``, so no cost rescaling) this is
    r_p / (2 d_p): a further factor of 2 smaller, from mean_k lambda_k = 1/2.
  * the surrogate is smoothed at radius ~ r_p eps / 2, not r_p eps, and it is a
    mixture over radii rather than a single one.

Part 2 measures a real loss at real probe locations, forms the real softmax
weights, and compares against a numerically differentiated Lmix, reporting the
empirically fitted coefficient rather than assuming one.
"""

import numpy as np

RNG = np.random.default_rng(0)

DP = 4
V = 2 * DP
K = 3
LAM = np.arange(1, K + 1) / (K + 1)  # k/(K+1): 0.25, 0.5, 0.75
ETA_MAX = 0.3
EPS = 1.0  # smoothing scale
TAU = 1.0  # softmax temperature (ent_epsilon); scale_cost = 1.0 so no rescaling

# L(x) = g.x + B x_0 ||x||^2.  Chosen because its uniform-ball smoothing has a
# CLOSED FORM gradient that depends on the radius:
#     grad L^ball_s(0) = g + B s^2 e_0        (exact, no higher-order terms)
# so "which radius is the surrogate smoothed at" is a first-order, measurable
# question rather than a hidden O(s^2) correction.
G = np.array([0.6, -0.3, 0.2, 0.1])
B = 5.0


def loss(x):
    return x @ G + B * x[..., 0] * np.sum(x * x, axis=-1)


def haar_so(n, dp, rng):
    """n Haar-uniform samples from SO(dp) via QR (Mezzadri), batched."""
    z = rng.standard_normal((n, dp, dp))
    q, r = np.linalg.qr(z)
    q = q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[:, None, :]
    flip = np.linalg.det(q) < 0
    q[flip, :, 0] *= -1
    return q


def orthoplex(dp):
    """The 2*dp unit vertices {+-e_j}."""
    return np.vstack([np.eye(dp), -np.eye(dp)])


# --- Part 1: the matrix identity ---------------------------------------------


def check_matrix_identity():
    n_samples = 200_000
    for dp in (2, 4, 8):
        verts = orthoplex(dp)

        # Pointwise, for EVERY orthogonal R: sum_v (Rv)(Rv)^T = R (2I) R^T = 2I.
        # No expectation needed, and no Monte Carlo error to hide behind.
        for R in haar_so(64, dp, RNG):
            rot = verts @ R.T
            assert np.allclose(rot.T @ rot, 2.0 * np.eye(dp), atol=1e-12), (dp, R)

        # The per-vertex statement is the one that genuinely needs E_R.
        R = haar_so(n_samples, dp, RNG)
        u = R[:, :, 0]  # R e_0, uniform on the sphere
        per_vertex = np.einsum("ni,nj->ij", u, u) / n_samples
        assert np.allclose(per_vertex, np.eye(dp) / dp, atol=8.0 / np.sqrt(n_samples))

        # The paper prints (2/dp) I for the summed quantity; that is wrong for dp > 1.
        assert not np.allclose(2.0 * np.eye(dp), (2.0 / dp) * np.eye(dp)) or dp == 1
        print(f"dp={dp}: sum_v (Rv)(Rv)^T = 2I pointwise (printed (2/dp)I); "
              f"E_R[(Ru)(Ru)^T] = I/{dp} confirmed")


# --- Part 2: the operational step constant -----------------------------------


def expected_step(r_p, n_rot, rng):
    """(E[step], E[linearised step]) for the real probe law and cost rule.

    C_v = mean_k L(x + r_p lambda_k (1+eta) eps R v_v), one eta per rotation
    (the optimizer jitters once per step, shared across probes and vertices).

    The second return value is the first-order softmax expansion
    -(1/(V tau)) sum_v C_v R v_v -- the object the Stein identity turns into
    -(E[s]/(d_p tau)) grad Lmix.  Returning it separately isolates the
    softmax-linearisation remainder from Monte-Carlo noise: for the orthoplex
    sum_v R v_v = 0 and sum_v (R v)(R v)^T = 2I pointwise, so the linearised step
    has NO rotation variance at all and the difference is almost noise-free.
    """
    verts = orthoplex(DP)
    R = haar_so(n_rot, DP, rng)
    directions = np.einsum("nij,vj->nvi", R, verts)  # (n, V, DP)
    eta = rng.uniform(-ETA_MAX, ETA_MAX, size=n_rot)
    s = r_p * EPS * np.outer(1.0 + eta, LAM)  # (n, K)
    pts = directions[:, :, None, :] * s[:, None, :, None]  # (n, V, K, DP)
    c = loss(pts).mean(axis=-1)  # (n, V)
    w = np.exp(-(c - c.min(axis=1, keepdims=True)) / TAU)
    w /= w.sum(axis=1, keepdims=True)
    step = np.einsum("nv,nvi->i", w, directions) / n_rot
    lin = -np.einsum("nv,nvi->i", c, directions) / (V * TAU * n_rot)
    return step, lin


def moments(r_p):
    """(E[s], E[s^3]) for s = r_p lambda (1+eta) eps, lambda uniform on LAM."""
    e_lam, e_lam3 = LAM.mean(), (LAM**3).mean()
    e_eta1, e_eta3 = 1.0, 1.0 + ETA_MAX**2  # E[1+eta], E[(1+eta)^3]
    return r_p * EPS * e_lam * e_eta1, (r_p * EPS) ** 3 * e_lam3 * e_eta3


def grad_lmix_mc(r_p, n, rng, h=0.01):
    """Central-difference gradient of Lmix at 0, sampled from the real probe law.

    Lmix(x) = E_{k,eta}[ s L^ball_s(x) ] / E[s]; sampled by drawing (k, eta),
    a uniform point in the unit ball, and importance-weighting by s / E[s].
    Common random numbers across the stencil.
    """
    e_s, _ = moments(r_p)
    eta = rng.uniform(-ETA_MAX, ETA_MAX, size=n)
    lam = rng.choice(LAM, size=n)
    s = r_p * EPS * lam * (1.0 + eta)
    gdir = rng.standard_normal((n, DP))
    gdir /= np.linalg.norm(gdir, axis=1, keepdims=True)
    u = s[:, None] * (rng.random(n) ** (1.0 / DP))[:, None] * gdir
    wt = s / e_s
    out = np.empty(DP)
    for j in range(DP):
        e = np.zeros(DP)
        e[j] = h
        out[j] = np.mean(wt * (loss(u + e) - loss(u - e))) / (2 * h)
    return out


def check_step_constant():
    n_rot = 300_000
    n_mc = 2_000_000
    r_ps = np.array([0.4, 0.2, 0.1, 0.05, 0.025])

    print(f"\nprobe scales lambda_k = {np.array2string(LAM, precision=3)} "
          f"(mean {LAM.mean():.3f}), jitter eta ~ U(+-{ETA_MAX}), eps={EPS}, tau={TAU}")
    print(f"{'r_p':>6} {'E[s]':>8} {'c_hat (fit)':>12} {'c_correct':>10} {'c_paper':>9} "
          f"{'rel.err correct':>16} {'rel.err paper':>14}")

    residuals, fitted = [], []
    for i, r_p in enumerate(r_ps):
        rng = np.random.default_rng(2000 + i)  # independent randomness per radius
        step, lin = expected_step(r_p, n_rot, rng)

        e_s, e_s3 = moments(r_p)
        # Closed form for this loss: grad L^ball_s(0) = g + B s^2 e_0, so
        # grad Lmix(0) = g + B (E[s^3]/E[s]) e_0.
        grad_mix = G + B * (e_s3 / e_s) * np.eye(DP)[0]
        grad_mix_mc = grad_lmix_mc(r_p, n_mc, rng)
        assert np.allclose(grad_mix_mc, grad_mix, rtol=0.02, atol=2e-3), (grad_mix_mc, grad_mix)

        c_correct = e_s / (DP * TAU)  # = r_p eps / (2 d_p tau)
        c_paper = r_p / DP  # printed constant, after fixing the matrix identity
        pred_correct = -c_correct * grad_mix
        # The paper's object: the ball smoothing at radius r_p eps, not r_p eps/2.
        pred_paper = -c_paper * (G + B * (r_p * EPS) ** 2 * np.eye(DP)[0])

        # The Stein/Flaxman half: the linearised step must equal -c grad Lmix
        # up to Monte-Carlo error on eta alone.
        assert np.allclose(lin, pred_correct, rtol=2e-3, atol=1e-9), (lin, pred_correct)

        c_hat = float(-step @ grad_mix / (grad_mix @ grad_mix))
        err_correct = np.linalg.norm(step - pred_correct) / np.linalg.norm(pred_correct)
        err_paper = np.linalg.norm(step - pred_paper) / np.linalg.norm(pred_paper)
        residuals.append(np.linalg.norm(step - lin))  # softmax nonlinearity only
        fitted.append(c_hat)

        print(f"{r_p:6.3f} {e_s:8.4f} {c_hat:12.5f} {c_correct:10.5f} {c_paper:9.5f} "
              f"{err_correct:16.3f} {err_paper:14.3f}")

    residuals, fitted = np.array(residuals), np.array(fitted)
    c_correct = np.array([moments(r)[0] for r in r_ps]) / (DP * TAU)

    # The fitted coefficient converges to E[s]/(d_p tau), i.e. to HALF the
    # r_p/d_p that the corrected matrix identity alone would give.
    assert np.allclose(fitted[-2:] / c_correct[-2:], 1.0, rtol=0.03), fitted / c_correct
    ratio = float(np.mean(fitted[-2:] / (r_ps[-2:] / DP)))
    print(f"\nfitted coefficient / (r_p/d_p) -> {ratio:.4f}   (theory 0.5 = mean_k lambda_k)")
    assert abs(ratio - 0.5) < 0.02, ratio

    # Against the constant as literally printed, r_p/d_p^2: the two errors partly
    # cancel.  The matrix identity is short by a factor d_p, mean_k lambda_k = 1/2
    # is long by a factor 2, so printed/true = 2/d_p -- exact only at d_p = 2.
    printed_ratio = float(np.mean(fitted[-2:] / (r_ps[-2:] / DP**2)))
    print(f"fitted coefficient / (r_p/d_p^2, as printed) -> {printed_ratio:.4f}   "
          f"(theory d_p/2 = {DP / 2})")
    assert abs(printed_ratio - DP / 2) < 0.05, printed_ratio

    # Softmax-linearisation remainder.  The r_p^5 term is still visible at r_p >= 0.2
    # (local slope 4.2 there), so the exponent is fitted on the asymptotic tail; the
    # printed residuals show the approach.
    slope = float(np.polyfit(np.log(r_ps[-3:]), np.log(residuals[-3:]), 1)[0])
    print(f"residuals ||E[step] - E[linearised step]|| = {np.array2string(residuals, precision=2)}")
    print(f"softmax remainder ~ r_p^{slope:+.2f} for r_p <= {r_ps[-3]}   (theory 3)")
    assert 2.5 < slope < 3.5, slope


# --- the plateau freeze: a consequence of the same antipodal symmetry ------------


def check_constant_row_freeze():
    """On a constant cost row the barycentric step is EXACTLY zero, for every
    centered polytope and every rotation.

    The step is sum_v T_v R v_v.  A constant row makes the softmax weights uniform,
    T_v = a_i / V, so the step is (a_i / V) R (sum_v v_v), and every polytope
    PolyStep provides is centered: sum_v v_v = 0.  The rotation factors out, so no
    choice of rotation law escapes it -- in particular biased rotation does not,
    even though it makes E[R] != 0.

    This is the same symmetry that makes sum_v (Rv)(Rv)^T = 2I hold POINTWISE for
    the orthoplex rather than only in expectation.  One symmetry, one strength, one
    limitation.
    """
    from polystep.geometry import POLYTOPE_MAP

    rng = np.random.default_rng(0)
    print("\nconstant-row step, by polytope (a plateau wider than the probe reach)")
    for name, fn in POLYTOPE_MAP.items():
        for dp in (2, 4, 8):
            verts = fn(dp).numpy().astype(float)
            vsum = float(np.linalg.norm(verts.sum(axis=0)))
            V = verts.shape[0]
            R = haar_so(500, dp, rng)
            # Uniform weights: what a constant cost row produces at any temperature.
            step = np.einsum("nij,vj->nvi", R, verts).mean(axis=1)
            worst = float(np.linalg.norm(step, axis=1).max())
            # The identity, not a magnitude: |step| = |sum_v v_v| / V for every R,
            # because the rotation factors out of the sum. Asserting this rather than
            # "step is small" keeps the check honest about the simplex generator,
            # which centres only to float32 precision.
            assert abs(worst - vsum / V) < 1e-12, (name, dp, worst, vsum / V)
            assert vsum < 1e-6, (name, dp, vsum)     # every provided polytope is centered
            print(f"  {name:10s} d_p={dp}  |sum_v v_v|={vsum:.1e}  "
                  f"max|step| over rotations = {worst:.2e}  (= |sum_v v_v|/V)")

    # Biased rotation does not help: the sum factors through R.
    dp = 4
    verts = POLYTOPE_MAP["orthoplex"](dp).numpy().astype(float)
    biased = haar_so(200, dp, rng)
    biased[:, :, 0] = np.abs(biased[:, :, 0])          # a crude E[R] != 0 bias
    step = np.einsum("nij,vj->nvi", biased, verts).mean(axis=1)
    assert float(np.linalg.norm(step, axis=1).max()) < 1e-12
    print("  biased rotation (E[R] != 0) does not move a constant row either: "
          "sum_v R v_v = R sum_v v_v = 0")


def demo():
    check_matrix_identity()
    check_step_constant()
    check_constant_row_freeze()


if __name__ == "__main__":
    demo()
    print("OK")
