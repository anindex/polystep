"""Log-domain Sinkhorn solver for entropic optimal transport.

PolyStep's OT problems are (n particles x m=V polytope vertices) with V small
(2*particle_dim), so the cost is O(n*V): a full-rank log-domain solve is
already optimal and there is no large dense kernel to approximate away.
"""

import functools
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from ._shared import (
    align_dual,
    align_marginal,
    exp_plan,
    prepare_cost,
    validate_positive,
    warn_tiny_temperature,
)


def _warn_dual_reset() -> None:
    """Announce that the duals went non-finite and were reset.

    ``converged=False`` was the only signal, and ``.matrix`` still returns a
    plausible-looking ``exp(-C/eps)`` whose rows do not sum to ``a``. The
    integrated path renormalizes and survives; a direct caller does not.
    """
    warnings.warn(
        "Sinkhorn duals went non-finite and were reset to zero. The returned plan is "
        "exp(-C/eps) and does not satisfy the marginals; check `converged` before use. "
        "A smaller epsilon or a larger cost scale usually triggers this.",
        stacklevel=3,
    )


@dataclass
class SinkhornResult:
    """Output from a Sinkhorn solve.

    Attributes:
        f: First dual potential of shape (n,). Together with ``g``, these
            dual potentials encode the optimal transport solution. They can
            be reused as warm-start initializations for the next solve,
            which typically reduces iterations from ~100 to ~10.
        g: Second dual potential of shape (m,). See ``f``.
        converged: Whether the solver converged within tolerance.
        n_iters: Number of iterations actually run.
        ent_reg_cost: Entropic dual objective <f, a> + <g, b> - eps*sum_ij P_ij
            (true plan mass), reported in the original cost frame. Equals the
            regularized transport cost at convergence.
        errors: Per-check marginal errors (for diagnostics).
    """

    f: torch.Tensor
    g: torch.Tensor
    converged: bool
    n_iters: int
    errors: Optional[List[float]] = None

    # Internal fields for the lazy .matrix and .ent_reg_cost properties
    _eps: float = float("nan")
    _cost_matrix: Optional[torch.Tensor] = None  # (n, m), recentered and scaled
    _a: Optional[torch.Tensor] = None
    _b: Optional[torch.Tensor] = None
    _cost_scale: float = 1.0
    _cost_shift: float = 0.0

    @functools.cached_property
    def matrix(self) -> torch.Tensor:
        """Transport plan, computed lazily: P_ij = exp((f_i + g_j - C_ij) / eps)."""
        return exp_plan(self.f, self.g, self._cost_matrix, self._eps)

    @functools.cached_property
    def ent_reg_cost(self) -> float:
        """Entropic dual objective, computed lazily and reported in the caller's frame.

        ``<f, a> + <g, b> - eps * sum_ij P_ij`` using the true plan mass, which equals
        ``eps * sum(a)`` only at convergence. Undoes both frame changes: multiply by
        the cost scale (the scaled problem at eps is the raw problem at eps*scale),
        then add back the recentering shift.

        Lazy because the monolithic step never reads it, and computing it eagerly costs
        a full (n, m) logsumexp plus a device sync on every solve.
        """
        log_P = (self.f.unsqueeze(1) + self.g.unsqueeze(0) - self._cost_matrix) / self._eps
        plan_mass = torch.exp(torch.logsumexp(log_P.reshape(-1), dim=0))
        dual = (self.f * self._a).sum() + (self.g * self._b).sum() - self._eps * plan_mass
        return (dual * self._cost_scale + self._cost_shift * self._a.sum()).item()


@dataclass
class SinkhornSolver:
    """Full-rank log-domain Sinkhorn solver for entropic optimal transport.

    Solves the entropic optimal transport problem by alternating row and
    column scaling in log domain. The entropic regularization parameter
    ``epsilon`` controls the trade-off between transport cost minimization
    and entropy maximization: higher epsilon gives a smoother, more diffuse
    transport plan (easier to solve but less precise), while lower epsilon
    gives a sharper plan closer to exact OT (but harder to solve numerically).

    The solver converges when the marginal constraint violation falls below
    ``threshold``. Warm-starting with previous dual potentials (f, g) from
    ``SinkhornResult`` typically reduces iterations from ~100 to ~10.

    Attributes:
        epsilon: Entropic regularization strength.
        max_iterations: Maximum number of Sinkhorn iterations.
        threshold: Convergence threshold on marginal error.
            Set <= 0 for fixed-iteration mode (no early stopping).
        check_every: Check convergence every N iterations.
        compile: Whether to use torch.compile for hot paths (requires CUDA).
    """

    epsilon: float = 0.1
    max_iterations: int = 2000
    threshold: float = 1e-6
    check_every: int = 10
    compile: bool = False
    omega: float = 1.0
    anderson_depth: int = 0  # 0 = disabled, >0 = ring buffer depth for Anderson acceleration
    adaptive_omega: bool = False  # False = static omega, True = residual-ratio dynamic omega (Lehmann 2022)
    data_dependent_init: bool = False  # False = zeros init, True = row-softmin init for cold starts

    def __post_init__(self):
        """Initialize compiled function registry and validate parameters."""
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon}. "
                f"A zero or negative epsilon causes division by zero in log-domain Sinkhorn iterations."
            )
        if self.omega < 0.5 or self.omega > 1.95:
            raise ValueError(
                f"omega must be in [0.5, 1.95], got {self.omega}. "
                f"Values < 0.5 cause divergence; values > 1.95 are numerically unstable. "
                f"Recommended range: [1.0, 1.8] for acceleration."
            )
        if self.check_every < 1:
            raise ValueError(
                f"check_every must be >= 1, got {self.check_every}. It is the modulus for periodic convergence checks."
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}. "
                f"Zero iterations leave the duals at their init and return an infeasible plan."
            )

        from .._compiled import CompiledFunctions

        self._compiled = CompiledFunctions(compile=self.compile and torch.cuda.is_available())

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
        init_eps: Optional[float] = None,
    ) -> SinkhornResult:
        """Solve entropic OT problem.

        Solves: min_P <C, P> + eps * sum_ij P_ij (log P_ij - 1)
        s.t.  P 1 = a,  P^T 1 = b,  P >= 0

        The negative-entropy form, which is what ``ent_reg_cost`` reports. Under the
        marginal constraints it has the same argmin as the KL(P || a x b) form, but a
        different value.

        Arguments are :meth:`Solver.solve`'s, plus:

        Args:
            init_eps: Epsilon the warm-start duals were computed under. When it differs
                from ``self.epsilon`` the duals are rescaled by ``self.epsilon/init_eps``,
                holding ``u = f/eps`` fixed. A heuristic, not an exact warm start:
                cost-unit potentials are not homogeneous in epsilon. Measured neutral for
                well-scaled costs and fewer non-converged solves at large cost / small
                epsilon, where un-rescaled duals risk ``exp`` overflow
                (``experiments/scripts/bench_eps_rescale.py``).

        Returns:
            SinkhornResult with dual potentials and transport plan access.
        """
        # Schedules mutate ``self.epsilon`` per step, so re-validate here (a
        # plain float check, no host sync) rather than only at construction.
        validate_positive(
            self.epsilon,
            "epsilon",
            "A zero/negative epsilon divides by zero in log-domain Sinkhorn.",
        )
        return self._solve_full_rank(
            cost_matrix,
            a,
            b,
            init_f,
            init_g,
            scale_cost,
            init_eps,
        )

    def _solve_full_rank(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor],
        b: Optional[torch.Tensor],
        init_f: Optional[torch.Tensor],
        init_g: Optional[torch.Tensor],
        scale_cost: Optional[Union[str, float]],
        init_eps: Optional[float] = None,
    ) -> SinkhornResult:
        """Full-rank log-domain Sinkhorn iterations."""
        # Balanced Sinkhorn needs sum(a) == sum(b). Check whenever the caller gave
        # either marginal: a supplied a=[1,1] against the default unit-mass b is
        # just as infeasible as two mismatched supplied marginals. The default
        # a=b=None is balanced by construction, so the hot path stays sync-free.
        marginals_given = a is not None or b is not None
        cost_matrix, a, cost_shift, cost_scale = prepare_cost(cost_matrix, a, scale_cost, "Sinkhorn")
        n, m = cost_matrix.shape
        device, dtype = cost_matrix.device, cost_matrix.dtype
        b = align_marginal(b, m, device, dtype, "b")
        if marginals_given and not torch.isclose(a.sum(), b.sum(), rtol=1e-3, atol=1e-6):
            warnings.warn(
                f"Sinkhorn marginals have unequal total mass (a.sum()={a.sum().item():.6g}, "
                f"b.sum()={b.sum().item():.6g}); balanced OT is infeasible, so 'converged' "
                f"and the returned plan may be meaningless. Use KLSoftmaxSolver for unbalanced OT.",
                stacklevel=2,
            )

        warn_tiny_temperature(self, float(self.epsilon), cost_matrix, "SinkhornSolver", "epsilon")

        # Log kernel: log_K = -C / eps
        eps = self.epsilon
        log_K = -cost_matrix / eps

        log_a = torch.log(torch.clamp(a, min=1e-30))
        log_b = torch.log(torch.clamp(b, min=1e-30))

        # Data-dependent initialization: set initial duals from cost matrix means
        # Only when no warm-start provided. Applied AFTER cost scaling on the scaled matrix.
        if self.data_dependent_init and init_f is None and init_g is None:
            # Closed form of the first f then g update from zero duals. The optimal
            # potentials satisfy f_i + g_j - C_ij ~ 0 where the plan carries mass, so f
            # tracks a row softmin, which keeps the exponent near zero.
            f = eps * (log_a - torch.logsumexp(log_K, dim=1))
            g = eps * (log_b - torch.logsumexp(f.unsqueeze(1) / eps + log_K, dim=0))
        else:
            # align_dual moves the warm start onto (device, dtype) and clones
            # it; it returns None on a shape mismatch -> fall back to zeros.
            f = align_dual(init_f, n, device, dtype, "init_f")
            g = align_dual(init_g, m, device, dtype, "init_g")
            if f is None:
                f = torch.zeros(n, device=device, dtype=dtype)
            if g is None:
                g = torch.zeros(m, device=device, dtype=dtype)

        # Rescale the warm-started duals when the caller changed epsilon since
        # the previous solve (see ``init_eps`` docstring above). Do this before
        # the clamp below so a large ``eps/init_eps`` ratio stays bounded and
        # cannot blow the plan to Inf/NaN.
        if init_eps is not None and init_eps > 0 and (init_f is not None or init_g is not None):
            if abs(init_eps - eps) / max(eps, 1e-9) > 1e-6:
                scale_factor = eps / init_eps
                f = f * scale_factor
                g = g * scale_factor

        # Validate warm-started dual potentials. Dual potentials scale with the
        # cost matrix magnitude, not epsilon. Kept on-device so the clamp does not
        # sync every solve. Runs after the rescale above so the bound holds. Distinct
        # from ``cost_scale``, which stays the cost divisor and undoes the frame in
        # ``ent_reg_cost`` below.
        clamp_scale = cost_matrix.abs().max().clamp(min=1e-6)
        max_abs_dual = 10.0 * clamp_scale
        # Device-side finite guard (no host sync): reset a non-finite warm start
        # to zero, else clamp to the cost-scaled bound. finite is a 0-dim mask,
        # avoiding an .all()-in-a-Python-if that would sync every solve.
        finite = torch.isfinite(f).all() & torch.isfinite(g).all()
        f = torch.where(finite, f.clamp(-max_abs_dual, max_abs_dual), torch.zeros_like(f))
        g = torch.where(finite, g.clamp(-max_abs_dual, max_abs_dual), torch.zeros_like(g))

        # Gauge f -> f + c, g -> g - c leaves f_i + g_j and the plan unchanged. The
        # half-difference balances their magnitudes so the warm-start clamp holds over a
        # long schedule. Independent mean subtraction is NOT a gauge and perturbs the
        # iterate under overrelaxation.
        c = 0.5 * (g.mean() - f.mean())
        f = f + c
        g = g - c

        fixed_mode = self.threshold <= 0

        if fixed_mode:
            if self.anderson_depth > 0:
                warnings.warn(
                    "anderson_depth > 0 has no effect in fixed-iteration mode "
                    "(threshold <= 0). Anderson acceleration is only supported "
                    "in convergence-checking mode.",
                    stacklevel=2,
                )
            if self.adaptive_omega:
                warnings.warn(
                    "adaptive_omega=True has no effect in fixed-iteration mode "
                    "(threshold <= 0). Adaptive omega is only supported "
                    "in convergence-checking mode.",
                    stacklevel=2,
                )

        converged = False
        n_iters = 0
        errors: List[float] = []

        omega = self.omega

        # Pin the iteration loop inside an autocast-disabled FP32
        # region so a caller running under ``autocast(bfloat16)`` can't
        # demote our log-sum-exp intermediates.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            if fixed_mode:
                # Compiled body, NaN check after the loop: a warm-started solve on a
                # well-conditioned cost rarely diverges mid-iteration, and a periodic
                # isfinite() would sync every check.
                sinkhorn_iter = self._compiled.sinkhorn_iter
                for i in range(self.max_iterations):
                    f, g = sinkhorn_iter(f, g, log_K, log_a, log_b, eps, omega)
                    n_iters = i + 1
                # Fixed mode runs a set iteration count with no residual check, so
                # ``converged`` reports numerical validity only (see SolverResult).
                # Reporting False on a finite result would make ProgressiveEpsilon
                # inflate epsilon after every solve.
                if not (torch.isfinite(f).all() and torch.isfinite(g).all()):
                    f.zero_()
                    g.zero_()
                    _warn_dual_reset()
                else:
                    converged = True
            else:
                # Convergence-checking path: stays fully eager with overrelaxation
                # Anderson acceleration: ring buffer for iterate mixing
                if self.anderson_depth > 0:
                    aa_history_x = []  # list of (f, g) pairs
                    aa_history_r = []  # list of (r_f, r_g) residual pairs

                # Adaptive omega: residual-ratio estimator state (Lehmann 2022)
                if self.adaptive_omega:
                    prev_err = None

                # Divergence detector for static omega. Track consecutive
                # growths of ``|f|.max + |g|.max``; if the iterate norm
                # keeps growing across ``_divergence_patience`` checks,
                # back omega off to 1.0 (Lehmann 2022's proven-safe value).
                _divergence_prev_norm = float("inf")
                _divergence_growth_count = 0
                _divergence_patience = 3
                # Latch the divergence back-off so the adaptive estimator below
                # cannot re-raise omega in the same check.
                _omega_capped = False

                def dual_objective(fv, gv):
                    # D(f,g) = <f,a> + <g,b> - eps * sum_ij exp((f_i+g_j-C_ij)/eps).
                    # The true Sinkhorn Lyapunov: unlike <f,a>+<g,b> alone, it is
                    # valid off the marginal constraint (sum(P) != 1).
                    log_P = fv.unsqueeze(1) / eps + gv.unsqueeze(0) / eps + log_K
                    mass_p = torch.exp(torch.logsumexp(log_P.reshape(-1), dim=0))
                    return (fv * a).sum() + (gv * b).sum() - eps * mass_p

                for i in range(self.max_iterations):
                    f_target = eps * (log_a - torch.logsumexp(log_K + g.unsqueeze(0) / eps, dim=1))
                    f_new = (1 - omega) * f + omega * f_target
                    g_target = eps * (log_b - torch.logsumexp(log_K + f_new.unsqueeze(1) / eps, dim=0))
                    g_new = (1 - omega) * g + omega * g_target

                    # Anderson acceleration
                    if self.anderson_depth > 0 and (i + 1) % self.check_every == 0:
                        r_f = f_new - f
                        r_g = g_new - g
                        aa_history_x.append((f.clone(), g.clone()))
                        aa_history_r.append((r_f.clone(), r_g.clone()))
                        m_depth = self.anderson_depth
                        if len(aa_history_x) > m_depth + 1:
                            aa_history_x.pop(0)
                            aa_history_r.pop(0)

                        if len(aa_history_r) >= 2:
                            k = len(aa_history_r) - 1
                            delta_r = torch.stack(
                                [
                                    torch.cat(
                                        [
                                            aa_history_r[j + 1][0] - aa_history_r[j][0],
                                            aa_history_r[j + 1][1] - aa_history_r[j][1],
                                        ]
                                    )
                                    for j in range(k)
                                ],
                                dim=1,
                            )  # (n+m, k)
                            current_r = torch.cat([r_f, r_g])  # (n+m,)

                            # Tikhonov-regularized least squares on the augmented
                            # system [dR; sqrt(lambda) I] alpha = [r; 0], which is more
                            # stable than the normal equations. lambda scales with
                            # ||dR||^2 to track conditioning.
                            try:
                                lam = 1e-8 * (delta_r * delta_r).sum().clamp(min=1e-30)
                                aug_A = torch.cat(
                                    [
                                        delta_r,
                                        torch.sqrt(lam) * torch.eye(k, device=delta_r.device, dtype=delta_r.dtype),
                                    ],
                                    dim=0,
                                )
                                aug_b = torch.cat(
                                    [current_r, torch.zeros(k, device=delta_r.device, dtype=delta_r.dtype)]
                                )
                                alpha = torch.linalg.lstsq(aug_A, aug_b.unsqueeze(1)).solution.squeeze(1)  # (k,)

                                # Guard against NaN/Inf and huge alpha from ill-conditioning
                                if torch.isfinite(alpha).all() and alpha.norm() < 1e3:
                                    delta_x = torch.stack(
                                        [
                                            torch.cat(
                                                [
                                                    aa_history_x[j + 1][0] - aa_history_x[j][0],
                                                    aa_history_x[j + 1][1] - aa_history_x[j][1],
                                                ]
                                            )
                                            for j in range(k)
                                        ],
                                        dim=1,
                                    )
                                    # Type-II Anderson: x_AA = G(x_k) - dG @ alpha with
                                    # dG = dX + dR, since G(x) = x + r(x). Subtracting dX
                                    # alone leaves an O(||r_k||) error in every accepted
                                    # step, because alpha was fitted to dR @ alpha ~ r_k.
                                    combined = torch.cat([f_new, g_new]) - (delta_x + delta_r) @ alpha
                                    # Validate combined result before assigning
                                    if torch.isfinite(combined).all():
                                        # Accept only when it beats the plain iterate
                                        # and does not regress below the previous one,
                                        # which stays monotone under overrelaxation
                                        # where the SOR step does not.
                                        f_combined = combined[:n]
                                        g_combined = combined[n:]
                                        lyap_prev = dual_objective(f, g)
                                        lyap_plain = dual_objective(f_new, g_new)
                                        lyap_combined = dual_objective(f_combined, g_combined)
                                        accept = (lyap_combined >= lyap_plain - 1e-6) & (
                                            lyap_combined >= lyap_prev - 1e-6
                                        )
                                        f_new = torch.where(accept, f_combined, f_new)
                                        g_new = torch.where(accept, g_combined, g_new)
                            except RuntimeError:
                                pass  # Fall back to standard iterate on solver failure (expected for ill-conditioned problems)

                    f, g = f_new, g_new
                    n_iters = i + 1

                    # Checked every check_every, with all scalars batched into one
                    # device->host transfer. The last iteration is always checked, or a
                    # clean solve whose budget is not a multiple of check_every reports
                    # unconverged and ProgressiveEpsilon inflates epsilon for it.
                    if (i + 1) % self.check_every == 0 or i == self.max_iterations - 1:
                        # Divergence check
                        # One sync, not four: `or` does not short-circuit on the
                        # all-finite common path, so each .any() cost a device transfer.
                        if not (torch.isfinite(f).all() & torch.isfinite(g).all()):
                            f.zero_()
                            g.zero_()
                            _warn_dual_reset()
                            break

                        log_P_row = f.unsqueeze(1) / eps + log_K + g.unsqueeze(0) / eps
                        marginal_a = torch.exp(torch.logsumexp(log_P_row, dim=1))
                        marginal_b = torch.exp(torch.logsumexp(log_P_row, dim=0))

                        # Batch all scalar measurements into one transfer.
                        err_a_t = torch.max(torch.abs(marginal_a - a))
                        err_b_t = torch.max(torch.abs(marginal_b - b))
                        if omega > 1.5:
                            dual_norm_t = f.abs().max() + g.abs().max()
                        else:
                            dual_norm_t = err_a_t  # placeholder, value unused
                        err_a, err_b, dual_norm_v = torch.stack([err_a_t, err_b_t, dual_norm_t]).tolist()
                        err = max(err_a, err_b)

                        # Lehmann et al. 2022 give omega in (0, 2 - rho); omega <= 1.5
                        # is safe on well-conditioned C, so only monitor above it.
                        # Sustained growth (>5%) three checks running, or this fires on
                        # the benign ripples near the fixed point.
                        if omega > 1.5:
                            if dual_norm_v > _divergence_prev_norm * 1.05:
                                _divergence_growth_count += 1
                                if _divergence_growth_count >= _divergence_patience:
                                    warnings.warn(
                                        f"Sinkhorn divergence detected with "
                                        f"omega={omega:.2f} after "
                                        f"{_divergence_patience} consecutive "
                                        f"growth checks (>5% per check); "
                                        f"backing omega off to 1.0 (safe).",
                                        stacklevel=2,
                                    )
                                    omega = 1.0
                                    _omega_capped = True
                                    _divergence_growth_count = 0
                            else:
                                _divergence_growth_count = 0
                            _divergence_prev_norm = dual_norm_v

                        # Adaptive omega, Lehmann residual-ratio estimator
                        # (arXiv:2012.12562): read the linear rate off successive marginal
                        # errors and pick the optimal overrelaxation. Only while omega is
                        # 1.0, since the formula maps the unrelaxed rate and an already
                        # overrelaxed one ratchets omega to its cap. ``r ** (1/check_every)``
                        # assumes a full cycle, so skip the off-cycle final check.
                        _on_cycle = (i + 1) % self.check_every == 0
                        if self.adaptive_omega and _on_cycle and not _omega_capped and omega == 1.0:
                            # Only estimate while the unrelaxed iteration is actually
                            # contracting. A rising residual is not a convergence rate;
                            # clamping such a ratio to 0.99 would map a diverging
                            # transient to omega ~ 1.94 and latch it there, because this
                            # branch runs once (only while omega == 1.0).
                            if prev_err is not None and prev_err > 1e-12 and 0.0 < err < prev_err:
                                r = err / prev_err
                                omega = 2.0 / (1.0 + math.sqrt(1.0 - r ** (1.0 / self.check_every)))
                                omega = min(max(omega, 1.0), 1.95)
                            prev_err = err

                        errors.append(err)
                        if err < self.threshold:
                            converged = True
                            break

        return SinkhornResult(
            f=f,
            g=g,
            converged=converged,
            n_iters=n_iters,
            errors=errors if errors else None,
            _eps=eps,
            _cost_matrix=cost_matrix,
            _a=a,
            _b=b,
            _cost_scale=cost_scale,
            _cost_shift=cost_shift,
        )
