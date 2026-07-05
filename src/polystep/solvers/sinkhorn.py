"""Log-domain Sinkhorn solver for entropic optimal transport.

PolyStep's OT problems are (n particles x m=V polytope vertices) with V small
(2*particle_dim), so the cost is O(n*V) -- a full-rank log-domain solve is
already optimal and there is no large dense kernel to approximate away.
"""
import functools
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from ..costs import scale_cost_matrix
from ._prelude import align_dual, align_marginal, sanitize_cost, validate_positive


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
        ent_reg_cost: Entropic dual objective <f, a> + <g, b> - eps*sum(a),
            which equals the regularized transport cost at convergence.
        errors: Per-check marginal errors (for diagnostics).
    """

    f: torch.Tensor
    g: torch.Tensor
    converged: bool
    n_iters: int
    ent_reg_cost: float
    errors: Optional[List[float]] = None

    # Internal fields for lazy .matrix computation
    _eps: float = float('nan')
    _cost_matrix: Optional[torch.Tensor] = None  # (n, m)

    @functools.cached_property
    def matrix(self) -> torch.Tensor:
        """Transport plan, computed lazily: P_ij = exp((f_i + g_j - C_ij) / eps)."""
        log_P = (
            self.f.unsqueeze(1) / self._eps
            + self.g.unsqueeze(0) / self._eps
            - self._cost_matrix / self._eps
        )
        return torch.exp(log_P)


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
    anderson_depth: int = 0         # 0 = disabled, >0 = ring buffer depth for Anderson acceleration
    adaptive_omega: bool = False    # False = static omega, True = residual-ratio dynamic omega (Lehmann 2022)
    data_dependent_init: bool = False  # False = zeros init, True = cost-mean init for cold starts

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
                f"check_every must be >= 1, got {self.check_every}. "
                f"It is the modulus for periodic convergence checks."
            )

        from .._compiled import CompiledFunctions

        self._compiled = CompiledFunctions(
            compile=self.compile and torch.cuda.is_available()
        )

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

        Solves: min_P <C, P> + eps * KL(P || a x b)
        s.t.  P 1 = a,  P^T 1 = b,  P >= 0

        Args:
            cost_matrix: Cost matrix C of shape (n, m).
            a: Source marginal of shape (n,). Defaults to uniform 1/n.
            b: Target marginal of shape (m,). Defaults to uniform 1/m.
            init_f: Warm-start first dual potential of shape (n,).
            init_g: Warm-start second dual potential of shape (m,).
            scale_cost: Cost scaling: 'mean', 'max_cost', float, or None.
            init_eps: Epsilon under which ``init_f`` / ``init_g`` were
                computed by a previous solve. When provided and different
                from ``self.epsilon``, the duals are rescaled by
                ``self.epsilon / init_eps`` (holding the log-domain scaling
                ``u = f / eps`` fixed). This is a numerical *heuristic*, not
                an exact warm start -- entropic cost-unit potentials are not
                homogeneous in epsilon. Benchmarks
                (``experiments/scripts/bench_eps_rescale.py``) show it is
                ~neutral for well-scaled costs and modestly reduces
                non-converged solves in the large-cost / small-epsilon
                regime, where un-rescaled duals risk ``exp`` overflow.

        Returns:
            SinkhornResult with dual potentials and transport plan access.
        """
        # Schedules mutate ``self.epsilon`` per step, so re-validate here (a
        # plain float check, no host sync) rather than only at construction.
        validate_positive(
            self.epsilon, "epsilon",
            "A zero/negative epsilon divides by zero in log-domain Sinkhorn.",
        )
        return self._solve_full_rank(
            cost_matrix, a, b, init_f, init_g, scale_cost, init_eps,
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
        # Shared prelude: FP32 promotion (BF16's 7 mantissa bits collapse the
        # log-sum-exp row-max trick past ~15 nats) + device-side non-finite
        # cost handling (no host sync), then device/dtype-aligned marginals.
        cost_matrix = sanitize_cost(cost_matrix)
        n, m = cost_matrix.shape
        device, dtype = cost_matrix.device, cost_matrix.dtype
        a = align_marginal(a, n, device, dtype, "a")
        b = align_marginal(b, m, device, dtype, "b")

        # Scale cost matrix (division creates a new tensor; no clone needed)
        cost_matrix = scale_cost_matrix(cost_matrix, scale_cost)

        # Log kernel: log_K = -C / eps
        eps = self.epsilon
        log_K = -cost_matrix / eps

        log_a = torch.log(torch.clamp(a, min=1e-30))
        log_b = torch.log(torch.clamp(b, min=1e-30))

        # Initialize dual potentials with warm-start validation
        # Data-dependent initialization: set initial duals from cost matrix means
        # Only when no warm-start provided. Applied AFTER cost scaling on the scaled matrix.
        if self.data_dependent_init and init_f is None and init_g is None:
            f = -cost_matrix.mean(dim=1)
            g = -cost_matrix.mean(dim=0)
        else:
            # align_dual moves the warm start onto (device, dtype) and clones
            # it; it returns None on a shape mismatch -> fall back to zeros.
            f = align_dual(init_f, n, device, dtype, "init_f")
            g = align_dual(init_g, m, device, dtype, "init_g")
            if f is None:
                f = torch.zeros(n, device=device, dtype=dtype)
            if g is None:
                g = torch.zeros(m, device=device, dtype=dtype)

        # Validate warm-started dual potentials. Dual potentials scale
        # with the cost matrix magnitude, not epsilon. ``cost_scale`` is
        # kept on-device so the clamp does not sync every solve.
        cost_scale = cost_matrix.abs().max().clamp(min=1e-6)
        max_abs_dual = 10.0 * cost_scale
        if not (torch.isfinite(f).all() and torch.isfinite(g).all()):
            f.zero_()
            g.zero_()
        else:
            f.clamp_(-max_abs_dual, max_abs_dual)
            g.clamp_(-max_abs_dual, max_abs_dual)

        # Rescale the warm-started duals when the caller changed epsilon
        # since the previous solve (see ``init_eps`` docstring above).
        if (init_eps is not None and init_eps > 0
                and (init_f is not None or init_g is not None)):
            if abs(init_eps - eps) / max(eps, 1e-9) > 1e-6:
                scale_factor = eps / init_eps
                f = f * scale_factor
                g = g * scale_factor

        # Re-center using the only valid dual gauge ``f -> f + c, g -> g - c``,
        # which leaves ``f_i + g_j`` (and hence the plan ``P``) unchanged. The
        # half-difference shift balances the two potentials' magnitudes so the
        # warm-start clamp above keeps them bounded across a long schedule.
        # NB: independent mean subtraction (``f -= f.mean(); g -= g.mean()``)
        # is NOT a valid gauge and perturbs the iterate under overrelaxation.
        c = 0.5 * (g.mean() - f.mean())
        f = f + c
        g = g - c

        # Determine if we should check convergence
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
        with torch.no_grad(), \
                torch.amp.autocast("cuda", enabled=False), \
                torch.amp.autocast("cpu", enabled=False):
            if fixed_mode:
                # Fixed-iteration path: compiled body with post-loop NaN check.
                # Warm-started solver with well-conditioned cost matrix rarely
                # diverges mid-iteration; checking only at the end avoids
                # GPU-CPU sync overhead from periodic isfinite() reductions.
                sinkhorn_iter = self._compiled.sinkhorn_iter
                for i in range(self.max_iterations):
                    f, g = sinkhorn_iter(f, g, log_K, log_a, log_b, eps, omega)
                    n_iters = i + 1
                # Post-loop NaN check: zero duals if solver diverged
                if not (torch.isfinite(f).all() and torch.isfinite(g).all()):
                    f.zero_()
                    g.zero_()
            else:
                # Convergence-checking path: stays fully eager with overrelaxation
                # Anderson acceleration: ring buffer for iterate mixing
                if self.anderson_depth > 0:
                    aa_history_x = []   # list of (f, g) pairs
                    aa_history_r = []   # list of (r_f, r_g) residual pairs

                # Adaptive omega: residual-ratio estimator state (Lehmann 2022)
                if self.adaptive_omega:
                    prev_err = None

                # Divergence detector for static omega. Track consecutive
                # growths of ``|f|.max + |g|.max``; if the iterate norm
                # keeps growing across ``_divergence_patience`` checks,
                # back omega off to 1.0 (Lehmann 2022's proven-safe value).
                _divergence_prev_norm = float('inf')
                _divergence_growth_count = 0
                _divergence_patience = 3

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
                            # Build residual difference matrix
                            delta_r = torch.stack([
                                torch.cat([aa_history_r[j+1][0] - aa_history_r[j][0],
                                           aa_history_r[j+1][1] - aa_history_r[j][1]])
                                for j in range(k)
                            ], dim=1)  # (n+m, k)
                            current_r = torch.cat([r_f, r_g])  # (n+m,)

                            # Tikhonov-regularized least-squares for stability
                            try:
                                alpha, _, _, _ = torch.linalg.lstsq(delta_r, current_r.unsqueeze(1))
                                alpha = alpha.squeeze(1)  # (k,)

                                # Guard against NaN/Inf and huge alpha from ill-conditioning
                                if torch.isfinite(alpha).all() and alpha.norm() < 1e3:
                                    delta_x = torch.stack([
                                        torch.cat([aa_history_x[j+1][0] - aa_history_x[j][0],
                                                   aa_history_x[j+1][1] - aa_history_x[j][1]])
                                        for j in range(k)
                                    ], dim=1)
                                    combined = torch.cat([f_new, g_new]) - delta_x @ alpha
                                    # Validate combined result before assigning
                                    if torch.isfinite(combined).all():
                                        # Lyapunov regression check (Chizat 2020):
                                        # the entropic dual objective increases
                                        # monotonically in plain Sinkhorn, so only
                                        # accept the Anderson step when it does not
                                        # regress vs the plain iterate. Without this
                                        # guard acceleration can push the iterate to
                                        # a worse Lyapunov on ill-conditioned C.
                                        f_combined = combined[:n]
                                        g_combined = combined[n:]
                                        lyap_plain = dual_objective(f_new, g_new)
                                        lyap_combined = dual_objective(f_combined, g_combined)
                                        # Device-side accept gate avoids a
                                        # GPU->CPU sync per Anderson iter; the
                                        # math is unchanged (broadcast scalar
                                        # bool selects between the two iterates).
                                        accept = (lyap_combined >= lyap_plain - 1e-6)
                                        f_new = torch.where(accept, f_combined, f_new)
                                        g_new = torch.where(accept, g_combined, g_new)
                            except RuntimeError:
                                pass  # Fall back to standard iterate on solver failure (expected for ill-conditioned problems)

                    f, g = f_new, g_new
                    n_iters = i + 1

                    # Convergence and divergence check (every check_every iterations
                    # to avoid GPU-CPU sync on every iteration). All scalar
                    # measurements are batched into a single device->host
                    # transfer per check to amortize the sync cost.
                    if (i + 1) % self.check_every == 0:
                        # Divergence check
                        if (torch.isnan(f).any() or torch.isinf(f).any()
                                or torch.isnan(g).any() or torch.isinf(g).any()):
                            f.zero_()
                            g.zero_()
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
                        err_a, err_b, dual_norm_v = torch.stack(
                            [err_a_t, err_b_t, dual_norm_t]
                        ).tolist()
                        err = max(err_a, err_b)

                        # Static-omega divergence detector. Lehmann et al.
                        # 2022 give a safe range ``omega in (0, 2 - rho)``
                        # where rho is the linearised spectral radius.
                        # Empirically ``omega <= 1.5`` is safe on
                        # well-conditioned C; only monitor at the aggressive
                        # end. Require both sustained growth (>5% per check)
                        # and three-in-a-row to avoid firing on the benign
                        # iterate ripples that occur near the fixed point.
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
                                    _divergence_growth_count = 0
                            else:
                                _divergence_growth_count = 0
                            _divergence_prev_norm = dual_norm_v

                        # Adaptive omega via the Lehmann residual-ratio estimator
                        # (arXiv:2012.12562): estimate the linear rate from the
                        # ratio of successive marginal errors and pick the optimal
                        # overrelaxation. Converges to omega_opt instead of
                        # oscillating like a fixed step-up/step-down heuristic.
                        if self.adaptive_omega:
                            if prev_err is not None and prev_err > 1e-12 and err > 0.0:
                                r = min(err / prev_err, 0.99)
                                omega = 2.0 / (1.0 + math.sqrt(1.0 - r ** (1.0 / self.check_every)))
                                omega = min(max(omega, 1.0), 1.95)
                            prev_err = err

                        errors.append(err)
                        if err < self.threshold:
                            converged = True
                            break

        # Entropic dual objective D(f,g) = <f,a> + <g,b> - eps*sum(a). The mass
        # term makes it the true regularized value (a bare <f,a>+<g,b> overstates
        # it by eps*sum(a)); at convergence sum(P) equals sum(a). One host transfer.
        ent_reg_cost = (
            torch.stack([(f * a).sum(), (g * b).sum()]).sum() - eps * a.sum()
        ).item()

        result = SinkhornResult(
            f=f,
            g=g,
            converged=converged,
            n_iters=n_iters,
            ent_reg_cost=ent_reg_cost,
            errors=errors if errors else None,
            _eps=eps,
            _cost_matrix=cost_matrix,
        )
        return result
