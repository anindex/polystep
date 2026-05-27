"""Unified Sinkhorn solver for entropic optimal transport.

Implements full-rank log-domain Sinkhorn with optional low-rank cost approximation
via randomized SVD, and auto-selection based on problem size.
"""
import functools
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from ..costs import scale_cost_matrix


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
        ent_reg_cost: Entropic regularized cost = <f, a> + <g, b>.
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
    _cost_matrix: Optional[torch.Tensor] = None  # Full-rank: (n, m)
    _Q: Optional[torch.Tensor] = None             # Low-rank: (n, r)
    _R: Optional[torch.Tensor] = None             # Low-rank: (m, r)
    _g_lr: Optional[torch.Tensor] = None           # Low-rank: (r,)

    @functools.cached_property
    def matrix(self) -> torch.Tensor:
        """Compute the transport matrix lazily.

        Full-rank: P_ij = exp((f_i + g_j - C_ij) / eps)
        Low-rank:  P = Q @ diag(1/g_lr) @ R^T
        """
        if self._Q is not None:
            # Low-rank mode - guard against division by zero
            # Note: SVD truncation may introduce small negative entries (~1e-3).
            # These are negligible and clamping would break marginal constraints.
            g_lr_safe = torch.clamp(self._g_lr, min=1e-10)
            return self._Q @ torch.diag(1.0 / g_lr_safe) @ self._R.T
        else:
            # Full-rank mode
            log_P = (
                self.f.unsqueeze(1) / self._eps
                + self.g.unsqueeze(0) / self._eps
                - self._cost_matrix / self._eps
            )
            return torch.exp(log_P)


@dataclass
class SinkhornSolver:
    """Unified Sinkhorn solver with full-rank and low-rank modes.

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
        rank: None for full-rank (with auto-selection), or int for low-rank.
        gamma: Mirror descent step size inverse for low-rank mode.
        auto_rank_threshold: Total particles (n+m) above which auto-selects low-rank.
        compile: Whether to use torch.compile for hot paths (requires CUDA).
    """

    epsilon: float = 0.1
    max_iterations: int = 2000
    threshold: float = 1e-6
    check_every: int = 10
    rank: Optional[int] = None
    gamma: float = 10.0
    auto_rank_threshold: int = 50_000
    compile: bool = False
    omega: float = 1.0
    anderson_depth: int = 0         # 0 = disabled, >0 = ring buffer depth for Anderson acceleration
    adaptive_omega: bool = False    # False = static omega, True = Lyapunov-based dynamic omega
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
        seed: Optional[int] = None,
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
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.
            init_eps: Epsilon under which ``init_f`` / ``init_g`` were
                computed by a previous solve. When provided and different
                from ``self.epsilon``, the duals are rescaled by
                ``self.epsilon / init_eps``; the cost-units potential
                ``f = eps * u`` (with ``u`` in log-units), so a fixed
                ``u`` corresponds to a rescaled ``f`` of
                ``(eps_new / eps_old) * f_old``. Without this rescale,
                warm-starting across an epsilon schedule inflates the
                iteration count by 5-10x.
            seed: Optional seed for randomized SVD in low-rank mode.
                When None, defaults to 0 for backward compatibility.

        Returns:
            SinkhornResult with dual potentials and transport plan access.
        """
        n, m = cost_matrix.shape

        # Auto-selection: switch to low-rank for large problems
        rank = self.rank
        if rank is None and (n + m) > self.auto_rank_threshold:
            rank = min(n, m) // 2

        if rank is not None:
            if self.anderson_depth > 0:
                warnings.warn(
                    "anderson_depth > 0 has no effect in low-rank mode. "
                    "Anderson acceleration is only supported in full-rank "
                    "convergence-checking mode.",
                    stacklevel=2,
                )
            if self.adaptive_omega:
                warnings.warn(
                    "adaptive_omega=True has no effect in low-rank mode. "
                    "Adaptive omega is only supported in full-rank "
                    "convergence-checking mode.",
                    stacklevel=2,
                )
            if init_f is not None or init_g is not None:
                warnings.warn(
                    "init_f/init_g warm-start potentials are ignored in "
                    "low-rank mode. Low-rank Sinkhorn always starts from "
                    "zeros (or data-dependent init).",
                    stacklevel=2,
                )
            return self._solve_low_rank(
                cost_matrix, a, b, rank, scale_cost, seed=seed,
            )
        else:
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
        # log-sum-exp needs FP32 mantissa precision: BF16's 7 mantissa
        # bits collapse the row-max-subtract trick once the cost spread
        # exceeds ~15 nats. Promote half-precision inputs and run the
        # iteration inside an autocast-disabled region (see below) so
        # an outer mixed-precision context can't undo the promotion.
        if cost_matrix.dtype in (torch.bfloat16, torch.float16):
            cost_matrix = cost_matrix.to(torch.float32)
        n, m = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        # Default uniform marginals
        if a is None:
            a = torch.ones(n, device=device, dtype=dtype) / n
        if b is None:
            b = torch.ones(m, device=device, dtype=dtype) / m

        # Validate cost matrix is finite (NaN/Inf in cost propagates silently
        # through log_K and all iterations; catching it here gives a clear error).
        if not torch.isfinite(cost_matrix).all():
            n_bad = (~torch.isfinite(cost_matrix)).sum().item()
            warnings.warn(
                f"Cost matrix has {n_bad} non-finite entries. "
                "Replacing with max finite value + 1 penalty.",
                stacklevel=2,
            )
            finite_mask = torch.isfinite(cost_matrix)
            if finite_mask.any():
                penalty = cost_matrix[finite_mask].abs().max().item() * 2.0 + 1.0
            else:
                penalty = 1e6
            cost_matrix = torch.where(finite_mask, cost_matrix,
                                      torch.full_like(cost_matrix, penalty))

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
            f = torch.zeros(n, device=device, dtype=dtype)
            g = torch.zeros(m, device=device, dtype=dtype)
            if init_f is not None:
                if init_f.shape == (n,):
                    f = init_f.clone()
                else:
                    warnings.warn(
                        f"warm-start init_f shape mismatch: expected ({n},), "
                        f"got {tuple(init_f.shape)}. Falling back to zeros.",
                        stacklevel=2,
                    )
            if init_g is not None:
                if init_g.shape == (m,):
                    g = init_g.clone()
                else:
                    warnings.warn(
                        f"warm-start init_g shape mismatch: expected ({m},), "
                        f"got {tuple(init_g.shape)}. Falling back to zeros.",
                        stacklevel=2,
                    )

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

        # Re-center so ``|f|, |g|`` stay bounded under repeated solves
        # where the cost matrix mean drifts. The entropic OT problem is
        # invariant to ``f -> f + c, g -> g - c``; without this drift,
        # warm-started duals can grow unboundedly across a long schedule.
        f = f - f.mean()
        g = g - g.mean()

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

                # Adaptive omega: Lyapunov monitoring for dynamic overrelaxation
                if self.adaptive_omega:
                    prev_lyapunov = float('-inf')

                # Divergence detector for static omega. Track consecutive
                # growths of ``|f|.max + |g|.max``; if the iterate norm
                # keeps growing across ``_divergence_patience`` checks,
                # back omega off to 1.0 (Lehmann 2022's proven-safe value).
                _divergence_prev_norm = float('inf')
                _divergence_growth_count = 0
                _divergence_patience = 3

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
                                        # the dual objective <f,a> + <g,b>
                                        # increases monotonically in plain
                                        # Sinkhorn, so only accept the Anderson
                                        # step when it does not regress vs the
                                        # plain iterate. Without this guard
                                        # acceleration can push the iterate to
                                        # a worse Lyapunov on ill-conditioned C.
                                        f_combined = combined[:n]
                                        g_combined = combined[n:]
                                        lyap_plain = (f_new * a).sum() + (g_new * b).sum()
                                        lyap_combined = (f_combined * a).sum() + (g_combined * b).sum()
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
                        if self.adaptive_omega:
                            lyap_t = (f * a).sum() + (g * b).sum()
                        else:
                            lyap_t = err_a_t  # placeholder, value unused
                        err_a, err_b, dual_norm_v, lyap_v = torch.stack(
                            [err_a_t, err_b_t, dual_norm_t, lyap_t]
                        ).tolist()

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

                        # Adaptive omega: adjust based on Lyapunov function
                        if self.adaptive_omega:
                            if lyap_v >= prev_lyapunov:
                                omega = min(omega * 1.05, 1.8)  # Good progress, cautiously increase
                            else:
                                omega = max(omega * 0.8, 1.0)   # Overshot, back off toward standard
                            prev_lyapunov = lyap_v

                        err = max(err_a, err_b)
                        errors.append(err)
                        if err < self.threshold:
                            converged = True
                            break

        # Entropic regularized cost: <f, a> + <g, b>. One host transfer.
        ent_reg_cost = torch.stack([(f * a).sum(), (g * b).sum()]).sum().item()

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

    def _solve_low_rank(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor],
        b: Optional[torch.Tensor],
        rank: int,
        scale_cost: Optional[Union[str, float]],
        seed: Optional[int] = None,
    ) -> SinkhornResult:
        """Low-rank Sinkhorn solver for large-scale OT problems.

        Approximates the cost matrix via randomized SVD to rank r, then runs
        standard log-domain Sinkhorn on the approximation. The transport plan
        is computed from dual potentials via P_ij = exp((f_i + g_j - C_ij) / eps).

        Note: The cost matrix approximation reduces rank, but the Sinkhorn
        iterations still materialize full O(nm) log-kernel matrices. A warning
        is emitted when the estimated memory exceeds 2GB.

        Args:
            seed: Optional seed for the randomized SVD. When None, defaults
                to 0 for backward compatibility.
        """
        n, m = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype
        eps = self.epsilon

        rank = min(rank, n, m)

        # Warn if materializing full O(nm) matrices would use excessive memory
        estimated_bytes = n * m * 4 * 4  # 4 copies (log_K, log_P, etc.), float32
        if estimated_bytes > 2e9:  # 2GB threshold
            warnings.warn(
                f"Low-rank Sinkhorn will materialize ~{estimated_bytes / 1e9:.1f}GB "
                f"of dense matrices for n={n}, m={m}. Consider reducing problem size.",
                stacklevel=2,
            )

        if a is None:
            a = torch.ones(n, device=device, dtype=dtype) / n
        if b is None:
            b = torch.ones(m, device=device, dtype=dtype) / m

        cost_matrix = scale_cost_matrix(cost_matrix, scale_cost)
        fixed_mode = self.threshold <= 0
        omega = self.omega

        # Randomized SVD approximation of cost matrix: C ~ L @ M^T
        gen = torch.Generator(device=device)
        gen.manual_seed(seed if seed is not None else 0)
        Omega = torch.randn(m, rank, device=device, dtype=dtype, generator=gen)
        Y = cost_matrix @ Omega                                  # (n, rank)
        Q_orth, _ = torch.linalg.qr(Y)                          # (n, rank)
        B_proj = Q_orth.T @ cost_matrix                          # (rank, m)
        U_s, S_s, Vt_s = torch.linalg.svd(B_proj, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S_s[:rank], min=1e-30))
        L = Q_orth @ U_s[:, :rank] * sqrt_S.unsqueeze(0)        # (n, rank)
        M = Vt_s[:rank].T * sqrt_S.unsqueeze(0)                 # (m, rank)
        cost_approx = L @ M.T                                    # (n, m)

        # Standard log-domain Sinkhorn on the approximated cost
        log_K = -cost_approx / eps
        log_a = torch.log(torch.clamp(a, min=1e-30))
        log_b = torch.log(torch.clamp(b, min=1e-30))

        # Data-dependent initialization: low-rank path has no warm-start
        if self.data_dependent_init:
            f = -cost_approx.mean(dim=1)
            g = -cost_approx.mean(dim=0)
        else:
            f = torch.zeros(n, device=device, dtype=dtype)
            g = torch.zeros(m, device=device, dtype=dtype)

        converged = False
        n_iters = 0
        errors: List[float] = []

        with torch.no_grad(), \
                torch.amp.autocast("cuda", enabled=False), \
                torch.amp.autocast("cpu", enabled=False):
            if fixed_mode:
                # Fixed-iteration path: compiled body with post-loop NaN check
                sinkhorn_iter = self._compiled.sinkhorn_iter
                for i in range(self.max_iterations):
                    f, g = sinkhorn_iter(f, g, log_K, log_a, log_b, eps, omega)
                    n_iters = i + 1
                if not (torch.isfinite(f).all() and torch.isfinite(g).all()):
                    f.zero_()
                    g.zero_()
            else:
                # Convergence-checking path: stays fully eager with overrelaxation
                for i in range(self.max_iterations):
                    f_target = eps * (log_a - torch.logsumexp(log_K + g.unsqueeze(0) / eps, dim=1))
                    f = (1 - omega) * f + omega * f_target
                    g_target = eps * (log_b - torch.logsumexp(log_K + f.unsqueeze(1) / eps, dim=0))
                    g = (1 - omega) * g + omega * g_target

                    n_iters = i + 1

                    if (i + 1) % self.check_every == 0:
                        # NaN/Inf check (batched with convergence check to minimize GPU syncs)
                        if (torch.isnan(f).any() or torch.isinf(f).any()
                                or torch.isnan(g).any() or torch.isinf(g).any()):
                            f.zero_()
                            g.zero_()
                            break

                        log_P_row = f.unsqueeze(1) / eps + log_K + g.unsqueeze(0) / eps
                        marginal_a_hat = torch.exp(torch.logsumexp(log_P_row, dim=1))
                        marginal_b_hat = torch.exp(torch.logsumexp(log_P_row, dim=0))
                        # One host transfer for both marginal errors.
                        err_a, err_b = torch.stack([
                            torch.max(torch.abs(marginal_a_hat - a)),
                            torch.max(torch.abs(marginal_b_hat - b)),
                        ]).tolist()
                        err = max(err_a, err_b)
                        errors.append(err)
                        if err < self.threshold:
                            converged = True
                            break

        # Entropic regularized cost: <f, a> + <g, b>. One host transfer.
        ent_reg_cost = torch.stack([(f * a).sum(), (g * b).sum()]).sum().item()

        # Store transport plan via dual potentials + approximated cost matrix.
        # The .matrix property computes P_ij = exp((f_i + g_j - C_ij) / eps)
        # directly, avoiding an extra O(nm) SVD that would not improve accuracy
        # (since g_lr = ones makes Q @ diag(1/g_lr) @ R^T = Q @ R^T anyway).
        return SinkhornResult(
            f=f,
            g=g,
            converged=converged,
            n_iters=n_iters,
            ent_reg_cost=ent_reg_cost,
            errors=errors if errors else None,
            _eps=eps,
            _cost_matrix=cost_approx,
        )
