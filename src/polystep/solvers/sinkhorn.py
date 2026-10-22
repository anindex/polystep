"""Log-domain Sinkhorn solver for entropic optimal transport."""

import functools
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from .._compiled import CompiledFunctions
from ._shared import (
    align_dual,
    align_marginal,
    exp_plan,
    precision_floor,
    prepare_cost,
    validate_positive,
    warn_tiny_temperature,
)


def _warn_dual_reset() -> None:
    """Warn that the duals went non-finite and were reset."""
    warnings.warn(
        "Sinkhorn duals went non-finite and were reset to zero. The returned plan is "
        "exp(-C/eps) and does not satisfy the marginals; check `converged` before use. "
        "A smaller epsilon or a larger cost scale usually triggers this.",
        stacklevel=3,
    )


@dataclass
class SinkhornResult:
    """Output from a Sinkhorn solve."""

    f: torch.Tensor
    g: torch.Tensor
    converged: bool
    n_iters: int
    errors: Optional[List[float]] = None

    # Internal fields for the lazy .matrix and .ent_reg_cost properties.
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
        """Entropic dual objective, computed lazily and reported in the caller's frame."""
        log_P = (self.f.unsqueeze(1) + self.g.unsqueeze(0) - self._cost_matrix) / self._eps
        plan_mass = torch.exp(torch.logsumexp(log_P.reshape(-1), dim=0))
        dual = (self.f * self._a).sum() + (self.g * self._b).sum() - self._eps * plan_mass
        return (dual * self._cost_scale + self._cost_shift * self._a.sum()).item()


@dataclass
class SinkhornSolver:
    """Log-domain Sinkhorn solver for entropic optimal transport.

    ``threshold`` bounds the relative marginal error, floored at what the dtype can resolve once the error stops falling.
    """

    epsilon: float = 0.1
    max_iterations: int = 2000
    threshold: float = 3e-5
    check_every: int = 10
    compile: bool = False
    omega: float = 1.0
    anderson_depth: int = 0  # 0 = disabled, >0 = ring buffer depth for Anderson acceleration
    adaptive_omega: bool = False  # True = residual-ratio dynamic omega (Lehmann 2022)

    def __post_init__(self):
        """Initialize compiled function registry and validate parameters."""
        validate_positive(self.epsilon, "epsilon")
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

        self._compiled = CompiledFunctions(compile=self.compile)

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SinkhornResult:
        """Solve entropic OT."""
        # Schedules mutate self.epsilon per step, so re-validate here.
        validate_positive(
            self.epsilon,
            "epsilon",
            "A zero/negative epsilon divides by zero in log-domain Sinkhorn.",
        )
        # Default marginals are balanced; validate caller-supplied masses.
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

        eps = self.epsilon
        log_K = -cost_matrix / eps

        log_a = torch.log(torch.clamp(a, min=1e-30))
        log_b = torch.log(torch.clamp(b, min=1e-30))
        inv_a = torch.where(a > 0, 1 / a, 0)
        inv_b = torch.where(b > 0, 1 / b, 0)

        # align_dual returns None on a shape mismatch, so fall back to zeros.
        f = align_dual(init_f, n, device, dtype, "init_f")
        g = align_dual(init_g, m, device, dtype, "init_g")
        warm = f is not None or g is not None
        if f is None:
            f = torch.zeros(n, device=device, dtype=dtype)
        if g is None:
            g = torch.zeros(m, device=device, dtype=dtype)

        # Dual potentials scale with the cost magnitude, not epsilon. Kept on-device to avoid a sync per solve.
        clamp_scale = cost_matrix.abs().max().clamp(min=1e-6)
        max_abs_dual = 10.0 * clamp_scale
        # Reset a non-finite warm start to zero, else clamp to the cost-scaled bound. Kept device-side to avoid a host sync.
        finite = torch.isfinite(f).all() & torch.isfinite(g).all()
        f = torch.where(finite, f.clamp(-max_abs_dual, max_abs_dual), torch.zeros_like(f))
        g = torch.where(finite, g.clamp(-max_abs_dual, max_abs_dual), torch.zeros_like(g))

        # Gauge f -> f + c, g -> g - c leaves the plan unchanged; the half-difference
        # balances magnitudes so the clamp holds. Subtracting each mean independently is
        # NOT a gauge and perturbs the iterate under overrelaxation.
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
        sinkhorn_iter = self._compiled.sinkhorn_iter

        # Disable autocast so a bfloat16 caller cannot demote the log-sum-exp intermediates.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            if fixed_mode:
                # NaN check after the loop; a periodic isfinite() would sync every check.
                for i in range(self.max_iterations):
                    f, g = sinkhorn_iter(f, g, log_K, log_a, log_b, eps, omega)
                    n_iters = i + 1
                # In fixed mode, converged reports numerical validity for ProgressiveEpsilon.
                if not (torch.isfinite(f).all() & torch.isfinite(g).all()):
                    f.zero_()
                    g.zero_()
                    _warn_dual_reset()
                else:
                    converged = True
            else:
                # Convergence-checking path with overrelaxation.
                if self.anderson_depth > 0:
                    aa_history_x = []  # list of (f, g) pairs
                    aa_history_r = []  # list of (r_f, r_g) residual pairs

                # Adaptive omega: residual-ratio estimator state (Lehmann 2022)
                if self.adaptive_omega:
                    prev_err = None

                # Back omega off to 1.0 if the iterate norm keeps growing across checks (Lehmann 2022).
                _divergence_prev_norm = float("inf")
                _divergence_growth_count = 0
                _divergence_patience = 3
                # Latch the back-off so the adaptive estimator cannot re-raise omega in the same check.
                _omega_capped = False
                early = None

                def dual_objective(fv, gv):
                    # True Sinkhorn Lyapunov; valid off the marginal constraint, unlike <f,a>+<g,b> alone.
                    log_P = fv.unsqueeze(1) / eps + gv.unsqueeze(0) / eps + log_K
                    mass_p = torch.exp(torch.logsumexp(log_P.reshape(-1), dim=0))
                    return (fv * a).sum() + (gv * b).sum() - eps * mass_p

                for i in range(self.max_iterations):
                    f_new, g_new = sinkhorn_iter(f, g, log_K, log_a, log_b, eps, omega)

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

                            # Tikhonov-regularized least squares on the augmented system; more stable than the normal equations.
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

                                # Guard against NaN/Inf and huge alpha from ill-conditioning.
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
                                    # Type-II Anderson: x_AA = G(x_k) - dG @ alpha with dG = dX + dR.
                                    combined = torch.cat([f_new, g_new]) - (delta_x + delta_r) @ alpha
                                    if torch.isfinite(combined).all():
                                        # Accept only when it beats the plain iterate and the previous one.
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
                                pass  # Fall back to the standard iterate on solver failure.

                    check_due = (i + 1) % self.check_every == 0 or i == self.max_iterations - 1
                    if warm and i == 1 and omega == 1.0 and not check_due:
                        early = (f, g, torch.linalg.vector_norm(f - f_new, float("inf")))
                    f, g = f_new, g_new
                    n_iters = i + 1

                    # Batch all scalars into one device->host transfer; always check the last iteration or ProgressiveEpsilon inflates epsilon.
                    if check_due:
                        log_P_row = f.unsqueeze(1) / eps + log_K + g.unsqueeze(0) / eps
                        err_a_t = ((torch.exp(torch.logsumexp(log_P_row, dim=1)) - a).abs() * inv_a).amax()
                        if omega != 1.0 or self.anderson_depth > 0 or self.adaptive_omega:
                            err_b_t = ((torch.exp(torch.logsumexp(log_P_row, dim=0)) - b).abs() * inv_b).amax()
                        else:
                            err_b_t = err_a_t

                        # Batch all scalar measurements into one transfer; a dual's max is nonfinite exactly when the dual is.
                        scalars = [err_a_t, err_b_t, f.abs().max(), g.abs().max(), clamp_scale]
                        err_a, err_b, f_max, g_max, c_max, *early_norm = torch.stack(
                            scalars if early is None else [*scalars, early[2]]
                        ).tolist()
                    if early is not None and (
                        check_due or (i == 1 and (not f.is_cuda or torch.cuda.current_stream(f.device).query()))
                    ):
                        step = (early_norm[0] if check_due else early[2].item()) / eps
                        if step < math.log1p(self.threshold):
                            f, g, n_iters, converged = early[0], early[1], 1, True
                            errors.append(math.expm1(step))
                            break
                        early = None
                    if check_due:
                        if not (math.isfinite(f_max) and math.isfinite(g_max)):
                            f.zero_()
                            g.zero_()
                            _warn_dual_reset()
                            break
                        dual_norm_v = f_max + g_max
                        err = max(err_a, err_b)

                        # omega <= 1.5 is safe on well-conditioned C, so only monitor above it; require three checks of sustained growth.
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

                        # Estimate the linear rate from residual ratios at omega=1 (arXiv:2012.12562).
                        _on_cycle = (i + 1) % self.check_every == 0
                        if self.adaptive_omega and _on_cycle and not _omega_capped and omega == 1.0:
                            # Only estimate while the unrelaxed iteration is contracting.
                            if prev_err is not None and prev_err > 1e-12 and 0.0 < err < prev_err:
                                r = err / prev_err
                                omega = 2.0 / (1.0 + math.sqrt(1.0 - r ** (1.0 / self.check_every)))
                                omega = min(max(omega, 1.0), 1.95)
                            prev_err = err

                        errors.append(err)
                        stalled = len(errors) > 1 and errors[-2] <= err
                        if err < self.threshold or (
                            stalled and err < precision_floor(dtype, eps, f_max + g_max, c_max)
                        ):
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
