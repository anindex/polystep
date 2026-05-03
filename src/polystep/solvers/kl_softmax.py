"""KL-penalized one-sided OT solver (interpolates softmax ↔ Sinkhorn).

Implements the soft-target-marginal formulation:

    min_P  <C, P> + epsilon * H(P) + lam * KL(P^T 1 || b)
    s.t.   P 1 = a

with `lam ∈ [0, ∞]`. The two limits are exact:

- `lam = 0`  ≡ ``SoftmaxSolver`` (only row marginal enforced).
- `lam → ∞` ≡ ``SinkhornSolver`` (both row and column marginals).

Algorithm (log-domain alternating updates):

    α = lam / (lam + epsilon)            ∈ [0, 1]
    f_i = epsilon * (log a_i - LSE_j((g_j - C_ij) / epsilon))   # exact
    g_j = α * epsilon * (log b_j - LSE_i((f_i - C_ij) / epsilon))   # soft

Setting α = 0 freezes g at zero, recovering ``SoftmaxSolver`` (one
iteration suffices). Setting α = 1 (lam = ∞) recovers standard
Sinkhorn alternating projections. Intermediate α produces a smooth
interpolation, with `KL(P^T 1 || b)` decreasing monotonically as α
grows.

The α-scaling is the Chizat-Roussillon-Léger-Vialard (2018) "scaling
algorithm for unbalanced OT" form. We also support `lam = inf`
explicitly (the user-facing default for "go to full Sinkhorn") so
that downstream code can pass `float('inf')` without arithmetic on
infinity.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ..costs import scale_cost_matrix
from .base import SolverResult


@dataclass
class KLSoftmaxSolver:
    """KL-penalized one-sided entropic OT solver.

    Attributes:
        epsilon: Entropic regularization (temperature). Must be > 0.
        lam: KL penalty weight on the column marginal.
            `0` reduces to ``SoftmaxSolver``; `inf` reduces to
            ``SinkhornSolver``. Must be >= 0.
        max_iterations: Maximum dual-update iterations.
        threshold: Convergence tolerance on max(|Δf|, |Δg|).
        compile: Placeholder for API compatibility (unused).
    """

    epsilon: float = 0.1
    lam: float = float("inf")
    max_iterations: int = 2000
    threshold: float = 1e-6
    compile: bool = False

    # KL(P^T 1 || b) recorded on every solve() call; None until the first solve.
    last_marginal_violation: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon}. "
                "epsilon is the entropic temperature and must be positive."
            )
        if self.lam < 0:
            raise ValueError(
                f"lam must be >= 0, got {self.lam}. "
                "lam is the KL penalty on the column marginal."
            )
        if self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )

    @property
    def alpha(self) -> float:
        """The KL-scaling coefficient α = lam / (lam + epsilon) ∈ [0, 1]."""
        if self.lam == 0.0:
            return 0.0
        if math.isinf(self.lam):
            return 1.0
        return float(self.lam / (self.lam + self.epsilon))

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        n, m = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        # Default uniform marginals
        if a is None:
            a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        if b is None:
            b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)

        # Promote BF16/FP16 to FP32 for numerical stability inside LSE
        if cost_matrix.dtype in (torch.bfloat16, torch.float16):
            cost_matrix = cost_matrix.to(torch.float32)
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            dtype = torch.float32

        # Optional cost rescaling
        C = scale_cost_matrix(cost_matrix.clone(), scale_cost)

        # Replace +Inf entries with a large finite penalty (matches SoftmaxSolver)
        if not torch.isfinite(C).all():
            finite_mask = torch.isfinite(C)
            if finite_mask.any():
                penalty = C[finite_mask].abs().max().item() * 2.0 + 1.0
            else:
                penalty = 1e6
            C = torch.where(finite_mask, C, torch.full_like(C, penalty))

        eps = float(self.epsilon)
        alpha = self.alpha

        log_a = a.clamp(min=1e-30).log()
        log_b = b.clamp(min=1e-30).log()

        # Disable any outer mixed-precision autocast inside the iteration -
        # downcast LSE to BF16 collapses the dual potentials.
        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            f = init_f.to(dtype=dtype, device=device).clone() if init_f is not None else torch.zeros(n, device=device, dtype=dtype)
            g = init_g.to(dtype=dtype, device=device).clone() if init_g is not None else torch.zeros(m, device=device, dtype=dtype)

            # Special-case alpha == 0 (softmax limit): single closed-form.
            if alpha == 0.0:
                f = eps * (log_a - torch.logsumexp(-C / eps, dim=1))
                g = torch.zeros_like(g)
                converged = True
                n_iters = 1
            else:
                converged = False
                n_iters = self.max_iterations
                for it in range(self.max_iterations):
                    # f-update: exact row-marginal enforcement
                    f_new = eps * (log_a - torch.logsumexp(
                        (g.unsqueeze(0) - C) / eps, dim=1
                    ))
                    # g-update: α-fraction of full-Sinkhorn target.
                    g_target = eps * (log_b - torch.logsumexp(
                        (f_new.unsqueeze(1) - C) / eps, dim=0
                    ))
                    g_new = alpha * g_target

                    df = (f_new - f).abs().max().item()
                    dg = (g_new - g).abs().max().item()
                    f, g = f_new, g_new
                    if max(df, dg) < self.threshold:
                        converged = True
                        n_iters = it + 1
                        break

            log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
            P = log_P.exp()

            # Numerical hygiene
            if not torch.isfinite(P).all():
                warnings.warn(
                    "KLSoftmaxSolver produced non-finite transport entries; "
                    "consider raising epsilon or lowering lam.",
                    stacklevel=2,
                )
                P = torch.where(torch.isfinite(P), P, torch.zeros_like(P))

            cost = (C * P).sum().item()

            # Theorem 4.1 instrumentation: KL(P^T 1 || b).
            # P^T 1 is the realized column marginal; b is the target.
            # KL = sum_j q_j * log(q_j / b_j) with q = P^T 1.
            try:
                col_marginal = P.sum(dim=0).clamp(min=1e-30)
                b_safe = b.clamp(min=1e-30)
                kl = (col_marginal * (col_marginal.log() - b_safe.log())).sum().item()
                self.last_marginal_violation = float(kl)
            except Exception:  # noqa: BLE001
                self.last_marginal_violation = float("nan")

        return SolverResult(
            matrix=P,
            cost=cost,
            f=f,
            g=g,
            converged=converged,
            n_iters=n_iters,
            ent_reg_cost=cost,
        )
