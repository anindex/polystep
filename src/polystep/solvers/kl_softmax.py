"""KL-penalized one-sided OT solver: alpha = lam / (lam + epsilon) interpolates softmax and Sinkhorn. See Chizat et al., arXiv:1607.05816."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ._shared import (
    align_dual,
    align_marginal,
    exp_plan,
    prepare_cost,
    validate_positive,
    warn_tiny_temperature,
)
from .base import SolverResult


@dataclass
class KLSoftmaxSolver:
    """KL-penalized one-sided entropic OT solver."""

    epsilon: float = 0.1
    lam: float = float("inf")
    max_iterations: int = 2000
    threshold: float = 1e-6
    compile: bool = False

    # KL(P^T 1 || b) from the last solve; None until the first solve.
    last_marginal_violation: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon}. epsilon is the entropic temperature and must be positive."
            )
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}. lam is the KL penalty on the column marginal.")
        if self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

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
        # Re-validate: schedulers rewrite epsilon between solves.
        validate_positive(self.epsilon, "epsilon", "the entropic temperature")
        # lam is mutable too; a finite negative lam would give alpha < 0.
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam!r}.")
        if not math.isfinite(self.lam) and self.lam != float("inf"):
            raise ValueError(f"lam must be a non-negative number or +inf, got {self.lam!r}.")
        C, a, cost_shift, cost_scale = prepare_cost(cost_matrix, a, scale_cost, "KLSoftmaxSolver")
        n, m = C.shape
        device, dtype = C.device, C.dtype
        b = align_marginal(b, m, device, dtype, "b")

        warn_tiny_temperature(self, float(self.epsilon), C, "KLSoftmaxSolver", "epsilon")

        eps = float(self.epsilon)
        alpha = self.alpha

        log_a = a.clamp(min=1e-30).log()
        log_b = b.clamp(min=1e-30).log()

        # Disable outer autocast; a BF16 downcast collapses the dual potentials.
        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            # align_dual rejects a shape mismatch; a raw .to().clone() would let a (1, m) init_g broadcast.
            f = align_dual(init_f, n, device, dtype, "init_f")
            g = align_dual(init_g, m, device, dtype, "init_g")
            if f is None:
                f = torch.zeros(n, device=device, dtype=dtype)
            if g is None:
                g = torch.zeros(m, device=device, dtype=dtype)
            # Drop a non-finite warm start rather than propagate it into the LSE.
            if not (torch.isfinite(f).all() and torch.isfinite(g).all()):
                f = torch.zeros(n, device=device, dtype=dtype)
                g = torch.zeros(m, device=device, dtype=dtype)

            # Softmax limit, closed form in one iteration. The bound is the dtype's smallest normal: below it the /alpha residual becomes 0/0 = NaN.
            if alpha < torch.finfo(dtype).tiny:
                f = eps * (log_a - torch.logsumexp(-C / eps, dim=1))
                g = torch.zeros_like(g)
                converged = True
                n_iters = 1
            else:
                converged = False
                n_iters = self.max_iterations
                # Sync the convergence flag once per check_every to keep the dual updates GPU-resident.
                check_every = max(1, self.max_iterations // 20)
                threshold = float(self.threshold)
                for it in range(self.max_iterations):
                    # f-update: exact row-marginal enforcement.
                    f_new = eps * (log_a - torch.logsumexp((g.unsqueeze(0) - C) / eps, dim=1))
                    # g-update: α-fraction of full-Sinkhorn target.
                    g_target = eps * (log_b - torch.logsumexp((f_new.unsqueeze(1) - C) / eps, dim=0))
                    g_new = alpha * g_target

                    if (it + 1) % check_every == 0 or it == self.max_iterations - 1:
                        # The g-update is damped by alpha, so divide its increment back out to keep the residual comparable at any lam.
                        delta = torch.maximum(
                            (f_new - f).abs().amax(),
                            (g_new - g).abs().amax() / alpha,
                        )
                        # <= so threshold=0 means "converge on an exact fixed point", not "never converge".
                        if delta.item() <= threshold:
                            f, g = f_new, g_new
                            converged = True
                            n_iters = it + 1
                            break
                    f, g = f_new, g_new

                # One more f-update so P1 == a holds, since the loop leaves f one step behind g.
                f = eps * (log_a - torch.logsumexp((g.unsqueeze(0) - C) / eps, dim=1))

            # Clamped exponent, not a zero-fill: zeroing overflowed entries would drop the mass that makes P1 == a.
            P = exp_plan(f, g, C, eps)

            # Undo both frame changes so cost is <C_raw, P> (sum(P) == a.sum()).
            cost = ((C * P).sum() * cost_scale + cost_shift * a.sum()).item()

            # Generalized KL(P^T 1 || b), the divergence this solver actually penalizes. The -q+b mass terms keep it non-negative when sum(q) != sum(b).
            col_marginal = P.sum(dim=0).clamp(min=1e-30)
            b_safe = b.clamp(min=1e-30)
            kl = (col_marginal * (col_marginal.log() - b_safe.log()) - col_marginal + b_safe).sum().item()
            self.last_marginal_violation = float(kl)

        return SolverResult(
            matrix=P,
            cost=cost,
            f=f,
            g=g,
            converged=converged,
            n_iters=n_iters,
            ent_reg_cost=cost,
        )
