"""SoftmaxSolver: direct softmax weighting for subspace modes.

Replaces iterative Sinkhorn OT with a single softmax(-C/epsilon) pass.
Mathematically equivalent when the OT target marginal constraint is
naturally satisfied (few particles in subspace mode). See paper Section 5.10.

Key properties:
    - Row sums of transport matrix equal source marginal a
    - No dual potentials (f, g are None)
    - Single iteration (converged=True, n_iters=1)
    - Numerical stability via PyTorch's built-in softmax (subtracts row-max)
"""
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch

from ..costs import scale_cost_matrix
from .base import SolverResult


# Below this ratio of `epsilon` to `max|C|`, ``-C/epsilon`` overflows
# before ``torch.softmax`` gets a chance to subtract the row max.
# Tested empirically at FP32 / BF16: ratios above ~1e-6 stay finite.
_TINY_EPSILON_RATIO = 1e-6

SoftmaxResult = SolverResult


@dataclass
class SoftmaxSolver:
    """Direct softmax weighting solver for subspace modes.

    Computes transport weights via ``softmax(-C / epsilon)`` and scales
    by the source marginal to produce a transport matrix whose row sums
    equal ``a``. This is equivalent to entropic OT when the target marginal
    constraint is naturally satisfied (few particles, subspace mode).

    Attributes:
        epsilon: Temperature parameter (entropic regularization strength).
            Controls sharpness of the softmax: lower epsilon gives sharper
            (more selective) weights.
        compile: Placeholder for API compatibility with SinkhornSolver.
            Currently unused since softmax is a single torch op.
    """

    epsilon: float = 0.1
    compile: bool = False

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        """Compute softmax weights from cost matrix.

        Args:
            cost_matrix: Cost matrix C of shape (P, V).
            a: Source marginal of shape (P,). Defaults to uniform 1/P.
            b: Target marginal (accepted but ignored for softmax).
            init_f: Warm-start dual potential (accepted but ignored).
            init_g: Warm-start dual potential (accepted but ignored).
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.

        Returns:
            SolverResult with transport matrix, cost, and metadata.

        Raises:
            ValueError: If epsilon <= 0.
        """
        if self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be > 0, got {self.epsilon}. "
                f"Epsilon is the temperature parameter in softmax(-C/epsilon); "
                f"zero or negative values cause division by zero or undefined behavior."
            )

        P, V = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        # Default uniform source marginal
        if a is None:
            a = torch.ones(P, device=device, dtype=dtype) / P

        # SoftmaxSolver is one-sided: it only enforces row sums equal `a`.
        # If a caller passes a non-uniform `b` they probably meant to call
        # SinkhornSolver; warn loudly so the constraint isn't silently dropped.
        if b is not None:
            uniform = torch.full_like(b, 1.0 / V)
            if not torch.allclose(b.to(uniform.dtype), uniform, atol=1e-6):
                warnings.warn(
                    "SoftmaxSolver ignores the target marginal `b`: it only "
                    "enforces row sums equal to the source marginal `a`. "
                    "Pass solver='sinkhorn' for two-sided OT.",
                    stacklevel=2,
                )

        # Promote half-precision inputs to FP32. PyTorch's softmax
        # subtracts the row max for stability, but the subtraction happens
        # AFTER an outer autocast may have already downcast ``-C/epsilon``
        # to BF16 (~7 mantissa bits), which collapses the transport plan
        # to near-uniform once the cost spread exceeds ~15 nats.
        if cost_matrix.dtype == torch.bfloat16 or cost_matrix.dtype == torch.float16:
            cost_matrix = cost_matrix.to(torch.float32)
            dtype = torch.float32

        # +Inf is a useful way to model a hard constraint ("never pick this
        # vertex"). Replace it with a large finite penalty so the masked
        # entry still gets near-zero weight without sending the whole row
        # to -Inf inside ``-C/epsilon`` (which would NaN the softmax).
        if not torch.isfinite(cost_matrix).all():
            finite_mask = torch.isfinite(cost_matrix)
            if finite_mask.any():
                penalty = cost_matrix[finite_mask].abs().max().item() * 2.0 + 1.0
            else:
                penalty = 1e6
            cost_matrix = torch.where(
                finite_mask, cost_matrix,
                torch.full_like(cost_matrix, penalty),
            )

        C = scale_cost_matrix(cost_matrix.clone(), scale_cost)

        # ``epsilon > 0`` is enough to avoid division-by-zero, but
        # eps=1e-30 with cost_max~10 still overflows -C/epsilon before
        # softmax can subtract the row max. Warn at extreme ratios.
        cost_max = C.detach().abs().max().item() if C.numel() > 0 else 0.0
        if cost_max > 0 and self.epsilon < _TINY_EPSILON_RATIO * cost_max:
            warnings.warn(
                f"SoftmaxSolver epsilon={self.epsilon:.2e} is very small "
                f"relative to the cost-matrix scale (max |C|={cost_max:.2e}); "
                f"-C/epsilon may underflow / overflow before the row-max "
                f"subtraction inside torch.softmax. Consider rescaling the "
                f"cost or raising epsilon.",
                stacklevel=2,
            )

        # PyTorch's softmax subtracts the row max internally for stability.
        # Pin the whole block inside an autocast-disabled context so an outer
        # mixed-precision region can't downcast intermediates back to BF16.
        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            W = torch.softmax(-C / self.epsilon, dim=-1)

            # Row sums equal source marginal a
            transport = W * a.to(W.dtype).unsqueeze(-1)

            # Compute entropic cost
            ent_cost = (C * transport).sum().item()

        return SolverResult(
            matrix=transport,
            cost=ent_cost,
            f=None,
            g=None,
            converged=True,
            n_iters=1,
            ent_reg_cost=ent_cost,
        )
