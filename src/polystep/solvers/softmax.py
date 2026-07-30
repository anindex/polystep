"""SoftmaxSolver: direct softmax weighting for subspace modes.

Replaces iterative Sinkhorn OT with a single softmax(-C/epsilon) pass. This is
the exact lambda -> 0 unbalanced limit, and coincides with balanced OT only when
the column sums happen to hit the target marginal. See paper Section 5.10.

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

from ._shared import solve_softmax
from .base import SolverResult


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
        # One-sided: only row sums are enforced. A caller passing a non-uniform `b`
        # probably wanted SinkhornSolver.
        if b is not None:
            uniform = torch.full_like(b, 1.0 / cost_matrix.shape[1])
            if not torch.allclose(b.to(device=uniform.device, dtype=uniform.dtype), uniform, atol=1e-6):
                warnings.warn(
                    "SoftmaxSolver ignores the target marginal `b`: it only "
                    "enforces row sums equal to the source marginal `a`. "
                    "Pass solver='sinkhorn' for two-sided OT.",
                    stacklevel=2,
                )

        return solve_softmax(self, cost_matrix, a, self.epsilon, scale_cost, "SoftmaxSolver", "epsilon")
