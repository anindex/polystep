"""Low-rank subspace whose candidates never materialize a weight.

A 2D parameter ``W`` of shape ``(d_out, d_in)`` is perturbed by ``dW = A @ B``, with
``B`` a fixed random matrix with orthonormal rows and the coordinates being ``A``.
``B`` is fixed within a rotation epoch, so ``dW`` is linear in the coordinates and
absorb, displacement history and rotation work as for :class:`HybridSubspace`.

For an input ``x``::

    x (W + A B)^T  =  x W^T  +  (x B^T) A^T

so ``N`` candidates need one GEMM against the shared base weight plus a rank-``r``
correction each, costing ``O(batch * r * (d_in + d_out))``.
:class:`~polystep.cost_nn.FactoredEvaluator` does this;
:meth:`reconstruct_batch` is the materializing fallback for other models.

Orthonormal rows in ``B`` give ``||A @ B||_F == ||A||_F``, matching the unit-gain
convention :class:`HybridSubspace` uses for tall layers, so step radii carry over.

Cheaper per step than :class:`HybridSubspace` but lower per-step progress: ``dW``
is confined to the ``rank`` input directions ``B`` spans. See ``docs/performance.md``
for the trade-off.

Reference: arXiv:2511.16652, which also requires a perturbation scale of
``o(d^-1/2)`` for the linearization to hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .solvers._shared import thin_qr
from .subspace import ProjectedAbsorbMixin, absorb_due

from .hybrid_subspace import LayerProjectionSpec, _stable_entry_seed


@dataclass(frozen=True)
class FactoredSubspace(ProjectedAbsorbMixin):
    """Per-layer low-rank subspace whose coordinates are the ``A`` factors.

    Args:
        specs: One :class:`LayerProjectionSpec` per parameter entry. For projected
            entries ``num_coords == d_out * rank``; 1D entries pass through unprojected.
        subspace_dim: Total coordinate count.
        compression_ratio: ``subspace_dim / total_params``.
        seed: Base seed for the fixed ``B`` factors.
        ranks: Effective rank per entry key.
        rotation_interval: Redraw ``B`` every N steps. ``0`` (default) holds the basis
            fixed, matching :class:`HybridSubspace`.
        absorb_mode: ``"stagnation"`` or ``"periodic"``.
        absorb_patience: Stagnation steps before an absorb.
        absorb_interval: Step interval for ``absorb_mode="periodic"``.
        displacement_history_size: Rolling displacement buffer length.
    """

    specs: Tuple[LayerProjectionSpec, ...]
    subspace_dim: int
    compression_ratio: float
    seed: int = 0
    ranks: Dict[str, int] = field(default_factory=dict)
    rotation_interval: int = 0
    absorb_mode: str = "stagnation"
    absorb_patience: int = 20
    absorb_interval: int = 0
    displacement_history_size: int = 10
    _total_params: int = 0

    @classmethod
    def from_layout(
        cls,
        layout,
        rank: int = 8,
        seed: int = 0,
        **kwargs,
    ) -> "FactoredSubspace":
        """Build specs from a :class:`~polystep.transform.ParamLayout`.

        Args:
            layout: Source layout.
            rank: Requested rank per projected entry, clipped to ``min(d_in, d_out)``.
            seed: Base seed for the fixed ``B`` factors.
            **kwargs: Forwarded to the constructor (``rotation_interval``, ``absorb_*``).

        Returns:
            A :class:`FactoredSubspace` covering every entry in ``layout``.
        """
        specs = []
        ranks: Dict[str, int] = {}
        offset = 0

        for entry in layout.entries:
            shape = entry.shape
            if len(shape) >= 2:
                d_out = shape[0]
                d_in = math.prod(shape[1:])
                r = max(1, min(rank, d_in, d_out))
                num_coords = d_out * r
                ranks[entry.key] = r
                projected = True
            else:
                # 1D params (bias, LayerNorm) are cheap and carry no matrix structure
                # to factor; perturb them directly, as HybridSubspace does.
                num_coords = entry.numel
                projected = False

            specs.append(
                LayerProjectionSpec(
                    entry_key=entry.key,
                    original_shape=shape,
                    num_params=math.prod(shape) if len(shape) >= 2 else entry.numel,
                    num_coords=num_coords,
                    flat_start=offset,
                    flat_end=offset + num_coords,
                    is_projected=projected,
                )
            )
            offset += num_coords

        total = layout.total_params
        return cls(
            specs=tuple(specs),
            subspace_dim=offset,
            compression_ratio=offset / total if total > 0 else 0.0,
            seed=seed,
            ranks=ranks,
            _total_params=total,
            **kwargs,
        )

    def _make_b(self, spec: LayerProjectionSpec, device, dtype, step: int) -> torch.Tensor:
        """Fixed ``(rank, d_in)`` factor with orthonormal rows, from a stable seed.

        Generated on CPU in fp32 so the same seed gives the same factor on CPU and
        CUDA, then moved. QR of a ``(d_in, rank)`` Gaussian gives orthonormal columns;
        transposing yields orthonormal rows, hence ``||A @ B||_F == ||A||_F``.
        """
        r = self.ranks[spec.entry_key]
        d_in = spec.num_params // spec.original_shape[0]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_stable_entry_seed(self.seed, spec.entry_key, step))
        G = torch.randn(d_in, r, generator=gen, dtype=torch.float32, device="cpu")
        Q, _ = thin_qr(G)  # (d_in, r), orthonormal columns
        return Q.t().contiguous().to(device=device, dtype=dtype)

    def init_projections(self, device, dtype, step: int = 0) -> Dict[str, torch.Tensor]:
        """Build the ``B`` factor for every projected entry."""
        return {spec.entry_key: self._make_b(spec, device, dtype, step) for spec in self.specs if spec.is_projected}

    def rotate_all(
        self,
        projections: Dict[str, torch.Tensor],
        step: int,
        total_steps: Optional[int] = None,
        displacement_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Redraw every ``B`` on the rotation interval, else return ``projections``.

        Returning the same object signals "nothing changed" to the caller, which uses
        identity to decide whether to drop warm-started duals.
        """
        if self.rotation_interval <= 0 or step <= 0 or step % self.rotation_interval != 0:
            return projections
        # Empty when every parameter is a vector: nothing is projected, nothing to redraw.
        if not projections:
            return projections
        device = next(iter(projections.values())).device
        dtype = next(iter(projections.values())).dtype
        return self.init_projections(device, dtype, step=step)

    def should_absorb(self, stagnation_count: int, iteration: int) -> bool:
        """Whether to fold the perturbation into the base weights this step."""
        return absorb_due(
            self.absorb_mode,
            self.absorb_patience,
            self.absorb_interval,
            stagnation_count,
            iteration,
        )

    def _delta(self, spec: LayerProjectionSpec, B: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """``dW`` for one entry from its coordinate slice.

        ``coords`` is ``(..., num_coords)``; the result is ``(..., num_params)``.
        """
        if not spec.is_projected:
            return coords
        d_out = spec.original_shape[0]
        r = self.ranks[spec.entry_key]
        A = coords.reshape(*coords.shape[:-1], d_out, r)
        return (A @ B.to(A.dtype)).reshape(*coords.shape[:-1], spec.num_params)

    def apply_perturbation(
        self,
        projections: Dict[str, torch.Tensor],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full state_dict for a single coordinate vector."""
        out = dict(base_sd)
        for spec in self.specs:
            base = base_sd.get(spec.entry_key)
            if base is None:
                continue
            coords = flat_subspace[spec.flat_start : spec.flat_end]
            delta = self._delta(spec, projections.get(spec.entry_key), coords)
            out[spec.entry_key] = base + delta.reshape(base.shape).to(base.dtype)
        return out

    def reconstruct_batch(
        self,
        projections: Dict[str, torch.Tensor],
        base_sd: Dict[str, torch.Tensor],
        flat_subspace_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Materialize ``{key: (N, *shape)}``.

        Only used for models :class:`~polystep.cost_nn.FactoredEvaluator` cannot handle.
        This subspace exists to avoid this call.
        """
        N = flat_subspace_batch.shape[0]
        out = {}
        for spec in self.specs:
            base = base_sd.get(spec.entry_key)
            if base is None:
                continue
            coords = flat_subspace_batch[:, spec.flat_start : spec.flat_end]
            delta = self._delta(spec, projections.get(spec.entry_key), coords)
            out[spec.entry_key] = base.reshape(1, -1) + delta.to(base.dtype)
            out[spec.entry_key] = out[spec.entry_key].reshape(N, *spec.original_shape)
        return out
