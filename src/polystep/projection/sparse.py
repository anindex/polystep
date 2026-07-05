"""Sparse random projection for memory-efficient large-scale subspace.

Implements an Achlioptas-style sparse projection (Rademacher entries at
randomly sampled positions) at the very-sparse density of Li, Hastie &
Church (2006), preserving Johnson-Lindenstrauss distance guarantees
while using O(nnz) memory.

References:
    Achlioptas, D. (2003). "Database-friendly random projections:
        Johnson-Lindenstrauss with binary coins."
    Li, P., Hastie, T. & Church, K. (2006). "Very sparse random
        projections."
"""

import math
import warnings
from typing import Optional

import torch


# Below this ratio of subspace-dim to source-dim, the Johnson-
# Lindenstrauss distance guarantees stop holding for typical
# optimization workloads. Empirical floor: GPT-2 124M projected to
# 128-dim (ratio ~ 1e-6) collapsed to random predictions in early
# experiments.
_EXTREME_COMPRESSION_RATIO = 1e-5


class SparseRandomProjection:
    """Sparse random projection for memory-efficient large-scale subspace.

    Achlioptas-style sparse projection: each column has Rademacher (+1/-1)
    entries at randomly sampled positions, scaled by 1/sqrt(nnz_per_col).
    The default density ``1/sqrt(full_dim)`` follows Li, Hastie & Church
    (2006), which preserves JL distance guarantees while keeping memory
    at O(nnz).

    The projection matrix P is (full_dim x subspace_dim) with:
    - Each column has exactly nnz_per_col = density * full_dim nonzeros
    - Values are +1 or -1 (Rademacher) scaled by 1/sqrt(nnz_per_col)

    Memory usage: O(density * full_dim * subspace_dim) = O(nnz)
    For 1M params, rank-256, density 1/sqrt(1M) = 0.1%:
    - Dense: 1M * 256 * 4 bytes = 1GB
    - Sparse: 1M * 0.001 * 256 * 4 bytes = 1MB

    Attributes:
        full_dim: Dimension of the full parameter space.
        subspace_dim: Dimension of the low-rank subspace.
        density: Fraction of nonzeros per column (default: 1/sqrt(full_dim)).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        full_dim: int,
        subspace_dim: int,
        density: Optional[float] = None,
        seed: int = 0,
    ):
        """Initialize sparse random projection.

        Args:
            full_dim: Dimension of the full parameter space.
            subspace_dim: Dimension of the low-rank subspace.
            density: Fraction of nonzeros per column. Default: 1/sqrt(full_dim).
            seed: Random seed for reproducibility.
        """
        self.full_dim = full_dim
        self.subspace_dim = subspace_dim
        self.density = density if density is not None else 1.0 / math.sqrt(full_dim)
        self.seed = seed

        # Warn loudly at extreme compression ratios -- see
        # ``_EXTREME_COMPRESSION_RATIO`` above.
        if full_dim > 0:
            ratio = subspace_dim / full_dim
            if ratio < _EXTREME_COMPRESSION_RATIO:
                warnings.warn(
                    f"SparseRandomProjection: subspace_dim={subspace_dim} / "
                    f"full_dim={full_dim} = {ratio:.2e} is below the empirical "
                    f"floor ({_EXTREME_COMPRESSION_RATIO:.0e}) where the "
                    f"projection preserves distances meaningfully. The paper "
                    f"reports GPT-2 124M with 128-dim projection collapsed to "
                    f"random predictions in this regime. Consider increasing "
                    f"subspace_dim.",
                    stacklevel=2,
                )

        # Compute nnz per column (at least 1)
        self._nnz_per_col = max(1, int(self.density * full_dim))

        # Lazy initialization: indices and values created on first use
        self._indices: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._sparse_matrix: Optional[torch.Tensor] = None

    def _init_sparse_matrix(self, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize sparse projection matrix on demand.

        Creates row indices via vectorized random sampling and Rademacher
        values (+1/-1) scaled by 1/sqrt(nnz_per_col).

        Uses ``torch.randint`` for row index generation which samples with
        replacement. At typical sparsity (density = 1/sqrt(n)), the collision
        rate is negligible: for n=786K and nnz_per_col=886, expected collisions
        per column is ~0.5 (0.06% of entries). Any duplicate indices in a
        column are summed by ``coalesce()``, which marginally reduces that
        column's effective nonzeros but does not affect JL distance
        preservation guarantees in practice.

        Args:
            device: Device for the sparse matrix.
            dtype: Data type for values.
        """
        # Use CPU generator for consistent behavior (avoids CUDA generator issues)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        total_nnz = self._nnz_per_col * self.subspace_dim

        # Scale factor for unit variance
        scale = 1.0 / math.sqrt(self._nnz_per_col)

        # Vectorized row index sampling (with replacement -- negligible collisions)
        row_indices = torch.randint(
            0,
            self.full_dim,
            (total_nnz,),
            generator=generator,
        )

        # Column indices: each column gets nnz_per_col consecutive entries
        col_indices = (
            torch.arange(self.subspace_dim)
            .unsqueeze(1)
            .expand(
                -1,
                self._nnz_per_col,
            )
            .reshape(-1)
        )

        # Rademacher values: +1 or -1 with equal probability, vectorized
        signs = (
            torch.randint(
                0,
                2,
                (total_nnz,),
                generator=generator,
            )
            * 2
            - 1
        )
        values = signs.to(dtype) * scale

        # Store indices as (2, nnz) for sparse_coo_tensor
        self._indices = torch.stack([row_indices, col_indices]).to(device)
        self._values = values.to(device)
        self._device = device
        self._dtype = dtype

    def _get_sparse_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get sparse projection matrix, initializing if needed.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Sparse COO tensor of shape (full_dim, subspace_dim).
        """
        # Initialize on first use or if device/dtype changed.
        if self._indices is None or self._device != device or self._dtype != dtype:
            self._init_sparse_matrix(device, dtype)
            self._sparse_matrix = None

        if self._sparse_matrix is None:
            self._sparse_matrix = torch.sparse_coo_tensor(
                self._indices,
                self._values,
                size=(self.full_dim, self.subspace_dim),
                device=device,
                dtype=dtype,
            ).coalesce()

        return self._sparse_matrix

    def project(self, coords: torch.Tensor) -> torch.Tensor:
        """Project subspace coordinates to full parameter space.

        Computes: full = P @ coords where P is (full_dim x subspace_dim).

        Args:
            coords: Subspace coordinates of shape (subspace_dim,) or (batch, subspace_dim).

        Returns:
            Full parameter space vector of shape (full_dim,) or (batch, full_dim).
        """
        is_1d = coords.dim() == 1
        if is_1d:
            coords = coords.unsqueeze(0)  # (1, subspace_dim)

        # Get sparse matrix
        P = self._get_sparse_matrix(coords.device, coords.dtype)

        # Sparse-dense matmul: P @ coords.T -> (full_dim, batch)
        # coords is (batch, subspace_dim), need (subspace_dim, batch)
        result = torch.sparse.mm(P, coords.T)  # (full_dim, batch)
        result = result.T  # (batch, full_dim)

        if is_1d:
            result = result.squeeze(0)

        return result

    def project_transpose(self, full: torch.Tensor) -> torch.Tensor:
        """Project full parameters back to subspace (transpose).

        Computes: coords = P^T @ full where P is (full_dim x subspace_dim).

        Args:
            full: Full parameter space vector of shape (full_dim,) or (batch, full_dim).

        Returns:
            Subspace coordinates of shape (subspace_dim,) or (batch, subspace_dim).
        """
        is_1d = full.dim() == 1
        if is_1d:
            full = full.unsqueeze(0)  # (1, full_dim)

        # Get sparse matrix and transpose
        P = self._get_sparse_matrix(full.device, full.dtype)
        P_T = P.t()  # (subspace_dim, full_dim)

        # Sparse-dense matmul: P^T @ full.T -> (subspace_dim, batch)
        result = torch.sparse.mm(P_T, full.T)  # (subspace_dim, batch)
        result = result.T  # (batch, subspace_dim)

        if is_1d:
            result = result.squeeze(0)

        return result

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Data type of the projection matrix (None if not yet initialized)."""
        return self._dtype

    @property
    def device(self) -> Optional[torch.device]:
        """Device of the projection matrix (None if not yet initialized)."""
        return self._device

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes.

        Memory breakdown:
        - Indices: 2 * nnz * 8 bytes (int64 for row and col)
        - Values: nnz * element_size bytes (depends on dtype)

        Returns:
            Estimated memory usage in bytes.
        """
        total_nnz = self._nnz_per_col * self.subspace_dim
        # 2 index arrays (row, col) with int64
        indices_bytes = 2 * total_nnz * 8
        element_size = self._dtype.itemsize if self._dtype else 4
        values_bytes = total_nnz * element_size
        return indices_bytes + values_bytes

    @property
    def nnz(self) -> int:
        """Total number of nonzeros in projection matrix."""
        return self._nnz_per_col * self.subspace_dim

    @property
    def actual_density(self) -> float:
        """Actual density (nnz / total_elements)."""
        return self.nnz / (self.full_dim * self.subspace_dim)

    def __repr__(self) -> str:
        return (
            f"SparseRandomProjection(full_dim={self.full_dim}, "
            f"subspace_dim={self.subspace_dim}, density={self.density:.4f}, "
            f"nnz_per_col={self._nnz_per_col}, memory={self.memory_bytes / 1e6:.2f}MB)"
        )
