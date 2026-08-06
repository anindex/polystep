"""Sparse random projection for memory-efficient large-scale subspace."""

import math
import warnings
from typing import Optional

import torch


# Below this subspace-to-source ratio the projection stops preserving distances.
_EXTREME_COMPRESSION_RATIO = 1e-5


class SparseRandomProjection:
    """Sparse random projection for memory-efficient large-scale subspace.

    Each column has Rademacher (+1/-1) entries at sampled positions, scaled by
    1/sqrt(nnz_per_col). Default density 1/sqrt(full_dim) follows Li, Hastie &
    Church (2006).
    """

    def __init__(
        self,
        full_dim: int,
        subspace_dim: int,
        density: Optional[float] = None,
        seed: int = 0,
    ):
        """Initialize the sparse projection."""
        self.full_dim = full_dim
        self.subspace_dim = subspace_dim
        self.density = density if density is not None else 1.0 / math.sqrt(full_dim)
        self.seed = seed

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

        self._nnz_per_col = max(1, int(self.density * full_dim))

        # A row missed by every column stays frozen while the basis stands.
        if full_dim > 0 and subspace_dim > 0:
            covered = 1.0 - (1.0 - self._nnz_per_col / full_dim) ** subspace_dim
            if covered < 0.5:
                warnings.warn(
                    f"SparseRandomProjection: this basis reaches only {covered:.1%} of the "
                    f"{full_dim} parameters, so the rest cannot move while it stands. "
                    f"Raise density (currently {self.density:.2e}) or subspace_dim "
                    f"(currently {subspace_dim}); coverage is "
                    f"1 - (1 - density)**subspace_dim.",
                    stacklevel=2,
                )

        self._indices: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._sparse_matrix: Optional[torch.Tensor] = None
        self._csr_matrix: Optional[torch.Tensor] = None

    def _init_sparse_matrix(self, device: torch.device, dtype: torch.dtype) -> None:
        """Build the sparse projection matrix on first use."""
        # Keep device="cpu" explicit so factories don't follow torch.set_default_device.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        total_nnz = self._nnz_per_col * self.subspace_dim

        scale = 1.0 / math.sqrt(self._nnz_per_col)

        row_indices = torch.randint(
            0,
            self.full_dim,
            (total_nnz,),
            generator=generator,
            device="cpu",
        )

        col_indices = (
            torch.arange(self.subspace_dim, device="cpu")
            .unsqueeze(1)
            .expand(
                -1,
                self._nnz_per_col,
            )
            .reshape(-1)
        )

        signs = (
            torch.randint(
                0,
                2,
                (total_nnz,),
                generator=generator,
                device="cpu",
            )
            * 2
            - 1
        )
        values = signs.to(dtype) * scale

        self._indices = torch.stack([row_indices, col_indices]).to(device)
        self._values = values.to(device)
        self._device = device
        self._dtype = dtype

    def _get_sparse_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the sparse COO matrix, building it on first use."""
        if self._indices is None or self._device != device or self._dtype != dtype:
            self._init_sparse_matrix(device, dtype)
            self._sparse_matrix = None
            self._csr_matrix = None
            self._csr_matrix = None

        if self._sparse_matrix is None:
            # Indices are in range by construction; the context-manager opt-out is process-global.
            self._sparse_matrix = torch.sparse_coo_tensor(
                self._indices,
                self._values,
                size=(self.full_dim, self.subspace_dim),
                device=device,
                dtype=dtype,
                check_invariants=False,
            ).coalesce()

        return self._sparse_matrix

    def _get_csr_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The same matrix in CSR, cached beside the COO one."""
        coo = self._get_sparse_matrix(device, dtype)
        if self._csr_matrix is None:
            self._csr_matrix = coo.to_sparse_csr()
        return self._csr_matrix

    def columns(self, cols: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """Row indices and values for the requested columns.

        Returns ``(rows, vals)`` shaped ``(len(cols), nnz_per_col)``. Duplicate rows
        within a column stay separate; scatter-add to match what coalesce would sum.
        """
        if self._indices is None or self._device != device or self._dtype != dtype:
            self._init_sparse_matrix(device, dtype)
            self._sparse_matrix = None
        rows = self._indices[0].view(self.subspace_dim, self._nnz_per_col)
        vals = self._values.view(self.subspace_dim, self._nnz_per_col)
        return rows.index_select(0, cols), vals.index_select(0, cols)

    def project(self, coords: torch.Tensor) -> torch.Tensor:
        """Project subspace coordinates to full space: full = P @ coords."""
        is_1d = coords.dim() == 1
        if is_1d:
            coords = coords.unsqueeze(0)

        # CSR on a contiguous operand is much faster than COO on the transposed view.
        P = self._get_csr_matrix(coords.device, coords.dtype)
        result = (P @ coords.T.contiguous()).T

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
        """Estimated memory usage in bytes (indices plus values)."""
        total_nnz = self.nnz
        indices_bytes = 2 * total_nnz * 8
        element_size = self._dtype.itemsize if self._dtype else 4
        values_bytes = total_nnz * element_size
        return indices_bytes + values_bytes

    @property
    def nnz(self) -> int:
        """Number of nonzeros after coalesce, or the pre-coalesce estimate before build."""
        if self._sparse_matrix is not None:
            return self._sparse_matrix._nnz()
        return self._nnz_per_col * self.subspace_dim

    def __repr__(self) -> str:
        return (
            f"SparseRandomProjection(full_dim={self.full_dim}, "
            f"subspace_dim={self.subspace_dim}, density={self.density:.4f}, "
            f"nnz_per_col={self._nnz_per_col}, memory={self.memory_bytes / 1e6:.2f}MB)"
        )
