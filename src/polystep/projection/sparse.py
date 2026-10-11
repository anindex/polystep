"""Sparse random projection for memory-efficient large-scale subspace."""

import math
import warnings
from typing import Optional

import torch

from ..solvers._shared import decomposition_dtype


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
        if not isinstance(full_dim, int) or isinstance(full_dim, bool) or full_dim < 1:
            raise ValueError("full_dim must be a positive integer")
        if not isinstance(subspace_dim, int) or isinstance(subspace_dim, bool) or subspace_dim < 1:
            raise ValueError("subspace_dim must be a positive integer")
        if density is not None and not (math.isfinite(density) and 0 < density <= 1):
            raise ValueError("density must be finite and in (0, 1]")
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

        self._rows: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._csr_matrix: Optional[torch.Tensor] = None
        self._csr_rows: Optional[torch.Tensor] = None

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

        self._rows = row_indices.view(self.subspace_dim, self._nnz_per_col).to(device)
        self._values = values.view(self.subspace_dim, self._nnz_per_col).to(device)
        self._device = device
        self._dtype = dtype
        self._csr_matrix = None
        self._csr_rows = None

    def _pointers_dominate(self, nnz: int, index_size: int) -> bool:
        """Whether ``full_dim + 1`` row pointers would outweigh an int64 map and pointers for ``nnz`` rows."""
        return self.full_dim * index_size > nnz * (index_size + 8)

    def _get_csr_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The coalesced matrix in CSR, built on first use.

        When full-height row pointers would outweigh a map of the touched rows, the CSR
        keeps only the rows that hold an entry and ``_csr_rows`` maps them back.
        """
        if self._rows is None or self._device != device or self._dtype != dtype:
            self._init_sparse_matrix(device, dtype)

        if self._csr_matrix is None:
            sub = self.subspace_dim
            # Indices are in range by construction; the context-manager opt-out is process-global.
            coo = torch.sparse_coo_tensor(
                (self._rows * sub).add_(torch.arange(sub, device=device)[:, None]).view(1, -1),
                self._values.reshape(-1),
                size=(self.full_dim * sub,),
                device=device,
                dtype=decomposition_dtype(dtype) if device.type == "cpu" else dtype,
                check_invariants=False,
            ).coalesce()
            keys, values = coo.indices()[0], coo.values().clone()
            del coo
            rows, cols = keys // sub, keys % sub
            del keys
            index_dtype = torch.int32 if max(self.full_dim, cols.numel()) < 2**31 else torch.int64
            if self._pointers_dominate(cols.numel(), index_dtype.itemsize):
                rows, counts = rows.unique_consecutive(return_counts=True)
                self._csr_rows = rows.clone()
            else:
                counts = torch.bincount(rows, minlength=self.full_dim)
            self._csr_matrix = torch.sparse_csr_tensor(
                torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(index_dtype),
                cols.to(index_dtype),
                values,
                size=(counts.numel(), self.subspace_dim),
                device=device,
                check_invariants=False,
            )

        return self._csr_matrix

    def columns(self, cols: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """Row indices and values for the requested columns.

        Returns ``(rows, vals)`` shaped ``(len(cols), nnz_per_col)``. Duplicate rows
        within a column stay separate; scatter-add to match what coalesce would sum.
        """
        if self._rows is None or self._device != device or self._dtype != dtype:
            self._init_sparse_matrix(device, dtype)
        return self._rows.index_select(0, cols), self._values.index_select(0, cols)

    def project(self, coords: torch.Tensor) -> torch.Tensor:
        """Project subspace coordinates to full space: full = P @ coords."""
        is_1d = coords.dim() == 1
        if is_1d:
            coords = coords.unsqueeze(0)

        # CSR on a contiguous operand is much faster than COO on the transposed view.
        P = self._get_csr_matrix(coords.device, coords.dtype)
        result = P @ coords.T.to(P.dtype).contiguous()
        if self._csr_rows is not None:
            result = result.new_zeros(self.full_dim, result.shape[1]).index_copy_(0, self._csr_rows, result)
        result = result.T.to(coords.dtype)

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
        """Bytes held by the entries and the cached CSR, estimated before the first ``project``."""
        if self._csr_matrix is not None:
            P = self._csr_matrix
            held = [self._rows, self._values, P.crow_indices(), P.col_indices(), P.values(), self._csr_rows]
            return sum(t.untyped_storage().nbytes() for t in held if t is not None)
        nnz = self.nnz
        index_size = 4 if max(self.full_dim, nnz) < 2**31 else 8
        value_size = self._dtype.itemsize if self._dtype else 4
        if self._pointers_dominate(nnz, index_size):
            pointers = nnz * 8 + (nnz + 1) * index_size
        else:
            pointers = (self.full_dim + 1) * index_size
        return nnz * (8 + value_size) + nnz * (index_size + value_size) + pointers

    @property
    def nnz(self) -> int:
        """Number of nonzeros after coalesce, or the pre-coalesce estimate before build."""
        if self._csr_matrix is not None:
            return self._csr_matrix._nnz()
        return self._nnz_per_col * self.subspace_dim

    def __repr__(self) -> str:
        return (
            f"SparseRandomProjection(full_dim={self.full_dim}, "
            f"subspace_dim={self.subspace_dim}, density={self.density:.4f}, "
            f"nnz_per_col={self._nnz_per_col}, memory={self.memory_bytes / 1e6:.2f}MB)"
        )
