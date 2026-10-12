"""Polytope templates and random rotations."""

import math
from typing import Callable, Dict, Optional

import torch

from .solvers._shared import single_thread_cpu, thin_qr


def get_orthoplex_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
) -> torch.Tensor:
    """``(2*dim, dim)``: the positive then negative unit vector along each axis."""
    eye = torch.eye(dim, dtype=dtype, device=device)
    points = torch.cat([eye, -eye], dim=0)
    return points * radius


def get_simplex_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
) -> torch.Tensor:
    """``(dim+1, dim)``: a regular simplex centered at the origin."""
    points = math.sqrt(1 + 1 / dim) * torch.eye(dim, dtype=dtype, device=device)
    points = points - ((math.sqrt(dim + 1) + 1) / math.sqrt(dim**3))

    last_vertex = (1 / math.sqrt(dim)) * torch.ones(1, dim, dtype=dtype, device=device)
    points = torch.cat([points, last_vertex], dim=0)

    # Centred and unit-norm by construction.
    return points * radius


def get_cube_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
) -> torch.Tensor:
    """``(2**dim, dim)``: every sign combination, normalized by ``sqrt(dim)``."""
    n_vertices = 2**dim

    indices = torch.arange(n_vertices, dtype=torch.int64, device=device).unsqueeze(1)
    shifts = torch.arange(dim, dtype=torch.int64, device=device).unsqueeze(0)

    bits = (indices >> shifts) & 1
    float_dtype = dtype if dtype is not None else torch.float32
    signs = 1.0 - 2.0 * bits.to(float_dtype)

    points = signs / math.sqrt(dim)
    return points * radius


POLYTOPE_MAP: Dict[str, Callable] = {
    "cube": get_cube_vertices,
    "orthoplex": get_orthoplex_vertices,
    "simplex": get_simplex_vertices,
}


# Widest batch where one batched QR still beat the reflection loop, indexed by is_cuda.
_QR_MAX_BATCH = {False: 256, True: 16}


def get_rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """``(...)`` angles to ``(..., 2, 2)`` rotation matrices."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    row1 = torch.stack([c, -s], dim=-1)
    row2 = torch.stack([s, c], dim=-1)
    return torch.stack([row1, row2], dim=-2)


def get_random_rotation_matrices(
    batch: int,
    dim: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Batched Haar-random rotations on SO(dim); see Diaconis-Shahshahani 1987 and Mezzadri 2007."""
    # Build in FP32 and cast back: half precision loses the rank-1 updates.
    resolved_dtype = dtype if dtype is not None else torch.float32
    compute_dtype = torch.float32 if resolved_dtype in (torch.bfloat16, torch.float16) else resolved_dtype

    # Sampling needs the generator's device to match the tensor's.
    sample_device = generator.device if generator is not None else device

    if dim == 2:
        angles = torch.empty(batch, device=sample_device, dtype=compute_dtype)
        angles.uniform_(0, 2 * math.pi, generator=generator)
        return get_rotation_matrix_2d(angles).to(device=device, dtype=dtype)

    # The QR runs on the sample's device, not the requested output device.
    work_device = sample_device if sample_device is not None else torch.device("cpu")
    if batch <= _QR_MAX_BATCH[work_device.type == "cuda"]:
        # Keep the whole branch: the batched det oversubscribes like the QR.
        with single_thread_cpu(work_device):
            A = torch.randn(batch, dim, dim, device=sample_device, dtype=compute_dtype, generator=generator)
            Q, R = thin_qr(A)
            # Q diag(sign(diag R)) is Haar on O(dim); a zero pivot maps to +1.
            diag = torch.diagonal(R, dim1=-2, dim2=-1)
            signs = torch.where(diag == 0, torch.ones_like(diag), diag.sign())
            # Negating one column on the det = -1 half maps onto SO(dim); folds into one multiply.
            parity = (torch.linalg.det(Q) * signs.prod(dim=-1)).sign()
            signs = torch.cat([signs[:, :1] * parity.unsqueeze(-1), signs[:, 1:]], dim=-1)
            return (Q * signs.unsqueeze(-2)).to(device=device, dtype=dtype)

    Q = torch.eye(dim, device=sample_device, dtype=compute_dtype).expand(batch, dim, dim).contiguous()
    # Fixed sign flip makes det(Q) = +1 without a batched det.
    if dim % 2 == 0:
        Q[:, dim - 1, dim - 1] = -1.0

    for k in range(2, dim + 1):
        u = torch.randn(batch, k, device=sample_device, dtype=compute_dtype, generator=generator)
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp(min=1e-30)

        w = -u
        w[:, 0] += 1.0
        wsq = (w * w).sum(dim=-1, keepdim=True)
        # Degenerate only at u == e_1; fall back to w = e_1, not H = I (which flips det).
        # On CUDA the guard runs unconditionally to avoid a host sync per iteration.
        degenerate = wsq < 1e-12
        if not (Q.is_cuda or degenerate.any()):
            rows = Q[:, dim - k :, :]
            rows -= (2.0 / wsq).unsqueeze(-1) * w.unsqueeze(-1) * (w.unsqueeze(1) @ rows)
            continue
        w = torch.where(degenerate, torch.zeros_like(w), w)
        w[:, 0] = torch.where(degenerate[:, 0], torch.ones_like(w[:, 0]), w[:, 0])
        wsq = torch.where(degenerate, torch.ones_like(wsq), wsq)

        # Q <- H_k Q, trailing k rows only.
        rows = Q[:, dim - k :, :]
        rows -= (2.0 / wsq).unsqueeze(-1) * w.unsqueeze(-1) * (w.unsqueeze(1) @ rows)

    return Q.to(device=device, dtype=dtype)


def apply_biased_rotation(rot_mats: torch.Tensor, bias_dir: torch.Tensor) -> torch.Tensor:
    """Point each frame's first axis along ``bias_dir`` with one Householder reflection.

    A column sign fix keeps det +1, and the other axes stay Haar on the complement
    of the bias. Rows with a zero or nonfinite ``bias_dir`` come back unchanged.
    """
    dim = rot_mats.shape[-1]
    if dim == 1:
        # SO(1) = {[[1]]}: a sign flip is a reflection, so bias is not representable.
        return torch.ones_like(rot_mats)

    out_dtype = rot_mats.dtype
    if out_dtype in (torch.bfloat16, torch.float16):
        rot_mats = rot_mats.float()
        bias_dir = bias_dir.float()

    norms = torch.linalg.vector_norm(bias_dir, dim=-1, keepdim=True)
    valid = torch.isfinite(norms) & (norms > 1e-8)
    b = torch.where(valid, bias_dir / norms.clamp(min=1e-10), torch.zeros_like(bias_dir))
    r0 = rot_mats[:, :, 0]
    s = torch.ones_like(norms).copysign((r0 * b).sum(dim=-1, keepdim=True))
    w = r0 + s * b
    scale = (2 / (w * w).sum(dim=-1, keepdim=True)).unsqueeze(-1)
    out = rot_mats - scale * w.unsqueeze(-1) * (w.unsqueeze(1) @ rot_mats)
    out = out * torch.cat([-s, torch.ones_like(r0[:, 2:]), s], dim=-1).unsqueeze(1)
    return torch.where(valid.unsqueeze(-1), out, rot_mats).to(out_dtype)
