"""Geometry module: polytope templates, rotation matrices, and probe generation.

Combines polytope vertex generators (orthoplex, simplex, cube), random rotation
via QR decomposition (Mezzadri method) or analytical 2D formula, and deterministic
probe point generation into a single module for PolyStep exploration directions.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch


def get_orthoplex_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Generate orthoplex (cross-polytope) vertices centered at the origin.

    Produces 2*dim vertices: the positive and negative unit vectors along each axis.

    Args:
        dim: Dimensionality of the vertices.
        device: Target device for the output tensor.
        dtype: Target dtype for the output tensor.
        radius: Scaling radius.

    Returns:
        Vertices of shape (2*dim, dim).
    """
    eye = torch.eye(dim, dtype=dtype, device=device)
    points = torch.cat([eye, -eye], dim=0)
    return points * radius


def get_simplex_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Generate regular simplex vertices centered at the origin.

    Produces dim+1 vertices forming a regular simplex.

    Args:
        dim: Dimensionality of the vertices.
        device: Target device for the output tensor.
        dtype: Target dtype for the output tensor.
        radius: Scaling radius.

    Returns:
        Vertices of shape (dim+1, dim).
    """
    points = math.sqrt(1 + 1 / dim) * torch.eye(dim, dtype=dtype, device=device)
    points = points - ((math.sqrt(dim + 1) + 1) / math.sqrt(dim**3))

    last_vertex = (1 / math.sqrt(dim)) * torch.ones(1, dim, dtype=dtype, device=device)
    points = torch.cat([points, last_vertex], dim=0)

    # Center simplex at origin (unlike orthoplex/cube, simplex is not inherently symmetric)
    centroid = points.mean(dim=0, keepdim=True)
    points = points - centroid

    return points * radius


def get_cube_vertices(
    dim: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    radius: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Generate hypercube vertices using bitwise logic.

    Produces 2^dim vertices: all sign combinations normalized by sqrt(dim).

    Args:
        dim: Dimensionality of the vertices.
        device: Target device for the output tensor.
        dtype: Target dtype for the output tensor.
        radius: Scaling radius.

    Returns:
        Vertices of shape (2^dim, dim).
    """
    n_vertices = 2**dim

    indices = torch.arange(n_vertices, dtype=torch.int32, device=device).unsqueeze(1)
    shifts = torch.arange(dim, dtype=torch.int32, device=device).unsqueeze(0)

    bits = (indices >> shifts) & 1
    # Resolve dtype for the float conversion (default to float32 if None)
    float_dtype = dtype if dtype is not None else torch.float32
    signs = 1.0 - 2.0 * bits.to(float_dtype)

    points = signs / math.sqrt(dim)
    return points * radius


POLYTOPE_MAP: Dict[str, Callable] = {
    "cube": get_cube_vertices,
    "orthoplex": get_orthoplex_vertices,
    "simplex": get_simplex_vertices,
}

POLYTOPE_NUM_VERTICES_MAP: Dict[str, Callable[[int], int]] = {
    "cube": lambda dim: 2**dim,
    "orthoplex": lambda dim: 2 * dim,
    "simplex": lambda dim: dim + 1,
}


def get_rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """Create 2x2 rotation matrices from angles using analytical formula.

    Args:
        theta: Angles tensor of shape (...).

    Returns:
        Rotation matrices of shape (..., 2, 2).
    """
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
    """Generate batched uniformly random rotation matrices on SO(dim).

    For dim=2, uses fast analytical rotation. For dim>2, uses batched QR
    decomposition with Mezzadri sign correction for Haar measure.

    Ref: F. Mezzadri, "How to generate random matrices from the
    classical compact groups" (arXiv:math-ph/0609050).

    Args:
        batch: Number of rotation matrices to generate.
        dim: Dimension of each rotation matrix.
        device: Target device.
        dtype: Target dtype.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Rotation matrices of shape (batch, dim, dim) with det(R) = +1.
    """
    if dim == 2:
        angles = torch.empty(batch, device=device, dtype=dtype)
        angles.uniform_(0, 2 * math.pi, generator=generator)
        return get_rotation_matrix_2d(angles)

    # Batched QR decomposition for dim > 2
    # QR has no bf16/fp16 kernel outside CUDA (geqrf_cpu, and MPS/others too);
    # generate in FP32, compute QR, then convert back to the target dtype.
    # Resolve None device/dtype to concrete values for comparison
    resolved_device = device if device is not None else torch.device("cpu")
    resolved_dtype = dtype if dtype is not None else torch.float32
    device_type = resolved_device.type if hasattr(resolved_device, "type") else str(resolved_device)
    needs_fp32_qr = resolved_dtype in (torch.bfloat16, torch.float16) and device_type != "cuda"
    compute_dtype = torch.float32 if needs_fp32_qr else resolved_dtype
    compute_device = "cpu" if needs_fp32_qr else resolved_device

    # Generator device must match tensor device for ``randn``. If the user
    # supplied a CUDA generator but we have to run QR on CPU (bfloat16
    # fallback), sample on the generator's device first, then move the
    # result to the QR compute device. This preserves reproducibility.
    gen_for_randn = generator
    sample_device = compute_device
    needs_post_move = False
    if generator is not None and hasattr(generator, "device"):
        gen_device_type = generator.device.type if hasattr(generator.device, "type") else str(generator.device)
        compute_device_type = "cpu" if needs_fp32_qr else device_type
        if gen_device_type != compute_device_type:
            sample_device = generator.device
            needs_post_move = True

    Z = torch.randn(batch, dim, dim, device=sample_device, dtype=compute_dtype, generator=gen_for_randn)
    if needs_post_move:
        Z = Z.to(device=compute_device)
    Q, R = torch.linalg.qr(Z)

    # Sign correction for Haar measure (Mezzadri method). Use a nonzero sign
    # (sign(0) := +1) so an underflowed zero R-diagonal can't null a column.
    d = torch.diagonal(R, dim1=-2, dim2=-1)  # (batch, dim)
    phases = torch.where(d == 0, torch.ones_like(d), torch.sign(d))
    Q = Q * phases.unsqueeze(-2)  # (batch, 1, dim) * (batch, dim, dim)

    # Ensure det = +1 (SO(n) not just O(n)): flip first column where det < 0.
    # Q is a fresh tensor (product above), so the in-place column flip is safe.
    flip = torch.where(torch.det(Q) < 0, -1.0, 1.0).to(Q.dtype)  # (batch,)
    Q[:, :, 0] = Q[:, :, 0] * flip.unsqueeze(-1)

    # Convert to target dtype and device
    Q = Q.to(device=device, dtype=dtype)

    return Q


def apply_biased_rotation(rot_mats: torch.Tensor, bias_dir: torch.Tensor) -> torch.Tensor:
    """Bias the first rotation axis toward ``bias_dir`` and re-orthonormalize.

    Replaces column 0 with the normalized bias direction, re-orthonormalizes the
    remaining columns against it, and flips the last column where needed so det = +1.
    Particles whose direction is too short to carry a heading keep their unbiased
    rotation.

    Uses modified Gram-Schmidt, not ``torch.linalg.qr``. Both give the same basis, but
    cuSOLVER's batched QR is 10x slower at (64, 2, 2) and 29x slower at (4096, 16, 16).
    ``dim`` is at most a few tens here, where Gram-Schmidt is accurate enough.

    Args:
        rot_mats: Rotation matrices of shape (batch, dim, dim).
        bias_dir: Bias directions of shape (batch, dim), any magnitude.

    Returns:
        Biased rotation matrices of shape (batch, dim, dim), det = +1.
    """
    dim = rot_mats.shape[-1]
    if dim == 1:
        # SO(1) = {[[1]]}: a sign flip is a reflection, so the bias is not
        # representable. Return identity.
        return torch.ones_like(rot_mats)

    # Normalize here rather than at each call site: column 0 must be unit or the
    # Gram-Schmidt below orthogonalizes the rest against a scaled axis and returns a
    # non-orthonormal frame. Clamping the divisor instead of zeroing would do exactly
    # that, turning a 1e-12 direction into a column of norm 1e-2. Zeroing routes those
    # particles to the ``bias_ok`` fallback below.
    norms = torch.linalg.vector_norm(bias_dir, dim=-1, keepdim=True)
    bias_dir_norm = torch.where(norms > 1e-8, bias_dir / norms.clamp(min=1e-10), torch.zeros_like(bias_dir))

    out = rot_mats.clone()
    out[:, :, 0] = bias_dir_norm
    for col in range(1, dim):
        v = out[:, :, col].clone()
        # Two sweeps: one pass loses orthogonality when a column is nearly in the span
        # of the earlier ones (Q^T Q off by >1e-4 at dim=8). Twice is enough.
        for _ in range(2):
            for prev in range(col):
                proj = (v * out[:, :, prev]).sum(dim=-1, keepdim=True)
                v = v - proj * out[:, :, prev]
        raw_norm = torch.norm(v, dim=-1, keepdim=True)
        # A collapsed column carries no direction; keep the original instead of a zero.
        keep = raw_norm > 1e-6
        out[:, :, col] = torch.where(keep, v / raw_norm.clamp(min=1e-10), rot_mats[:, :, col])

    # A zero or non-finite bias leaves column 0 degenerate, so +e0 and -e0 would
    # coincide. Fall back to the unbiased rotation.
    bias_ok = torch.isfinite(bias_dir_norm).all(dim=-1) & (bias_dir_norm.abs().amax(dim=-1) > 0)
    valid = torch.isfinite(out).all(dim=-1).all(dim=-1) & bias_ok  # (batch,)
    out = torch.where(valid[:, None, None], out, rot_mats)

    # Flip the last column for det = +1. det has no bf16/fp16 CPU kernel, use fp32.
    det_in = out.float() if out.dtype in (torch.bfloat16, torch.float16) else out
    flip = (torch.det(det_in) < 0).unsqueeze(-1)
    out[:, :, -1] = torch.where(flip, -out[:, :, -1], out[:, :, -1])
    return out


def get_probe_points(
    origin: torch.Tensor,
    directions: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float = 2.0,
) -> torch.Tensor:
    """Generate probe points at fixed scale intervals along directions.

    Args:
        origin: Center positions of shape (batch, dim).
        directions: Direction vectors of shape (batch, num_points, dim).
        scales: Scalar coefficients of shape (num_probe,).
        probe_radius: Maximum distance multiplier.

    Returns:
        Probe points of shape (batch, num_points, num_probe, dim).
    """
    # origin: (batch, 1, 1, dim)
    origin_exp = origin[:, None, None, :]
    # directions: (batch, num_points, 1, dim)
    directions_exp = directions[:, :, None, :]
    # scales: (1, 1, num_probe, 1)
    scales_exp = scales[None, None, :, None]

    return origin_exp + (directions_exp * probe_radius) * scales_exp


def get_sampled_polytope_vertices(
    origin: torch.Tensor,
    probes: torch.Tensor,
    polytope_vertices: torch.Tensor,
    step_radius: float = 1.0,
    probe_radius: float = 2.0,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate a polytope template and generate deterministic probes.

    Applies a random rotation to the polytope vertices for each particle,
    translates by the step radius, and generates probe points at multiple
    radii along each direction.

    Args:
        origin: Particle positions of shape (batch, dim) or (dim,).
        probes: Scalar probe scales of shape (num_probe,).
        polytope_vertices: Template vertices of shape (num_vertices, dim).
        step_radius: Step distance multiplier.
        probe_radius: Probe distance multiplier.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Tuple of (step_points, probe_points, rotated_vertices):
            - step_points: (batch, num_verts, dim) vertex positions after rotation + translation
            - probe_points: (batch, num_verts, num_probe, dim) probe points along each direction
            - rotated_vertices: (batch, num_verts, dim) rotated vertices before translation
    """
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)
    batch, dim = origin.shape

    # Generate rotation matrices (batch, dim, dim)
    rot_mats = get_random_rotation_matrices(
        batch,
        dim,
        device=origin.device,
        dtype=origin.dtype,
        generator=generator,
    )

    # Apply rotation: R @ v for each vertex
    # rot_mats: (batch, dim, dim), polytope_vertices: (num_verts, dim)
    # Result: (batch, num_verts, dim)
    rotated_vertices = torch.einsum("bji, ni -> bnj", rot_mats, polytope_vertices)

    # Translate step points
    step_points = rotated_vertices * step_radius + origin.unsqueeze(1)

    # Generate probes (deterministic scaling)
    probe_points = get_probe_points(origin, rotated_vertices, probes, probe_radius)

    return step_points, probe_points, rotated_vertices
