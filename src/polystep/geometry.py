"""Geometry module: polytope templates, rotation matrices, and probe generation.

Combines polytope vertex generators (orthoplex, simplex, cube), random rotation
via QR decomposition (Mezzadri method) or analytical 2D formula, and deterministic
probe point generation into a single module for PolyStep exploration directions.
"""
import math
from typing import Callable, Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Section 1: Polytope vertex templates
# ---------------------------------------------------------------------------


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
    points = points - ((math.sqrt(dim + 1) + 1) / math.sqrt(dim ** 3))

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
    n_vertices = 2 ** dim

    indices = torch.arange(n_vertices, dtype=torch.int32, device=device).unsqueeze(1)
    shifts = torch.arange(dim, dtype=torch.int32, device=device).unsqueeze(0)

    bits = (indices >> shifts) & 1
    # Resolve dtype for the float conversion (default to float32 if None)
    float_dtype = dtype if dtype is not None else torch.float32
    signs = 1.0 - 2.0 * bits.to(float_dtype)

    points = signs / math.sqrt(dim)
    return points * radius


POLYTOPE_MAP: Dict[str, Callable] = {
    'cube': get_cube_vertices,
    'orthoplex': get_orthoplex_vertices,
    'simplex': get_simplex_vertices,
}

POLYTOPE_NUM_VERTICES_MAP: Dict[str, Callable[[int], int]] = {
    'cube': lambda dim: 2 ** dim,
    'orthoplex': lambda dim: 2 * dim,
    'simplex': lambda dim: dim + 1,
}


# ---------------------------------------------------------------------------
# Section 2: Rotation matrices
# ---------------------------------------------------------------------------


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
    # QR decomposition requires FP32 on CPU (BF16 not supported for geqrf_cpu)
    # Generate in FP32, compute QR, then convert to target dtype
    # Resolve None device/dtype to concrete values for comparison
    resolved_device = device if device is not None else torch.device('cpu')
    resolved_dtype = dtype if dtype is not None else torch.float32
    device_type = resolved_device.type if hasattr(resolved_device, 'type') else str(resolved_device)
    is_cpu = device_type == 'cpu'
    needs_fp32_qr = resolved_dtype == torch.bfloat16 and is_cpu
    compute_dtype = torch.float32 if needs_fp32_qr else resolved_dtype
    compute_device = 'cpu' if needs_fp32_qr else resolved_device

    # Generator device must match tensor device for ``randn``. If the user
    # supplied a CUDA generator but we have to run QR on CPU (bfloat16
    # fallback), sample on the generator's device first, then move the
    # result to the QR compute device. This preserves reproducibility.
    gen_for_randn = generator
    sample_device = compute_device
    needs_post_move = False
    if generator is not None and hasattr(generator, 'device'):
        gen_device_type = generator.device.type if hasattr(generator.device, 'type') else str(generator.device)
        compute_device_type = 'cpu' if needs_fp32_qr else device_type
        if gen_device_type != compute_device_type:
            sample_device = generator.device
            needs_post_move = True

    Z = torch.randn(batch, dim, dim, device=sample_device, dtype=compute_dtype, generator=gen_for_randn)
    if needs_post_move:
        Z = Z.to(device=compute_device)
    Q, R = torch.linalg.qr(Z)

    # Sign correction for Haar measure (Mezzadri method)
    d = torch.diagonal(R, dim1=-2, dim2=-1)  # (batch, dim)
    phases = torch.sign(d)                     # (batch, dim)
    Q = Q * phases.unsqueeze(-2)               # (batch, 1, dim) * (batch, dim, dim)

    # Ensure det = +1 (SO(n) not just O(n)): flip first column if det = -1
    dets = torch.det(Q)                        # (batch,)
    flip = torch.sign(dets).unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
    Q = Q.clone()
    Q[:, :, 0] = Q[:, :, 0] * flip.squeeze(-1)

    # Convert to target dtype and device
    Q = Q.to(device=device, dtype=dtype)

    return Q


def get_sobol_rotation_matrices(
    batch: int,
    dim: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    engine: Optional[torch.quasirandom.SobolEngine] = None,
) -> torch.Tensor:
    """Generate rotation matrices from Sobol quasi-random sequences.

    Low-discrepancy sequences ensure better coverage of SO(d) over
    multiple calls than purely random rotations.

    Args:
        batch: Number of rotation matrices.
        dim: Dimension of the rotation.
        device: Target device.
        dtype: Target dtype.
        engine: Optional pre-initialized SobolEngine (for stateful sequencing).

    Returns:
        Rotation matrices of shape (batch, dim, dim) in SO(dim).
    """
    if dim == 2:
        # 2D: Sobol angle in [0, 2pi)
        if engine is None:
            engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
        uniform = engine.draw(batch).to(device=device, dtype=dtype)  # (batch, 1)
        angles = uniform[:, 0] * 2 * math.pi
        return get_rotation_matrix_2d(angles)

    # Higher dimensions: Sobol-driven QR (Mezzadri method with quasi-random input)
    sobol_dim = dim * dim
    if engine is None:
        engine = torch.quasirandom.SobolEngine(dimension=sobol_dim, scramble=True)
    uniform = engine.draw(batch)  # (batch, dim*dim) in [0, 1]

    # Transform uniform to normal via inverse CDF (erfinv)
    uniform = uniform.clamp(1e-6, 1 - 1e-6)
    normal = torch.erfinv(2 * uniform - 1) * math.sqrt(2)
    normal = normal.to(device=device, dtype=dtype)

    Z = normal.reshape(batch, dim, dim)
    Q, R = torch.linalg.qr(Z)

    # Mezzadri sign correction
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    phases = torch.sign(d)
    Q = Q * phases.unsqueeze(-2)

    # Ensure det = +1
    dets = torch.det(Q)
    flip = torch.sign(dets).unsqueeze(-1).unsqueeze(-1)
    Q = Q.clone()
    Q[:, :, 0] = Q[:, :, 0] * flip.squeeze(-1)

    return Q


# ---------------------------------------------------------------------------
# Section 3: Probe point generation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section 4: Main sampling pipeline
# ---------------------------------------------------------------------------


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

    # 1. Generate rotation matrices (batch, dim, dim)
    rot_mats = get_random_rotation_matrices(
        batch, dim, device=origin.device, dtype=origin.dtype, generator=generator,
    )

    # 2. Apply rotation: R @ v for each vertex
    # rot_mats: (batch, dim, dim), polytope_vertices: (num_verts, dim)
    # Result: (batch, num_verts, dim)
    rotated_vertices = torch.einsum('bji, ni -> bnj', rot_mats, polytope_vertices)

    # 3. Translate step points
    step_points = rotated_vertices * step_radius + origin.unsqueeze(1)

    # 4. Generate probes (deterministic scaling)
    probe_points = get_probe_points(origin, rotated_vertices, probes, probe_radius)

    return step_points, probe_points, rotated_vertices
