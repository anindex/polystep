#!/usr/bin/env sage -python
"""A2: generic linear sections of the discontinuity set.

The convergence proof needs the probes to miss the discontinuity set D almost
surely.  Probes for one particle live in a d_p-dimensional affine plane inside
R^d, so what matters is not "D has measure zero in R^d" (true but useless here)
but "D meets the particle plane in a set of measure zero *within that plane*".

For D definable with dim D <= d-1, o-minimal generic-section theory gives

      dim(D ∩ plane) <= dim D + d_p - d <= d_p - 1        for a generic plane,

which is exactly measure zero inside the plane -- and it is a statement about the
whole plane, so it holds uniformly over every iterate that plane contains.  That
removes the circularity of an iterate-dependent argument.

HOW THIS IS MEASURED, AND HOW IT USED TO BE MIS-MEASURED
--------------------------------------------------------
An earlier version of this script reported "0.00000 of probes on D" for the
generic plane, from a hard tolerance 1e-9 applied to points sampled on a fixed
extent.  That number is a TUBE VOLUME, not a dimension: it is roughly
tol / (extent * |B^T n|), so it can be driven anywhere between 0 and 1 by moving
either knob, and more samples only stabilise the artifact.  Section (c) below
demonstrates the artifact explicitly rather than relying on it.

The quantity that actually distinguishes containment from transversal
intersection is the RANK of the wall's normal restricted to the plane.  For a
wall {x : n.x = c} and a plane base + span(B):

    rank(B^T n) = 1  ->  D ∩ plane is an affine subspace of dimension d_p - 1:
                         a PROPER subspace of the plane, hence measure zero in it,
                         uniformly in the base point.
    rank(B^T n) = 0  ->  the plane is PARALLEL to the wall.  The section is then
                         either empty or the whole plane, decided entirely by the
                         base point.  No genericity is available and the argument
                         becomes iterate-dependent again.

Sections:
  (a) generic (Gaussian-spanned) plane: rank 1 for all three headline geometries.
  (b) axis-aligned coordinate plane: rank 0 AND base on the wall, i.e. the plane
      lies inside D.  Full-space mode therefore needs a separate transversality
      hypothesis; this half was already honest and is kept.
  (c) the tube statistic, shown to be an artifact.
  (d) the REAL projection.  The lemma assumes a global i.i.d. dense projection.
      ``polystep.hybrid_subspace.HybridSubspace`` is block-diagonal per layer, with
      IDENTITY blocks wherever num_coords == num_params (all 1D biases/LayerNorms,
      and any weight matrix whose rank budget does not compress it).  Section (d)
      builds a real HybridSubspace and reports transversality per block type.
"""

import numpy as np
import torch
import torch.nn as nn

from polystep.hybrid_subspace import HybridSubspace, create_hybrid_blocks
from polystep.transform import ParamLayout

RNG = np.random.default_rng(0)

D_AMBIENT = 6
D_PARTICLE = 2


# --- the three headline discontinuity geometries -----------------------------
# Each is a wall {x : n.x = c} of codimension 1 (the staircase locally so).

GEOMETRIES = {
    # LIF spike threshold: a coordinate hyperplane.
    "lif_threshold": (np.eye(D_AMBIENT)[2], 0.0, lambda x: x[..., 2]),
    # Argmax / hard-MoE routing: a Voronoi-cell boundary, a coordinate difference.
    "argmax_voronoi": (np.eye(D_AMBIENT)[0] - np.eye(D_AMBIENT)[1], 0.0,
                       lambda x: x[..., 0] - x[..., 1]),
    # floor() staircase: union of integer level sets, locally a coordinate hyperplane.
    "floor_staircase": (np.eye(D_AMBIENT)[3], 0.0, lambda x: np.sin(np.pi * x[..., 3])),
}

# An axis-aligned plane contained in D, for each geometry: (base point, axes).
CONTAINED_AXIS_PLANES = {
    "lif_threshold": (np.zeros(D_AMBIENT), (0, 1)),      # x_2 = 0 identically
    "argmax_voronoi": (np.zeros(D_AMBIENT), (2, 3)),     # x_0 = x_1 = 0 identically
    "floor_staircase": (np.zeros(D_AMBIENT), (0, 1)),    # x_3 = 0 identically
}


def section_dim(normal, offset, base, basis, tol=1e-10):
    """Dimension of (wall ∩ plane) inside the plane, and a label.

    ``basis`` is (d, d_p).  Returns (dim, label) where dim is -1 for an empty
    section.  This is exact linear algebra, not a sampling statistic.
    """
    d_p = basis.shape[1]
    restricted = basis.T @ normal                      # the constraint, in plane coords
    if np.linalg.norm(restricted) > tol:
        return d_p - 1, "proper subspace"
    return (d_p, "plane inside wall") if abs(normal @ base - offset) <= tol else (-1, "empty")


def axis_basis(axes):
    b = np.zeros((D_AMBIENT, len(axes)))
    for k, a in enumerate(axes):
        b[a, k] = 1.0
    return b


def tube_fraction(g, base, basis, n, rng, tol, extent):
    """The OLD statistic: fraction of sampled plane points with |g| <= tol.

    Kept only to show that it is a function of tol/extent, not of dimension.
    """
    coeffs = rng.uniform(-extent, extent, size=(n, basis.shape[1]))
    return float(np.mean(np.abs(g(base + coeffs @ basis.T)) <= tol))


def check_dimensions():
    print("(a,b) dimension of D ∩ plane, inside a plane of dimension "
          f"{D_PARTICLE} (measure zero in the plane iff dim <= {D_PARTICLE - 1})")
    for name, (normal, offset, g) in GEOMETRIES.items():
        base = np.zeros(D_AMBIENT)
        assert abs(g(base[None, :])[0]) <= 1e-12, name          # base is on D

        # (a) generic plane: Gaussian-spanned, through a point ON D.
        dims = []
        for _ in range(200):
            basis = RNG.standard_normal((D_AMBIENT, D_PARTICLE))
            dim, label = section_dim(normal, offset, base, basis)
            dims.append(dim)
        assert all(d == D_PARTICLE - 1 for d in dims), (name, set(dims))

        # (b) axis-aligned plane: contained in D.  Exact, not a sampling claim.
        ab, axes = CONTAINED_AXIS_PLANES[name]
        adim, alabel = section_dim(normal, offset, ab, axis_basis(axes))
        assert adim == D_PARTICLE, (name, adim)
        # ...and the containment is algebraic: g vanishes identically on the plane.
        pts = ab + RNG.uniform(-1e3, 1e3, size=(20_000, len(axes))) @ axis_basis(axes).T
        assert np.max(np.abs(g(pts))) <= 1e-12, name

        print(f"  {name:16s} generic plane: dim {D_PARTICLE - 1} ({label}) | "
              f"axis-aligned plane: dim {adim} ({alabel})")


def check_tube_artifact():
    """The retired statistic reports whatever tol/extent you choose."""
    n = 20_000
    normal, _, g = GEOMETRIES["lif_threshold"]
    base = np.zeros(D_AMBIENT)
    basis = RNG.standard_normal((D_AMBIENT, D_PARTICLE))

    wide = tube_fraction(g, base, basis, n, RNG, tol=1e-9, extent=1.0)
    narrow = tube_fraction(g, base, basis, n, RNG, tol=1e-9, extent=1e-9)
    loose = tube_fraction(g, base, basis, n, RNG, tol=1e-1, extent=1.0)
    print(f"\n(c) tube statistic on the SAME generic plane (dim of section is {D_PARTICLE - 1} throughout):")
    print(f"      tol=1e-9, extent=1e+0 -> {wide:.5f}   <- the number the old script reported")
    print(f"      tol=1e-9, extent=1e-9 -> {narrow:.5f}")
    print(f"      tol=1e-1, extent=1e+0 -> {loose:.5f}")
    print("    Same geometry, three answers: a tube volume, not evidence of measure zero.")
    assert wide < 1e-3, wide
    assert narrow > 0.2, narrow          # the "0.00000" is pure tolerance/extent
    assert loose > 0.02, loose


# --- (d) the real HybridSubspace projection ----------------------------------


def real_projection_planes(particle_dim=4):
    """(ambient dim, {block name: (is_projected, layer slice, [plane bases])}).

    Uses the real ``HybridSubspace`` + ``create_hybrid_blocks`` code path.  A plane
    basis column is obtained by pushing a unit subspace coordinate through
    ``apply_perturbation``, so it is the projection the optimizer actually uses.
    """
    model = nn.Sequential(nn.Linear(16, 12), nn.Tanh(), nn.Linear(12, 8)).to(torch.float64)
    layout = ParamLayout.from_module(model)
    hyb = HybridSubspace.from_layout(layout, rank=2)
    proj = hyb.init_projections(torch.device("cpu"), torch.float64)
    keys = [s.entry_key for s in hyb.specs]
    base_sd = {k: torch.zeros_like(model.state_dict()[k], dtype=torch.float64) for k in keys}
    sizes = [int(base_sd[k].numel()) for k in keys]
    d_ambient = sum(sizes)

    def ambient_column(j):
        coords = torch.zeros(hyb.subspace_dim, dtype=torch.float64)
        coords[j] = 1.0
        sd = hyb.apply_perturbation(proj, base_sd, coords)
        return torch.cat([sd[k].reshape(-1) for k in keys]).numpy()

    cols = np.stack([ambient_column(j) for j in range(hyb.subspace_dim)], axis=1)

    starts = np.cumsum([0] + sizes)
    layer_slice = {k: slice(int(starts[i]), int(starts[i + 1])) for i, k in enumerate(keys)}

    out = {}
    for block, spec in zip(create_hybrid_blocks(hyb, particle_dim), hyb.specs):
        # No inter-layer padding in this layout, so block ranges match spec ranges.
        assert (block.flat_start, block.flat_end) == (spec.flat_start, spec.flat_end), block.name
        planes = [cols[:, s:s + particle_dim]
                  for s in range(spec.flat_start, spec.flat_end, particle_dim)]
        out[block.name] = (spec.is_projected, layer_slice[block.name], planes)
    return d_ambient, out


def check_real_projection():
    d_ambient, blocks = real_projection_planes()
    axes = np.eye(d_ambient)

    print(f"\n(d) real HybridSubspace, ambient dim {d_ambient}. "
          "Fraction of walls each particle plane is TRANSVERSAL to (rank(B^T n) = 1):")
    print(f"    {'block':10s} {'proj':>5} {'layer dim':>10} {'coord-axis walls':>17} "
          f"{'argmax walls':>13} {'dense normals':>14}")

    pairs = axes[RNG.choice(d_ambient, 400)] - axes[RNG.choice(d_ambient, 400)]
    dense = RNG.standard_normal((200, d_ambient))
    results = {}
    for name, (is_proj, lsl, planes) in blocks.items():
        def frac(normals, planes=planes):
            r = [np.linalg.norm(B.T @ normals.T, axis=0) > 1e-9 for B in planes]
            return float(np.mean(r))

        f_axis, f_pair, f_dense = frac(axes), frac(pairs), frac(dense)
        results[name] = (is_proj, f_axis)
        layer_dim = lsl.stop - lsl.start
        print(f"    {name:10s} {str(is_proj):>5} {layer_dim:10d} {f_axis:17.4f} "
              f"{f_pair:13.4f} {f_dense:14.4f}")

        # Dense (fully generic) normals are transversal to every block, always.
        assert f_dense == 1.0, (name, f_dense)
        if is_proj:
            # A Gaussian block sees its whole layer and nothing else.
            assert abs(f_axis - layer_dim / d_ambient) < 1e-9, (name, f_axis)
        else:
            # An identity block sees only its own particle_dim coordinates.
            assert abs(f_axis - 4 / d_ambient) < 1e-9, (name, f_axis)

    # The lemma's hypothesis: one global i.i.d. dense projection.  Transversal to
    # every coordinate axis, so the conclusion holds for the paper's own geometries.
    global_p = RNG.standard_normal((d_ambient, 4))
    assert np.all(np.linalg.norm(global_p.T @ axes.T, axis=0) > 1e-9)
    print(f"    {'global iid':10s} {'True':>5} {d_ambient:10d} {1.0:17.4f} {1.0:13.4f} {1.0:14.4f}"
          "   <- what the lemma assumes")

    proj_f = [f for is_p, f in results.values() if is_p]
    ident_f = [f for is_p, f in results.values() if not is_p]
    print("\n    Weight (projected) blocks reach "
          f"{min(proj_f):.1%}-{max(proj_f):.1%} of coordinate-axis walls; "
          f"bias (identity) blocks reach {max(ident_f):.1%}.")
    print("    For every wall a block is NOT transversal to, the plane is parallel to it,")
    print("    so the section is empty or the entire plane depending on the base point.")
    print("    That is exactly the iterate-dependent case the lemma was meant to remove,")
    print("    and the paper's three headline geometries are all coordinate-aligned.")
    assert max(ident_f) < min(proj_f), (ident_f, proj_f)


def demo():
    check_dimensions()
    check_tube_artifact()
    check_real_projection()


if __name__ == "__main__":
    demo()
    print("OK")
