"""Regression tests for subspace projections.

- ``HybridSubspace`` reconstruction is exact at saturation
  (``r >= min(d_in, d_out)``).
- 1D parameters (biases) keep ``is_projected=False`` and pass
  through with one coord per element.
- Tied weights produce one ``LayerProjectionSpec`` per storage
  rather than per state_dict key.
- ``AdaptiveSubspace`` step 0 (no displacement history) falls back
  to a seeded random rotation.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from polystep import HybridSubspace, AdaptiveSubspace, ParamLayout


# ---------------------------------------------------------------------------
# 2D.1 HybridSubspace exact reconstruction at saturation
# ---------------------------------------------------------------------------


def test_hybrid_subspace_exact_reconstruction_at_saturation():
    """When r >= min(d_in, d_out), num_coords = d_out*r + r*d_in >= d_out*d_in
    and the projection is surjective onto the parameter space (over-
    determined linear system). For r = d_in = d_out = 4, any 4x4 target
    delta must be reachable by some choice of coords.
    """
    model = nn.Linear(4, 4, bias=False)
    layout = ParamLayout.from_module(model, particle_dim=2)
    hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

    # Single linear layer -> single spec
    assert len(hybrid.specs) == 1
    spec = hybrid.specs[0]
    assert spec.is_projected
    # num_coords = d_out*r + r*d_in = 4*4 + 4*4 = 32 (>= num_params=16)
    assert spec.num_coords == 32, f"expected num_coords=32, got {spec.num_coords}"
    assert spec.num_params == 16

    device = torch.device("cpu")
    projections = hybrid.init_projections(device, torch.float32)
    P = projections[spec.entry_key]  # shape (num_params, num_coords) = (16, 32)
    assert P.shape == (16, 32)

    # P has rank 16 with high probability for random Gaussian projections.
    rank = torch.linalg.matrix_rank(P).item()
    assert rank == 16, (
        f"HybridSubspace projection at saturation should span the full 16-dim "
        f"param space; got rank {rank}"
    )

    # Round-trip a target delta: solve for coords, apply, recover delta exactly.
    target_delta = torch.randn(4, 4)
    target_flat = target_delta.reshape(-1)  # (16,)
    # Solve P @ coords = target_flat (16 eqs, 32 unknowns -> least-norm soln)
    coords = torch.linalg.lstsq(P, target_flat).solution  # shape (32,)

    base_sd = {spec.entry_key: torch.zeros(4, 4)}
    perturbed = hybrid.apply_perturbation(projections, base_sd, coords)
    recovered = perturbed[spec.entry_key]

    assert torch.allclose(recovered, target_delta, atol=1e-4), (
        f"saturated HybridSubspace failed to reconstruct target delta exactly; "
        f"max diff {(recovered - target_delta).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2D.2 Bias handling: 1D params stay full-dim
# ---------------------------------------------------------------------------


def test_hybrid_subspace_bias_is_full_dim_identity():
    """Biases (1D params) should not be compressed: the spec carries
    is_projected=False and num_coords == num_params, so each coord
    drives exactly one bias element (identity mapping)."""
    model = nn.Linear(4, 8, bias=True)  # weight 8x4, bias (8,)
    layout = ParamLayout.from_module(model, particle_dim=2)
    hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

    bias_specs = [s for s in hybrid.specs if s.entry_key == "bias"]
    assert len(bias_specs) == 1
    spec = bias_specs[0]

    assert not spec.is_projected, "bias must not be projected"
    assert spec.num_params == 8
    assert spec.num_coords == 8, (
        f"bias must keep full dim (1 coord per element); got num_coords={spec.num_coords}"
    )

    # Drive only the bias slice and verify per-element correspondence.
    device = torch.device("cpu")
    projections = hybrid.init_projections(device, torch.float32)
    coords = torch.zeros(hybrid.subspace_dim)
    delta_bias = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    coords[spec.flat_start:spec.flat_end] = delta_bias

    base_sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    perturbed = hybrid.apply_perturbation(projections, base_sd, coords)

    assert torch.equal(perturbed["bias"], delta_bias), (
        f"bias chunk did not pass through identity; got {perturbed['bias']}"
    )


# ---------------------------------------------------------------------------
# 2D.3 Tied weights deduplicated through HybridSubspace
# ---------------------------------------------------------------------------


class _TiedHead(nn.Module):
    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embedding.weight


def test_hybrid_subspace_dedupes_tied_weights():
    """ParamLayout deduplicates tied weights to a single canonical entry,
    and HybridSubspace builds exactly one LayerProjectionSpec per entry,
    so the tied embedding/lm_head pair is projected ONCE rather than
    twice."""
    model = _TiedHead(vocab=8, dim=4)
    layout = ParamLayout.from_module(model, particle_dim=2)

    # Layout: only embedding.weight is canonical; lm_head.weight is aliased.
    canonical_keys = [e.key for e in layout.entries]
    assert "embedding.weight" in canonical_keys
    assert "lm_head.weight" not in canonical_keys

    hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)

    # Hybrid has exactly len(layout.entries) specs - no duplicate spec for
    # the tied weight.
    assert len(hybrid.specs) == len(layout.entries)
    spec_keys = [s.entry_key for s in hybrid.specs]
    assert spec_keys.count("embedding.weight") == 1


# ---------------------------------------------------------------------------
# 2D.4 AdaptiveSubspace step-0 fallback + determinism
# ---------------------------------------------------------------------------


def test_adaptive_subspace_step0_rotation_is_deterministic_with_seed():
    """At iteration 0 the displacement history is empty so AdaptiveSubspace
    falls back to a random rotation. With a seeded torch.Generator the
    fallback must be reproducible."""
    full_dim = 32
    subspace_dim = 8

    def make_proj():
        gen = torch.Generator(device="cpu").manual_seed(123)
        sub = AdaptiveSubspace(full_dim=full_dim, subspace_dim=subspace_dim)
        return sub.init_projection(generator=gen)

    p1 = make_proj()
    p2 = make_proj()
    assert torch.equal(p1, p2), (
        "AdaptiveSubspace.init_projection is not seed-deterministic"
    )

    # Now perform a "rotation" with empty history (None passed in).
    # _rotate_random must be invoked.
    sub = AdaptiveSubspace(full_dim=full_dim, subspace_dim=subspace_dim)
    gen = torch.Generator(device="cpu").manual_seed(456)
    new_proj = sub.rotate(
        projection=p1,
        step=0,
        total_steps=100,
        displacement_history=None,
        generator=gen,
    )
    # Orthogonal columns
    gram = new_proj.T @ new_proj
    assert torch.allclose(
        gram, torch.eye(subspace_dim), atol=1e-5,
    ), "step-0 random rotation should still produce orthonormal columns"
