"""Regression tests for vmap-safe layers and parameter layout.

Covers:

- ``ParamLayout.from_module`` deduplicates tied weights and emits a
  logger.info so the dedup is not silent
- ``dp`` divisibility padding round-trips through ``flatten`` /
  ``unflatten`` without leaking padding bytes into the state_dict
- upstream ``nn.LSTM`` fails under ``torch.vmap`` (PyTorch #105982)
- ``VmapSafeMultiHeadAttention``: scales by ``sqrt(head_dim)``,
  treats bool ``attn_mask`` as mask-fill(-inf), and raises
  NotImplementedError for ``kdim``, ``vdim``, ``add_bias_kv``,
  ``add_zero_attn``, ``batch_first=False``, ``need_weights``,
  ``is_causal``
- ``VmapSafeLSTM``: raises NotImplementedError for ``bidirectional``,
  ``proj_size``, ``batch_first=False``, and ``PackedSequence`` input
- ``state_dict`` round-trip is bitwise-identical for BF16 weights with
  tied params
"""
from __future__ import annotations

import logging
import math
import warnings

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from polystep.layers import VmapSafeMultiHeadAttention, VmapSafeLSTM
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# ParamLayout dedup of tied weights
# ---------------------------------------------------------------------------


class _TiedHead(nn.Module):
    """embedding.weight = lm_head.weight: classic transformer tie."""

    def __init__(self, vocab=8, dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embedding.weight  # tie

    def forward(self, ids):
        h = self.embedding(ids)
        return self.lm_head(h)


def test_tied_weights_deduplicated_with_info_log(caplog):
    """ParamLayout.from_module must deduplicate tied weights into a single
    flat-param slot and emit a logger.info so users know the tie was
    detected.

    Pre-fix: dedup happens but is silent.
    Post-fix: a single info-level log line names the tied keys.
    """
    model = _TiedHead(vocab=8, dim=4)
    with caplog.at_level(logging.INFO, logger="polystep.transform"):
        layout = ParamLayout.from_module(model, particle_dim=2)

    # Embedding.weight is shared with lm_head.weight: one canonical entry.
    canonical_keys = [e.key for e in layout.entries]
    assert "embedding.weight" in canonical_keys
    assert "lm_head.weight" not in canonical_keys, (
        "lm_head.weight should be aliased to embedding.weight, not a "
        "separate flat-param entry"
    )

    # The canonical entry must record the alias.
    canonical = next(e for e in layout.entries if e.key == "embedding.weight")
    assert "lm_head.weight" in canonical.shared_with

    # And the dedup must be visible in the log.
    info_msgs = [r.getMessage().lower() for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        "tied" in m or "shared" in m or "alias" in m or "dedup" in m
        for m in info_msgs
    ), (
        f"expected an INFO log mentioning the tied weight; got: {info_msgs}"
    )


def test_tied_weights_unflatten_aliased():
    """After unflatten, both embedding.weight and lm_head.weight refer to
    the same tensor object."""
    model = _TiedHead(vocab=8, dim=4)
    layout = ParamLayout.from_module(model, particle_dim=2)
    flat = layout.flatten(model)
    sd = layout.unflatten(flat)
    assert sd["embedding.weight"].data_ptr() == sd["lm_head.weight"].data_ptr()


# ---------------------------------------------------------------------------
# dp divisibility padding round-trip
# ---------------------------------------------------------------------------


def test_dp_padding_round_trip_does_not_mutate_state_dict():
    """flatten -> unflatten must produce a state_dict identical to the
    original (within dtype rounding) and must not expose padding bytes
    as state_dict keys."""
    # 7-element model + particle_dim=2 -> requires 1 byte of padding
    # (sanity check: a single 4-param linear has no padding)
    _ = nn.Linear(3, 1, bias=True)
    model2 = nn.Sequential(nn.Linear(3, 1, bias=True), nn.Linear(1, 1, bias=False))
    # 4 + 1 = 5 params; padded to 6 with particle_dim=2 -> 1 element of padding

    layout = ParamLayout.from_module(model2, particle_dim=2)
    assert layout.total_params == 5
    assert layout.padded_size == 6, (
        f"expected padded_size=6 for 5 params with particle_dim=2; "
        f"got {layout.padded_size}"
    )

    flat = layout.flatten(model2)
    assert flat.numel() == 6
    sd = layout.unflatten(flat)

    # Padding must not appear as a state_dict key.
    expected_keys = {"0.weight", "0.bias", "1.weight"}
    assert set(sd.keys()) == expected_keys, (
        f"state_dict keys leaked padding: {set(sd.keys()) - expected_keys}"
    )

    # Values round-trip exactly.
    orig = model2.state_dict()
    for k in expected_keys:
        assert torch.allclose(sd[k], orig[k]), f"value mismatch on {k}"


# ---------------------------------------------------------------------------
# nn.LSTM under vmap (issue #105982) and nn.MultiheadAttention float
# mask (issue #107084) - record the upstream pitfalls so a future PyTorch
# release that fixes them will fail this test and prompt removal of the
# VmapSafe layers.
# ---------------------------------------------------------------------------


def test_upstream_nn_lstm_fails_under_vmap():
    """Documented PyTorch issue #105982. If this ever passes, the
    workaround in VmapSafeLSTM can be removed."""
    lstm = nn.LSTM(4, 8, num_layers=1, batch_first=True)
    params = {k: v.detach() for k, v in lstm.named_parameters()}
    buffers = {k: v.detach() for k, v in lstm.named_buffers()}
    x = torch.randn(2, 5, 4)

    def call(p):
        return functional_call(lstm, {**p, **buffers}, (x,))[0]

    # Stack 3 candidate parameter sets.
    stacked = {k: torch.stack([v, v, v], dim=0) for k, v in params.items()}
    try:
        vmap(call, in_dims=(0,))(stacked)
        pytest.skip(
            "nn.LSTM now works under vmap (PyTorch fixed #105982); "
            "consider removing VmapSafeLSTM."
        )
    except Exception:
        pass  # expected pre-fix in upstream


def test_vmap_safe_lstm_works_under_vmap():
    """The drop-in VmapSafeLSTM must succeed where nn.LSTM fails."""
    lstm = VmapSafeLSTM(input_size=4, hidden_size=8, num_layers=1)
    params = {k: v.detach() for k, v in lstm.named_parameters()}
    buffers = {k: v.detach() for k, v in lstm.named_buffers()}
    x = torch.randn(2, 5, 4)

    def call(p):
        return functional_call(lstm, {**p, **buffers}, (x,))[0]

    stacked = {k: torch.stack([v, v, v], dim=0) for k, v in params.items()}
    out = vmap(call, in_dims=(0,))(stacked)
    assert out.shape == (3, 2, 5, 8)


# ---------------------------------------------------------------------------
# sqrt(head_dim) scale
# ---------------------------------------------------------------------------


def test_vmap_safe_attention_scales_by_sqrt_head_dim():
    embed_dim, num_heads = 64, 8
    head_dim = embed_dim // num_heads  # 8
    attn = VmapSafeMultiHeadAttention(embed_dim, num_heads)
    assert math.isclose(attn.scale, 1.0 / math.sqrt(head_dim))
    assert not math.isclose(attn.scale, 1.0 / math.sqrt(embed_dim))


# ---------------------------------------------------------------------------
# bool attn_mask must mask-fill (-inf), not add (P1 BUG)
# ---------------------------------------------------------------------------


def test_vmap_safe_attention_bool_mask_zeros_attention():
    """Upstream nn.MultiheadAttention treats bool=True in attn_mask as
    "do not attend" (mask-fill -inf). VmapSafeMultiHeadAttention used to
    silently add the bool tensor, multiplying logits by True/False
    instead of masking. Post-fix: bool mask matches upstream semantics.
    """
    torch.manual_seed(0)
    embed_dim, num_heads, B, T = 8, 2, 1, 4
    attn = VmapSafeMultiHeadAttention(embed_dim, num_heads, bias=False)

    x = torch.randn(B, T, embed_dim)
    # Mask the second key position for every query: shape (T, T)
    bool_mask = torch.zeros(T, T, dtype=torch.bool)
    bool_mask[:, 1] = True   # mask out key index 1

    # Compute attention weights manually and check key-1 weight is zero.
    Q = attn.W_q(x).view(B, T, num_heads, embed_dim // num_heads).transpose(1, 2)
    K = attn.W_k(x).view(B, T, num_heads, embed_dim // num_heads).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
    # Replicate the post-fix mask logic:
    expected_scores = scores.clone()
    expected_scores = expected_scores.masked_fill(bool_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    expected_weights = torch.softmax(expected_scores, dim=-1)

    # Now run the actual layer with the bool mask and inspect attention weights.
    # Easiest: run forward and check the output respects the mask via a probe.
    out = attn(x, x, x, attn_mask=bool_mask)

    # Replay attention manually with the post-fix semantics and compare.
    V = attn.W_v(x).view(B, T, num_heads, embed_dim // num_heads).transpose(1, 2)
    expected_context = torch.matmul(expected_weights, V)
    expected_context = expected_context.transpose(1, 2).reshape(B, T, embed_dim)
    expected_out = attn.W_o(expected_context)

    assert torch.allclose(out, expected_out, atol=1e-5), (
        "VmapSafeMultiHeadAttention does not treat bool attn_mask as "
        "mask-fill(-inf). Post-fix path expected a zero-weight at masked "
        "key positions; got mismatch."
    )


# ---------------------------------------------------------------------------
# NotImplementedError for unsupported configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(kdim=16),
    dict(vdim=16),
    dict(add_bias_kv=True),
    dict(add_zero_attn=True),
    dict(batch_first=False),
])
def test_vmap_safe_attention_raises_on_unsupported_kwargs(kwargs):
    """Constructor must raise a clear NotImplementedError for any
    unsupported nn.MultiheadAttention argument."""
    with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
        VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4, **kwargs)


@pytest.mark.parametrize("forward_kwargs", [
    dict(need_weights=True),
    dict(is_causal=True),
])
def test_vmap_safe_attention_raises_on_unsupported_forward_kwargs(forward_kwargs):
    """Forward must reject need_weights / is_causal explicitly."""
    attn = VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
        attn(x, x, x, **forward_kwargs)


# ---------------------------------------------------------------------------
# VmapSafeLSTM raises on unsupported configs / PackedSequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    dict(bidirectional=True),
    dict(proj_size=4),
    dict(batch_first=False),
])
def test_vmap_safe_lstm_raises_on_unsupported_kwargs(kwargs):
    with pytest.raises(NotImplementedError, match="VmapSafeLSTM"):
        VmapSafeLSTM(input_size=4, hidden_size=8, **kwargs)


def test_vmap_safe_lstm_raises_on_packed_sequence():
    lstm = VmapSafeLSTM(input_size=4, hidden_size=8)
    x = torch.randn(3, 5, 4)
    lengths = torch.tensor([5, 3, 2])
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    with pytest.raises(NotImplementedError, match="PackedSequence"):
        lstm(packed)


# ---------------------------------------------------------------------------
# state_dict round-trip BF16 + tied weights
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip_bf16_with_tied_weights():
    """flatten -> unflatten -> load_state_dict -> flatten must reproduce
    the original flat tensor bit-for-bit on BF16 weights, with tied
    weights aliased through the round-trip."""
    model = _TiedHead(vocab=8, dim=4).to(dtype=torch.bfloat16)
    layout = ParamLayout.from_module(model, particle_dim=2)

    flat1 = layout.flatten(model)
    sd1 = layout.unflatten(flat1)

    # Reload into a fresh model with identical architecture.
    fresh = _TiedHead(vocab=8, dim=4).to(dtype=torch.bfloat16)
    # Drop strict=True since aliased keys may appear extra
    fresh.load_state_dict(sd1, strict=False)
    flat2 = layout.flatten(fresh)

    assert torch.equal(flat1, flat2), (
        f"BF16 round-trip not bitwise stable: max diff "
        f"{(flat1.float() - flat2.float()).abs().max().item():.3e}"
    )

    # Aliased: only one flat-param slot for both keys.
    assert sd1["embedding.weight"].data_ptr() == sd1["lm_head.weight"].data_ptr()
