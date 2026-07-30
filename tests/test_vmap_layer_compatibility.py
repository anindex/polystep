"""Contracts of the vmap-safe layers and the parameter layout.

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

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from polystep.layers import VmapSafeMultiHeadAttention, VmapSafeLSTM
from polystep.transform import ParamLayout


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
    flat-param slot and emit a log message so users know the tie was detected.
    """
    model = _TiedHead(vocab=8, dim=4)
    with caplog.at_level(logging.DEBUG, logger="polystep.transform"):
        layout = ParamLayout.from_module(model, particle_dim=2)

    # Embedding.weight is shared with lm_head.weight: one canonical entry.
    canonical_keys = [e.key for e in layout.entries]
    assert "embedding.weight" in canonical_keys
    assert "lm_head.weight" not in canonical_keys, (
        "lm_head.weight should be aliased to embedding.weight, not a separate flat-param entry"
    )

    # The canonical entry must record the alias.
    canonical = next(e for e in layout.entries if e.key == "embedding.weight")
    assert "lm_head.weight" in canonical.shared_with

    # The dedup must be visible in the log at any level.
    all_msgs = [r.getMessage().lower() for r in caplog.records]
    assert any("tied" in m or "shared" in m or "alias" in m or "dedup" in m for m in all_msgs), (
        f"expected a log message mentioning the tied weight; got: {all_msgs}"
    )


def test_tied_weights_unflatten_aliased():
    """After unflatten, both embedding.weight and lm_head.weight refer to
    the same tensor object."""
    model = _TiedHead(vocab=8, dim=4)
    layout = ParamLayout.from_module(model, particle_dim=2)
    flat = layout.flatten(model)
    sd = layout.unflatten(flat)
    assert sd["embedding.weight"].data_ptr() == sd["lm_head.weight"].data_ptr()


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
    assert layout.padded_size == 6, f"expected padded_size=6 for 5 params with particle_dim=2; got {layout.padded_size}"

    flat = layout.flatten(model2)
    assert flat.numel() == 6
    sd = layout.unflatten(flat)

    # Padding must not appear as a state_dict key.
    expected_keys = {"0.weight", "0.bias", "1.weight"}
    assert set(sd.keys()) == expected_keys, f"state_dict keys leaked padding: {set(sd.keys()) - expected_keys}"

    # Values round-trip exactly.
    orig = model2.state_dict()
    for k in expected_keys:
        assert torch.allclose(sd[k], orig[k]), f"value mismatch on {k}"


def test_upstream_nn_lstm_fails_under_vmap():
    """Documented PyTorch issue #105982, the reason VmapSafeLSTM exists.

    Matching the specific error keeps this honest. A bare ``except Exception: pass``
    passes on any failure, including a typo in the test itself, so it could never fail.
    """
    lstm = nn.LSTM(4, 8, num_layers=1, batch_first=True)
    params = {k: v.detach() for k, v in lstm.named_parameters()}
    buffers = {k: v.detach() for k, v in lstm.named_buffers()}
    x = torch.randn(2, 5, 4)

    def call(p):
        return functional_call(lstm, {**p, **buffers}, (x,))[0]

    stacked = {k: torch.stack([v, v, v], dim=0) for k, v in params.items()}
    with pytest.raises(RuntimeError, match="Batching rule not implemented|does not support|Cannot access data pointer"):
        vmap(call, in_dims=(0,))(stacked)


def test_vmap_safe_attention_scales_by_sqrt_head_dim():
    embed_dim, num_heads = 64, 8
    head_dim = embed_dim // num_heads  # 8
    attn = VmapSafeMultiHeadAttention(embed_dim, num_heads)
    assert math.isclose(attn.scale, 1.0 / math.sqrt(head_dim))
    assert not math.isclose(attn.scale, 1.0 / math.sqrt(embed_dim))


def test_vmap_safe_attention_bool_mask_matches_upstream():
    """A True entry in a bool attn_mask means "do not attend", as in nn.MultiheadAttention.

    Compared against the upstream module rather than a replay of this one's own
    forward: a replay reuses ``attn.scale`` and would cancel an error in it, leaving
    only the mask semantics actually tested.
    """
    torch.manual_seed(0)
    embed_dim, num_heads, B, T = 8, 2, 1, 4
    reference = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
    attn = VmapSafeMultiHeadAttention(embed_dim, num_heads, bias=False)
    with torch.no_grad():
        for layer, w in zip((attn.W_q, attn.W_k, attn.W_v), reference.in_proj_weight.chunk(3, dim=0)):
            layer.weight.copy_(w)
        attn.W_o.weight.copy_(reference.out_proj.weight)

    x = torch.randn(B, T, embed_dim)
    bool_mask = torch.zeros(T, T, dtype=torch.bool)
    bool_mask[:, 1] = True  # key index 1 is unreachable from every query

    expected, _ = reference(x, x, x, attn_mask=bool_mask, need_weights=False)
    torch.testing.assert_close(attn(x, x, x, attn_mask=bool_mask), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kdim=16),
        dict(vdim=16),
        dict(add_bias_kv=True),
        dict(add_zero_attn=True),
        dict(batch_first=False),
    ],
)
def test_vmap_safe_attention_raises_on_unsupported_kwargs(kwargs):
    """Constructor must raise a clear NotImplementedError for any
    unsupported nn.MultiheadAttention argument."""
    with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
        VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4, **kwargs)


@pytest.mark.parametrize(
    "forward_kwargs",
    [
        dict(need_weights=True),
        dict(is_causal=True),
    ],
)
def test_vmap_safe_attention_raises_on_unsupported_forward_kwargs(forward_kwargs):
    """Forward must reject need_weights / is_causal explicitly."""
    attn = VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
        attn(x, x, x, **forward_kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bidirectional=True),
        dict(proj_size=4),
        dict(batch_first=False),
    ],
)
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
        f"BF16 round-trip not bitwise stable: max diff {(flat1.float() - flat2.float()).abs().max().item():.3e}"
    )

    # Aliased: only one flat-param slot for both keys.
    assert sd1["embedding.weight"].data_ptr() == sd1["lm_head.weight"].data_ptr()
