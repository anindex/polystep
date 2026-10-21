"""Unit tests for the vmap-safe layers: correctness and vmap compatibility."""

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator
from polystep.layers import (
    VmapSafeMultiHeadAttention,
    VmapSafeLSTMCell,
    VmapSafeLSTM,
)


class TestVmapSafeMultiHeadAttention:
    @pytest.fixture
    def attn(self):
        """Create a test attention module."""
        return VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)

    def test_cross_attention_keeps_query_length(self, attn):
        """The upstream comparison below is self-attention only; cross-attention has to
        keep the query's sequence length, and the gradient has to reach the projections."""
        torch.manual_seed(0)
        query, key = torch.randn(2, 5, 64), torch.randn(2, 15, 64)
        out = attn(query, key, key)
        assert out.shape == (2, 5, 64)

        out.sum().backward()
        assert attn.W_q.weight.grad.abs().max() > 0

    @pytest.mark.parametrize("mask", [None, "attn_mask", "key_padding_mask"])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
            ),
        ],
    )
    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_attention_matches_upstream_mha(self, num_heads, device, mask):
        """The class is a drop-in for nn.MultiheadAttention, so the values must agree on
        the CPU and CUDA paths, with and without masks; the upstream module is an
        independent oracle for the scaling, head split, masking, and output projection."""
        torch.manual_seed(0)
        embed = 32
        reference = nn.MultiheadAttention(embed, num_heads, batch_first=True).double().to(device)
        ours = VmapSafeMultiHeadAttention(embed_dim=embed, num_heads=num_heads).double().to(device)

        with torch.no_grad():
            q_w, k_w, v_w = reference.in_proj_weight.chunk(3, dim=0)
            q_b, k_b, v_b = reference.in_proj_bias.chunk(3, dim=0)
            for layer, (w, b) in zip((ours.W_q, ours.W_k, ours.W_v), ((q_w, q_b), (k_w, k_b), (v_w, v_b))):
                layer.weight.copy_(w)
                layer.bias.copy_(b)
            ours.W_o.weight.copy_(reference.out_proj.weight)
            ours.W_o.bias.copy_(reference.out_proj.bias)

        query = torch.randn(2, 6, embed, dtype=torch.float64, device=device)
        key = torch.randn(2, 9, embed, dtype=torch.float64, device=device)
        masks = {}
        if mask == "attn_mask":
            masks[mask] = torch.ones(6, 9, dtype=torch.bool, device=device).triu(1)
        elif mask == "key_padding_mask":
            masks[mask] = torch.zeros(2, 9, dtype=torch.bool, device=device)
            masks[mask][1, 7:] = True
        expected, _ = reference(query, key, key, need_weights=False, **masks)
        torch.testing.assert_close(ours(query, key, key, **masks), expected)

    def test_causal_mask_hides_later_positions(self, attn):
        """Under a causal mask, position i attends only to 0..i, so changing the tail
        of the sequence must leave the earlier outputs bit-identical."""
        torch.manual_seed(0)
        seq_len = 10
        x = torch.randn(2, seq_len, 64)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        out = attn(x, x, x, attn_mask=mask)
        assert out.shape == (2, seq_len, 64)

        perturbed = x.clone()
        perturbed[:, -3:] = torch.randn(2, 3, 64)
        out_perturbed = attn(perturbed, perturbed, perturbed, attn_mask=mask)

        torch.testing.assert_close(out_perturbed[:, :-3], out[:, :-3])
        assert not torch.allclose(out_perturbed[:, -1], out[:, -1]), "the tail should have moved"

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
            ),
        ],
    )
    def test_causal_flag_matches_triangular_mask(self, device):
        reference = nn.MultiheadAttention(8, 2, batch_first=True).double().to(device)
        ours = VmapSafeMultiHeadAttention(8, 2).double().to(device)
        with torch.no_grad():
            for index, layer in enumerate((ours.W_q, ours.W_k, ours.W_v)):
                layer.weight.copy_(reference.in_proj_weight.chunk(3)[index])
                layer.bias.copy_(reference.in_proj_bias.chunk(3)[index])
            ours.W_o.load_state_dict(reference.out_proj.state_dict())
        x = torch.randn(2, 5, 8, dtype=torch.float64, device=device)
        mask = torch.ones(5, 5, dtype=torch.bool, device=device).triu(1)
        actual = ours(x, x, x, attn_mask=mask, is_causal=True)
        torch.testing.assert_close(actual, ours(x, x, x, attn_mask=mask))
        torch.testing.assert_close(actual, reference(x, x, x, attn_mask=mask, is_causal=True, need_weights=False)[0])
        with pytest.raises(RuntimeError, match="Need attn_mask"):
            ours(x, x, x, is_causal=True)
        half, x_half = ours.half(), x.half()
        torch.testing.assert_close(
            half(x_half, x_half, x_half, attn_mask=mask, is_causal=True), half(x_half, x_half, x_half, attn_mask=mask)
        )

    def test_key_padding_mask_excludes_masked_keys(self, attn):
        """Masked keys must not reach the output, so changing them must change nothing."""
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)
        key_padding_mask[1, 7:] = True

        out = attn(x, x, x, key_padding_mask=key_padding_mask)
        assert out.shape == (2, 10, 64)

        # Row 1 masks keys 7+ so its output must not move; row 0 masks nothing, so it must.
        perturbed = x.clone()
        perturbed[1, 7:] = torch.randn(3, 64)
        perturbed[0, 7:] = torch.randn(3, 64)
        out_perturbed = attn(perturbed, perturbed, perturbed, key_padding_mask=key_padding_mask)

        torch.testing.assert_close(out_perturbed[1, :7], out[1, :7])
        assert not torch.allclose(out_perturbed[0], out[0]), "unmasked row should have moved"

    def test_attention_head_dim_validation(self):
        """embed_dim must be divisible by num_heads."""
        with pytest.raises(ValueError):
            VmapSafeMultiHeadAttention(embed_dim=64, num_heads=5)

    @pytest.mark.parametrize("num_models,batch", [(5, 1), (3, 4)])
    def test_attention_vmap(self, num_models, batch):
        """nn.MultiheadAttention fails under vmap on mask validation; this must not.
        Each instance gets distinct weights and is checked against its own call:
        identical weights would let a vmap that ignored the parameter axis pass."""
        torch.manual_seed(0)
        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
        attn.eval()
        params = dict(attn.named_parameters())

        def forward_fn(params_dict, x):
            return torch.func.functional_call(attn, params_dict, (x, x, x))

        batched_params = {k: torch.stack([v + 0.1 * i for i in range(num_models)]) for k, v in params.items()}
        x = torch.randn(batch, 10, 64)

        out = torch.vmap(forward_fn, in_dims=(0, None))(batched_params, x)

        assert out.shape == (num_models, batch, 10, 64)
        for i in range(num_models):
            single = forward_fn({k: v[i] for k, v in batched_params.items()}, x)
            torch.testing.assert_close(out[i], single, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("heads_mask", [False, True])
    def test_float_padding_head_mask_match_pytorch(self, heads_mask):
        reference = nn.MultiheadAttention(8, 2, batch_first=True).double().eval()
        ours = VmapSafeMultiHeadAttention(8, 2).double().eval()
        with torch.no_grad():
            for index, layer in enumerate((ours.W_q, ours.W_k, ours.W_v)):
                layer.weight.copy_(reference.in_proj_weight.chunk(3)[index])
                layer.bias.copy_(reference.in_proj_bias.chunk(3)[index])
            ours.W_o.load_state_dict(reference.out_proj.state_dict())
        q, kv = torch.randn(2, 3, 8, dtype=torch.float64), torch.randn(2, 5, 8, dtype=torch.float64)
        mask = torch.randn(4, 3, 5, dtype=torch.float64) if heads_mask else torch.zeros(3, 5, dtype=torch.float64)
        padding = torch.tensor([[0.0, 0.0, -float("inf"), -0.4, 0.0], [0.0, -0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
        expected = reference(q, kv, kv, attn_mask=mask, key_padding_mask=padding, need_weights=False)[0]
        torch.testing.assert_close(ours(q, kv, kv, attn_mask=mask, key_padding_mask=padding), expected)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
            ),
        ],
    )
    def test_fully_masked_attention_row_stays_finite(self, device):
        """Softmax of an all -inf row is NaN, which would spread through the value mix."""
        torch.manual_seed(0)
        attn = VmapSafeMultiHeadAttention(embed_dim=8, num_heads=2).to(device)
        x = torch.randn(2, 3, 8, device=device)
        mask = torch.zeros(2, 3, dtype=torch.bool, device=device)
        mask[1] = True  # every key of the second sequence is padding
        assert torch.isfinite(attn(x, x, x, key_padding_mask=mask)).all()

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
    def test_attention_rejects_unsupported_kwargs(self, kwargs):
        """Constructor must raise a clear NotImplementedError for any
        unsupported nn.MultiheadAttention argument."""
        with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
            VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4, **kwargs)

    def test_attention_rejects_need_weights(self):
        """Forward must reject need_weights explicitly."""
        attn = VmapSafeMultiHeadAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        with pytest.raises(NotImplementedError, match="VmapSafeMultiHeadAttention"):
            attn(x, x, x, need_weights=True)


class TestVmapSafeLSTMCell:
    @pytest.fixture
    def cell(self):
        """Create a test LSTM cell."""
        return VmapSafeLSTMCell(input_size=32, hidden_size=64)

    def test_lstm_cell_state_update(self, cell):
        """One step must move both h and c, and keep every tensor (batch, hidden)."""
        x = torch.randn(4, 32)
        h = torch.zeros(4, 64)
        c = torch.zeros(4, 64)
        h_new, (h_out, c_out) = cell(x, (h, c))
        assert h_new.shape == h_out.shape == c_out.shape == (4, 64)
        assert not torch.allclose(h_new, h)
        assert not torch.allclose(c_out, c)

    def test_lstm_cell_vmap(self):
        """nn.LSTM fails under vmap on CuDNN .data access; this cell must not.
        Distinct weights per instance, each row checked against its own call."""
        torch.manual_seed(0)
        cell = VmapSafeLSTMCell(input_size=32, hidden_size=64)
        params = dict(cell.named_parameters())

        def forward_fn(params_dict, x, h, c):
            h_new, _ = torch.func.functional_call(cell, params_dict, (x, (h, c)))
            return h_new

        num_models = 5
        batched_params = {k: torch.stack([v + 0.1 * i for i in range(num_models)]) for k, v in params.items()}
        x = torch.randn(32)
        h = torch.zeros(64)
        c = torch.zeros(64)

        out = torch.vmap(forward_fn, in_dims=(0, None, None, None))(batched_params, x, h, c)

        assert out.shape == (num_models, 64)
        for i in range(num_models):
            single = forward_fn({k: v[i] for k, v in batched_params.items()}, x, h, c)
            torch.testing.assert_close(out[i], single, rtol=1e-5, atol=1e-6)


class TestVmapSafeLSTM:
    @pytest.fixture
    def lstm(self):
        """Create a test LSTM."""
        return VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2)

    def test_lstm_with_initial_state(self, lstm):
        """Non-zero initial state."""
        x = torch.randn(4, 10, 32)
        h0 = torch.randn(2, 4, 64)
        c0 = torch.randn(2, 4, 64)
        out, (h_n, c_n) = lstm(x, (h0, c0))
        assert out.shape == (4, 10, 64)
        assert not torch.allclose(h_n, h0)

    def test_lstm_with_dropout(self):
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=3, dropout=0.5)
        lstm.train()  # Enable dropout
        x = torch.randn(4, 10, 32)
        out1, _ = lstm(x)
        out2, _ = lstm(x)
        # Two train-mode passes differ with dropout=0.5 (equality is possible but unlikely).
        assert not torch.allclose(out1, out2)

    def test_lstm_differentiable(self, lstm):
        """Gradients must reach every layer with real magnitude; ``grad is not None``
        alone would pass on the all-zero gradient a broken gate chain produces."""
        out, _ = lstm(torch.randn(4, 10, 32))
        out.sum().backward()
        for i, cell in enumerate(lstm.cells):
            for name, param in (("W_i", cell.W_i.weight), ("W_h", cell.W_h.weight)):
                assert param.grad is not None, f"layer {i} {name}"
                assert param.grad.abs().max() > 0, f"layer {i} {name} got an all-zero gradient"

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_lstm_matches_upstream_nn_lstm(self, num_layers):
        """The class is a drop-in for nn.LSTM, so it has to compute the same thing;
        this pins the recurrence, gate order, and layer stacking against the reference."""
        torch.manual_seed(0)
        reference = nn.LSTM(input_size=8, hidden_size=6, num_layers=num_layers, batch_first=True)
        ours = VmapSafeLSTM(input_size=8, hidden_size=6, num_layers=num_layers)

        # nn.LSTM packs the same four gates in the same i, f, g, o order.
        with torch.no_grad():
            for layer, cell in enumerate(ours.cells):
                cell.W_i.weight.copy_(getattr(reference, f"weight_ih_l{layer}"))
                cell.W_h.weight.copy_(getattr(reference, f"weight_hh_l{layer}"))
                cell.W_i.bias.copy_(getattr(reference, f"bias_ih_l{layer}"))
                cell.W_h.bias.copy_(getattr(reference, f"bias_hh_l{layer}"))

        x = torch.randn(3, 7, 8)
        expected, (h_ref, c_ref) = reference(x)
        got, (h, c) = ours(x)

        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h, h_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(c, c_ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("num_models,batch", [(5, 1), (3, 4)])
    def test_lstm_vmap(self, num_models, batch):
        """nn.LSTM fails under vmap on CuDNN .data access. vmap over params, broadcast
        over the input: a batch of model instances scored on the same data, which is
        what a PolyStep candidate sweep does."""
        device = "cpu"
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2).to(device)

        params = dict(lstm.named_parameters())

        def forward_fn(params_dict, x):
            out, _ = torch.func.functional_call(lstm, params_dict, (x,))
            return out

        # Distinct rows, not clones: with identical parameters a vmap that ignored the
        # candidate axis would still pass.
        gen = torch.Generator().manual_seed(0)
        batched_params = {
            k: v.unsqueeze(0).expand(num_models, *v.shape) + 0.1 * torch.randn(num_models, *v.shape, generator=gen)
            for k, v in params.items()
        }

        x = torch.randn(batch, 10, 32, device=device)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        expected = (num_models, batch, 10, 64)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

        # Each row must match its own call, and no two rows may agree.
        for i in range(num_models):
            row = forward_fn({k: v[i] for k, v in batched_params.items()}, x)
            torch.testing.assert_close(out[i], row, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(out[0], out[1], atol=1e-6)

    def test_lstm_state_deeper_than_layers_raises(self):
        """Extra (h, c) layers would otherwise be dropped silently."""
        lstm = VmapSafeLSTM(input_size=4, hidden_size=3, num_layers=1)
        h = torch.zeros(2, 5, 3)
        with pytest.raises(ValueError, match="initial h and c"):
            lstm(torch.randn(5, 2, 4), (h, h.clone()))

    def test_upstream_nn_lstm_fails_under_vmap(self):
        """Documented PyTorch issue #105982, the reason VmapSafeLSTM exists.

        Matching the specific error keeps this honest. A bare ``except Exception: pass``
        passes on any failure, including a typo in the test itself, so it could never fail.
        """
        lstm = nn.LSTM(4, 8, num_layers=1, batch_first=True)
        params = {k: v.detach() for k, v in lstm.named_parameters()}
        buffers = {k: v.detach() for k, v in lstm.named_buffers()}
        x = torch.randn(2, 5, 4)

        def call(p):
            return torch.func.functional_call(lstm, {**p, **buffers}, (x,))[0]

        stacked = {k: torch.stack([v, v, v], dim=0) for k, v in params.items()}
        with pytest.raises(
            RuntimeError, match="Batching rule not implemented|does not support|Cannot access data pointer"
        ):
            torch.vmap(call, in_dims=(0,))(stacked)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(bidirectional=True),
            dict(proj_size=4),
            dict(batch_first=False),
        ],
    )
    def test_lstm_rejects_unsupported_kwargs(self, kwargs):
        with pytest.raises(NotImplementedError, match="VmapSafeLSTM"):
            VmapSafeLSTM(input_size=4, hidden_size=8, **kwargs)

    def test_lstm_rejects_packed_sequence(self):
        lstm = VmapSafeLSTM(input_size=4, hidden_size=8)
        x = torch.randn(3, 5, 4)
        lengths = torch.tensor([5, 3, 2])
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        with pytest.raises(NotImplementedError, match="PackedSequence"):
            lstm(packed)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_half_attention_accepts_float_masks():
    for dtype in (torch.float16, torch.bfloat16):
        model = VmapSafeMultiHeadAttention(8, 2).to(device="cuda", dtype=dtype).eval()
        x = torch.randn(2, 3, 8, device="cuda", dtype=dtype)
        expected = model(x, x, x, attn_mask=torch.zeros(3, 3, dtype=torch.bool, device="cuda"))
        actual = model(
            x,
            x,
            x,
            attn_mask=torch.zeros(3, 3, device="cuda"),
            key_padding_mask=torch.zeros(2, 3, device="cuda"),
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding"])
def test_vmap_sdpa_cuda_matches_explicit(monkeypatch, mask_kind):
    torch.manual_seed(0)
    attn = VmapSafeMultiHeadAttention(32, 4).cuda().eval()
    params = {k: torch.stack([v + 0.05 * i for i in range(3)]) for k, v in attn.named_parameters()}
    x = torch.randn(2, 7, 32, device="cuda")
    mask = torch.ones(7, 7, dtype=torch.bool, device="cuda").triu(1) & (mask_kind == "causal")
    padding = torch.zeros(2, 7, dtype=torch.bool, device="cuda")
    padding[1, 5:] = mask_kind == "padding"
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", lambda *a, **k: calls.append(1) or sdpa(*a, **k)
    )

    def forward(p, kwargs):
        return torch.func.functional_call(attn, p, (x, x, x), kwargs)

    fused_kwargs = {
        "none": {},
        "causal": dict(attn_mask=mask, is_causal=True),
        "padding": dict(key_padding_mask=padding),
    }[mask_kind]
    fused = torch.vmap(forward, in_dims=(0, None))(params, fused_kwargs)
    assert bool(calls) == (mask_kind != "padding")
    calls.clear()
    explicit = torch.vmap(forward, in_dims=(0, None))(params, dict(attn_mask=mask | padding[:, None, :]))
    assert not calls
    torch.testing.assert_close(fused, explicit, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_meta_attention_forward_keeps_shape(dtype):
    model = VmapSafeMultiHeadAttention(8, 2).to(device="meta", dtype=dtype)
    x = torch.randn(2, 3, 8, device="meta", dtype=dtype)
    assert model(x, x, x).shape == (2, 3, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("attn_mask", [None, torch.tensor([[False, True], [False, False]])])
def test_fp16_cuda_attention_stays_finite(attn_mask, autocast):
    model = VmapSafeMultiHeadAttention(2, 1, bias=False).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.eye(2))
    x = torch.tensor([[[400.0, 0.0], [0.0, 400.0]]])
    expected = model(x, x, x, attn_mask=attn_mask)
    dtype = torch.float32 if autocast else torch.float16
    model = model.to(device="cuda", dtype=dtype)
    x = x.to(device="cuda", dtype=dtype)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        actual = model(x, x, x, attn_mask=None if attn_mask is None else attn_mask.cuda())
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=1e-3, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bias", [False, True])
def test_fp16_autocast_attention_cost_stays_finite(bias):
    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = VmapSafeMultiHeadAttention(2, 1, bias=bias)

        def forward(self, x):
            return self.attn(x, x, x)

    model = SelfAttention().cuda()
    params = {
        name: torch.eye(2, device="cuda").repeat(3, 1, 1) if p.dim() == 2 else torch.zeros(3, 2, device="cuda")
        for name, p in model.named_parameters()
    }
    x = torch.tensor([[[400.0, 0.0], [0.0, 400.0]]], device="cuda")
    evaluator = NNCostEvaluator(model, nn.MSELoss(), autocast_dtype=torch.float16)
    torch.testing.assert_close(evaluator.evaluate(params, x, x), torch.zeros(3, device="cuda"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_autocast_keeps_fp64_attention_exact():
    torch.manual_seed(0)
    model = VmapSafeMultiHeadAttention(16, 4).to(device="cuda", dtype=torch.float64).eval()
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.float64)
    attn_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool, device="cuda"), diagonal=1)
    key_padding_mask = torch.zeros(2, 5, dtype=torch.bool, device="cuda")
    key_padding_mask[1, 3:] = True
    expected = model(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = model(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
