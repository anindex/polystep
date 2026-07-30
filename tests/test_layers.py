"""Unit tests for vmap-safe layers.

Tests VmapSafeMultiHeadAttention, VmapSafeLSTMCell, and VmapSafeLSTM
for correctness and vmap compatibility.
"""

import pytest
import torch
import torch.nn as nn

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

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_attention_output_shape_follows_the_query(self, num_heads):
        """Self- and cross-attention: the output takes the query's sequence length."""
        torch.manual_seed(0)
        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=num_heads)
        x = torch.randn(4, 20, 64)
        assert attn(x, x, x).shape == (4, 20, 64)

        query, key = torch.randn(2, 5, 64), torch.randn(2, 15, 64)
        assert attn(query, key, key).shape == (2, 5, 64)

        attn(x, x, x).sum().backward()
        assert attn.W_q.weight.grad is not None

    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_attention_matches_upstream_mha(self, num_heads):
        """The class is a drop-in for nn.MultiheadAttention, so the values must agree.

        Replaying the projections, scale and softmax here would reuse the module's own
        ``scale`` and cancel an error in it; the upstream module is an independent
        oracle for the scaling, the head split and the output projection alike.
        """
        torch.manual_seed(0)
        embed = 32
        reference = nn.MultiheadAttention(embed, num_heads, batch_first=True)
        ours = VmapSafeMultiHeadAttention(embed_dim=embed, num_heads=num_heads)

        with torch.no_grad():
            q_w, k_w, v_w = reference.in_proj_weight.chunk(3, dim=0)
            q_b, k_b, v_b = reference.in_proj_bias.chunk(3, dim=0)
            for layer, (w, b) in zip((ours.W_q, ours.W_k, ours.W_v), ((q_w, q_b), (k_w, k_b), (v_w, v_b))):
                layer.weight.copy_(w)
                layer.bias.copy_(b)
            ours.W_o.weight.copy_(reference.out_proj.weight)
            ours.W_o.bias.copy_(reference.out_proj.bias)

        query, key = torch.randn(2, 6, embed), torch.randn(2, 9, embed)
        expected, _ = reference(query, key, key, need_weights=False)
        torch.testing.assert_close(ours(query, key, key), expected, rtol=1e-5, atol=1e-6)

    def test_a_causal_mask_hides_later_positions(self, attn):
        """Shape alone cannot see a mask that is silently dropped.

        Under a causal mask, position i attends only to 0..i, so changing the tail of
        the sequence must leave the earlier outputs bit-identical.
        """
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

    def test_a_key_padding_mask_excludes_those_keys(self, attn):
        """Masked keys must not reach the output, so changing them must change nothing."""
        torch.manual_seed(0)
        x = torch.randn(2, 10, 64)
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)
        key_padding_mask[1, 7:] = True

        out = attn(x, x, x, key_padding_mask=key_padding_mask)
        assert out.shape == (2, 10, 64)

        # Rewrite only the padded keys of row 1. Row 1's output must be unchanged, and
        # row 0, which masks nothing, must move because its own keys moved.
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

        Each model instance gets its own weights, and every row is checked against the
        same call made on its own: identical weights would let a vmap that ignored the
        parameter axis pass.
        """
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

        Distinct weights per instance, each row checked against its own call.
        """
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
        # Final states should differ from initial
        assert not torch.allclose(h_n, h0)

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_lstm_multi_layer(self, num_layers):
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=num_layers)
        out, (h_n, c_n) = lstm(torch.randn(4, 10, 32))
        assert out.shape == (4, 10, 64)
        assert h_n.shape == c_n.shape == (num_layers, 4, 64)

    def test_lstm_with_dropout(self):
        """Test LSTM with dropout between layers."""
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=3, dropout=0.5)
        lstm.train()  # Enable dropout
        x = torch.randn(4, 10, 32)
        out1, _ = lstm(x)
        out2, _ = lstm(x)
        # With dropout, two forward passes should differ
        # (small chance they're equal, but very unlikely with dropout=0.5)
        assert not torch.allclose(out1, out2)

    def test_lstm_differentiable(self, lstm):
        """Gradients must reach every layer with real magnitude.

        ``grad is not None`` alone passes on an all-zero gradient, which is what a
        broken chain through the gates would produce.
        """
        out, _ = lstm(torch.randn(4, 10, 32))
        out.sum().backward()
        for i, cell in enumerate(lstm.cells):
            for name, param in (("W_i", cell.W_i.weight), ("W_h", cell.W_h.weight)):
                assert param.grad is not None, f"layer {i} {name}"
                assert param.grad.abs().max() > 0, f"layer {i} {name} got an all-zero gradient"

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_lstm_matches_upstream_nn_lstm(self, num_layers):
        """The class is a drop-in for nn.LSTM, so it has to compute the same thing.

        Shape checks would pass on ``torch.zeros(shape)``; this pins the recurrence,
        the gate order and the layer stacking against the reference implementation.
        """
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
        """The reason this class exists: nn.LSTM fails under vmap on CuDNN .data access.

        vmap over params, broadcast over the input: a batch of model instances scored on
        the same data, which is what a PolyStep candidate sweep does.
        """
        device = "cpu"
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2).to(device)

        params = dict(lstm.named_parameters())

        def forward_fn(params_dict, x):
            out, _ = torch.func.functional_call(lstm, params_dict, (x,))
            return out

        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Input with batch dimension: (batch, seq, input)
        # The module expects 3D input, so we keep the batch dim
        x = torch.randn(batch, 10, 32, device=device)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, hidden)
        expected = (num_models, batch, 10, 64)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"


def test_attention_lstm_pipeline():
    """Test attention followed by LSTM."""
    attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
    lstm = VmapSafeLSTM(input_size=64, hidden_size=128, num_layers=1)

    x = torch.randn(4, 10, 64)
    attn_out = attn(x, x, x)
    lstm_out, _ = lstm(attn_out)

    assert lstm_out.shape == (4, 10, 128)
