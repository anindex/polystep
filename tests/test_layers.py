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


# =============================================================================
# VmapSafeMultiHeadAttention Tests
# =============================================================================


class TestVmapSafeMultiHeadAttention:
    """Tests for VmapSafeMultiHeadAttention."""

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
        """CRITICAL: Verify attention works under torch.vmap.

        This is the main reason for this implementation - nn.MultiheadAttention
        fails under vmap with mask validation bugs (Issue #151558).

        The vmap pattern used: vmap over params, broadcast over input.
        This simulates evaluating a batch of model instances on the same input.
        """
        # Use CPU for testing to avoid CUDA generator issues
        device = "cpu"
        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        attn.eval()  # Disable dropout for deterministic testing

        # Get parameters for functional_call
        params = dict(attn.named_parameters())

        def forward_fn(params_dict, x):
            return torch.func.functional_call(attn, params_dict, (x, x, x))

        # Create batched params (simulating multiple model instances)
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Input with batch dimension: (batch, seq, embed)
        # The module expects 3D input, so we keep the batch dim
        x = torch.randn(batch, 10, 64, device=device)

        # This should NOT error (unlike nn.MultiheadAttention)
        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, embed)
        expected = (num_models, batch, 10, 64)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"


# =============================================================================
# VmapSafeLSTMCell Tests
# =============================================================================


class TestVmapSafeLSTMCell:
    """Tests for VmapSafeLSTMCell."""

    @pytest.fixture
    def cell(self):
        """Create a test LSTM cell."""
        return VmapSafeLSTMCell(input_size=32, hidden_size=64)

    def test_lstm_cell_shapes(self, cell):
        """Verify single step shapes."""
        x = torch.randn(4, 32)  # (batch, input)
        h = torch.zeros(4, 64)
        c = torch.zeros(4, 64)
        h_new, (h_out, c_out) = cell(x, (h, c))
        assert h_new.shape == (4, 64)
        assert h_out.shape == (4, 64)
        assert c_out.shape == (4, 64)

    def test_lstm_cell_state_update(self, cell):
        """Verify h and c change after forward pass."""
        x = torch.randn(4, 32)
        h = torch.zeros(4, 64)
        c = torch.zeros(4, 64)
        h_new, (h_out, c_out) = cell(x, (h, c))
        # States should have changed (non-zero)
        assert not torch.allclose(h_new, h)
        assert not torch.allclose(c_out, c)

    def test_lstm_cell_differentiable(self, cell):
        """Verify cell is differentiable."""
        x = torch.randn(4, 32)
        h = torch.zeros(4, 64)
        c = torch.zeros(4, 64)
        h_new, _ = cell(x, (h, c))
        loss = h_new.sum()
        loss.backward()
        assert cell.W_i.weight.grad is not None
        assert cell.W_h.weight.grad is not None

    def test_lstm_cell_vmap(self):
        """CRITICAL: Verify LSTM cell works under vmap."""
        device = "cpu"
        cell = VmapSafeLSTMCell(input_size=32, hidden_size=64).to(device)

        params = dict(cell.named_parameters())

        def forward_fn(params_dict, x, h, c):
            h_new, _ = torch.func.functional_call(cell, params_dict, (x, (h, c)))
            return h_new

        num_models = 5
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        x = torch.randn(32, device=device)  # Single input
        h = torch.zeros(64, device=device)
        c = torch.zeros(64, device=device)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None, None, None))
        out = vmapped(batched_params, x, h, c)

        assert out.shape == (5, 64), f"Expected (5, 64), got {out.shape}"


# =============================================================================
# VmapSafeLSTM Tests
# =============================================================================


class TestVmapSafeLSTM:
    """Tests for VmapSafeLSTM."""

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

    def test_lstm_multi_layer(self):
        """Test LSTM across layer counts."""
        for num_layers in [1, 2, 3, 4]:
            lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=num_layers)
            x = torch.randn(4, 10, 32)
            out, (h_n, c_n) = lstm(x)
            assert out.shape == (4, 10, 64)
            assert h_n.shape == (num_layers, 4, 64)
            assert c_n.shape == (num_layers, 4, 64)

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
        """Verify LSTM is differentiable."""
        x = torch.randn(4, 10, 32)
        out, _ = lstm(x)
        loss = out.sum()
        loss.backward()
        # Check gradients exist for all layers
        for cell in lstm.cells:
            assert cell.W_i.weight.grad is not None
            assert cell.W_h.weight.grad is not None

    @pytest.mark.parametrize("num_models,batch", [(5, 1), (3, 4)])
    def test_lstm_vmap(self, num_models, batch):
        """CRITICAL: Verify LSTM works under vmap.

        This is the main reason for this implementation - nn.LSTM
        fails under vmap with CuDNN .data access errors.

        The vmap pattern used: vmap over params, broadcast over input.
        This simulates evaluating a batch of model instances on the same input.
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


# =============================================================================
# Integration Tests
# =============================================================================


class TestLayersIntegration:
    """Integration tests combining layers."""

    def test_attention_lstm_pipeline(self):
        """Test attention followed by LSTM."""
        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
        lstm = VmapSafeLSTM(input_size=64, hidden_size=128, num_layers=1)

        x = torch.randn(4, 10, 64)
        attn_out = attn(x, x, x)
        lstm_out, _ = lstm(attn_out)

        assert lstm_out.shape == (4, 10, 128)
