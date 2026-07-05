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

    def test_attention_shapes(self, attn):
        """Verify output shape matches (batch, seq, embed)."""
        x = torch.randn(2, 10, 64)
        out = attn(x, x, x)
        assert out.shape == (2, 10, 64)

    def test_attention_self_attention(self, attn):
        """Self-attention: Q=K=V same tensor."""
        x = torch.randn(4, 20, 64)
        out = attn(x, x, x)
        assert out.shape == (4, 20, 64)
        # Output should be differentiable
        loss = out.sum()
        loss.backward()
        assert attn.W_q.weight.grad is not None

    def test_attention_cross_attention(self, attn):
        """Cross-attention: Q different from K/V."""
        query = torch.randn(2, 5, 64)
        key = torch.randn(2, 15, 64)
        value = torch.randn(2, 15, 64)
        out = attn(query, key, value)
        # Output seq length matches query
        assert out.shape == (2, 5, 64)

    def test_attention_with_mask(self, attn):
        """Attention with additive attention mask."""
        x = torch.randn(2, 10, 64)
        # Causal mask: prevent attending to future positions
        seq_len = 10
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        out = attn(x, x, x, attn_mask=mask)
        assert out.shape == (2, 10, 64)

    def test_attention_with_padding_mask(self, attn):
        """Attention with key padding mask."""
        x = torch.randn(2, 10, 64)
        # Mask last 3 positions in second sequence
        key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)
        key_padding_mask[1, 7:] = True
        out = attn(x, x, x, key_padding_mask=key_padding_mask)
        assert out.shape == (2, 10, 64)

    def test_attention_different_heads(self):
        """Test with different number of heads."""
        for num_heads in [1, 2, 8]:
            attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=num_heads)
            x = torch.randn(2, 10, 64)
            out = attn(x, x, x)
            assert out.shape == (2, 10, 64)

    def test_attention_head_dim_validation(self):
        """embed_dim must be divisible by num_heads."""
        with pytest.raises(ValueError):
            VmapSafeMultiHeadAttention(embed_dim=64, num_heads=5)

    def test_attention_vmap(self):
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
        num_models = 5
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Input with batch dimension: (batch=1, seq, embed)
        # The module expects 3D input, so we keep the batch dim
        x = torch.randn(1, 10, 64, device=device)

        # This should NOT error (unlike nn.MultiheadAttention)
        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, embed)
        assert out.shape == (5, 1, 10, 64), f"Expected (5, 1, 10, 64), got {out.shape}"

    def test_attention_vmap_with_batch(self):
        """Verify vmap works with batched inputs too."""
        device = "cpu"
        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4).to(device)
        attn.eval()

        params = dict(attn.named_parameters())

        def forward_fn(params_dict, x):
            return torch.func.functional_call(attn, params_dict, (x, x, x))

        # Batched params
        num_models = 3
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Batched input: (batch, seq, embed)
        x = torch.randn(4, 10, 64, device=device)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, embed)
        assert out.shape == (3, 4, 10, 64), f"Expected (3, 4, 10, 64), got {out.shape}"


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

    def test_lstm_shapes(self, lstm):
        """Verify sequence output shapes."""
        x = torch.randn(4, 10, 32)  # (batch, seq, input)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (4, 10, 64)
        assert h_n.shape == (2, 4, 64)  # (num_layers, batch, hidden)
        assert c_n.shape == (2, 4, 64)

    def test_lstm_with_initial_state(self, lstm):
        """Non-zero initial state."""
        x = torch.randn(4, 10, 32)
        h0 = torch.randn(2, 4, 64)
        c0 = torch.randn(2, 4, 64)
        out, (h_n, c_n) = lstm(x, (h0, c0))
        assert out.shape == (4, 10, 64)
        # Final states should differ from initial
        assert not torch.allclose(h_n, h0)

    def test_lstm_single_layer(self):
        """Test single layer LSTM."""
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=1)
        x = torch.randn(4, 10, 32)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (4, 10, 64)
        assert h_n.shape == (1, 4, 64)

    def test_lstm_multi_layer(self):
        """Test multi-layer LSTM (num_layers > 1)."""
        for num_layers in [2, 3, 4]:
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

    def test_lstm_vmap(self):
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

        num_models = 5
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Input with batch dimension: (batch=1, seq, input)
        # The module expects 3D input, so we keep the batch dim
        x = torch.randn(1, 10, 32, device=device)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, hidden)
        assert out.shape == (5, 1, 10, 64), f"Expected (5, 1, 10, 64), got {out.shape}"

    def test_lstm_vmap_with_batch(self):
        """Verify vmap works with batched sequence inputs."""
        device = "cpu"
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64, num_layers=2).to(device)

        params = dict(lstm.named_parameters())

        def forward_fn(params_dict, x):
            out, _ = torch.func.functional_call(lstm, params_dict, (x,))
            return out

        num_models = 3
        batched_params = {k: v.unsqueeze(0).expand(num_models, *v.shape).clone() for k, v in params.items()}

        # Batched sequence
        x = torch.randn(4, 10, 32, device=device)  # (batch, seq, input)

        vmapped = torch.vmap(forward_fn, in_dims=(0, None))
        out = vmapped(batched_params, x)

        # Output: (num_models, batch, seq, hidden)
        assert out.shape == (3, 4, 10, 64), f"Expected (3, 4, 10, 64), got {out.shape}"


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

    def test_export_from_main_package(self):
        """Verify layers are exported from main polystep package."""
        from polystep import VmapSafeMultiHeadAttention, VmapSafeLSTMCell, VmapSafeLSTM

        attn = VmapSafeMultiHeadAttention(embed_dim=64, num_heads=4)
        cell = VmapSafeLSTMCell(input_size=32, hidden_size=64)
        lstm = VmapSafeLSTM(input_size=32, hidden_size=64)

        assert attn is not None
        assert cell is not None
        assert lstm is not None
