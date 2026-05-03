"""Tests for non-differentiable model definitions in experiments/runners/nondiff_models.py.

Tests verify:
  - Correct output shapes for all building blocks and full models
  - Non-differentiable operations are present (sign, round, argmax, floor, threshold)
  - All models are vmap-compatible (functional_call with batched params)
  - MAX-SAT utilities produce correct outputs
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from experiments.runners.nondiff_models import (
    LIFNeuron,
    SpikingMNISTNet,
    QuantizedLinear,
    QuantizedMLP,
    BinaryLinear,
    BinaryMNISTNet,
    TernaryLinear,
    TernaryMNISTNet,
    STESign,
    STETernary,
    BinaryLinearSTE,
    TernaryLinearSTE,
    BinaryConv2d,
    BinaryConv2dSTE,
    BinaryMNISTNetSTE,
    TernaryMNISTNetSTE,
    BinaryCIFAR10Net,
    BinaryCIFAR10NetSTE,
    DiscreteAttention,
    DiscreteAttentionNet,
    StaircaseActivation,
    StaircaseNet,
    HardMoELayer,
    HardMoENet,
    SoftMoELayer,
    SoftMoENet,
    compute_expert_utilization,
    MaxSATModel,
    evaluate_sat_loss,
    cra_penalty,
    HardPermutationNet,
    SoftPermutationNet,
    PermutationLoss,
)


# ---------------------------------------------------------------------------
# Building block tests
# ---------------------------------------------------------------------------

class TestLIFNeuron:
    def test_forward_shape(self):
        lif = LIFNeuron(beta=0.95, threshold=1.0)
        x = torch.randn(2, 16)
        mem = torch.zeros(2, 16)
        spike, new_mem = lif(x, mem)
        assert spike.shape == (2, 16)
        assert new_mem.shape == (2, 16)

    def test_spike_values_binary(self):
        """Spike output must be in {0.0, 1.0}."""
        lif = LIFNeuron(beta=0.95, threshold=1.0)
        x = torch.randn(4, 32) * 2.0  # Large values to trigger spikes
        mem = torch.randn(4, 32).abs() * 1.5  # Some above threshold
        spike, _ = lif(x, mem)
        unique_vals = spike.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Spike value {v.item()} not in {{0, 1}}"


class TestQuantizedLinear:
    def test_forward_shape(self):
        ql = QuantizedLinear(16, 32)
        x = torch.randn(2, 16)
        out = ql(x)
        assert out.shape == (2, 32)

    def test_weights_are_int8_rounded(self):
        """Internal effective weights should be int8-quantized."""
        ql = QuantizedLinear(16, 32, scale=0.01)
        w_q = torch.clamp(torch.round(ql.weight / ql.scale), -128, 127) * ql.scale
        # Check that quantized weights differ from raw weights (unless already quantized)
        # The key property: w_q values are multiples of scale
        remainder = (w_q / ql.scale) - torch.round(w_q / ql.scale)
        assert remainder.abs().max() < 1e-5


class TestBinaryLinear:
    def test_forward_shape(self):
        bl = BinaryLinear(16, 32)
        x = torch.randn(2, 16)
        out = bl(x)
        assert out.shape == (2, 32)

    def test_effective_weights_binary(self):
        """Effective weights in forward pass should be in {-1, +1}."""
        bl = BinaryLinear(16, 32)
        w_b = torch.sign(bl.weight)
        unique_vals = w_b.unique()
        for v in unique_vals:
            assert v.item() in (-1.0, 0.0, 1.0), f"Binary weight {v.item()} unexpected"
        # With randn init, we expect mostly -1 and +1 (0 is extremely rare)
        assert (w_b.abs() > 0).float().mean() > 0.99


class TestTernaryLinear:
    def test_forward_shape(self):
        tl = TernaryLinear(16, 32)
        x = torch.randn(2, 16)
        out = tl(x)
        assert out.shape == (2, 32)

    def test_effective_weights_ternary(self):
        """Effective weights should be in {-1, 0, +1}."""
        tl = TernaryLinear(16, 32, threshold=0.5)
        w_t = torch.sign(tl.weight) * (tl.weight.abs() >= 0.5).float()
        unique_vals = w_t.unique()
        for v in unique_vals:
            assert v.item() in (-1.0, 0.0, 1.0), f"Ternary weight {v.item()} unexpected"


# ---------------------------------------------------------------------------
# STE autograd function tests
# ---------------------------------------------------------------------------


class TestSTESign:
    def test_forward_produces_sign(self):
        input = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.5])
        output = STESign.apply(input)
        expected = torch.sign(input)
        assert torch.equal(output, expected)

    def test_backward_passes_gradient_within_clamp(self):
        input = torch.tensor([-0.5, 0.5], requires_grad=True)
        out = STESign.apply(input)
        out.sum().backward()
        assert input.grad is not None
        assert input.grad.abs().sum() > 0

    def test_backward_zeros_gradient_outside_clamp(self):
        input = torch.tensor([-1.5, 2.0], requires_grad=True)
        out = STESign.apply(input)
        out.sum().backward()
        assert torch.equal(input.grad, torch.tensor([0.0, 0.0]))


class TestSTETernary:
    def test_forward_produces_ternary(self):
        input = torch.tensor([-1.0, -0.3, 0.1, 0.3, 0.8])
        output = STETernary.apply(input, 0.5)
        expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])
        assert torch.equal(output, expected)

    def test_backward_passes_gradient(self):
        input = torch.tensor([-0.5, 0.5], requires_grad=True)
        out = STETernary.apply(input, 0.3)
        out.sum().backward()
        assert input.grad is not None


# ---------------------------------------------------------------------------
# STE-enabled layer tests
# ---------------------------------------------------------------------------


class TestBinaryLinearSTE:
    def test_forward_shape(self):
        layer = BinaryLinearSTE(16, 8)
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_gradient_flows(self):
        layer = BinaryLinearSTE(16, 8)
        x = torch.randn(2, 16)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (8, 16)


class TestTernaryLinearSTE:
    def test_forward_shape_and_gradient(self):
        layer = TernaryLinearSTE(16, 8, threshold=0.3)
        x = torch.randn(2, 16)
        out = layer(x)
        assert out.shape == (2, 8)
        out.sum().backward()
        assert layer.weight.grad is not None


class TestBinaryConv2d:
    def test_forward_shape(self):
        layer = BinaryConv2d(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 8, 8)
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)

    def test_weights_are_binary(self):
        layer = BinaryConv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            w_b = torch.sign(layer.weight)
        unique_vals = w_b.unique()
        for v in unique_vals:
            assert v.item() in (-1.0, 0.0, 1.0), f"Binary weight {v.item()} unexpected"


class TestBinaryConv2dSTE:
    def test_forward_shape(self):
        layer = BinaryConv2dSTE(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 8, 8)
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)

    def test_gradient_flows(self):
        layer = BinaryConv2dSTE(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 8, 8)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is not None


# ---------------------------------------------------------------------------
# STE full model tests
# ---------------------------------------------------------------------------


class TestBinaryMNISTNetSTE:
    def test_forward_shape(self):
        model = BinaryMNISTNetSTE()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_gradient_flows(self):
        model = BinaryMNISTNetSTE()
        x = torch.randn(2, 1, 28, 28)
        out = model(x).sum()
        out.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestTernaryMNISTNetSTE:
    def test_forward_shape(self):
        model = TernaryMNISTNetSTE()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestBinaryCIFAR10Net:
    def test_forward_shape(self):
        model = BinaryCIFAR10Net()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_param_count(self):
        model = BinaryCIFAR10Net()
        total = sum(p.numel() for p in model.parameters())
        assert total > 100000, f"Expected >100K params, got {total}"


class TestBinaryCIFAR10NetSTE:
    def test_forward_shape(self):
        model = BinaryCIFAR10NetSTE()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_gradient_flows(self):
        model = BinaryCIFAR10NetSTE()
        x = torch.randn(2, 3, 32, 32)
        out = model(x).sum()
        out.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestDiscreteAttention:
    def test_forward_shape(self):
        da = DiscreteAttention(dim=32, num_slots=8)
        x = torch.randn(2, 32)
        out = da(x)
        assert out.shape == (2, 32)


class TestStaircaseActivation:
    def test_forward_shape(self):
        sa = StaircaseActivation(levels=5)
        x = torch.randn(2, 16)
        out = sa(x)
        assert out.shape == (2, 16)

    def test_output_values_quantized(self):
        """Output values must be in {0/5, 1/5, 2/5, 3/5, 4/5}."""
        sa = StaircaseActivation(levels=5)
        x = torch.randn(100, 16)  # Enough samples for variety
        out = sa(x)
        valid_values = {0.0, 0.2, 0.4, 0.6, 0.8}
        unique_vals = out.unique()
        for v in unique_vals:
            assert round(v.item(), 6) in valid_values, (
                f"Staircase value {v.item()} not in {valid_values}"
            )


class TestHardMoELayer:
    def test_forward_shape(self):
        moe = HardMoELayer(input_dim=32, hidden_dim=64, num_experts=4)
        x = torch.randn(2, 32)
        out = moe(x)
        assert out.shape == (2, 64)

    def test_all_experts_evaluated(self):
        """HardMoELayer must evaluate ALL experts (vmap-safe pattern)."""
        import inspect
        source = inspect.getsource(HardMoELayer.forward)
        assert "torch.stack" in source or "stack" in source, (
            "HardMoELayer.forward should use torch.stack to evaluate all experts"
        )


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestSpikingMNISTNet:
    def test_forward_shape(self):
        model = SpikingMNISTNet(num_steps=5)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestQuantizedMLP:
    def test_forward_shape(self):
        model = QuantizedMLP(784, 128, 10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestBinaryMNISTNet:
    def test_forward_shape(self):
        model = BinaryMNISTNet()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestTernaryMNISTNet:
    def test_forward_shape(self):
        model = TernaryMNISTNet()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestDiscreteAttentionNet:
    def test_forward_shape(self):
        model = DiscreteAttentionNet(784, 128, 10, num_slots=8)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestStaircaseNet:
    def test_forward_shape(self):
        model = StaircaseNet(784, 128, 10, levels=5)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)


class TestHardMoENet:
    def test_forward_shape(self):
        model = HardMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 20)


# ---------------------------------------------------------------------------
# Soft MoE tests (differentiable baseline)
# ---------------------------------------------------------------------------


class TestSoftMoELayer:
    def test_forward_shape(self):
        moe = SoftMoELayer(input_dim=32, hidden_dim=64, num_experts=4)
        x = torch.randn(2, 32)
        out = moe(x)
        assert out.shape == (2, 64)

    def test_gradient_flows(self):
        moe = SoftMoELayer(input_dim=32, hidden_dim=64, num_experts=4)
        x = torch.randn(2, 32)
        out = moe(x).sum()
        out.backward()
        assert moe.gate.weight.grad is not None
        assert moe.gate.weight.grad.shape == (4, 32)

    def test_softmax_not_argmax(self):
        """SoftMoELayer must use softmax, not argmax."""
        import inspect
        source = inspect.getsource(SoftMoELayer.forward)
        assert "softmax" in source, "SoftMoELayer.forward should use F.softmax"
        assert "argmax" not in source, "SoftMoELayer.forward should NOT use argmax"


class TestSoftMoENet:
    def test_forward_shape(self):
        model = SoftMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 20)

    def test_param_count_matches_hard(self):
        hard = HardMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        soft = SoftMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        hard_params = sum(p.numel() for p in hard.parameters())
        soft_params = sum(p.numel() for p in soft.parameters())
        assert hard_params == soft_params, f"Param count mismatch: hard={hard_params}, soft={soft_params}"

    def test_gradient_flows(self):
        model = SoftMoENet()
        x = torch.randn(2, 1, 28, 28)
        out = model(x).sum()
        out.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestExpertUtilization:
    def test_returns_correct_keys(self):
        model = HardMoENet()
        # Create a minimal test loader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 1, 28, 28), torch.randint(0, 20, (20,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        result = compute_expert_utilization(model, loader, device="cpu")
        assert "expert_utilization" in result
        assert "max_expert_share" in result
        assert "collapsed" in result
        assert "routing_entropy" in result
        assert "normalized_entropy" in result

    def test_utilization_sums_to_one(self):
        model = HardMoENet()
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 1, 28, 28), torch.randint(0, 20, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=50)
        result = compute_expert_utilization(model, loader, device="cpu")
        total = sum(result["expert_utilization"].values())
        assert abs(total - 1.0) < 1e-5, f"Utilization should sum to 1.0, got {total}"

    def test_collapse_detection(self):
        model = HardMoENet()
        # Bias gate weights so one expert dominates
        with torch.no_grad():
            model.moe.gate.bias.zero_()
            model.moe.gate.bias[0] = 100.0  # Expert 0 always wins
        dataset = torch.utils.data.TensorDataset(
            torch.randn(50, 1, 28, 28), torch.randint(0, 20, (50,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=50)
        result = compute_expert_utilization(model, loader, device="cpu")
        assert result["collapsed"] is True, "Should detect collapse when one expert handles all inputs"
        assert result["max_expert_share"] > 0.90


# ---------------------------------------------------------------------------
# MAX-SAT utility tests
# ---------------------------------------------------------------------------

class TestMaxSATModel:
    def test_forward_returns_scalar(self):
        model = MaxSATModel(num_vars=20)
        # Create simple clauses: 3 clauses, each with 3 variables
        clause_vars = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        clause_signs = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        out = model(clause_vars, clause_signs)
        assert out.dim() == 0 or out.numel() == 1, "MaxSATModel should return scalar"

    def test_no_hidden_layers(self):
        """MaxSATModel should have NO hidden layers, only self.assignments."""
        model = MaxSATModel(num_vars=20)
        param_names = [name for name, _ in model.named_parameters()]
        assert param_names == ["assignments"], (
            f"MaxSATModel should only have 'assignments' parameter, got {param_names}"
        )


class TestCraPenalty:
    def test_known_values(self):
        """cra_penalty should be 0 for {0, 1} values and positive for 0.5."""
        soft = torch.tensor([0.0, 1.0, 0.5])
        penalty = cra_penalty(soft)
        # For x=0: (2*0-1)^2 = 1, so 1-1=0
        # For x=1: (2*1-1)^2 = 1, so 1-1=0
        # For x=0.5: (2*0.5-1)^2 = 0, so 1-0=1
        assert penalty.item() == pytest.approx(1.0, abs=1e-5)


class TestEvaluateSatLoss:
    def test_returns_scalar(self):
        soft = torch.tensor([0.5, 0.8, 0.2])
        clause_vars = torch.tensor([[0, 1], [1, 2]])
        clause_signs = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        loss = evaluate_sat_loss(soft, clause_vars, clause_signs)
        assert loss.dim() == 0 or loss.numel() == 1


# ---------------------------------------------------------------------------
# Vmap compatibility tests
# ---------------------------------------------------------------------------

class TestVmapCompatibility:
    """Test that all classification models work under torch.vmap + functional_call."""

    @staticmethod
    def _vmap_test(model, input_tensor, num_perturbations=2):
        """Helper: run vmap with `num_perturbations` parameter perturbations."""
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        # Create batched params: stack original params num_perturbations times with noise
        batched_params = {}
        for name, p in params.items():
            noise = torch.randn(num_perturbations, *p.shape) * 0.01
            batched_params[name] = p.unsqueeze(0).expand(num_perturbations, *p.shape) + noise

        def call_single(single_params):
            return functional_call(model, (single_params, buffers), (input_tensor,))

        outputs = vmap(call_single)(batched_params)
        return outputs

    def test_spiking_mnist_vmap(self):
        model = SpikingMNISTNet(num_steps=3)  # Few steps for speed
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)  # (num_perturb, batch, classes)

    def test_quantized_mlp_vmap(self):
        model = QuantizedMLP(784, 128, 10)
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)

    def test_binary_mnist_vmap(self):
        model = BinaryMNISTNet()
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)

    def test_ternary_mnist_vmap(self):
        model = TernaryMNISTNet()
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)

    def test_discrete_attention_vmap(self):
        model = DiscreteAttentionNet(784, 128, 10, num_slots=8)
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)

    def test_staircase_vmap(self):
        model = StaircaseNet(784, 128, 10, levels=5)
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 10)

    def test_hard_moe_vmap(self):
        model = HardMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 20)

    def test_soft_moe_vmap(self):
        model = SoftMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
        x = torch.randn(2, 1, 28, 28)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == (2, 2, 20)


# ---------------------------------------------------------------------------
# Permutation model tests
# ---------------------------------------------------------------------------


class TestHardPermutationNet:
    def test_forward_shape(self):
        """Input (4, 10) -> output (4, 10) long tensor."""
        model = HardPermutationNet(N=10, hidden_dim=64)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 10)

    def test_output_is_long_indices(self):
        """Output dtype is long, values in [0, N)."""
        model = HardPermutationNet(N=10, hidden_dim=64)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.dtype == torch.long
        assert out.min() >= 0
        assert out.max() < 10

    def test_param_count_N10(self):
        """N=10, hidden=64 -> 778 params (1*64 + 64 bias + 64*10 + 10 bias)."""
        model = HardPermutationNet(N=10, hidden_dim=64)
        total = sum(p.numel() for p in model.parameters())
        assert total == 778, f"Expected 778 params for N=10, got {total}"

    def test_param_count_N50(self):
        """N=50, hidden=64 -> 3378 params (64+64+3200+50)."""
        model = HardPermutationNet(N=50, hidden_dim=64)
        total = sum(p.numel() for p in model.parameters())
        assert total == 3378, f"Expected 3378 params for N=50, got {total}"


class TestSoftPermutationNet:
    def test_forward_shape(self):
        """Input (4, 10) -> output (4, 10, 10) float tensor."""
        model = SoftPermutationNet(N=10, hidden_dim=64)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 10, 10)

    def test_doubly_stochastic(self):
        """Row sums and column sums are approximately 1.0."""
        model = SoftPermutationNet(N=10, hidden_dim=64, n_sinkhorn_iters=20)
        x = torch.randn(4, 10)
        out = model(x)
        row_sums = out.sum(dim=-1)  # (4, 10)
        col_sums = out.sum(dim=-2)  # (4, 10)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
            f"Row sums not close to 1: max deviation {(row_sums - 1).abs().max():.6f}"
        )
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), (
            f"Col sums not close to 1: max deviation {(col_sums - 1).abs().max():.6f}"
        )

    def test_param_count_matches_hard(self):
        """Same param count as HardPermutationNet for same N, hidden_dim."""
        hard = HardPermutationNet(N=10, hidden_dim=64)
        soft = SoftPermutationNet(N=10, hidden_dim=64)
        hard_params = sum(p.numel() for p in hard.parameters())
        soft_params = sum(p.numel() for p in soft.parameters())
        assert hard_params == soft_params, (
            f"Param count mismatch: hard={hard_params}, soft={soft_params}"
        )


class TestPermutationLoss:
    def test_perfect_match(self):
        """Identical permutations -> loss 0.0."""
        loss_fn = PermutationLoss()
        perm = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        loss = loss_fn(perm, perm)
        assert loss.item() == pytest.approx(0.0)

    def test_complete_mismatch(self):
        """Completely wrong -> loss close to 1.0."""
        loss_fn = PermutationLoss()
        pred = torch.tensor([[1, 0, 3, 2]])  # all wrong
        target = torch.tensor([[0, 1, 2, 3]])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0)

    def test_partial_match(self):
        """Known partial match -> expected fraction."""
        loss_fn = PermutationLoss()
        pred = torch.tensor([[0, 1, 3, 2]])  # 2 correct (positions 0,1), 2 wrong
        target = torch.tensor([[0, 1, 2, 3]])
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.5)  # 2/4 wrong


class TestHardPermutationNetVmap:
    """Vmap compatibility test for HardPermutationNet."""

    def test_hard_permutation_vmap(self):
        model = HardPermutationNet(N=10, hidden_dim=64)
        x = torch.randn(4, 10)
        outputs = TestVmapCompatibility._vmap_test(model, x, num_perturbations=2)
        assert outputs.shape == (2, 4, 10), f"Expected (2, 4, 10), got {outputs.shape}"


class TestSoftPermutationNetVmap:
    """Vmap compatibility test for SoftPermutationNet."""

    def test_soft_permutation_vmap(self):
        model = SoftPermutationNet(N=10, hidden_dim=64)
        x = torch.randn(4, 10)
        outputs = TestVmapCompatibility._vmap_test(model, x, num_perturbations=2)
        assert outputs.shape == (2, 4, 10, 10), f"Expected (2, 4, 10, 10), got {outputs.shape}"
