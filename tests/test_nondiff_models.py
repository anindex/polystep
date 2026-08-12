"""Tests for non-differentiable model definitions in experiments/runners/nondiff_models.py.

Tests verify:
  - Correct output shapes for all building blocks and full models
  - Non-differentiable operations are present (sign, round, argmax, floor, threshold)
  - All models are vmap-compatible (functional_call with batched params)
  - MAX-SAT utilities produce correct outputs
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.func import functional_call, vmap

# experiments/ is the paper reproduction harness and is not part of the distributed
# package, so this module is unimportable when the tests run from an sdist.
if not (Path(__file__).resolve().parent.parent / "experiments" / "runners").is_dir():
    pytest.skip("experiments/runners not present (running outside the repo)", allow_module_level=True)

from experiments.runners.nondiff_models import (  # noqa: E402
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
    DiscreteAttentionNet,
    StaircaseActivation,
    StaircaseNet,
    HardMoELayer,
    HardMoENet,
    SoftMoELayer,
    SoftMoENet,
    compute_expert_utilization,
    MaxSATModel,
    cra_penalty,
    HardPermutationNet,
    SoftPermutationNet,
    PermutationLoss,
)


def test_spike_values_binary():
    """Spike output must be in {0.0, 1.0}."""
    lif = LIFNeuron(beta=0.95, threshold=1.0)
    x = torch.randn(4, 32) * 2.0  # Large values to trigger spikes
    mem = torch.randn(4, 32).abs() * 1.5  # Some above threshold
    spike, _ = lif(x, mem)
    unique_vals = spike.unique()
    for v in unique_vals:
        assert v.item() in (0.0, 1.0), f"Spike value {v.item()} not in {{0, 1}}"


def test_weights_are_int8_rounded():
    """forward() must apply int8 round-quantization to the weights it uses."""
    ql = QuantizedLinear(2, 1, scale=0.5)
    with torch.no_grad():
        ql.weight.copy_(torch.tensor([[0.4, -0.8]]))  # /0.5=[0.8,-1.6]->round[1,-2]->*0.5=[0.5,-1.0]
        ql.bias.zero_()
    x = torch.tensor([[1.0, 1.0]])
    out = ql(x)
    assert torch.allclose(out, torch.tensor([[-0.5]]), atol=1e-6)  # uses quantized weights
    assert not torch.allclose(out, x @ ql.weight.t(), atol=1e-3)  # not the raw linear (-0.4)


def test_effective_weights_binary():
    """forward() must binarize weights to {-1, +1} via sign()."""
    bl = BinaryLinear(2, 1)
    with torch.no_grad():
        bl.weight.copy_(torch.tensor([[0.3, -0.7]]))  # sign -> [+1, -1]
        bl.bias.zero_()
    x = torch.tensor([[2.0, 5.0]])
    out = bl(x)
    assert torch.allclose(out, torch.tensor([[-3.0]]))  # 2*(+1) + 5*(-1)
    assert not torch.allclose(out, x @ bl.weight.t())  # not the raw linear (-2.9)


def test_effective_weights_ternary():
    """forward() must ternarize weights to {-1, 0, +1} using the threshold."""
    tl = TernaryLinear(3, 1, threshold=0.5)
    with torch.no_grad():
        tl.weight.copy_(torch.tensor([[0.9, -0.1, 0.6]]))  # |.|>=0.5 -> [+1, 0, +1]
        tl.bias.zero_()
    x = torch.tensor([[1.0, 1.0, 1.0]])
    out = tl(x)
    assert torch.allclose(out, torch.tensor([[2.0]]))  # +1 + 0 + +1
    assert not torch.allclose(out, x @ tl.weight.t())  # not the raw linear (1.4)


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


@pytest.mark.parametrize(
    "build, input_shape",
    [
        (lambda: BinaryLinearSTE(16, 8), (2, 16)),
        (lambda: TernaryLinearSTE(16, 8, threshold=0.3), (2, 16)),
        (lambda: BinaryConv2dSTE(3, 16, 3, padding=1), (2, 3, 8, 8)),
        (lambda: SoftMoELayer(input_dim=32, hidden_dim=64, num_experts=4), (2, 32)),
        (BinaryMNISTNetSTE, (2, 1, 28, 28)),
        (SoftMoENet, (2, 1, 28, 28)),
    ],
)
def test_ste_passes_gradient_to_every_weight(build, input_shape):
    """The reason the STE exists: a hard forward must still reach every parameter."""
    model = build()
    model(torch.randn(*input_shape)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.shape == p.shape, name
        # An all-zero gradient reaches the parameter and teaches it nothing, which is
        # exactly the saturation the STE exists to avoid.
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().max() > 0, name


def test_weights_are_binary():
    """forward() must convolve with sign-binarized weights, not the raw ones."""
    layer = BinaryConv2d(1, 1, 1, padding=0)  # 1x1 conv -> per-pixel scale by the (binarized) weight
    x = torch.randn(1, 1, 4, 4)
    with torch.no_grad():
        layer.bias.zero_()
        layer.weight.copy_(torch.tensor([[[[0.3]]]]))  # sign -> +1
    assert torch.allclose(layer(x), x, atol=1e-6)  # +1 -> identity
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[[-0.2]]]]))  # sign -> -1
    out = layer(x)
    assert torch.allclose(out, -x, atol=1e-6)  # -1 -> negate
    assert not torch.allclose(out, -0.2 * x, atol=1e-3)  # not the raw (-0.2) conv


def test_output_values_quantized():
    """Output values must be in {0/5, 1/5, 2/5, 3/5, 4/5}."""
    sa = StaircaseActivation(levels=5)
    x = torch.randn(100, 16)  # Enough samples for variety
    out = sa(x)
    valid_values = {0.0, 0.2, 0.4, 0.6, 0.8}
    unique_vals = out.unique()
    for v in unique_vals:
        assert round(v.item(), 6) in valid_values, f"Staircase value {v.item()} not in {valid_values}"


def test_hard_routing_selects_argmax_expert():
    """forward() must return the argmax-gated expert's output (hard top-1),
    not a soft average of the experts."""
    torch.manual_seed(0)
    moe = HardMoELayer(input_dim=8, hidden_dim=6, num_experts=4)
    x = torch.randn(5, 8)
    out = moe(x)
    gate_idx = moe.gate(x).argmax(dim=-1)
    for i in range(x.shape[0]):
        selected = moe.experts[int(gate_idx[i])](x[i : i + 1])[0]
        assert torch.allclose(out[i], selected, atol=1e-5)
    avg = torch.stack([e(x) for e in moe.experts], dim=1).mean(dim=1)
    assert not torch.allclose(out, avg, atol=1e-4)  # hard threshold, not averaging


def test_soft_routing_is_softmax_weighted():
    """forward() must return the softmax-weighted expert mix, not hard top-1."""
    torch.manual_seed(0)
    moe = SoftMoELayer(input_dim=8, hidden_dim=6, num_experts=4)
    x = torch.randn(5, 8)
    out = moe(x)
    weights = torch.softmax(moe.gate(x), dim=-1)
    all_out = torch.stack([e(x) for e in moe.experts], dim=1)
    expected = (all_out * weights.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(out, expected, atol=1e-5)  # soft-weighted mix
    hard = all_out[torch.arange(5), moe.gate(x).argmax(dim=-1)]
    assert not torch.allclose(out, hard, atol=1e-4)  # not hard argmax


def test_param_count_matches_hard():
    hard = HardMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
    soft = SoftMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4)
    hard_params = sum(p.numel() for p in hard.parameters())
    soft_params = sum(p.numel() for p in soft.parameters())
    assert hard_params == soft_params, f"Param count mismatch: hard={hard_params}, soft={soft_params}"


class TestExpertUtilization:
    def test_utilization_sums_to_one(self):
        model = HardMoENet()
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 1, 28, 28), torch.randint(0, 20, (100,)))
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
        dataset = torch.utils.data.TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 20, (50,)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=50)
        result = compute_expert_utilization(model, loader, device="cpu")
        assert result["collapsed"] is True, "Should detect collapse when one expert handles all inputs"
        assert result["max_expert_share"] > 0.90


class TestMaxSATModel:
    def test_no_hidden_layers(self):
        """MaxSATModel should have NO hidden layers, only self.assignments."""
        model = MaxSATModel(num_vars=20)
        param_names = [name for name, _ in model.named_parameters()]
        assert param_names == ["assignments"], (
            f"MaxSATModel should only have 'assignments' parameter, got {param_names}"
        )


def test_known_values():
    """cra_penalty should be 0 for {0, 1} values and positive for 0.5."""
    soft = torch.tensor([0.0, 1.0, 0.5])
    penalty = cra_penalty(soft)
    # For x=0: (2*0-1)^2 = 1, so 1-1=0
    # For x=1: (2*1-1)^2 = 1, so 1-1=0
    # For x=0.5: (2*0.5-1)^2 = 0, so 1-0=1
    assert penalty.item() == pytest.approx(1.0, abs=1e-5)


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

    @pytest.mark.parametrize(
        "model_fn, input_shape, expected_shape",
        [
            (lambda: SpikingMNISTNet(num_steps=3), (2, 1, 28, 28), (2, 2, 10)),
            (lambda: QuantizedMLP(784, 128, 10), (2, 1, 28, 28), (2, 2, 10)),
            (lambda: BinaryMNISTNet(), (2, 1, 28, 28), (2, 2, 10)),
            (lambda: TernaryMNISTNet(), (2, 1, 28, 28), (2, 2, 10)),
            (lambda: DiscreteAttentionNet(784, 128, 10, num_slots=8), (2, 1, 28, 28), (2, 2, 10)),
            (lambda: StaircaseNet(784, 128, 10, levels=5), (2, 1, 28, 28), (2, 2, 10)),
            (
                lambda: HardMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4),
                (2, 1, 28, 28),
                (2, 2, 20),
            ),
            (
                lambda: SoftMoENet(input_dim=784, hidden_dim=128, num_classes=20, num_experts=4),
                (2, 1, 28, 28),
                (2, 2, 20),
            ),
            (lambda: HardPermutationNet(N=10, hidden_dim=64), (4, 10), (2, 4, 10)),
            (lambda: SoftPermutationNet(N=10, hidden_dim=64), (4, 10), (2, 4, 10, 10)),
        ],
    )
    def test_spiking_mnist_vmap(self, model_fn, input_shape, expected_shape):
        model = model_fn()
        x = torch.randn(*input_shape)
        outputs = self._vmap_test(model, x)
        assert outputs.shape == expected_shape


def test_output_is_long_indices():
    """Output dtype is long, values in [0, N)."""
    model = HardPermutationNet(N=10, hidden_dim=64)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.dtype == torch.long
    assert out.max() < 10


class TestSoftPermutationNet:
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
        assert hard_params == soft_params, f"Param count mismatch: hard={hard_params}, soft={soft_params}"


@pytest.mark.parametrize(
    "pred, target, expected",
    [
        (
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            0.0,
        ),
        (torch.tensor([[1, 0, 3, 2]]), torch.tensor([[0, 1, 2, 3]]), 1.0),
        (torch.tensor([[0, 1, 3, 2]]), torch.tensor([[0, 1, 2, 3]]), 0.5),
    ],
)
def test_perfect_match(pred, target, expected):
    loss_fn = PermutationLoss()
    loss = loss_fn(pred, target)
    assert loss.item() == pytest.approx(expected)
