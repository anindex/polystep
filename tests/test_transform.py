"""Tests for ParamLayout flatten/unflatten round-trip correctness.

Covers the common architectures (MLP, CNN, tied weights, BatchNorm),
``float64`` round-trip, parameterless modules, particle shape,
metadata preservation, device handling, and deterministic generators.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from polystep.transform import ParamEntry, ParamLayout, get_device, create_generator


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class SharedWeightsModel(nn.Module):
    """Model where fc2.weight is tied to fc1.weight (shared storage)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        # Tie weights: fc2.weight IS fc1.weight
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundtripMLP:
    """Test 1: Round-trip flatten/unflatten for a simple MLP."""

    def test_roundtrip_mlp(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        assert set(recovered.keys()) == set(sd.keys())
        for key in sd:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


class TestRoundtripCNN:
    """Test 2: Round-trip for CNN with BatchNorm (includes buffers)."""

    def test_roundtrip_cnn(self):
        model = SimpleCNN()
        # Run a forward pass so BatchNorm running stats are non-trivial
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 3, 8, 8))

        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        # Non-trainable buffers (running_mean/var, num_batches_tracked)
        # are excluded from the particle layout - only trainable
        # parameters and their shared aliases are included.
        trainable_ptrs = {p.data_ptr() for n, p in model.named_parameters() if p.requires_grad}
        expected_keys = {k for k, v in sd.items() if v.data_ptr() in trainable_ptrs}
        assert set(recovered.keys()) == expected_keys
        for key in recovered:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


class TestRoundtripSharedParams:
    """Test 3: Shared parameters are deduplicated and reconstructed."""

    def test_roundtrip_shared_params(self):
        model = SharedWeightsModel()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        # Round-trip correctness
        for key in sd:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

        # Deduplication: particle array should be smaller than naive concat
        naive_total = sum(p.numel() for p in sd.values())
        assert layout.total_params < naive_total, (
            f"Shared params not deduplicated: total_params={layout.total_params}, naive={naive_total}"
        )


class TestParamLayoutFrozen:
    """Test 4: ParamLayout is a frozen dataclass."""

    def test_paramlayout_frozen(self):
        model = nn.Linear(2, 2)
        layout = ParamLayout.from_module(model)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            layout.total_params = 999


class TestParticleShape:
    """Test 5: Flatten returns correct 2D shape."""

    def test_particle_shape(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)

        assert particles.ndim == 2, f"Expected 2D, got {particles.ndim}D"
        assert particles.shape[1] == layout.particle_dim
        # Total elements must accommodate all params
        assert particles.shape[0] * particles.shape[1] >= layout.total_params


class TestEmptyModule:
    """Test 6: Empty module does not crash."""

    def test_empty_module(self):
        model = nn.Module()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)

        assert layout.total_params == 0
        assert particles.shape[0] == 0
        assert particles.ndim == 2


class TestParamEntriesMetadata:
    """Test 7: ParamEntry fields are populated correctly."""

    def test_param_entries_metadata(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)

        sd = model.state_dict()
        # Build requires_grad lookup from named_parameters (state_dict
        # values are always detached, so we need the live parameters).
        param_grad = {n: p.requires_grad for n, p in model.named_parameters()}

        for entry in layout.entries:
            assert entry.key in sd, f"Entry key {entry.key} not in state_dict"
            tensor = sd[entry.key]
            assert entry.shape == tuple(tensor.shape), f"{entry.key}: shape {entry.shape} != {tuple(tensor.shape)}"
            assert entry.dtype == tensor.dtype, f"{entry.key}: dtype {entry.dtype} != {tensor.dtype}"
            assert entry.numel == tensor.numel(), f"{entry.key}: numel {entry.numel} != {tensor.numel()}"
            expected_grad = param_grad.get(entry.key, False)
            assert entry.requires_grad == expected_grad, (
                f"{entry.key}: requires_grad {entry.requires_grad} != {expected_grad}"
            )
            # module_path is the prefix before the last dot
            if "." in entry.key:
                expected_path = entry.key.rsplit(".", 1)[0]
            else:
                expected_path = ""
            assert entry.module_path == expected_path, (
                f"{entry.key}: module_path {entry.module_path!r} != {expected_path!r}"
            )


# ---------------------------------------------------------------------------
# Device and determinism tests
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()


class TestFlattenDeviceCPU:
    """``flatten`` produces particles on CPU when the model is on CPU."""

    def test_flatten_device_cpu(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        assert particles.device == torch.device("cpu")


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestFlattenDeviceCUDA:
    """``flatten`` produces particles on CUDA when the model is on CUDA."""

    def test_flatten_device_cuda(self):
        model = SimpleMLP().cuda()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        assert particles.is_cuda


class TestUnflattenPreservesDevice:
    """``unflatten`` returns tensors on the same device as the particles."""

    def test_unflatten_preserves_device(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)
        for key, tensor in recovered.items():
            assert tensor.device == particles.device, f"{key} on {tensor.device}, expected {particles.device}"


class TestGetDeviceCPU:
    """``get_device`` returns CPU for a CPU model."""

    def test_get_device_cpu(self):
        model = nn.Linear(2, 2)
        assert get_device(model) == torch.device("cpu")


class TestGetDeviceEmpty:
    """``get_device`` returns CPU for an empty module."""

    def test_get_device_empty(self):
        model = nn.Module()
        assert get_device(model) == torch.device("cpu")


class TestCreateGeneratorCPU:
    """``create_generator`` returns a CPU generator with deterministic output."""

    def test_create_generator_cpu(self):
        gen = create_generator(seed=42, device=torch.device("cpu"))
        assert isinstance(gen, torch.Generator)
        t1 = torch.randn(5, generator=gen)
        # Reseed and draw again
        gen2 = create_generator(seed=42, device=torch.device("cpu"))
        t2 = torch.randn(5, generator=gen2)
        assert torch.equal(t1, t2)


class TestCreateGeneratorDeterminism:
    """Same seed yields identical tensors; different seeds differ."""

    def test_create_generator_determinism(self):
        gen_a = create_generator(seed=42, device=torch.device("cpu"))
        gen_b = create_generator(seed=42, device=torch.device("cpu"))
        ta = torch.randn(10, generator=gen_a)
        tb = torch.randn(10, generator=gen_b)
        assert torch.equal(ta, tb), "Same seed must produce identical tensors"

        gen_c = create_generator(seed=99, device=torch.device("cpu"))
        tc = torch.randn(10, generator=gen_c)
        assert not torch.equal(ta, tc), "Different seed must produce different tensors"


class TestDoubleDtype:
    """``float64`` parameters survive the flatten / unflatten round-trip
    bitwise and preserve their dtype."""

    def test_double_dtype_round_trip(self):
        model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2)).double()
        layout = ParamLayout.from_module(model)

        assert layout.dominant_dtype == torch.float64

        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        for key in sd:
            assert recovered[key].dtype == sd[key].dtype, f"{key}: dtype {recovered[key].dtype} != {sd[key].dtype}"
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


class TestSolverDeterminism:
    """``PolyStep.run`` with the same seed produces identical trajectories."""

    def test_solver_determinism(self):
        from polystep import PolyStep, Ackley

        obj = Ackley(dim=2)
        solver = PolyStep(objective_fn=obj, dim=2, max_iterations=5)
        X = torch.randn(8, 2)

        gen1 = create_generator(seed=42, device=torch.device("cpu"))
        gen2 = create_generator(seed=42, device=torch.device("cpu"))

        s1 = solver.run(X.clone(), generator=gen1)
        s2 = solver.run(X.clone(), generator=gen2)

        assert torch.equal(s1.X, s2.X), "Solver must be deterministic with same seed"
