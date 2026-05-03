"""Parameter round-trip flatten/unflatten tests for various nn.Module architectures.

Complements test_transform.py with additional architectures and edge cases:
MLP with ReLU (Sequential), deeper CNN, shared weights with bias, BatchNorm buffers,
metadata preservation, float64 dtype, and parameterless modules.
"""
import torch
import torch.nn as nn

from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# Helper models (distinct from test_transform.py models)
# ---------------------------------------------------------------------------


class SequentialMLP(nn.Module):
    """MLP built with nn.Sequential (tests unnamed layers)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        return self.net(x)


class DeeperCNN(nn.Module):
    """CNN with two conv layers, pooling, and a classifier head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SharedBiasModel(nn.Module):
    """Model where two layers share the same weight tensor."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        # Tie weights (not biases)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class BatchNormModel(nn.Module):
    """Model with BatchNorm1d that has running_mean/var buffers."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 5)

    def forward(self, x):
        return self.out(self.bn(torch.relu(self.fc(x))))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParamRoundTrip:
    """Round-trip flatten/unflatten tests for various architectures."""

    def test_mlp_round_trip(self):
        """Sequential MLP: flatten then unflatten produces bitwise-identical params."""
        model = SequentialMLP()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        assert set(recovered.keys()) == set(sd.keys())
        for key in sd:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

    def test_cnn_round_trip(self):
        """Deeper CNN with Conv2d, pooling, and Linear: bitwise round-trip."""
        model = DeeperCNN()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        assert set(recovered.keys()) == set(sd.keys())
        for key in sd:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

    def test_shared_parameters_round_trip(self):
        """Shared weight model: deduplicates and recovers correctly."""
        model = SharedBiasModel()
        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        for key in sd:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

        # Verify deduplication happened (shared weight counted once)
        naive_total = sum(t.numel() for t in sd.values())
        assert layout.total_params < naive_total, \
            f"Expected deduplication: total_params={layout.total_params}, naive={naive_total}"

        # Verify shared group recorded
        assert len(layout.shared_groups) > 0, "No shared groups found"

    def test_model_with_buffers_round_trip(self):
        """BatchNorm1d model with running_mean/var buffers: bitwise round-trip."""
        model = BatchNormModel()
        # Run a forward pass to populate running stats
        model.train()
        with torch.no_grad():
            model(torch.randn(8, 10))
        model.eval()

        layout = ParamLayout.from_module(model)
        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        # Non-trainable entries (all buffers: running_mean, running_var,
        # num_batches_tracked) are excluded from the particle layout.
        # Only requires_grad=True parameters are optimized.
        trainable_ptrs = {p.data_ptr() for n, p in model.named_parameters()
                          if p.requires_grad}
        expected_keys = {k for k, v in sd.items()
                         if v.data_ptr() in trainable_ptrs}
        assert set(recovered.keys()) == expected_keys
        for key in recovered:
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

        # Verify all buffers (float and integer) are excluded
        buffer_keys = {k for k, _ in model.named_buffers()}
        for bk in buffer_keys:
            assert bk not in recovered, f"Integer buffer {bk} should be excluded from layout"

    def test_layout_metadata_preserved(self):
        """ParamLayout preserves parameter metadata: keys, shapes, flat_size."""
        model = DeeperCNN()
        layout = ParamLayout.from_module(model)
        sd = model.state_dict()

        # All canonical entry keys should be in state_dict
        for entry in layout.entries:
            assert entry.key in sd, f"Entry key {entry.key} not in state_dict"
            assert entry.shape == tuple(sd[entry.key].shape), \
                f"{entry.key}: shape mismatch {entry.shape} vs {tuple(sd[entry.key].shape)}"

        # flat_size consistency
        particles = layout.flatten(model)
        flat_size = particles.numel()
        assert flat_size >= layout.total_params, \
            f"Flat size {flat_size} < total_params {layout.total_params}"
        assert flat_size == layout.padded_size, \
            f"Flat size {flat_size} != padded_size {layout.padded_size}"

    def test_double_dtype_round_trip(self):
        """Float64 model: flatten/unflatten preserves float64 dtype."""
        model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2)).double()
        layout = ParamLayout.from_module(model)

        assert layout.dominant_dtype == torch.float64, \
            f"Expected float64, got {layout.dominant_dtype}"

        particles = layout.flatten(model)
        recovered = layout.unflatten(particles)

        sd = model.state_dict()
        for key in sd:
            assert recovered[key].dtype == sd[key].dtype, \
                f"{key}: dtype {recovered[key].dtype} != {sd[key].dtype}"
            assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

    def test_empty_module_handling(self):
        """Parameterless module (nn.ReLU) handles gracefully."""
        model = nn.ReLU()
        layout = ParamLayout.from_module(model)

        assert layout.total_params == 0
        assert len(layout.entries) == 0

        particles = layout.flatten(model)
        assert particles.ndim == 2
        assert particles.shape[0] == 0

        recovered = layout.unflatten(particles)
        assert len(recovered) == 0
