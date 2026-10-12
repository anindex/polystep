"""Tests for ParamLayout flatten/unflatten round-trip correctness.

Covers MLP, CNN, tied weights, BatchNorm, float64, parameterless modules,
particle shape and padding, metadata, device handling, and deterministic generators.
"""

import dataclasses
import logging

import pytest
import torch
import torch.nn as nn

from polystep.transform import ParamLayout, create_generator


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


class SharedWeightsModel(nn.Module):
    """Model where fc2.weight is tied to fc1.weight (shared storage)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_roundtrip_mlp():
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    recovered = layout.unflatten(particles)

    sd = model.state_dict()
    assert set(recovered.keys()) == set(sd.keys())
    for key in sd:
        assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


def test_roundtrip_cnn():
    """Round-trip for CNN with BatchNorm (includes buffers)."""
    model = SimpleCNN()
    # Run a forward pass so BatchNorm running stats are non-trivial
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 8, 8))

    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    recovered = layout.unflatten(particles)

    sd = model.state_dict()
    # Non-trainable buffers are excluded from the layout; only trainable
    # parameters and their shared aliases are included.
    trainable_ptrs = {p.data_ptr() for n, p in model.named_parameters() if p.requires_grad}
    expected_keys = {k for k, v in sd.items() if v.data_ptr() in trainable_ptrs}
    assert set(recovered.keys()) == expected_keys
    for key in recovered:
        assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


def test_roundtrip_shared_params():
    """Shared parameters are deduplicated and reconstructed."""
    model = SharedWeightsModel()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    recovered = layout.unflatten(particles)

    sd = model.state_dict()
    for key in sd:
        assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"

    naive_total = sum(p.numel() for p in sd.values())
    assert layout.total_params < naive_total, (
        f"Shared params not deduplicated: total_params={layout.total_params}, naive={naive_total}"
    )


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


def test_bf16_tied_state_dict_roundtrip():
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


def test_paramlayout_frozen():
    model = nn.Linear(2, 2)
    layout = ParamLayout.from_module(model)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        layout.total_params = 999


def test_particle_shape():
    """Flatten returns correct 2D shape."""
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)

    assert particles.ndim == 2, f"Expected 2D, got {particles.ndim}D"
    assert particles.shape[1] == layout.particle_dim
    assert particles.shape[0] * particles.shape[1] >= layout.total_params


# nn.Linear(13, 7) is 98 params: pdim 1 needs no padding, 3 and 8 do.
@pytest.mark.parametrize("particle_dim", [1, 3, 8])
def test_roundtrip_various_particle_dims(particle_dim):
    model = nn.Linear(13, 7)
    layout = ParamLayout.from_module(model, particle_dim=particle_dim)

    assert layout.padded_size % particle_dim == 0
    flat = layout.flatten(model)
    assert flat.shape[1] == particle_dim

    recovered = layout.unflatten(flat)
    sd = model.state_dict()
    for key in recovered:
        assert torch.equal(sd[key], recovered[key])


def test_padded_roundtrip_preserves_state_dict():
    """flatten -> unflatten must produce a state_dict identical to the
    original (within dtype rounding) and must not expose padding bytes
    as state_dict keys."""
    model = nn.Sequential(nn.Linear(3, 1, bias=True), nn.Linear(1, 1, bias=False))
    # 4 + 1 = 5 params; padded to 6 with particle_dim=2 -> 1 element of padding

    layout = ParamLayout.from_module(model, particle_dim=2)
    assert layout.total_params == 5
    assert layout.padded_size == 6, f"expected padded_size=6 for 5 params with particle_dim=2; got {layout.padded_size}"

    flat = layout.flatten(model)
    assert flat.numel() == 6
    sd = layout.unflatten(flat)

    # Padding must not appear as a state_dict key.
    expected_keys = {"0.weight", "0.bias", "1.weight"}
    assert set(sd.keys()) == expected_keys, f"state_dict keys leaked padding: {set(sd.keys()) - expected_keys}"

    # Values round-trip exactly.
    orig = model.state_dict()
    for k in expected_keys:
        assert torch.allclose(sd[k], orig[k]), f"value mismatch on {k}"


def test_empty_module():
    model = nn.Module()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)

    assert layout.total_params == 0
    assert particles.shape[0] == 0
    assert particles.ndim == 2


def test_param_entries_metadata():
    """ParamEntry fields are populated correctly."""
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)

    sd = model.state_dict()
    # state_dict values are detached, so requires_grad must come from the live parameters.
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
        if "." in entry.key:
            expected_path = entry.key.rsplit(".", 1)[0]
        else:
            expected_path = ""
        assert entry.module_path == expected_path, (
            f"{entry.key}: module_path {entry.module_path!r} != {expected_path!r}"
        )


HAS_CUDA = torch.cuda.is_available()


def test_flatten_device_cpu():
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    assert particles.device == torch.device("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_flatten_device_cuda():
    model = SimpleMLP().cuda()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    assert particles.is_cuda


def test_unflatten_preserves_device():
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particles = layout.flatten(model)
    recovered = layout.unflatten(particles)
    for key, tensor in recovered.items():
        assert tensor.device == particles.device, f"{key} on {tensor.device}, expected {particles.device}"


def test_create_generator_determinism():
    """Same seed yields identical tensors; different seeds differ."""
    gen_a = create_generator(seed=42, device=torch.device("cpu"))
    gen_b = create_generator(seed=42, device=torch.device("cpu"))
    ta = torch.randn(10, generator=gen_a)
    tb = torch.randn(10, generator=gen_b)
    assert torch.equal(ta, tb), "Same seed must produce identical tensors"

    gen_c = create_generator(seed=99, device=torch.device("cpu"))
    tc = torch.randn(10, generator=gen_c)
    assert not torch.equal(ta, tc), "Different seed must produce different tensors"


def test_double_dtype_round_trip():
    """``float64`` parameters survive the flatten / unflatten round-trip
    bitwise and preserve their dtype."""
    model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2)).double()
    layout = ParamLayout.from_module(model)

    assert layout.dominant_dtype == torch.float64

    particles = layout.flatten(model)
    recovered = layout.unflatten(particles)

    sd = model.state_dict()
    for key in sd:
        assert recovered[key].dtype == sd[key].dtype, f"{key}: dtype {recovered[key].dtype} != {sd[key].dtype}"
        assert torch.equal(sd[key], recovered[key]), f"Mismatch in {key}"


def test_solver_determinism():
    """``PolyStep.run`` with the same seed produces identical trajectories."""
    from polystep import Ackley
    from polystep.solver import PolyStep

    obj = Ackley(dim=2)
    solver = PolyStep(objective_fn=obj, dim=2, max_iterations=5)
    X = torch.randn(8, 2)

    gen1 = create_generator(seed=42, device=torch.device("cpu"))
    gen2 = create_generator(seed=42, device=torch.device("cpu"))

    s1 = solver.run(X.clone(), generator=gen1)
    s2 = solver.run(X.clone(), generator=gen2)

    assert torch.equal(s1.X, s2.X), "Solver must be deterministic with same seed"


def test_differently_shaped_buffer_views_rejected():
    """data_ptr alone does not make two tensors the same parameter."""

    class ReshapedAlias(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3))
            # Same storage, different shape: unflatten would write (4, 3) to both.
            self.flat = nn.Parameter(self.weight.detach().view(12))
            self.flat.data = self.weight.data.view(12)

        def forward(self, x):
            return x @ self.weight.T

    with pytest.raises(ValueError, match="share storage but are different views"):
        ParamLayout.from_module(ReshapedAlias())


def test_transposed_alias_rejected():
    """A square weight and its transpose match on data_ptr AND shape, so a tie test
    on those alone would pass and unflatten would write the canonical tensor to
    both, silently transposing one of them.
    """

    class TransposedAlias(nn.Module):
        def __init__(self):
            super().__init__()
            base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            self.weight = nn.Parameter(base)
            self.weight_t = nn.Parameter(base)
            self.weight_t.data = base.t()

        def forward(self, x):
            return x @ self.weight

    model = TransposedAlias()
    assert model.weight.data_ptr() == model.weight_t.data_ptr()
    assert model.weight.shape == model.weight_t.shape
    assert model.weight.stride() != model.weight_t.stride()

    with pytest.raises(ValueError, match="share storage but are different views"):
        ParamLayout.from_module(model)
