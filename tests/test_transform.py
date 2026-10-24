"""Tests for ParamLayout flatten/unflatten round-trip correctness.

Covers MLP, CNN, tied weights, BatchNorm, float64, parameterless modules,
particle shape and padding, metadata, device handling, and deterministic generators.
"""

import dataclasses
import logging
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from polystep import transform
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
        self.lm_head.weight = self.embedding.weight

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

    canonical = next(e for e in layout.entries if e.key == "embedding.weight")
    assert "lm_head.weight" in canonical.shared_with

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

    expected_keys = {"0.weight", "0.bias", "1.weight"}
    assert set(sd.keys()) == expected_keys, f"state_dict keys leaked padding: {set(sd.keys()) - expected_keys}"

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

    with pytest.raises(ValueError, match="share storage but are different tensors"):
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

    with pytest.raises(ValueError, match="share storage but are different tensors"):
        ParamLayout.from_module(model)


@pytest.mark.parametrize("frozen", [False, True])
def test_layout_distinct_parameter_alias_raises(frozen):
    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 3))
    model[2].weight.data = model[0].weight.data
    model[2].weight.requires_grad_(not frozen)

    with pytest.raises(ValueError, match="share storage but are different tensors"):
        ParamLayout.from_module(model)


@pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors:UserWarning")
def test_layout_frozen_offset_alias_raises():
    model = nn.Module()
    flat = torch.arange(6.0)
    model.a = nn.Parameter(flat[2:4])
    model.before = nn.Parameter(flat[:2], requires_grad=False)
    model.after = nn.Parameter(flat[4:], requires_grad=False)
    cols = torch.arange(8.0).view(2, 4)
    model.cols_train = nn.Parameter(cols[:, :2])
    model.cols_frozen = nn.Parameter(cols[:, 2:], requires_grad=False)
    model.register_buffer("sparse", torch.eye(2).to_sparse())
    assert [e.key for e in ParamLayout.from_module(model).entries] == ["a", "cols_train"]

    model.register_buffer("nested", torch.nested.as_nested_tensor(cols))
    with pytest.raises(ValueError, match="'nested' and trainable parameter 'cols_train' share storage"):
        ParamLayout.from_module(model)

    del model.nested
    model.overlap = nn.Parameter(flat[1:3], requires_grad=False)
    with pytest.raises(ValueError, match="'overlap' and trainable parameter 'a' share storage"):
        ParamLayout.from_module(model)

    del model.overlap
    model.cols_shifted = nn.Parameter(cols[:, 1:3], requires_grad=False)
    with pytest.raises(ValueError, match="'cols_shifted' and trainable parameter 'cols_train' share storage"):
        ParamLayout.from_module(model)

    del model.cols_shifted
    grid = torch.arange(8.0).view(2, 4)
    model.grid_frozen = nn.Parameter(grid[:, :2], requires_grad=False)
    model.grid_train = nn.Parameter(grid[:, 1:3])
    with pytest.raises(ValueError, match="'grid_frozen' and trainable parameter 'grid_train' share storage"):
        ParamLayout.from_module(model)

    model.grid_train = nn.Parameter(grid[:, :3])
    model.grid_frozen = nn.Parameter(grid[:, 2:3], requires_grad=False)
    with pytest.raises(ValueError, match="'grid_frozen' and trainable parameter 'grid_train' share storage"):
        ParamLayout.from_module(model)

    del model.grid_frozen, model.grid_train
    wide = torch.zeros(8, dtype=torch.float64)
    model.wide_frozen = nn.Parameter(wide.view(2, 4)[:, :1], requires_grad=False)
    model.narrow_train = nn.Parameter(wide.view(torch.float32)[4:12].view(2, 4)[:, :1])
    with pytest.raises(ValueError, match="'wide_frozen' and trainable parameter 'narrow_train' share storage"):
        ParamLayout.from_module(model)

    del model.wide_frozen, model.narrow_train
    model.rows_frozen = nn.Parameter(grid[:, :1], requires_grad=False)
    model.pairs_train = nn.Parameter(grid.view(-1)[2:6].view(2, 2)[:, :1])
    with pytest.raises(ValueError, match="'rows_frozen' and trainable parameter 'pairs_train' share storage"):
        ParamLayout.from_module(model)


def test_alias_cache_misses_changed_model():
    model, matrix = nn.Module(), torch.arange(6.0).view(2, 3)
    model.a = nn.Parameter(matrix[:, 0])
    model.b = nn.Parameter(matrix[:, 1], requires_grad=False)
    model.register_buffer("c", matrix[:, 1])
    ParamLayout.from_module(model)

    model.b.requires_grad_(True)
    with pytest.raises(ValueError, match="'c' and trainable parameter 'b' share storage"):
        ParamLayout.from_module(model)

    model.b.requires_grad_(False)
    ParamLayout.from_module(model)
    model.c = matrix.view(-1)[1:5]
    with pytest.raises(ValueError, match="'c' and trainable parameter 'a' share storage"):
        ParamLayout.from_module(model)


def test_alias_cache_misses_changed_strides():
    storage, model = torch.zeros(40), nn.Module()
    model.w = nn.Parameter(storage.as_strided((4, 4), (8, 2), 0))
    model.register_buffer("b", storage.as_strided((4, 4), (8, 2), 1))
    ParamLayout.from_module(model)

    model.b = storage.as_strided((4, 4), (9, 1), 1)
    with pytest.raises(ValueError, match="'b' and trainable parameter 'w' share storage"):
        ParamLayout.from_module(model)


def test_numpy_wrapper_alias_raises():
    backing = np.arange(6, dtype=np.float32)
    model = nn.Module()
    model.a = nn.Parameter(torch.from_numpy(backing[2:4]))
    model.frozen = nn.Parameter(torch.from_numpy(backing[1:3]), requires_grad=False)
    assert model.a.untyped_storage().data_ptr() != model.frozen.untyped_storage().data_ptr()

    with pytest.raises(ValueError, match="share storage but are different tensors"):
        ParamLayout.from_module(model)


def test_disjoint_slices_layout_builds_fast(monkeypatch):
    calls = []
    monkeypatch.setattr(transform, "_shares_bytes", lambda a, b: calls.append((a, b)) or True)
    model, flat = nn.Module(), torch.ones(400)
    for i in range(100):
        model.register_parameter(f"p{i}", nn.Parameter(flat[4 * i : 4 * i + 4], requires_grad=i % 2 == 0))

    assert ParamLayout.from_module(model).total_params == 200
    assert calls == []


def test_strided_parameter_gap_buffer_builds():
    model, flat = nn.Module(), torch.arange(12.0)
    model.w = nn.Parameter(flat[0::4])
    model.a = nn.Parameter(flat[1:3])
    model.register_buffer("odd", flat[3:4])

    assert ParamLayout.from_module(model).total_params == 5


def test_strided_parameter_late_buffer_raises():
    model, flat = nn.Module(), torch.arange(12.0)
    model.w = nn.Parameter(flat[0::4])
    model.a = nn.Parameter(flat[1:3])
    model.register_buffer("late", flat[8:9])

    with pytest.raises(ValueError, match="'late' and trainable parameter 'w' share storage"):
        ParamLayout.from_module(model)


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize(
    "view",
    [lambda w: w, lambda w: w.detach(), lambda w: w.detach().reshape(1, 2), lambda w: w.detach().view(torch.int32)],
    ids=["parameter", "same", "reshaped", "int32"],
)
def test_buffer_alias_parameter_raises(persistent, view):
    model = nn.Module()
    model.w = nn.Parameter(torch.tensor([1.0, 2.0]))
    model.register_buffer("alias", view(model.w), persistent=persistent)

    with pytest.raises(ValueError, match="'alias' and trainable parameter 'w' share storage"):
        ParamLayout.from_module(model)


def test_many_ties_layout_builds_linear(monkeypatch):
    spans, byte_span = [], transform._byte_span
    monkeypatch.setattr(transform, "_byte_span", lambda t: spans.append(t) or byte_span(t))
    model, weight, table = nn.Module(), nn.Parameter(torch.ones(4)), torch.zeros(4)
    for i in range(200):
        child = nn.Module()
        child.weight = weight
        child.register_buffer("table", table)
        model.add_module(f"m{i}", child)

    layout = ParamLayout.from_module(model)
    assert layout.total_params == 4 and len(layout.shared_groups[0]) == 200
    assert len(spans) == 2


@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta:UserWarning")
@pytest.mark.parametrize("part", ["coo_values", "csr_values", "coo_indices", "csr_indices"])
def test_sparse_buffer_alias_raises(part):
    model = nn.Module()
    model.w = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    rows, cols, values = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), model.w.detach().flatten()
    if part == "coo_values":
        alias = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (2, 2), check_invariants=True)
    elif part == "csr_values":
        alias = torch.sparse_csr_tensor(torch.tensor([0, 2, 4]), cols, values, (2, 2), check_invariants=True)
    elif part == "coo_indices":
        indices = model.w.detach().view(torch.int64)
        alias = torch.sparse_coo_tensor(indices, torch.ones(2), (2, 2), check_invariants=True)
    else:
        crow = model.w.detach().flatten().view(torch.int64)
        alias = torch.sparse_csr_tensor(crow, cols[:0], torch.zeros(0), (3, 2), check_invariants=True)
    model.register_buffer("alias", alias)

    with pytest.raises(ValueError, match="'alias' and trainable parameter 'w' share storage"):
        ParamLayout.from_module(model)


@pytest.mark.parametrize("field", ["offsets", "lengths"])
def test_jagged_offsets_alias_raises(field):
    offsets, lengths = torch.tensor([0, 2, 4]), torch.tensor([2, 2])
    model = nn.Module()
    model.w = nn.Parameter((offsets if field == "offsets" else lengths).view(torch.float64))
    model.register_buffer("nested", torch.nested.nested_tensor_from_jagged(torch.ones(4, 2), offsets, lengths=lengths))

    with pytest.raises(ValueError, match="'nested' and trainable parameter 'w' share storage"):
        ParamLayout.from_module(model)


def test_reordered_state_dict_keeps_tie_name():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.z, self.a = nn.Linear(2, 2), nn.Linear(2, 2)
            self.a.weight = self.z.weight
            self.register_state_dict_post_hook(self.sort_keys)

        @staticmethod
        def sort_keys(module, state, prefix, local_metadata):
            for key in sorted(state):
                state.move_to_end(key)

    layout = ParamLayout.from_module(Tied())
    assert [e.key for e in layout.entries] == ["a.bias", "z.bias", "z.weight"]
    assert layout.shared_groups == (("z.weight", "a.weight"),)


def test_strided_span_sweep_stays_linear():
    events, previous = [], sys.gettrace()

    def count(frame, event, arg):
        events.append(event)
        return count

    n, model = 4000, nn.Module()
    flat = torch.ones(n + 2)
    model.long = nn.Parameter(flat[:: n + 1])
    for i in range(n):
        model.register_parameter(f"p{i}", nn.Parameter(flat[i + 1 : i + 2]))

    sys.settrace(lambda frame, event, arg: count if frame.f_code.co_filename == transform.__file__ else None)
    try:
        assert ParamLayout.from_module(model).total_params == n + 2
    finally:
        sys.settrace(previous)
    assert len(events) < 200 * n
