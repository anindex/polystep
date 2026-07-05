"""Tests for NNCostEvaluator, ParamLayout.batch_unflatten(), chunked eval, and cost matrix."""
import warnings

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from polystep.transform import ParamLayout
from polystep.cost_nn import NNCostEvaluator, compute_nn_cost_matrix, auto_detect_chunk_size


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=5, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SharedWeightsModel(nn.Module):
    """Model where fc2.weight is tied to fc1.weight."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.bn(self.fc1(x))))


class VmapIncompatibleModel(nn.Module):
    """Model that calls .item() in forward -- incompatible with vmap."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self._scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # .item() is not traceable by vmap
        s = self._scale.item()
        return self.fc(x) * s


# ---------------------------------------------------------------------------
# batch_unflatten tests
# ---------------------------------------------------------------------------


class TestBatchUnflattenShape:
    """Test 1: batch_unflatten returns correct shapes."""

    def test_batch_unflatten_shape(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 8
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        sd = model.state_dict()
        for key in sd:
            assert key in stacked, f"Missing key {key}"
            expected_shape = (N, *sd[key].shape)
            assert stacked[key].shape == expected_shape, (
                f"{key}: expected {expected_shape}, got {stacked[key].shape}"
            )


class TestBatchUnflattenValues:
    """Test 2: batch_unflatten agrees with per-particle unflatten."""

    def test_batch_unflatten_values(self):
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 4
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        for i in range(N):
            single = layout.unflatten(batch[i])
            for key in single:
                torch.testing.assert_close(
                    stacked[key][i],
                    single[key],
                    msg=lambda m: f"Mismatch at particle {i}, key {key}: {m}",
                )


class TestBatchUnflattenSharedParams:
    """Test 3: Shared params produce aliased tensors."""

    def test_batch_unflatten_shared_params(self):
        model = SharedWeightsModel()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 3
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        # Both canonical and alias key must be present
        assert "fc1.weight" in stacked
        assert "fc2.weight" in stacked
        # They must be the same tensor object (identity)
        assert stacked["fc2.weight"] is stacked["fc1.weight"]


# ---------------------------------------------------------------------------
# NNCostEvaluator tests
# ---------------------------------------------------------------------------


class TestEvaluatorMlpVmap:
    """Test 4: Evaluator returns (N,) losses via vmap for MLP."""

    def test_evaluator_mlp_vmap(self):
        from polystep.cost_nn import NNCostEvaluator

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)
        )
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 16
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.01
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs = torch.randn(32, 4)
        targets = torch.randint(0, 2, (32,))
        losses = evaluator.evaluate(stacked, inputs, targets)

        assert losses.shape == (N,), f"Expected ({N},), got {losses.shape}"
        assert losses.isfinite().all(), "Non-finite losses"


class TestEvaluatorUnsupervised:
    """Test 5: Unsupervised loss (targets=None) works."""

    def test_evaluator_unsupervised(self):
        from polystep.cost_nn import NNCostEvaluator

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 8
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(
            model, loss_fn=lambda output: output.pow(2).mean()
        )
        inputs = torch.randn(16, 4)
        losses = evaluator.evaluate(stacked, inputs, targets=None)

        assert losses.shape == (N,)
        assert losses.isfinite().all()


class TestEvaluatorDifferentParams:
    """Test 6: Different params produce different losses."""

    def test_evaluator_different_params_different_losses(self):
        from polystep.cost_nn import NNCostEvaluator

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 10
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 1.0  # large perturbation
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs = torch.randn(32, 4)
        targets = torch.randint(0, 2, (32,))
        losses = evaluator.evaluate(stacked, inputs, targets)

        # Not all losses should be identical
        assert not torch.all(losses == losses[0]), (
            "All losses identical despite different params"
        )


class TestEvaluatorBatchNorm:
    """Test 7: Evaluator handles BatchNorm (frozen buffers)."""

    def test_evaluator_batchnorm(self):
        from polystep.cost_nn import NNCostEvaluator

        model = MLPWithBatchNorm()
        # Run a forward pass in train mode to populate running stats
        model.train()
        with torch.no_grad():
            model(torch.randn(32, 4))

        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 8
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.01
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))
        losses = evaluator.evaluate(stacked, inputs, targets)

        assert losses.shape == (N,)
        assert losses.isfinite().all(), "Non-finite losses with BatchNorm"


class TestEvaluatorFallbackWarning:
    """Test 8: Fallback to loop with warning for vmap-incompatible model."""

    def test_evaluator_fallback_warning(self):
        from polystep.cost_nn import NNCostEvaluator

        model = VmapIncompatibleModel()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 4
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            losses = evaluator.evaluate(stacked, inputs, targets)

        # Check warning was emitted
        fallback_warnings = [
            x for x in w if "Falling back" in str(x.message)
        ]
        assert len(fallback_warnings) > 0, "Expected 'Falling back' warning"

        # Results should still be valid
        assert losses.shape == (N,)
        assert losses.isfinite().all()


class TestEvaluatorNoGrad:
    """Test 9: Evaluation does not attach gradients to model params."""

    def test_evaluator_no_grad(self):
        from polystep.cost_nn import NNCostEvaluator

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 5
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))
        evaluator.evaluate(stacked, inputs, targets)

        for name, param in model.named_parameters():
            assert param.grad is None, f"{name} has gradient attached"


class TestBatchedLinearRespectsCrossEntropyConfig:
    """Configured CrossEntropyLoss must not take the bmm fast path (it hardcodes
    mean reduction and no class weights)."""

    def test_vanilla_ce_uses_fast_path(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        assert evaluator._batched_linear is not None

    def test_label_smoothing_falls_back_and_matches(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 6
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.05
        stacked = layout.batch_unflatten(batch)

        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 3, (16,))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        evaluator = NNCostEvaluator(model, loss_fn)
        # Configured loss must fall back so the real (smoothed) loss is used.
        assert evaluator._batched_linear is None

        losses = evaluator.evaluate(stacked, inputs, targets)

        model.eval()
        buffers = dict(model.named_buffers())
        expected = torch.zeros(N)
        for i in range(N):
            sd = {k: v[i] for k, v in stacked.items()}
            with torch.no_grad():
                out = functional_call(model, {**sd, **buffers}, (inputs,))
                expected[i] = loss_fn(out, targets)

        torch.testing.assert_close(losses, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Chunked evaluation tests# ---------------------------------------------------------------------------


class TestChunkSizeProducesSameResult:
    """Test 10: Chunked evaluation matches unchunked."""

    def test_chunk_size_produces_same_result(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 50
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.01
        stacked = layout.batch_unflatten(batch)

        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))

        ev_full = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=None)
        ev_chunk = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=4)

        losses_full = ev_full.evaluate(stacked, inputs, targets)
        losses_chunk = ev_chunk.evaluate(stacked, inputs, targets)

        torch.testing.assert_close(losses_chunk, losses_full, atol=1e-5, rtol=1e-5)


class TestChunkSizeOne:
    """Test 11: chunk_size=1 (extreme) still produces correct results."""

    def test_chunk_size_one(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 10
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.01
        stacked = layout.batch_unflatten(batch)

        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))

        ev_full = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=None)
        ev_one = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=1)

        losses_full = ev_full.evaluate(stacked, inputs, targets)
        losses_one = ev_one.evaluate(stacked, inputs, targets)

        torch.testing.assert_close(losses_one, losses_full, atol=1e-5, rtol=1e-5)


class TestAutoDetectChunkSizeCpu:
    """Test 12: auto_detect_chunk_size returns None for CPU model (even on GPU machine)."""

    def test_auto_detect_chunk_size_cpu(self):
        model = nn.Linear(100, 50)  # CPU model
        result = auto_detect_chunk_size(model)
        assert result is None, f"Expected None for CPU model, got {result}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestAutoDetectChunkSizeGpu:
    """Test 13: auto_detect_chunk_size returns positive int for GPU model."""

    def test_auto_detect_chunk_size_returns_positive(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).cuda()
        result = auto_detect_chunk_size(model)
        assert isinstance(result, int), f"Expected int, got {type(result)}"
        assert result > 0, f"Expected positive, got {result}"


class TestEvaluatorAutoChunkSize:
    """Test 14: NNCostEvaluator with chunk_size='auto' works correctly."""

    def test_evaluator_auto_chunk_size(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 20
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.01
        stacked = layout.batch_unflatten(batch)

        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size="auto")
        # CPU model should always resolve to None, even on GPU machines
        resolved = evaluator.chunk_size
        assert resolved is None, f"Expected None for CPU model, got {resolved}"

        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))
        losses = evaluator.evaluate(stacked, inputs, targets)

        assert losses.shape == (N,)
        assert losses.isfinite().all()


class TestAutoChunkSizeCached:
    """Test 14b: Auto chunk_size should be computed once and cached."""

    def test_auto_chunk_size_cached(self):
        model = nn.Linear(4, 2)
        evaluator = NNCostEvaluator(model, nn.MSELoss(), chunk_size="auto")
        cs1 = evaluator.chunk_size
        cs2 = evaluator.chunk_size
        assert cs1 == cs2  # same value (None on CPU)
        # Verify internal cache attribute exists and sentinel was replaced
        assert hasattr(evaluator, '_chunk_size_cached')
        from polystep.cost_nn import _UNSET
        assert evaluator._chunk_size_cached is not _UNSET


# ---------------------------------------------------------------------------
# compute_nn_cost_matrix tests# ---------------------------------------------------------------------------


def _make_probe_setup(in_dim=4, hidden=8, out_dim=2, P=10, V=4, K=3):
    """Helper: create model, layout, evaluator, and probe array."""
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    layout = ParamLayout.from_module(model)
    p = layout.flatten(model)
    D = p.shape[0] * p.shape[1]  # flat size
    X_probe = p.reshape(1, 1, 1, D).expand(P, V, K, D).clone()
    X_probe += torch.randn(P, V, K, D) * 0.01
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs = torch.randn(16, in_dim)
    targets = torch.randint(0, out_dim, (16,))
    return model, layout, evaluator, X_probe, inputs, targets


class TestComputeNNCostMatrixShape:
    """Test 15: compute_nn_cost_matrix returns correct (P, V) shape."""

    def test_compute_nn_cost_matrix_shape(self):
        P, V, K = 10, 4, 3
        model, layout, evaluator, X_probe, inputs, targets = _make_probe_setup(P=P, V=V, K=K)
        cost_mat = compute_nn_cost_matrix(evaluator, X_probe, layout, inputs, targets)
        assert cost_mat.shape == (P, V), f"Expected ({P},{V}), got {cost_mat.shape}"


class TestComputeNNCostMatrixValues:
    """Test 16: Cost matrix has finite, non-identical values."""

    def test_compute_nn_cost_matrix_values(self):
        model, layout, evaluator, X_probe, inputs, targets = _make_probe_setup(P=10, V=4, K=3)
        cost_mat = compute_nn_cost_matrix(evaluator, X_probe, layout, inputs, targets)
        assert cost_mat.isfinite().all(), "Non-finite costs"
        assert not torch.all(cost_mat == cost_mat[0, 0]), "All costs identical"


class TestComputeNNCostMatrixChunked:
    """Test 17: Chunked compute_nn_cost_matrix matches unchunked."""

    def test_compute_nn_cost_matrix_chunked(self):
        P, V, K = 6, 3, 2
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        p = layout.flatten(model)
        D = p.shape[0] * p.shape[1]
        X_probe = p.reshape(1, 1, 1, D).expand(P, V, K, D).clone()
        X_probe += torch.randn(P, V, K, D) * 0.01
        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))

        ev_full = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=None)
        ev_chunk = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=8)

        cost_full = compute_nn_cost_matrix(ev_full, X_probe, layout, inputs, targets)
        cost_chunk = compute_nn_cost_matrix(ev_chunk, X_probe, layout, inputs, targets)

        torch.testing.assert_close(cost_chunk, cost_full, atol=1e-5, rtol=1e-5)


class TestComputeNNCostMatrixMatchesManual:
    """Test 18: compute_nn_cost_matrix matches manual per-probe evaluation."""

    def test_compute_nn_cost_matrix_matches_manual(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        layout = ParamLayout.from_module(model)
        p = layout.flatten(model)
        D = p.shape[0] * p.shape[1]

        P, V, K = 2, 2, 1
        X_probe = p.reshape(1, 1, 1, D).expand(P, V, K, D).clone()
        X_probe += torch.randn(P, V, K, D) * 0.01
        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16,))
        loss_fn = nn.CrossEntropyLoss()

        # Manual evaluation
        model.eval()
        buffers = dict(model.named_buffers())
        expected = torch.zeros(P, V)
        for pi in range(P):
            for vi in range(V):
                flat_particle = X_probe[pi, vi, 0]  # K=1, so single probe
                sd = layout.unflatten(flat_particle.reshape(-1, layout.particle_dim))
                full_dict = {**sd, **buffers}
                with torch.no_grad():
                    output = functional_call(model, full_dict, (inputs,))
                    expected[pi, vi] = loss_fn(output, targets)

        evaluator = NNCostEvaluator(model, loss_fn)
        cost_mat = compute_nn_cost_matrix(evaluator, X_probe, layout, inputs, targets)

        torch.testing.assert_close(cost_mat, expected, atol=1e-5, rtol=1e-5)
