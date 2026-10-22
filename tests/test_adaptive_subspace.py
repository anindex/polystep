"""Tests for AdaptiveSubspace: rotating projection, displacement-biased rotation, and factory methods."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cost_nn import NNCostEvaluator
from polystep.solvers._shared import orthonormal_columns, thin_qr
from polystep.transform import ParamLayout


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SimpleMLP()


@pytest.fixture
def layout(model):
    return ParamLayout.from_module(model)


@pytest.fixture
def adaptive_sub(model):
    return AdaptiveSubspace.auto_from_params(model)


class TestInitProjection:
    def test_init_projection_shape_and_orthogonality(self, adaptive_sub):
        """init_projection returns P with correct shape and P.T @ P ~ I."""
        P = adaptive_sub.init_projection()
        assert P.shape == (adaptive_sub.full_dim, adaptive_sub.subspace_dim)

        PtP = P.T @ P
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4), (
            f"Orthogonality error: max deviation = {(PtP - eye).abs().max().item()}"
        )

    def test_init_projection_deterministic_with_generator(self, adaptive_sub):
        gen1 = torch.Generator().manual_seed(123)
        P1 = adaptive_sub.init_projection(generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        P2 = adaptive_sub.init_projection(generator=gen2)

        assert torch.allclose(P1, P2, atol=1e-6), "Same seed should produce same P"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_generator_does_not_freeze_basis(self, adaptive_sub):
        """A seeded CUDA generator must advance across successive basis draws,
        not regenerate a byte-identical projection (the initial_seed() freeze)."""
        gen = torch.Generator(device="cuda").manual_seed(123)
        dev = torch.device("cuda")
        P1 = adaptive_sub.init_projection(generator=gen, device=dev)
        P2 = adaptive_sub.init_projection(generator=gen, device=dev)
        assert not torch.allclose(P1, P2), "seeded CUDA basis is frozen"


def test_rotate_random_produces_orthogonal_basis(adaptive_sub):
    """Random rotation produces orthogonal P different from input."""
    sub = AdaptiveSubspace(
        full_dim=adaptive_sub.full_dim,
        subspace_dim=adaptive_sub.subspace_dim,
        rotation_mode="random",
        _entry_specs=adaptive_sub._entry_specs,
    )
    P_old = sub.init_projection(generator=torch.Generator().manual_seed(0))
    P_new = sub.rotate(P_old, step=5, total_steps=100, generator=torch.Generator().manual_seed(99))

    PtP = P_new.T @ P_new
    eye = torch.eye(sub.subspace_dim)
    assert torch.allclose(PtP, eye, atol=1e-4)

    assert not torch.allclose(P_old, P_new, atol=1e-3), "Rotated P should differ from original"


class TestRotateDisplacement:
    @pytest.mark.parametrize(
        "num_rows, seed, scale",
        [
            (3, 77, 0.1),
            (1, 55, 1.0),
            (None, 66, 1.0),
        ],
    )
    def test_rotate_displacement_produces_orthogonal_basis(self, adaptive_sub, num_rows, seed, scale):
        """Displacement rotation with non-zero history produces orthogonal P."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        rows = num_rows if num_rows is not None else adaptive_sub.displacement_history_size
        torch.manual_seed(seed)
        disp_history = torch.randn(rows, adaptive_sub.subspace_dim) * scale

        P_new = adaptive_sub.rotate(
            P, step=5, total_steps=100, displacement_history=disp_history, generator=torch.Generator().manual_seed(42)
        )

        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4), f"Orthogonality error: {(PtP - eye).abs().max().item()}"

    def test_rotate_displacement_bf16_stays_orthonormal(self, adaptive_sub):
        """Rotation upcasts for SVD/QR, so the bf16 result is still a basis."""
        sub = AdaptiveSubspace(
            full_dim=adaptive_sub.full_dim,
            subspace_dim=adaptive_sub.subspace_dim,
            rotation_mode="displacement",
            _entry_specs=adaptive_sub._entry_specs,
        )
        P = sub.init_projection(dtype=torch.bfloat16)
        disp = (torch.randn(3, sub.subspace_dim) * 0.1).to(torch.bfloat16)
        P_new = sub.rotate(P, step=5, total_steps=100, displacement_history=disp)
        assert P_new.dtype == torch.bfloat16
        assert P_new.shape == (sub.full_dim, sub.subspace_dim)
        # The contract is orthonormal columns; a degraded bf16 QR still passes a shape check.
        gram = P_new.float().T @ P_new.float()
        assert torch.allclose(gram, torch.eye(sub.subspace_dim), atol=5e-2), gram.diagonal()[:4]

    @pytest.mark.parametrize(
        "use_zero_history, step, seed, atol",
        [
            (True, 5, 42, 1e-4),
            (False, 0, 456, 1e-5),
        ],
    )
    def test_rotate_zero_history_redraws_random(self, adaptive_sub, use_zero_history, step, seed, atol):
        """Covers zero and None history."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        disp_history = torch.zeros(3, adaptive_sub.subspace_dim) if use_zero_history else None

        P_new = adaptive_sub.rotate(
            P,
            step=step,
            total_steps=100,
            displacement_history=disp_history,
            generator=torch.Generator().manual_seed(seed),
        )

        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=atol)

        # Orthogonality alone does not prove a fallback happened: returning the input
        # basis unchanged satisfies it. The result has to be the freshly drawn basis.
        assert not torch.allclose(P_new, P, atol=1e-6), "rotate returned the input basis"
        # Same seed, same basis: the fallback must be a pure function of the generator,
        # not of the (zero or absent) history.
        again = adaptive_sub.rotate(
            P,
            step=step,
            total_steps=100,
            displacement_history=disp_history,
            generator=torch.Generator().manual_seed(seed),
        )
        torch.testing.assert_close(P_new, again, rtol=1e-5, atol=1e-6)

    def test_rotate_displacement_incorporates_svd(self):
        """History has one dominant direction."""
        full_dim = 100
        subspace_dim = 20
        sub = AdaptiveSubspace(
            full_dim=full_dim,
            subspace_dim=subspace_dim,
            rotation_mode="displacement",
            svd_ratio_init=0.5,
            svd_ratio_final=0.5,
        )

        P = sub.init_projection(generator=torch.Generator().manual_seed(0))

        # Displacement history with one strong dominant direction.
        torch.manual_seed(42)
        dominant_dir = torch.randn(subspace_dim)
        dominant_dir = dominant_dir / dominant_dir.norm()

        disp_history = torch.zeros(5, subspace_dim)
        for i in range(5):
            noise = torch.randn(subspace_dim) * 0.01
            disp_history[i] = dominant_dir * 10.0 + noise

        dominant_full = P @ dominant_dir
        dominant_full = dominant_full / dominant_full.norm()

        P_new = sub.rotate(
            P, step=50, total_steps=100, displacement_history=disp_history, generator=torch.Generator().manual_seed(99)
        )

        # The new basis must have a column aligned with the dominant direction.
        dot_products = (P_new.T @ dominant_full).abs()
        max_dot = dot_products.max().item()

        # Random baseline: for 100-dim space with 20 columns, expected
        # max |dot| ~ sqrt(2*ln(20)/100) ~ 0.24. With SVD, should be much higher.
        assert max_dot > 0.3, (
            f"SVD direction not incorporated: max |dot| = {max_dot:.4f}, "
            "expected > 0.3 for displacement-biased rotation"
        )

    def test_svd_ratio_zero_keeps_random_basis(self):
        """No singular vector enters the basis."""
        sub = AdaptiveSubspace(
            full_dim=100, subspace_dim=20, rotation_mode="displacement", svd_ratio_init=0.0, svd_ratio_final=0.0
        )
        P = sub.init_projection(generator=torch.Generator().manual_seed(0))

        direction = torch.randn(20, generator=torch.Generator().manual_seed(1))
        history = (direction / direction.norm()).expand(5, 20) * 10.0
        dominant_full = P @ direction
        dominant_full = dominant_full / dominant_full.norm()

        P_new = sub.rotate(
            P, step=0, total_steps=100, displacement_history=history, generator=torch.Generator().manual_seed(99)
        )
        # A retained singular vector lands at |dot| = 1; a random basis stays near the
        # sqrt(2 ln k / n) ~ 0.24 baseline.
        assert (P_new.T @ dominant_full).abs().max().item() < 0.5


@pytest.mark.parametrize(
    "step, total, expected",
    [
        (0, 100, 0.0),
        (50, 100, 0.25),
        (100, 100, 0.5),
        (200, 100, 0.5),
    ],
)
def test_svd_ratio_interpolates_and_clamps(step, total, expected):
    """Clamps beyond total_steps."""
    sub = AdaptiveSubspace(full_dim=100, subspace_dim=10, svd_ratio_init=0.0, svd_ratio_final=0.5)
    assert sub.get_svd_ratio(step, total) == pytest.approx(expected)


def _absorb_schedules(**kwargs):
    """One subspace per class, all sharing the absorb schedule under test."""
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace

    base = AdaptiveSubspace(full_dim=100, subspace_dim=10, **kwargs)
    return {
        "adaptive": base,
        "hybrid": HybridSubspace(specs=(), subspace_dim=10, compression_ratio=0.1, **kwargs),
        "factored": FactoredSubspace(specs=(), subspace_dim=10, compression_ratio=0.1, **kwargs),
    }


class TestShouldAbsorb:
    """The absorb schedule, checked on every subspace class that exposes it."""

    @pytest.mark.parametrize("kind", ["adaptive", "hybrid", "factored"])
    def test_stagnation_fires_at_patience_boundary(self, kind):
        sub = _absorb_schedules(absorb_mode="stagnation", absorb_patience=20)[kind]
        assert not sub.should_absorb(stagnation_count=19, iteration=50)
        assert sub.should_absorb(stagnation_count=20, iteration=50)
        assert sub.should_absorb(stagnation_count=25, iteration=50)

    @pytest.mark.parametrize("kind", ["adaptive", "hybrid", "factored"])
    def test_periodic_fires_on_multiples_skips_zero(self, kind):
        sub = _absorb_schedules(absorb_mode="periodic", absorb_interval=10)[kind]
        assert not sub.should_absorb(stagnation_count=0, iteration=0)
        assert not sub.should_absorb(stagnation_count=0, iteration=5)
        assert not sub.should_absorb(stagnation_count=0, iteration=13)
        assert sub.should_absorb(stagnation_count=0, iteration=10)
        assert sub.should_absorb(stagnation_count=0, iteration=20)

    @pytest.mark.parametrize("kind", ["adaptive", "hybrid", "factored"])
    def test_periodic_zero_interval_never_fires(self, kind):
        sub = _absorb_schedules(absorb_mode="periodic", absorb_interval=0)[kind]
        assert not sub.should_absorb(stagnation_count=100, iteration=100)


@pytest.mark.parametrize("kind", ["adaptive", "hybrid"])
def test_absorb_keeps_progress_baseline(model, layout, kind):
    """Baseline is the measured best-vertex mean, not the cost mean."""
    from torch.func import functional_call, vmap

    from polystep.hybrid_subspace import HybridSubspace

    cls = AdaptiveSubspace if kind == "adaptive" else HybridSubspace
    sub = cls.from_layout(layout, rank=4, absorb_mode="periodic", absorb_interval=1)
    opt = PolyStepOptimizer(model, subspace=sub, solver="softmax", seed=0)
    x, y = torch.randn(16, 20), torch.randn(16, 5)

    def closure(params):
        return vmap(lambda p: (functional_call(model, p, (x,)) - y).pow(2).mean())(params)

    for absorbs in (1, 2):
        opt.step(closure)
        state = opt._state
        assert state.absorb_count == absorbs
        assert state.prev_loss < state.costs[-1]


class TestFactoryMethods:
    @pytest.mark.parametrize(
        "full_dim, expected",
        [
            (100, 64),  # 5% of 100 is under min_rank, so the floor binds
            (10_000, 500),  # 5% sits between the bounds
            (10_000_000, 4096),  # 5% is over max_rank, so the ceiling binds
            (16, 16),  # a model smaller than min_rank cannot exceed its own width
        ],
    )
    def test_auto_from_params_defaults(self, full_dim, expected):
        """5% of the parameter count, clamped to the default [64, 4096]."""
        sub = AdaptiveSubspace.auto_from_params(nn.Linear(full_dim, 1, bias=False))
        assert sub.full_dim == full_dim
        assert sub.subspace_dim == expected
        assert sub.compression_ratio == pytest.approx(expected / full_dim)

    def test_from_layout(self, model, layout):
        """from_layout with explicit rank creates correct subspace."""
        sub = AdaptiveSubspace.from_layout(layout, rank=16)
        assert sub.full_dim == layout.total_params
        assert sub.subspace_dim == 16

    def test_from_layout_rank_clamped(self, model, layout):
        """from_layout clamps rank to full_dim."""
        huge_rank = layout.total_params + 100
        sub = AdaptiveSubspace.from_layout(layout, rank=huge_rank)
        assert sub.subspace_dim == layout.total_params

    def test_entry_specs_cover_all_params(self, adaptive_sub):
        """Sum of (flat_end - flat_start) for all entry_specs equals full_dim."""
        total_covered = sum(spec.flat_end - spec.flat_start for spec in adaptive_sub._entry_specs)
        assert total_covered == adaptive_sub.full_dim

    def test_entry_specs_contiguous(self, adaptive_sub):
        """entry_specs are contiguous (no gaps or overlaps)."""
        prev_end = 0
        for spec in adaptive_sub._entry_specs:
            assert spec.flat_start == prev_end, (
                f"Gap or overlap at {spec.entry_key}: expected start={prev_end}, got {spec.flat_start}"
            )
            assert spec.flat_end > spec.flat_start
            prev_end = spec.flat_end


def test_displacement_mode_productivity():
    """Displacement mode must beat random on a controlled quadratic (final cost or AUC)."""
    torch.manual_seed(42)
    full_dim = 100
    subspace_dim = 20
    num_steps = 30

    A = torch.randn(full_dim, full_dim)
    b = torch.randn(full_dim)

    def objective(x_flat):
        """Evaluate quadratic cost for a flat parameter vector."""
        return ((A @ x_flat - b) ** 2).sum().item()

    def run_optimization(rotation_mode: str, seed: int):
        """Run simple subspace optimization with given rotation mode."""
        sub = AdaptiveSubspace(
            full_dim=full_dim,
            subspace_dim=subspace_dim,
            rotation_mode=rotation_mode,
            svd_ratio_init=0.3,
            svd_ratio_final=0.6,
            displacement_history_size=5,
        )

        gen = torch.Generator().manual_seed(seed)
        P = sub.init_projection(generator=gen)

        x_base = torch.zeros(full_dim)
        costs = []
        disp_history_list = []

        for step in range(num_steps):
            gen_step = torch.Generator().manual_seed(seed + step * 1000)
            num_candidates = 50
            candidates = torch.randn(num_candidates, subspace_dim, generator=gen_step) * 0.5

            best_cost = float("inf")
            best_coords = torch.zeros(subspace_dim)
            for j in range(num_candidates):
                x_perturbed = x_base + P @ candidates[j]
                cost = objective(x_perturbed)
                if cost < best_cost:
                    best_cost = cost
                    best_coords = candidates[j].clone()

            costs.append(best_cost)

            displacement = P @ best_coords
            x_base = x_base + displacement

            # Track displacement for the next rotation.
            disp_history_list.append(best_coords.clone())
            if len(disp_history_list) > sub.displacement_history_size:
                disp_history_list.pop(0)

            disp_tensor = torch.stack(disp_history_list) if disp_history_list else None
            gen_rot = torch.Generator().manual_seed(seed + step * 2000 + 1)
            P = sub.rotate(P, step=step, total_steps=num_steps, displacement_history=disp_tensor, generator=gen_rot)

        return costs

    costs_displacement = run_optimization("displacement", seed=42)
    costs_random = run_optimization("random", seed=42)

    final_disp = costs_displacement[-1]
    final_rand = costs_random[-1]

    # Area under curve: lower means faster convergence.
    auc_disp = sum(costs_displacement)
    auc_rand = sum(costs_random)

    displacement_wins = (final_disp < final_rand) or (auc_disp < auc_rand)
    assert displacement_wins, (
        f"Displacement mode did not outperform random.\n"
        f"  Final cost - displacement: {final_disp:.4f}, random: {final_rand:.4f}\n"
        f"  AUC - displacement: {auc_disp:.4f}, random: {auc_rand:.4f}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_generator_draws_on_cuda():
    """The basis returns on the projection's device."""
    sub = AdaptiveSubspace(
        full_dim=100,
        subspace_dim=10,
        rotation_mode="random",
    )
    P = sub.init_projection(generator=torch.Generator().manual_seed(0))
    P_rotated = sub.rotate(P, step=0, total_steps=100, generator=torch.Generator(device="cuda").manual_seed(42))

    Z = torch.randn(100, 10, generator=torch.Generator(device="cuda").manual_seed(42), device="cuda").cpu()
    Q, R = torch.linalg.qr(Z)
    assert P_rotated.device == P.device
    assert torch.allclose(P_rotated, Q * torch.sign(torch.diagonal(R)), atol=1e-5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_history_redraws_random_basis(bad):
    """The norm check catches the bad entry."""
    sub = AdaptiveSubspace(full_dim=50, subspace_dim=6, svd_ratio_init=0.5, svd_ratio_final=0.5)
    projection = sub.init_projection(generator=torch.Generator().manual_seed(0))
    history = torch.randn(3, 50, generator=torch.Generator().manual_seed(1))
    history[1, 7] = bad

    rotated = sub.rotate(projection, 5, 10, history, generator=torch.Generator().manual_seed(2), history_is_full=True)
    expected = sub._rotate_random(projection.device, projection.dtype, torch.Generator().manual_seed(2))
    assert torch.equal(rotated, expected)


def test_rank_deficient_history_draws_random_columns():
    """Null directions of the history are redrawn, not kept as near-axis columns."""
    sub = AdaptiveSubspace(full_dim=200, subspace_dim=20, svd_ratio_init=0.5, svd_ratio_final=0.5)
    history = torch.zeros(4, 200)
    history[[0, 2]] = torch.randn(2, 200, generator=torch.Generator().manual_seed(0))
    projection = sub.init_projection(generator=torch.Generator().manual_seed(1))

    rotated = sub.rotate(projection, 1, 1, history, generator=torch.Generator().manual_seed(2), history_is_full=True)
    torch.testing.assert_close(rotated.mT @ rotated, torch.eye(20), atol=1e-5, rtol=0)
    assert rotated.abs().max() < 0.9


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_orthonormal_columns_tall_matches_householder(dtype):
    """CholQR2 on a tall basis of condition about 90 reproduces the positive-diagonal Householder Q."""
    Z = torch.randn(4096, 64, generator=torch.Generator().manual_seed(0), dtype=dtype).cumsum(1)
    Q, R = thin_qr(Z)
    householder = Q * torch.sign(torch.diagonal(R))
    P = orthonormal_columns(Z)
    torch.testing.assert_close(P, householder, atol=1e-5, rtol=0)
    assert not torch.equal(P, householder)
    torch.testing.assert_close(P.mT @ P, torch.eye(64, dtype=dtype), atol=100 * torch.finfo(dtype).eps, rtol=0)


def raise_mixed_api():
    raise RuntimeError("mix of the legacy and new APIs")


@pytest.mark.parametrize("precision_query", [raise_mixed_api, lambda: "high"])
def test_orthonormal_columns_reduced_precision_uses_householder(monkeypatch, precision_query):
    """TF32 matmuls, or a precision query that raises after mixed TF32 APIs, keep the tall basis on Householder."""
    monkeypatch.setattr(torch, "get_float32_matmul_precision", precision_query)
    Z = torch.randn(4096, 64, generator=torch.Generator().manual_seed(0)).cumsum(1)
    Q, R = thin_qr(Z)
    assert torch.equal(orthonormal_columns(Z), Q * torch.sign(torch.diagonal(R)))


@pytest.mark.parametrize("shape", [(512, 512), (1386, 16), (2**18, 4)])
def test_orthonormal_columns_excluded_shapes_use_householder(shape):
    """Square bases can fail FP32 Cholesky; small or narrow tall ones run faster under Householder."""
    Z = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    Q, R = thin_qr(Z)
    assert torch.equal(orthonormal_columns(Z), Q * torch.sign(torch.diagonal(R)))


@pytest.mark.parametrize("gap", [0.0, 1e-4])
def test_cholqr_ill_conditioned_falls_back(gap):
    """A tall basis with a near-duplicate column fails CholQR2 and takes the Householder Q."""
    Z = torch.randn(32768, 8, generator=torch.Generator().manual_seed(0))
    Z[:, 1] = Z[:, 0] + gap * Z[:, 1]
    Q, R = thin_qr(Z)
    assert torch.equal(orthonormal_columns(Z), Q * torch.sign(torch.diagonal(R)))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_displacement_rotation_ten_million_rows_succeeds():
    """A tall cuSOLVER SVD overflows its workspace query near 8M rows."""
    sub = AdaptiveSubspace(full_dim=10_000_000, subspace_dim=2, svd_ratio_init=0.5, svd_ratio_final=0.5)
    gen = torch.Generator(device="cuda").manual_seed(0)
    projection = sub.init_projection(generator=gen, device="cuda")
    history = torch.randn(2, sub.full_dim, device="cuda", generator=gen) * torch.tensor([[1.0], [0.1]], device="cuda")

    rotated = sub.rotate(projection, 1, 10, history, generator=gen, history_is_full=True)
    torch.testing.assert_close(rotated.mT @ rotated, torch.eye(2, device="cuda"), atol=1e-4, rtol=0)
    assert (rotated[:, 0] @ history[0]).abs() / history[0].norm() > 0.999


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gram_rotation_under_autocast(device, dtype):
    """Autocast must not hand ``eigh`` or Cholesky a half-precision Gram matrix."""
    sub = AdaptiveSubspace(full_dim=2**17, subspace_dim=8, svd_ratio_init=1.0, svd_ratio_final=1.0)
    history = torch.arange(1.0, 2**17 + 1.0, device=device).unsqueeze(0) * 300
    with torch.autocast(device, dtype=dtype):
        rotated = sub.rotate(torch.eye(2**17, 8, device=device), 1, 1, history, history_is_full=True)
    assert rotated.dtype == torch.float32
    torch.testing.assert_close(rotated[:, 0].abs(), history[0] / history[0].norm())


def test_rotation_keeps_uncentered_displacement_direction():
    """Rotation must not centre the history; the offset IS the direction to keep."""
    full_dim, sub_dim, history_len = 200, 8, 16
    sub = AdaptiveSubspace(
        full_dim=full_dim,
        subspace_dim=sub_dim,
        rotation_mode="displacement",
        svd_ratio_init=0.25,
        svd_ratio_final=0.25,
    )
    projection = torch.linalg.qr(torch.randn(full_dim, sub_dim, generator=torch.Generator().manual_seed(0)))[0]

    # Every displacement is a large common offset plus small noise, so the dominant
    # uncentered direction is the offset itself. Centering removes exactly that.
    offset = torch.ones(full_dim) / full_dim**0.5
    noise = torch.randn(history_len, full_dim, generator=torch.Generator().manual_seed(1)) * 1e-3
    history = offset.unsqueeze(0) + noise

    # Seeded and unseeded both take the exact SVD; neither may centre.
    for gen in (None, torch.Generator().manual_seed(3)):
        rotated = sub.rotate(projection, 0, 10, history, generator=gen, history_is_full=True)
        alignment = (rotated.T @ offset).abs().max()
        assert alignment > 0.9, f"rotation lost the dominant direction: {alignment:.3f}"


@pytest.mark.parametrize("interval, expected_steps", [(1, [1, 2, 3, 4]), (2, [2, 4]), (3, [3]), (0, [])])
def test_rotation_fires_after_interval_multiples(interval, expected_steps):
    """Rotation runs after steps k, 2k, ... and never before."""
    from torch.func import functional_call

    model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
    sub = AdaptiveSubspace.auto_from_params(model, compression_target=0.15)
    sub.rotation_interval = interval
    opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.1, seed=0, compile=False)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))

    seen = [opt.state.projection.clone()]
    for _ in range(4):
        opt.step(
            lambda bp: torch.stack(
                [
                    functional_call(model, {k: v[i] for k, v in bp.items()}, (x,)).pow(2).mean()
                    for i in range(next(iter(bp.values())).shape[0])
                ]
            )
        )
        seen.append(opt.state.projection.clone())

    changed = [k for k, (a, b) in enumerate(zip(seen, seen[1:]), 1) if not torch.equal(a, b)]
    assert changed == expected_steps


@pytest.mark.parametrize("kind", ["adaptive", "hybrid", "factored"])
@pytest.mark.parametrize("option", [{"amortize_steps": 2}, {"biased_rotation": True}, {"use_momentum": True}])
def test_per_step_rotation_warns_cleared_state(kind, option):
    """Basis changes clear velocity and cached directions."""
    import warnings

    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    layout = ParamLayout.from_module(model)
    build = {
        "adaptive": lambda k: AdaptiveSubspace.auto_from_params(model, rotation_interval=k),
        "hybrid": lambda k: HybridSubspace.from_layout(layout, rank=2, rotation_interval=k),
        "factored": lambda k: FactoredSubspace.from_layout(layout, rank=2, rotation_interval=k),
    }[kind]
    with pytest.warns(UserWarning, match="changes its basis every step"):
        PolyStepOptimizer(model, subspace=build(1), **option)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(model, **option)
        for k in (0, 2):
            PolyStepOptimizer(model, subspace=build(k), **option)
    assert not [w for w in caught if "rotation_interval" in str(w.message)]


@pytest.mark.parametrize(
    "kind, kwargs",
    [
        ("adaptive", dict(rank=24, rotation_interval=2, absorb_mode="none")),
        ("hybrid", dict(rank=2, rotation_interval=2, absorb_mode="none")),
        ("adaptive", dict(rank=24, rotation_interval=0, absorb_mode="periodic", absorb_interval=2)),
        ("hybrid", dict(rank=2, absorb_mode="periodic", absorb_interval=2)),
    ],
)
def test_amortized_interval_still_fires(kind, kwargs):
    """Coasting steps do not advance rotation or absorb intervals."""
    from torch.func import functional_call, vmap

    from polystep.hybrid_subspace import HybridSubspace

    cls = AdaptiveSubspace if kind == "adaptive" else HybridSubspace
    model = nn.Sequential(nn.Linear(6, 10), nn.Tanh(), nn.Linear(10, 1))
    sub = cls.from_layout(ParamLayout.from_module(model), **kwargs)
    opt = PolyStepOptimizer(model, subspace=sub, amortize_steps=2, seed=0, compile=False)
    x = torch.randn(32, 6, generator=torch.Generator().manual_seed(1))
    p0, h0 = opt.state.projection, opt.state.hybrid_projections
    for _ in range(8):
        opt.step(lambda bp: vmap(lambda p: functional_call(model, p, (x,)).pow(2).mean())(bp))
    s = opt.state
    assert s.absorb_count > 0 or s.projection is not p0 or s.hybrid_projections is not h0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_offsets_match_reconstruct_batch(model, layout, dtype):
    """Moving one particle adds that particle's columns of P."""
    sub = AdaptiveSubspace.from_layout(layout, rank=21)
    gen = torch.Generator().manual_seed(0)
    projection = sub.init_projection(generator=gen, dtype=dtype)
    base = {k: v.detach().to(dtype) for k, v in model.named_parameters()}
    coords = torch.randn(sub.subspace_dim, generator=gen, dtype=dtype)
    G, C, pdim = 2, 3, 4
    col_start = sub.subspace_dim - 1 - G * pdim
    dcoords = torch.randn(G, C, pdim, generator=gen, dtype=dtype)

    bary = torch.cat([v.reshape(-1) for v in sub.apply_perturbation(projection, base, coords).values()])
    got = sub.reconstruct_offsets(projection, bary, col_start, dcoords)

    moved = coords.repeat(G * C, 1)
    for g in range(G):
        moved[g * C : (g + 1) * C, col_start + g * pdim : col_start + (g + 1) * pdim] += dcoords[g]
    want = sub.reconstruct_batch(projection, base, moved)
    out = torch.full((G + 1, C, sub.full_dim), torch.nan, dtype=dtype)
    into = sub.reconstruct_offsets(projection, bary, col_start, dcoords, out[:G])
    assert got.keys() == want.keys() == into.keys()
    for key in want:
        torch.testing.assert_close(got[key], want[key])
        assert torch.equal(into[key], got[key])
        assert into[key].untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    assert out[G].isnan().all()


@pytest.mark.parametrize("chunk_size", [3, None, 10**6])
@pytest.mark.parametrize("polytope_type", ["simplex", "orthoplex"])
def test_step_with_offsets_matches_full_reconstruction(layout, polytope_type, chunk_size):
    """Offsets cover every whole particle; the padded tail particle keeps reconstruct_batch."""
    from unittest.mock import patch

    from torch.func import functional_call, vmap

    x, y = torch.randn(16, 20), torch.randn(16, 5)
    runs, delegated, buffers, storages = [], [], [], []
    for reference in (False, True):
        torch.manual_seed(42)
        model = SimpleMLP()
        sub = AdaptiveSubspace.from_layout(layout, rank=21, rotation_interval=0)
        opt = PolyStepOptimizer(
            model,
            subspace=sub,
            subspace_particle_dim=4,
            polytope_type=polytope_type,
            num_probe=2,
            chunk_size=chunk_size,
            solver="softmax",
            seed=0,
        )

        captured = {}
        compute_probe_points = opt._compiled.compute_probe_points

        def capture_probes(X, *args):
            captured.update(X=X.clone(), probes=compute_probe_points(X, *args), next=0)
            return captured["probes"]

        opt._compiled.compute_probe_points = capture_probes

        def full_reconstruction(self, projection, bary_flat, col_start, dcoords, out):
            G, C, p = dcoords.shape
            buffers.append(out)
            g0 = captured["next"]
            assert col_start == g0 * p
            moved = captured["X"].reshape(-1)[: self.subspace_dim].repeat(G * C, 1)
            for g in range(G):
                moved[g * C : (g + 1) * C, (g0 + g) * p : (g0 + g + 1) * p] = captured["probes"][g0 + g].reshape(C, p)
            captured["next"] = g0 + G
            delegated.append(G * C)
            return self.reconstruct_batch(projection, opt.state.base_params, moved)

        seen = set()
        storages.append(seen)

        def closure(params):
            seen.add(next(iter(params.values())).untyped_storage().data_ptr())
            return vmap(lambda p: (functional_call(model, p, (x,)) - y).pow(2).mean())(params)

        with patch.object(AdaptiveSubspace, "reconstruct_offsets", full_reconstruction) if reference else nullcontext():
            for _ in range(3):
                opt.step(closure)
        runs.append(opt)

    group = opt._polytope_vertices.shape[0] * 2
    assert sum(delegated) == 3 * (opt.state.X.shape[0] - 1) * group
    assert len({b.untyped_storage().data_ptr() for b in buffers}) == 1
    assert all(r._step_buffers[7].shape[0] <= r.state.X.shape[0] for r in runs)
    assert storages[0] == {runs[0]._step_buffers[7].untyped_storage().data_ptr()}
    torch.testing.assert_close(runs[0].state.X, runs[1].state.X, rtol=1e-5, atol=1e-6)
    assert runs[0].state.costs == pytest.approx(runs[1].state.costs, rel=1e-6)


def test_bf16_step_uses_full_reconstruction(layout):
    """Keeps one rounding model per cost matrix."""
    from unittest.mock import patch

    from torch.func import functional_call, vmap

    model = SimpleMLP().to(torch.bfloat16)
    x, y = torch.randn(16, 20, dtype=torch.bfloat16), torch.randn(16, 5, dtype=torch.bfloat16)
    opt = PolyStepOptimizer(model, subspace=AdaptiveSubspace.from_layout(layout, rank=8), solver="softmax", seed=0)

    def closure(params):
        return vmap(lambda p: (functional_call(model, p, (x,)) - y).pow(2).mean())(params)

    with patch.object(AdaptiveSubspace, "reconstruct_offsets", side_effect=AssertionError("offsets under bf16")):
        opt.step(closure)
    assert torch.isfinite(opt.state.X).all()


def test_offsets_keep_layout_after_param_freeze(model, layout):
    from torch.func import functional_call, vmap

    sub = AdaptiveSubspace.from_layout(layout, rank=21, rotation_interval=0)
    model.fc1.bias.requires_grad_(False)
    opt = PolyStepOptimizer(model, subspace=sub, subspace_particle_dim=4, solver="softmax", seed=0)
    x, y = torch.randn(16, 20), torch.randn(16, 5)

    def closure(params):
        return vmap(lambda p: (functional_call(model, p, (x,)) - y).pow(2).mean())(params)

    for _ in range(2):
        opt.step(closure)
    assert torch.isfinite(opt.state.X).all()


def test_partial_cast_offsets_use_projection_dtype(model, layout):
    """Casting one layer and resyncing leaves a mixed-dtype base under one projection."""
    from torch.func import functional_call, vmap

    sub = AdaptiveSubspace.from_layout(layout, rank=21)
    opt = PolyStepOptimizer(model, subspace=sub, subspace_particle_dim=4, solver="softmax", seed=0)
    x, y = torch.randn(16, 20), torch.randn(16, 5)

    def closure(params):
        return vmap(lambda p: (functional_call(model, p, (x,)).float() - y).pow(2).mean())(params)

    opt.step(closure)
    model.fc2.double()
    opt.resync_from_model()
    for _ in range(2):
        opt.step(closure)
    assert torch.isfinite(opt.state.X).all()


def _mlp_and_closure(seed, in_dim=20, out_dim=4, samples=64):
    """Build an MLP plus a batched-params closure over fixed random data."""
    gen = torch.Generator().manual_seed(1000 + seed)
    inputs = torch.randn(samples, in_dim, generator=gen)
    targets = torch.randn(samples, out_dim, generator=gen)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    def true_loss():
        with torch.no_grad():
            return nn.functional.mse_loss(model(inputs), targets).item()

    return model, closure, true_loss


class TestRotationPreservesPoint:
    """The represented point is ``base + P @ coords``.

    Replacing ``P`` while coords are non-zero moves the weights with no evaluation
    behind it, so every rotation must fold coords into base first.
    """

    def test_coords_are_zero_after_rotation(self):
        """Rotation re-anchors the origin, so coords must be exactly zero."""
        model, closure, _ = _mlp_and_closure(0)
        subspace = AdaptiveSubspace.auto_from_params(model)
        opt = PolyStepOptimizer(model, subspace=subspace, epsilon=0.1, max_iterations=10)

        prev_projection = None
        for _ in range(5):
            opt.step(closure)
            coords = opt.state.X.reshape(-1)[: subspace.subspace_dim]
            assert torch.count_nonzero(coords) == 0
            if prev_projection is not None:
                teleport = torch.linalg.vector_norm((opt.state.projection - prev_projection) @ coords)
                assert teleport.item() == 0.0
            prev_projection = opt.state.projection.clone()

    def test_loss_reduction_floor(self):
        """Median loss reduction across seeds must clear a floor."""
        reductions = []
        for seed in range(6):
            model, closure, true_loss = _mlp_and_closure(seed)
            opt = PolyStepOptimizer(
                model,
                subspace=AdaptiveSubspace.auto_from_params(model),
                epsilon=0.1,
                max_iterations=60,
            )
            before = true_loss()
            for _ in range(60):
                opt.step(closure)
            reductions.append((before - true_loss()) / before * 100.0)

        reductions.sort()
        median = 0.5 * (reductions[2] + reductions[3])
        assert median > 8.0, reductions


def test_rank_above_full_dim_rejected():
    """Reduced QR of a wide matrix returns ``full_dim`` columns, not the requested
    ``subspace_dim``, so the optimizer would allocate coordinates the projection
    cannot consume and the mismatch would surface only later, at reconstruction.
    """
    with pytest.raises(ValueError, match="subspace_dim"):
        AdaptiveSubspace(full_dim=2, subspace_dim=3)
