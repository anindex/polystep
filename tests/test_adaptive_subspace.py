"""Tests for AdaptiveSubspace: rotating projection, displacement-biased rotation, and factory methods."""
import pytest
import torch
import torch.nn as nn

from polystep.adaptive_subspace import AdaptiveSubspace, EntrySpec
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# Helper model fixture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# init_projection shape and orthogonality
# ---------------------------------------------------------------------------


class TestInitProjection:
    def test_init_projection_shape_and_orthogonality(self, adaptive_sub):
        """init_projection returns P with correct shape and P.T @ P ~ I."""
        P = adaptive_sub.init_projection()
        assert P.shape == (adaptive_sub.full_dim, adaptive_sub.subspace_dim)

        # Orthogonality check: P.T @ P should be close to identity
        PtP = P.T @ P
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4), (
            f"Orthogonality error: max deviation = {(PtP - eye).abs().max().item()}"
        )

    def test_init_projection_deterministic_with_generator(self, adaptive_sub):
        """Two calls with same Generator seed produce identical P."""
        gen1 = torch.Generator().manual_seed(123)
        P1 = adaptive_sub.init_projection(generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        P2 = adaptive_sub.init_projection(generator=gen2)

        assert torch.allclose(P1, P2, atol=1e-6), "Same seed should produce same P"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotateRandom:
    def test_rotate_random_produces_orthogonal_basis(self, adaptive_sub):
        """Random rotation produces orthogonal P different from input."""
        sub = AdaptiveSubspace(
            full_dim=adaptive_sub.full_dim,
            subspace_dim=adaptive_sub.subspace_dim,
            rotation_mode="random",
            _entry_specs=adaptive_sub._entry_specs,
        )
        P_old = sub.init_projection(generator=torch.Generator().manual_seed(0))
        P_new = sub.rotate(P_old, step=5, total_steps=100,
                           generator=torch.Generator().manual_seed(99))

        # Orthogonality
        PtP = P_new.T @ P_new
        eye = torch.eye(sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4)

        # Different from old
        assert not torch.allclose(P_old, P_new, atol=1e-3), "Rotated P should differ from original"


class TestRotateDisplacement:
    def test_rotate_displacement_produces_orthogonal_basis(self, adaptive_sub):
        """Displacement rotation with non-zero history produces orthogonal P."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        # Create non-zero displacement history
        torch.manual_seed(77)
        disp_history = torch.randn(3, adaptive_sub.subspace_dim) * 0.1

        P_new = adaptive_sub.rotate(P, step=5, total_steps=100,
                                    displacement_history=disp_history,
                                    generator=torch.Generator().manual_seed(42))

        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4), (
            f"Orthogonality error: {(PtP - eye).abs().max().item()}"
        )

    def test_rotate_displacement_zero_history_falls_back_to_random(self, adaptive_sub):
        """Zero displacement history falls back to random (still orthogonal)."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        disp_history = torch.zeros(3, adaptive_sub.subspace_dim)

        P_new = adaptive_sub.rotate(P, step=5, total_steps=100,
                                    displacement_history=disp_history,
                                    generator=torch.Generator().manual_seed(42))

        # Should still be orthogonal
        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4)

    def test_rotate_displacement_single_entry_history(self, adaptive_sub):
        """Single-entry displacement history works correctly."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        torch.manual_seed(55)
        disp_history = torch.randn(1, adaptive_sub.subspace_dim)

        P_new = adaptive_sub.rotate(P, step=5, total_steps=100,
                                    displacement_history=disp_history,
                                    generator=torch.Generator().manual_seed(42))

        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4)

    def test_rotate_displacement_full_history_buffer(self, adaptive_sub):
        """Full history buffer (history_size entries) works correctly."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))

        torch.manual_seed(66)
        disp_history = torch.randn(
            adaptive_sub.displacement_history_size, adaptive_sub.subspace_dim
        )

        P_new = adaptive_sub.rotate(P, step=5, total_steps=100,
                                    displacement_history=disp_history,
                                    generator=torch.Generator().manual_seed(42))

        PtP = P_new.T @ P_new
        eye = torch.eye(adaptive_sub.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4)

    def test_rotate_displacement_incorporates_svd(self):
        """Displacement rotation incorporates SVD directions when history has clear dominant direction."""
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

        # Create displacement history with a strong dominant direction
        # All rows point mostly in the same direction in subspace
        torch.manual_seed(42)
        dominant_dir = torch.randn(subspace_dim)
        dominant_dir = dominant_dir / dominant_dir.norm()

        disp_history = torch.zeros(5, subspace_dim)
        for i in range(5):
            noise = torch.randn(subspace_dim) * 0.01
            disp_history[i] = dominant_dir * 10.0 + noise

        # Project dominant direction to full space
        dominant_full = P @ dominant_dir
        dominant_full = dominant_full / dominant_full.norm()

        P_new = sub.rotate(P, step=50, total_steps=100,
                           displacement_history=disp_history,
                           generator=torch.Generator().manual_seed(99))

        # The new P should have at least one column correlated with the
        # projected dominant direction. Check max absolute dot product.
        dot_products = (P_new.T @ dominant_full).abs()
        max_dot = dot_products.max().item()

        # Random baseline: for 100-dim space with 20 columns, expected
        # max |dot| ~ sqrt(2*ln(20)/100) ~ 0.24. With SVD, should be much higher.
        assert max_dot > 0.3, (
            f"SVD direction not incorporated: max |dot| = {max_dot:.4f}, "
            "expected > 0.3 for displacement-biased rotation"
        )


# ---------------------------------------------------------------------------
# SVD ratio schedule
# ---------------------------------------------------------------------------


class TestSvdRatioSchedule:
    def test_svd_ratio_at_start(self):
        """SVD ratio at step=0 equals svd_ratio_init."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               svd_ratio_init=0.0, svd_ratio_final=0.5)
        assert sub.get_svd_ratio(0, 100) == pytest.approx(0.0)

    def test_svd_ratio_at_midpoint(self):
        """SVD ratio at step=total/2 equals midpoint."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               svd_ratio_init=0.0, svd_ratio_final=0.5)
        assert sub.get_svd_ratio(50, 100) == pytest.approx(0.25)

    def test_svd_ratio_at_end(self):
        """SVD ratio at step=total equals svd_ratio_final."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               svd_ratio_init=0.0, svd_ratio_final=0.5)
        assert sub.get_svd_ratio(100, 100) == pytest.approx(0.5)

    def test_svd_ratio_clamps_beyond_total(self):
        """SVD ratio beyond total_steps is clamped to svd_ratio_final."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               svd_ratio_init=0.0, svd_ratio_final=0.5)
        assert sub.get_svd_ratio(200, 100) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# apply_perturbation matches manual
# ---------------------------------------------------------------------------


class TestApplyPerturbation:
    def test_apply_perturbation_matches_manual(self, model, adaptive_sub):
        """apply_perturbation matches manual P @ flat_subspace computation."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(adaptive_sub.subspace_dim) * 0.01

        # Manual computation
        delta_flat = P @ flat_sub
        manual_result = {}
        for spec in adaptive_sub._entry_specs:
            chunk = delta_flat[spec.flat_start:spec.flat_end]
            manual_result[spec.entry_key] = base_sd[spec.entry_key] + chunk.reshape(spec.original_shape)

        # Method computation
        method_result = adaptive_sub.apply_perturbation(P, base_sd, flat_sub)

        for key in manual_result:
            assert torch.allclose(method_result[key], manual_result[key], atol=1e-6), (
                f"Key {key}: manual vs method mismatch"
            )

    def test_apply_perturbation_zero_is_identity(self, model, adaptive_sub):
        """Zero perturbation returns base params unchanged."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))
        base_sd = model.state_dict()
        flat_sub = torch.zeros(adaptive_sub.subspace_dim)

        result = adaptive_sub.apply_perturbation(P, base_sd, flat_sub)

        for key in base_sd:
            if key in result:
                assert torch.allclose(result[key], base_sd[key], atol=1e-6)


# ---------------------------------------------------------------------------
# reconstruct_batch matches loop
# ---------------------------------------------------------------------------


class TestReconstructBatch:
    def test_reconstruct_batch_matches_loop(self, model, adaptive_sub):
        """reconstruct_batch gives same result as looping apply_perturbation."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))
        base_sd = model.state_dict()

        N = 4
        torch.manual_seed(42)
        batch = torch.randn(N, adaptive_sub.subspace_dim) * 0.01

        batch_result = adaptive_sub.reconstruct_batch(P, base_sd, batch)

        for i in range(N):
            single_result = adaptive_sub.apply_perturbation(P, base_sd, batch[i])
            for key in single_result:
                assert torch.allclose(
                    batch_result[key][i], single_result[key], atol=1e-5
                ), f"Row {i}, key {key}: batch vs single mismatch"

    def test_reconstruct_batch_shapes(self, model, adaptive_sub):
        """reconstruct_batch produces (N, *shape) tensors."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))
        base_sd = model.state_dict()

        N = 3
        batch = torch.randn(N, adaptive_sub.subspace_dim) * 0.01
        result = adaptive_sub.reconstruct_batch(P, base_sd, batch)

        for spec in adaptive_sub._entry_specs:
            expected_shape = (N, *spec.original_shape)
            assert result[spec.entry_key].shape == expected_shape


# ---------------------------------------------------------------------------
# absorb zeros subspace
# ---------------------------------------------------------------------------


class TestAbsorb:
    def test_absorb_zeros_subspace(self, model, adaptive_sub):
        """After absorb, flat_subspace is all zeros and base_sd is updated."""
        P = adaptive_sub.init_projection(generator=torch.Generator().manual_seed(0))
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(adaptive_sub.subspace_dim) * 0.01

        # Expected base after absorb = apply_perturbation result
        expected_sd = adaptive_sub.apply_perturbation(P, base_sd, flat_sub)

        new_base, zeroed = adaptive_sub.absorb(P, base_sd, flat_sub)

        # Zeroed subspace
        assert torch.all(zeroed == 0)
        assert zeroed.shape == flat_sub.shape

        # New base matches expected
        for key in expected_sd:
            assert torch.allclose(new_base[key], expected_sd[key], atol=1e-6)


# ---------------------------------------------------------------------------
# should_absorb
# ---------------------------------------------------------------------------


class TestShouldAbsorb:
    def test_should_absorb_stagnation(self):
        """stagnation_count >= absorb_patience returns True."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               absorb_mode="stagnation", absorb_patience=20)
        assert not sub.should_absorb(stagnation_count=19, iteration=50)
        assert sub.should_absorb(stagnation_count=20, iteration=50)
        assert sub.should_absorb(stagnation_count=25, iteration=50)

    def test_should_absorb_periodic(self):
        """iteration % absorb_interval == 0 returns True (for iteration > 0)."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               absorb_mode="periodic", absorb_interval=10)
        assert not sub.should_absorb(stagnation_count=0, iteration=0)
        assert not sub.should_absorb(stagnation_count=0, iteration=5)
        assert sub.should_absorb(stagnation_count=0, iteration=10)
        assert sub.should_absorb(stagnation_count=0, iteration=20)
        assert not sub.should_absorb(stagnation_count=0, iteration=13)

    def test_should_absorb_periodic_disabled(self):
        """absorb_interval=0 with periodic mode never triggers."""
        sub = AdaptiveSubspace(full_dim=100, subspace_dim=10,
                               absorb_mode="periodic", absorb_interval=0)
        assert not sub.should_absorb(stagnation_count=100, iteration=100)


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    def test_auto_from_params(self, model):
        """auto_from_params matches model parameter count and gives reasonable rank."""
        sub = AdaptiveSubspace.auto_from_params(model)
        total = sum(p.numel() for p in model.parameters())
        assert sub.full_dim == total
        assert 8 <= sub.subspace_dim <= 512
        assert sub.subspace_dim <= sub.full_dim
        assert sub.compression_ratio > 0
        assert sub.compression_ratio <= 1.0

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
        total_covered = sum(
            spec.flat_end - spec.flat_start for spec in adaptive_sub._entry_specs
        )
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


# ---------------------------------------------------------------------------
# Displacement mode productivity validation
# ---------------------------------------------------------------------------


class TestDisplacementProductivity:
    def test_displacement_mode_productivity(self):
        """Displacement mode converges better than random on a controlled quadratic.

        This validates that incorporating SVD directions from displacement
        history actually helps optimization, not just that it runs correctly.
        """
        torch.manual_seed(42)
        full_dim = 100
        subspace_dim = 20
        num_steps = 30

        # Define a simple quadratic objective: f(x) = ||Ax - b||^2
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

            # Start from zero
            x_base = torch.zeros(full_dim)
            costs = []
            disp_history_list = []

            for step in range(num_steps):
                # Generate candidate perturbations in subspace
                gen_step = torch.Generator().manual_seed(seed + step * 1000)
                num_candidates = 50
                candidates = torch.randn(num_candidates, subspace_dim,
                                         generator=gen_step) * 0.5

                # Evaluate all candidates
                best_cost = float("inf")
                best_coords = torch.zeros(subspace_dim)
                for j in range(num_candidates):
                    x_perturbed = x_base + P @ candidates[j]
                    cost = objective(x_perturbed)
                    if cost < best_cost:
                        best_cost = cost
                        best_coords = candidates[j].clone()

                costs.append(best_cost)

                # Update base
                displacement = P @ best_coords
                x_base = x_base + displacement

                # Track displacement in subspace coords for next rotation
                disp_history_list.append(best_coords.clone())
                if len(disp_history_list) > sub.displacement_history_size:
                    disp_history_list.pop(0)

                # Rotate projection for next step
                disp_tensor = torch.stack(disp_history_list) if disp_history_list else None
                gen_rot = torch.Generator().manual_seed(seed + step * 2000 + 1)
                P = sub.rotate(P, step=step, total_steps=num_steps,
                               displacement_history=disp_tensor,
                               generator=gen_rot)

            return costs

        # Run both modes with same seed
        costs_displacement = run_optimization("displacement", seed=42)
        costs_random = run_optimization("random", seed=42)

        # Displacement mode should achieve lower final cost OR converge faster
        final_disp = costs_displacement[-1]
        final_rand = costs_random[-1]

        # Also check area under curve (lower = faster convergence)
        auc_disp = sum(costs_displacement)
        auc_rand = sum(costs_random)

        # At least one criterion should hold: lower final cost OR lower AUC
        displacement_wins = (final_disp < final_rand) or (auc_disp < auc_rand)
        assert displacement_wins, (
            f"Displacement mode did not outperform random.\n"
            f"  Final cost - displacement: {final_disp:.4f}, random: {final_rand:.4f}\n"
            f"  AUC - displacement: {auc_disp:.4f}, random: {auc_rand:.4f}"
        )


# ---------------------------------------------------------------------------
# Block-wise + AdaptiveSubspace mutual exclusion
# ---------------------------------------------------------------------------


class TestBlockwiseCombinedMode:
    def test_blockwise_adaptive_combined(self, model):
        """PolyStepOptimizer with subspace + block_strategy uses combined mode (combined subspace+block extension)."""
        from polystep.optimizer import PolyStepOptimizer

        sub = AdaptiveSubspace.auto_from_params(model)

        opt = PolyStepOptimizer(
            model,
            subspace=sub,
            block_strategy="per_layer",
            compile=False,
        )
        # Verify combined mode is active
        assert opt.subspace is not None
        assert opt.block_strategy == "per_layer"


# ---------------------------------------------------------------------------
# CUDA generator warning
# ---------------------------------------------------------------------------


class TestCudaGeneratorFallback:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_generator_creates_cpu_fallback(self):
        """CUDA generator should silently create a CPU generator for reproducibility."""
        sub = AdaptiveSubspace(
            full_dim=100,
            subspace_dim=10,
            rotation_mode="random",
        )
        P = sub.init_projection(generator=torch.Generator().manual_seed(0))
        cuda_gen = torch.Generator(device="cuda").manual_seed(42)
        # Should not warn - silently creates CPU generator from CUDA seed
        P_rotated = sub.rotate(P, step=0, total_steps=100, generator=cuda_gen)
        assert P_rotated.shape == P.shape
        assert torch.isfinite(P_rotated).all()
