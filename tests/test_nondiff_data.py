"""Tests for non-differentiable data generation utilities.

Tests cover:
- generate_maxsat_instance: PySAT-based random 3-SAT clause tensor generation
- generate_sorting_data: Random sequences and target sort permutations
- generate_multidomain_data: Combined MNIST + Fashion-MNIST with 20 classes
"""
from __future__ import annotations

import torch
import pytest


class TestGenerateMaxsatInstance:
    """Tests for generate_maxsat_instance."""

    def test_returns_dict_with_required_keys(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert isinstance(result, dict)
        assert "clause_vars" in result
        assert "clause_signs" in result
        assert "num_vars" in result
        assert "num_clauses" in result

    def test_clause_vars_shape_and_dtype(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert result["clause_vars"].shape == (86, 3)
        assert result["clause_vars"].dtype == torch.long

    def test_clause_signs_shape_and_dtype(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert result["clause_signs"].shape == (86, 3)
        assert result["clause_signs"].dtype == torch.float

    def test_clause_vars_in_valid_range(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert result["clause_vars"].min().item() >= 0
        assert result["clause_vars"].max().item() <= 19  # 0-indexed

    def test_clause_signs_binary_values(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        unique_vals = result["clause_signs"].unique().tolist()
        for v in unique_vals:
            assert v in [0.0, 1.0]

    def test_reproducible_with_same_seed(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        r1 = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3, seed=42)
        r2 = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3, seed=42)
        assert torch.equal(r1["clause_vars"], r2["clause_vars"])
        assert torch.equal(r1["clause_signs"], r2["clause_signs"])

    def test_critical_ratio_default(self):
        """generate_maxsat_instance with default ratio (alpha~4.27) generates valid instance."""
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=100)
        expected_clauses = int(100 * 4.27)
        assert result["num_vars"] == 100
        assert result["num_clauses"] == expected_clauses
        assert result["clause_vars"].shape == (expected_clauses, 3)

    def test_returns_cnf_object(self):
        pytest.importorskip("pysat", reason="python-sat not installed")
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert "cnf" in result
        # CNF object should have clauses
        assert len(result["cnf"].clauses) == 86


class TestGenerateSortingData:
    """Tests for generate_sorting_data."""

    def test_returns_correct_shapes(self):
        from experiments.runners.nondiff_data import generate_sorting_data

        sequences, permutations = generate_sorting_data(N=10, num_samples=50)
        assert sequences.shape == (50, 10)
        assert permutations.shape == (50, 10)

    def test_permutations_are_valid(self):
        """Each row should be a permutation of 0..N-1."""
        from experiments.runners.nondiff_data import generate_sorting_data

        sequences, permutations = generate_sorting_data(N=10, num_samples=50)
        for i in range(50):
            perm = permutations[i]
            assert set(perm.tolist()) == set(range(10))

    def test_permutations_correctly_sort(self):
        """Gathering sequences with permutations should yield sorted sequences."""
        from experiments.runners.nondiff_data import generate_sorting_data

        sequences, permutations = generate_sorting_data(N=10, num_samples=50)
        sorted_seqs = sequences.gather(1, permutations)
        for i in range(50):
            row = sorted_seqs[i]
            assert torch.all(row[:-1] <= row[1:]).item(), f"Row {i} not sorted"

    def test_reproducible_with_same_seed(self):
        from experiments.runners.nondiff_data import generate_sorting_data

        s1, p1 = generate_sorting_data(N=10, num_samples=50, seed=42)
        s2, p2 = generate_sorting_data(N=10, num_samples=50, seed=42)
        assert torch.equal(s1, s2)
        assert torch.equal(p1, p2)


class TestGenerateMultidomainData:
    """Tests for generate_multidomain_data."""

    def test_returns_dict_with_required_keys(self):
        pytest.importorskip("torchvision", reason="torchvision not installed")
        from experiments.runners.nondiff_data import generate_multidomain_data

        result = generate_multidomain_data()
        assert isinstance(result, dict)
        assert "train_loader" in result
        assert "test_loader" in result
        assert "num_classes" in result

    def test_num_classes_is_20(self):
        pytest.importorskip("torchvision", reason="torchvision not installed")
        from experiments.runners.nondiff_data import generate_multidomain_data

        result = generate_multidomain_data()
        assert result["num_classes"] == 20

    def test_labels_range_0_to_19(self):
        """Labels should span MNIST 0-9 and Fashion-MNIST 10-19."""
        pytest.importorskip("torchvision", reason="torchvision not installed")
        from experiments.runners.nondiff_data import generate_multidomain_data

        result = generate_multidomain_data()
        all_labels = []
        for images, labels in result["train_loader"]:
            all_labels.extend(labels.tolist())
            if len(all_labels) > 5000:
                break
        min_label = min(all_labels)
        max_label = max(all_labels)
        assert min_label >= 0
        assert max_label <= 19
        # Should have labels from both domains
        assert max_label >= 10, "Expected Fashion-MNIST labels (10-19)"
        assert min_label <= 9, "Expected MNIST labels (0-9)"

    def test_image_shape(self):
        """Each batch should have shape (batch, 1, 28, 28)."""
        pytest.importorskip("torchvision", reason="torchvision not installed")
        from experiments.runners.nondiff_data import generate_multidomain_data

        result = generate_multidomain_data(batch_size=32)
        images, labels = next(iter(result["train_loader"]))
        assert images.dim() == 4
        assert images.shape[1] == 1
        assert images.shape[2] == 28
        assert images.shape[3] == 28
