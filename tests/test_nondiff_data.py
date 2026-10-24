"""Tests for non-differentiable data generation utilities.

Tests cover:
- generate_maxsat_instance: PySAT-based random 3-SAT clause tensor generation
- generate_multidomain_data: Combined MNIST + Fashion-MNIST with 20 classes
"""

from __future__ import annotations

import torch
import pytest

pytest.importorskip("experiments.runners")


class TestGenerateMaxsatInstance:
    @pytest.fixture(autouse=True)
    def _pysat(self):
        pytest.importorskip("pysat", reason="python-sat not installed")

    @pytest.mark.parametrize(
        "key, expected_dtype",
        [("clause_vars", torch.long), ("clause_signs", torch.float)],
    )
    def test_clause_vars_shape_and_dtype(self, key, expected_dtype):
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert result[key].shape == (86, 3)
        assert result[key].dtype == expected_dtype

    def test_clause_vars_in_valid_range(self):
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert result["clause_vars"].min().item() >= 0
        assert result["clause_vars"].max().item() <= 19  # 0-indexed

    def test_clause_signs_binary_values(self):
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        unique_vals = result["clause_signs"].unique().tolist()
        for v in unique_vals:
            assert v in [0.0, 1.0]

    def test_reproducible_with_same_seed(self):
        from experiments.runners.nondiff_data import generate_maxsat_instance

        r1 = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3, seed=42)
        r2 = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3, seed=42)
        assert torch.equal(r1["clause_vars"], r2["clause_vars"])
        assert torch.equal(r1["clause_signs"], r2["clause_signs"])

    def test_critical_ratio_default(self):
        """generate_maxsat_instance with default ratio (alpha~4.27) generates valid instance."""
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=100)
        expected_clauses = int(100 * 4.27)
        assert result["num_vars"] == 100
        assert result["num_clauses"] == expected_clauses
        assert result["clause_vars"].shape == (expected_clauses, 3)

    def test_returns_cnf_object(self):
        from experiments.runners.nondiff_data import generate_maxsat_instance

        result = generate_maxsat_instance(num_vars=20, num_clauses=86, k=3)
        assert "cnf" in result
        assert len(result["cnf"].clauses) == 86


def test_combined_domains(monkeypatch, tmp_path):
    torchvision = pytest.importorskip("torchvision")
    import numpy as np

    from experiments.runners.nondiff_data import generate_multidomain_data

    def digits(root, *, train, download, transform):
        count = 10 if train else 4
        image = transform(np.full((28, 28), 255, dtype=np.uint8))
        return torch.utils.data.TensorDataset(image.expand(count, -1, -1, -1), torch.arange(count))

    monkeypatch.setattr(torchvision.datasets, "MNIST", digits)
    monkeypatch.setattr(torchvision.datasets, "FashionMNIST", digits)
    result = generate_multidomain_data(data_dir=str(tmp_path), batch_size=32)
    assert result["num_classes"] == 20
    for split, count in (("train_loader", 10), ("test_loader", 4)):
        images, labels = next(iter(result[split]))
        assert images.shape == (2 * count, 1, 28, 28)
        assert sorted(labels.tolist()) == [*range(count), *range(10, 10 + count)]
        torch.testing.assert_close(images[labels < 10], torch.full_like(images[:count], (1 - 0.1307) / 0.3081))
        torch.testing.assert_close(images[labels >= 10], torch.full_like(images[:count], (1 - 0.2860) / 0.3530))
