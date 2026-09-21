"""Validation reaches the runners; leaked results cannot enter reported tables."""

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- (a) the val split reaches every method, not only polystep -------------


def _tiny_loaders():
    """Four-sample MNIST-shaped train/test loaders."""
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=2)


def test_run_mnist_dispatch_gives_every_method_a_val_split(require_experiments, monkeypatch, tmp_path):
    """run_method must hand the held-out split to every method, not only polystep."""
    sys.path.insert(0, str(REPO_ROOT))
    import experiments.runners.run_mnist as run_mnist

    monkeypatch.setattr(run_mnist, "load_mnist", lambda **kwargs: _tiny_loaders())

    seen = {}
    for method in run_mnist.METHOD_RUNNERS:

        def recorder(seed, device, train_loader, test_loader, results_dir, _m=method, **kwargs):
            seen[_m] = kwargs

        monkeypatch.setitem(run_mnist.METHOD_RUNNERS, method, recorder)
        run_mnist.run_method(method, 42, "cpu", str(tmp_path), str(tmp_path), audit_no_leakage=True)

    assert set(seen) == set(run_mnist.METHOD_RUNNERS), "a method was never dispatched"
    for method, kwargs in seen.items():
        assert kwargs.get("val_loader") is not None, f"{method} was dispatched without a validation split"
        assert kwargs.get("audit_no_leakage") is True, f"{method} was dispatched without the leakage guard"


# --- (b) a leaked run cannot become a paper number -------------------------


def _write_result(results_dir, leaked):
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.common import save_result

    return save_result(
        benchmark="mnist",
        method="polystep",
        seed=42,
        metrics={
            "final_accuracy": 0.9,
            "best_accuracy": 0.95,
            "wall_time_seconds": 1.0,
            "peak_gpu_memory_mb": 0.0,
            "function_evals": 1,
            "total_steps": 1,
        },
        results_dir=str(results_dir),
        leaked=leaked,
    )


def test_leaked_run_is_stamped_and_refused_by_the_aggregator(require_experiments, tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.scripts.aggregate_results import LeakedResultError, aggregate_results

    path = _write_result(tmp_path, leaked=True)
    assert json.loads(Path(path).read_text())["leaked"] is True

    with pytest.raises(LeakedResultError):
        aggregate_results(str(tmp_path))


def test_honest_run_aggregates_on_the_selected_checkpoint(require_experiments, tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.scripts.aggregate_results import aggregate_results

    path = _write_result(tmp_path, leaked=False)
    saved = json.loads(Path(path).read_text())
    assert saved["leaked"] is False
    # Defaults to final_accuracy, never to the max-over-epochs best_accuracy.
    assert saved["metrics"]["test_accuracy_at_selected"] == 0.9

    df = aggregate_results(str(tmp_path))
    assert df.loc[0, "mean_accuracy"] == 0.9, "reported metric is not test_accuracy_at_selected"


def test_markdown_tables_use_the_selected_checkpoint(require_experiments, tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    from experiments.scripts import generate_tables as tables

    output = tmp_path / "README.md"
    output.write_text(f"{tables.README_START}\n{tables.README_END}\n")
    work = tmp_path / "code"
    work.mkdir()
    _write_result(work, leaked=False)
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(work), "--readme", str(output)])
    tables.main()
    assert "| MNIST (2-layer MLP) | 90.0 | - |" in output.read_text()
