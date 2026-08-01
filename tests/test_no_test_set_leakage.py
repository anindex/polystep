"""The main runners must expose --allow-test-leakage so test-set selection is
opt-in. A source scan avoids importing the runners' heavy optional deps.
"""

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

RUNNERS = ["run_mnist.py", "run_moe.py", "run_elevation.py", "run_timeseries.py"]
REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_DIR = REPO_ROOT / "experiments" / "runners"


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_exposes_allow_test_leakage(runner, require_experiments):
    """The flag must be a live argparse option, not a string that appears in the file.

    A substring scan passes on a commented-out or docstring mention.
    """
    path = RUNNER_DIR / runner
    assert path.exists(), f"{runner} is missing; the leakage guard cannot be checked"

    source = path.read_text()
    tree = ast.parse(source)
    added = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--allow-test-leakage" in added, (
        f"{runner} does not register --allow-test-leakage with add_argument; "
        f"found options: {sorted(o for o in added if o.startswith('--'))}"
    )

    # Registering the flag proves nothing on its own. The guard that decides which
    # split selects the reported model is ``audit_no_leakage``, so the flag has to
    # reach it: hardcoding ``audit_no_leakage=True`` leaves the option inert and the
    # registration check green.
    wired = [
        kw
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "audit_no_leakage" and "allow_test_leakage" in ast.dump(kw.value)
    ]
    assert wired, f"{runner} registers --allow-test-leakage but never passes it to audit_no_leakage"


# --- (a) the val split reaches every method, not only polystep -------------


def _tiny_loaders():
    """Four-sample MNIST-shaped train/test loaders."""
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=2)


def test_run_mnist_dispatch_gives_every_method_a_val_split(require_experiments, monkeypatch, tmp_path):
    """run_method must hand the held-out split to cmaes/es/spsa/adam too.

    The regression this guards: the dispatch used to pass ``val_loader``
    inside ``if method == "polystep"`` and drop it for everything else,
    so every baseline selected its reported checkpoint on the test set.
    """
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


@pytest.mark.parametrize("runner", ["run_moe.py", "run_elevation.py"])
def test_every_method_runner_loads_the_val_split(runner, require_experiments):
    """In these runners each method loads its own data, so each must go
    through ``_load_split`` -- the one place that carves the val slice."""
    tree = ast.parse((RUNNER_DIR / runner).read_text())
    runner_fns = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("run_") and node.name != "run_method"
    ]
    assert runner_fns, f"{runner} defines no run_* functions"

    for fn in runner_fns:
        calls = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_load_split" in calls, f"{runner}::{fn.name} loads data without carving the validation split"
        assert "audit_no_leakage" in {a.arg for a in fn.args.args}, (
            f"{runner}::{fn.name} cannot be told to run the honest protocol"
        )

    split_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_split")
    split_calls = {n.func.id for n in ast.walk(split_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "make_train_val_split" in split_calls, f"{runner}::_load_split does not actually hold out a val set"


def test_baselines_accept_a_validation_split(require_experiments):
    """A baseline with no val parameter can only select on the test set."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.baselines.openai_es import train_openai_es
    from experiments.baselines.sgd_baseline import train_sgd
    from experiments.baselines.spsa import train_spsa
    from polystep.benchmarks.baselines import train_cmaes

    for fn in (train_sgd, train_openai_es, train_spsa):
        assert "val_loader" in inspect.signature(fn).parameters, f"{fn.__name__} has no val_loader"
    assert "val_data" in inspect.signature(train_cmaes).parameters, "train_cmaes has no val_data"


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
    assert df.loc[0, "mean_accuracy"] == 0.9, "headline metric is not test_accuracy_at_selected"
