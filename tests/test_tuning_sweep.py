"""The validation-only hyperparameter sweep behind the two scale experiments.

Three properties the sweep must hold, and one thing that is easy to get wrong:

1. **A sweep must not touch the test split.** The sweep hands itself a
   :class:`~experiments.runners.fairness.TestSplitTripwire` and any read
   raises. The end-to-end test below proves the guard is live *and* that it is not
   vacuous, by showing the same run does read the test split when tuning is off.
2. **Equal tuning budget.** Every method's grid is the same size, so
   ``configs x seeds x evals-per-config`` is one number for the whole table.
3. **Deterministic selection.** Best validation score, ties broken by grid order.

And the easy mistake: an option registered but never wired, which
``tests/test_no_test_set_leakage.py`` already guards for ``--allow-test-leakage``.
Same treatment here for ``--tune``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "experiments" / "runners"


@pytest.fixture
def fairness(require_experiments):
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners import fairness

    return fairness


# --- 1. the test split is unreachable from a sweep --------------------------------


def test_tripwire_raises_on_both_ways_a_split_gets_read(fairness):
    """A DataLoader is iterated; an ``(x, y)`` split is unpacked. Both hit ``__iter__``."""
    trap = fairness.TestSplitTripwire()
    with pytest.raises(AssertionError, match="touched the test split"):
        for _ in trap:
            pass
    with pytest.raises(AssertionError, match="touched the test split"):
        _x, _y = trap


def _tiny_head_splits(device="cpu", n=64):
    """768-d features so the head is the real 1,538-parameter one."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 768, generator=g, device=device)
    y = (x[:, 0] > 0).long()
    return {"train": (x, y), "val": (x[: n // 2], y[: n // 2])}


@pytest.fixture
def headquant(require_experiments):
    pytest.importorskip("transformers", reason="run_gpt2_finetune imports the GPT-2 stack")
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners import run_gpt2_finetune

    return run_gpt2_finetune


def test_headquant_sweep_cannot_read_the_test_split(headquant, tmp_path, fairness):
    """A sweep trial runs to completion with the test split booby-trapped.

    The trial is handed a tripwire *by the caller*, so this is not a check that the
    runner substitutes one -- it is a check that no code path on the sweep route
    reads the test set at all.
    """
    splits = {**_tiny_head_splits(), "test": fairness.TestSplitTripwire()}
    out = headquant.run_headquant(
        "int8",
        "eggroll",
        seed=0,
        device="cpu",
        splits=splits,
        results_dir=str(tmp_path),
        budget=256,
        probe_every=64,
        point={"lr": 0.01, "sigma": 1.0},
        tune=True,
    )
    assert out["val_accuracy"] >= 0.0, "the sweep produced no validation score to select on"
    assert out["test_accuracy_at_selected"] != out["test_accuracy_at_selected"], (
        "a sweep trial reported a test number; selection must be validation-only"
    )


def test_the_tripwire_check_is_not_vacuous(headquant, tmp_path, fairness):
    """The same run with tuning off *does* read the test split, and so trips the wire.

    Without this, ``test_headquant_sweep_cannot_read_the_test_split`` would still pass
    if the runner had stopped scoring the test set entirely.
    """
    splits = {**_tiny_head_splits(), "test": fairness.TestSplitTripwire()}
    with pytest.raises(AssertionError, match="touched the test split"):
        headquant.run_headquant(
            "int8",
            "eggroll",
            seed=0,
            device="cpu",
            splits=splits,
            results_dir=str(tmp_path),
            budget=256,
            probe_every=64,
            point={"lr": 0.01, "sigma": 1.0},
            tune=False,
        )


@pytest.mark.parametrize(
    "runner,fns",
    [
        ("run_gpt2_finetune.py", ("run_headquant",)),
    ],
)
def test_tune_is_registered_and_wired(runner, fns, require_experiments):
    """``--tune`` must be a live option that actually reaches the split selection.

    A registered-but-inert flag would leave every check above green while the sweep
    read the test set anyway.
    """
    tree = ast.parse((RUNNER_DIR / runner).read_text())
    options = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--tune" in options, f"{runner} does not register --tune"

    for name in fns:
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        dumped = ast.dump(fn)
        assert "TestSplitTripwire" in dumped or "hide_test" in dumped, (
            f"{runner}::{name} never hides the test split, so --tune cannot be honest"
        )


# --- 2. equal tuning budget -------------------------------------------------------


def test_every_swept_method_gets_the_same_grid_size(fairness):
    """Equal tuning effort is the claim; equal grid size is what makes it checkable."""
    sizes = {m: len(g) for m, g in fairness.TUNING_GRID.items()}
    assert len(set(sizes.values())) == 1, f"grids differ in size, so tuning effort is not matched: {sizes}"
    assert "polystep" in sizes, "PolyStep must be swept on the same footing as the baselines"


def test_tuning_cost_is_the_number_the_paper_quotes(fairness):
    """configs x seeds x evals-per-config, and equal across methods at equal budget."""
    costs = [fairness.tuning_cost(m, 25_000, seeds=1) for m in fairness.TUNING_GRID]
    assert len({c["tuning_evals"] for c in costs}) == 1, costs
    one = costs[0]
    assert one["tuning_evals"] == one["configs"] * one["seeds_per_config"] * one["evals_per_config"]


# --- 3. deterministic selection ---------------------------------------------------


def test_selection_breaks_ties_by_grid_order(fairness):
    """Validation accuracy ties constantly on a small split; the rule must be fixed."""
    trials = [
        {"index": 2, "val": 0.5},
        {"index": 0, "val": 0.5},
        {"index": 1, "val": 0.4},
    ]
    assert fairness.select_best(trials)["index"] == 0
    assert fairness.select_best(list(reversed(trials)))["index"] == 0, "selection depends on input order"


def test_an_all_ties_sweep_is_flagged_uninformative(fairness, tmp_path):
    """Nine configs at the same score means the tie-break returned the untuned prior.

    That happens for real -- a reduced sweep budget can land below the horizon where a
    method separates at all -- and the paper must not quote the result as tuned.
    """
    path = str(tmp_path / "sel.json")
    trials = [
        {"showcase": "int8", "method": "polystep", "index": i, "name": f"c{i}", "point": {"epsilon": i}, "val": 0.558}
        for i in range(9)
    ]
    fairness.write_selection("gpt2_headquant", trials, {}, {}, path)
    entry, _ = fairness.load_selection("gpt2_headquant", "int8", "polystep", path)
    assert entry["informative"] is False and entry["grid_index"] == 0


def test_selection_roundtrips_through_disk(fairness, tmp_path):
    path = str(tmp_path / "sel.json")
    trials = [
        {"showcase": "snn", "method": "eggroll", "index": i, "name": f"c{i}", "point": {"lr": i}, "val": v}
        for i, v in enumerate([0.1, 0.9, 0.9])
    ]
    fairness.write_selection("elevation", trials, {"budget_per_config": 7}, {"eggroll": {"configs": 3}}, path)
    entry, provenance = fairness.load_selection("elevation", "snn", "eggroll", path)
    assert entry["point"] == {"lr": 1}, "ties must go to the earlier grid index"
    assert entry["informative"] is True and entry["distinct_val_scores"] == 2
    assert provenance["budget_per_config"] == 7 and "written" in provenance
    # An unswept method falls back rather than borrowing someone else's config.
    assert fairness.load_selection("elevation", "snn", "cma_es", path) == (None, None)
    assert fairness.load_selection("elevation", "snn", "eggroll", str(tmp_path / "nope.json")) == (None, None)


# --- how a grid point becomes hyperparameters -------------------------------------


def test_radius_keys_are_multipliers_on_the_shared_probe_scale(fairness):
    """PolyStep's radius is a norm, a baseline's sigma is per-coordinate; the grid is
    expressed in units of the shared scale so the two stay comparable."""
    out = fairness.apply_point({"sigma": 9.0, "lr": 0.01, "popsize": 32}, {"sigma": 2.0, "lr": 0.03}, probe_scale=0.5)
    assert out == {"sigma": 1.0, "lr": 0.03, "popsize": 32}


def test_polystep_multipliers_scale_both_ends_of_a_schedule(fairness):
    """A cosine-scheduled radius must keep its shape and only change scale."""
    cfg = {"epsilon_init": 5.0, "epsilon_target": 0.3, "step_radius": 8.0, "num_probe": 1}
    out = fairness.apply_polystep_multipliers(cfg, {"epsilon": 2.0, "step_radius": 0.5})
    assert out == {"epsilon_init": 10.0, "epsilon_target": 0.6, "step_radius": 4.0, "num_probe": 1}
    assert cfg["epsilon_init"] == 5.0, "apply_polystep_multipliers must not mutate its input"


# --- the two recorded asymmetries -------------------------------------------------


def test_eggroll_dimension_gap_is_recorded_not_hidden(fairness):
    """EGGROLL matches on rank, not dimension. The JSON has to say so.

    Matching dimension would force ``FactoredSubspace`` to rank 1, and a ``(d_out, 1)``
    coordinate matrix is already rank 1, so EGGROLL's ``A B^T`` sampler would have
    nothing to factor and it would become dense Gaussian ES.
    """
    import torch.nn as nn

    from polystep.transform import ParamLayout

    layout = ParamLayout.from_module(nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4)))
    shared = fairness.make_subspace(layout, rank=4, seed=0)
    factored = fairness.make_subspace(layout, rank=4, seed=0, method="eggroll")

    tag = fairness.subspace_tag(factored, 4, shared_dim=shared.subspace_dim)
    assert tag["subspace_class"] == "FactoredSubspace"
    assert tag["subspace_rank"] == 4, "rank is the axis that IS matched"
    assert tag["subspace_dim"] != tag["subspace_dim_shared"], "the fixture no longer exercises the gap"
    assert tag["subspace_dim_matched"] is False
    assert tag["matched_on"] == "rank"

    same = fairness.subspace_tag(shared, 4, shared_dim=shared.subspace_dim)
    assert same["subspace_dim_matched"] is True and same["matched_on"] == "rank+dimension"


def test_headquant_search_space_is_matched_and_eggrolls_rank_is_recorded(headquant):
    """Nothing is projected on the head-quant experiment, so there is no gap to report.

    The EGGROLL question only bites inside a projected subspace. Here every method
    searches the same 1,538-dimensional full parameter space, and what is worth
    recording instead is the rank its ``A B^T`` sampler runs at relative to the
    ``(2, 768)`` weight's maximum useful rank of 2.
    """
    for method in ("polystep", "mezo", "eggroll"):
        space = headquant._headquant_search_space(method, 1538)
        assert space["subspace_class"] is None and space["subspace_dim"] == 1538
        assert space["subspace_dim_matched"] is True

    egg = headquant._headquant_search_space("eggroll", 1538)
    assert egg["eggroll_rank"] < egg["eggroll_max_useful_rank"], (
        "rank >= max useful rank makes the perturbation full-rank, i.e. dense Gaussian ES"
    )
    assert egg["eggroll_degenerates_to_dense_es"] is False
    assert "eggroll_rank" not in headquant._headquant_search_space("mezo", 1538)


def test_eggroll_grid_tunes_sigma_learning_rate_and_population(fairness):
    """All three of EGGROLL's knobs must move somewhere in its nine configurations."""
    grid = fairness.TUNING_GRID["eggroll"]
    for axis in ("sigma", "lr", "popsize"):
        assert len({p[axis] for p in grid}) == 3, f"{axis} is not swept: {sorted({p[axis] for p in grid})}"
    assert all(p["popsize"] % 2 == 0 for p in grid), "EGGROLL samples antithetic pairs; popsize must be even"


def test_mezo_grid_tunes_eps_and_learning_rate(fairness):
    grid = fairness.TUNING_GRID["mezo"]
    assert len({p["eps"] for p in grid}) == 3 and len({p["lr"] for p in grid}) == 3


def test_binary_head_is_labelled_as_a_partial_control(headquant):
    """``BinaryLinear`` does not quantize its bias, so Adam keeps a live gradient there.

    int8 quantizes both and gives backprop *exactly* nothing. The rows are not the
    same claim, and the label in the result JSON is what says so.
    """
    variants = headquant.HEADQUANT_VARIANTS
    assert variants["int8"][1:] == (False, False), "int8 must be hard in weight and bias"
    assert variants["binary"][1:] == (False, True), "binary's bias is live; label it, do not hide it"

    # The label has to match the layer's actual behaviour, or it is just a comment.
    x, y = torch.randn(8, 768), torch.randint(0, 2, (8,))
    for name, (factory, _w, bias_live) in variants.items():
        head = factory()
        torch.nn.functional.cross_entropy(head(x), y).backward()
        grad = 0.0 if head.bias.grad is None else float(head.bias.grad.abs().sum())
        assert (grad > 0.0) is bias_live, f"{name}: bias gradient {grad} contradicts backprop_sees_the_bias={bias_live}"
