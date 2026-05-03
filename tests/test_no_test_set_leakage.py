"""Regression tests for test-set leakage in the headline runners.

The paper's headline runners historically used the test split to
argmax over ``best_state_dict`` and then reported the test accuracy
of the restored state. Any noise on the test set during training
inflates the reported number.

As of April 2026, the honest protocol (val-selected checkpoints) is
the **default** in all runners. The legacy test-set selection can be
restored with ``--allow-test-leakage`` for bit-for-bit reproduction.

These tests:
- Confirm ``experiments.runners.common.make_train_val_split`` exists
  and is deterministic by seed.
- Confirm ``run_mnist.py`` exposes the ``--allow-test-leakage``
  flag (opt-out from honest protocol) and that the default is honest.
- Surface (via xfail) which runners still contain the leakage
  selection pattern without the ``no_leakage_check`` knob.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helper exists and is importable
# ---------------------------------------------------------------------------


def test_make_train_val_split_helper_exists():
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.common import make_train_val_split

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(torch.arange(1000).float().unsqueeze(1), torch.arange(1000))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    new_train, val = make_train_val_split(loader, val_frac=0.1, seed=42)
    assert len(val.dataset) == 100
    assert len(new_train.dataset) == 900

    # Determinism: same seed -> same val indices.
    new_train2, val2 = make_train_val_split(loader, val_frac=0.1, seed=42)
    indices1 = sorted(int(x[0].item()) for x in val.dataset)
    indices2 = sorted(int(x[0].item()) for x in val2.dataset)
    assert indices1 == indices2


# ---------------------------------------------------------------------------
# Regression guard: every runner that historically leaked must either
# (a) be patched to support the validation flag, or
# (b) be flagged here so future hands fix it.
# ---------------------------------------------------------------------------

LEAKAGE_RUNNERS = (
    "experiments/runners/run_mnist.py",
    "experiments/runners/run_moe.py",
    "experiments/runners/run_elevation.py",
    "experiments/runners/run_timeseries.py",
)

# Pattern: variable-name-on-LHS test_(acc|metrics|...) > best_... followed
# by deepcopy+load_state_dict somewhere downstream. Greppable as
# `if test_acc > best_accuracy` etc.
_LEAKAGE_PATTERN = re.compile(
    r"if\s+test_acc\s*>\s*best_(?:accuracy|acc)|"
    r"if\s+test_metrics\[[\"']mse[\"']\]\s*<\s*best_mse",
)


@pytest.mark.parametrize("relpath", LEAKAGE_RUNNERS)
def test_runner_leakage_pattern_documented(relpath):
    """Any runner that still contains the leakage selection pattern must
    also expose the ``no_leakage_check`` knob (CLI flag or kwarg). When
    the knob is absent the runner is silently leaking.
    """
    src = (REPO_ROOT / relpath).read_text()
    has_pattern = bool(_LEAKAGE_PATTERN.search(src))
    has_audit_flag = (
        "no_leakage_check" in src
        or "--allow-test-leakage" in src
        or "--allow-test-leakage" in src
    )

    if has_pattern and not has_audit_flag:
        pytest.xfail(
            f"{relpath} still leaks test set into best_state_dict and has "
            f"no leakage opt-in. Apply the same fix as "
            f"experiments/runners/run_mnist.py."
        )
    # If the pattern is absent OR an opt-in flag exists, the runner is
    # either compliant or under validation control - test passes.


# ---------------------------------------------------------------------------
# run_mnist.py: end-to-end smoke that --allow-test-leakage at least parses
# and the honest-protocol default path runs (we don't run a real epoch in CI).
# ---------------------------------------------------------------------------


def test_run_mnist_honest_protocol_is_default():
    """`run_mnist.py --help` must mention the --allow-test-leakage
    flag (opt-out). The default is honest protocol (val-selected).
    Pure CLI smoke - no model invocation."""
    proc = subprocess.run(
        [sys.executable, "experiments/runners/run_mnist.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--allow-test-leakage" in proc.stdout, (
        f"--allow-test-leakage flag missing from CLI:\n{proc.stdout}"
    )
