#!/usr/bin/env python
"""Report which (benchmark, method, seed) cells are missing, and which failed.

The runners catch per-run exceptions, print ERROR, and carry on, so a job that
loses 39 of its 40 runs still exits 0 and the launcher logs "done".  Without
this check a table can be assembled from a grid that silently half-ran.

Usage:
    python experiments/scripts/check_grid_complete.py experiments/results/revision
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

SEEDS = [42, 123, 456, 789, 1337]
GF = ("openai_es", "spsa", "cma_es", "mezo", "random_search", "eggroll")
ALL_METHODS = ("polystep", "adam") + GF

#: Not a cross product: ``run_moe`` has no Adam arm.
GALLERY = {
    "snn": ALL_METHODS,
    "int8": ALL_METHODS,
    "argmax": ALL_METHODS,
    "staircase": ALL_METHODS,
    "mnist": ALL_METHODS,
    "moe": tuple(m for m in ALL_METHODS if m != "adam"),
}
#: ``timeseries`` is outside the cross-product, so leaving it out means its six
#: baselines are never checked.
TIMESERIES_METHODS = ("polystep", "adam", "persistence") + GF
PAIRS = [(b, m) for b, methods in GALLERY.items() for m in methods] + [("timeseries", m) for m in TIMESERIES_METHODS]


def expected():
    return [(b, m, s) for b, m in PAIRS for s in SEEDS]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]

    # --require fair,tuned: presence of a file says nothing about whether the run was
    # the run we meant. An untuned baseline and a tuned one produce identically named
    # JSON, and a table assembled from the first is a strawman comparison.
    require = []
    if "--require" in sys.argv:
        require = [k.strip() for k in sys.argv[sys.argv.index("--require") + 1].split(",") if k.strip()]

    have = {}
    leaked = []
    unmet = []
    # Only files directly in ``root``: a theory-mode run under ``theory/`` and a tuning
    # trial under ``tune_polystep/cell*/`` carry the same filename, so recursing would
    # let either stand in for a headline cell that never ran.
    for path in glob.glob(os.path.join(root, "*.json")):
        d = json.load(open(path))
        key = (d.get("benchmark"), d.get("method"), d.get("seed"))
        have[key] = path
        if d.get("leaked"):
            leaked.append(path)
        hp = d.get("hyperparameters") or {}
        # adam is gradient-based and has no tuned/fair grid; it is not a peer here. RL
        # and MAX-SAT are not gallery benchmarks and carry neither key.
        if d.get("method") != "adam" and d.get("benchmark") in GALLERY:
            for k in require:
                if not hp.get(k):
                    unmet.append((path, k, hp.get(k)))

    want = expected()
    missing = [k for k in want if k not in have]

    print(f"{len(want) - len(missing)}/{len(want)} cells present")
    if leaked:
        print(f"\nLEAKED files present ({len(leaked)}); a clean aggregate is impossible:")
        for p in leaked:
            print(f"  {p}")
    if missing:
        print(f"\nmissing {len(missing)}:")
        by_bm = {}
        for b, m, s in missing:
            by_bm.setdefault((b, m), []).append(s)
        for (b, m), ss in sorted(by_bm.items()):
            print(f"  {b:>7} {m:>14}  seeds {sorted(ss)}")

    # Failures the runner swallowed: an ERROR line in a log with no matching JSON.
    logdir = os.path.join(root, "logs")
    errs = []
    # Recursive: a launcher may write one log per cell in a subdirectory, and a flat
    # glob would see none of them.
    for lg in sorted(glob.glob(os.path.join(logdir, "**", "*.log"), recursive=True)):
        try:
            text = open(lg, errors="replace").read()
        except OSError:
            continue
        # Bare ERROR, not a stricter pattern: several runners and both tuning sweeps
        # print no seed or "failed:" field alongside it.
        for line in text.splitlines():
            if re.search(r"\bERROR\b", line):
                errs.append((os.path.basename(lg), line.strip()[:140]))
    if errs:
        print(f"\nswallowed run failures ({len(errs)}):")
        for lg, line in errs:
            print(f"  {lg:>28}  {line}")

    if unmet:
        print(f"\nruns missing a required hyperparameter ({len(unmet)}):")
        for path, k, v in unmet:
            print(f"  {os.path.basename(path):>44}  {k}={v!r}")

    if not missing and not errs and not leaked and not unmet:
        print("\ncomplete, no swallowed failures, nothing leaked")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
