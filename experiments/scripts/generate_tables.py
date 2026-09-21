"""Regenerate README and experiment-index Markdown tables from saved results."""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_results import load_single_result  # noqa: E402


DEFAULT_CUTOFF = "2026-08-01T11:02:28Z"


def collect(results_dir, cutoff):
    """Group post-cutoff runs by benchmark and method; reject test-selected runs."""
    runs = defaultdict(list)
    dropped = 0
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = load_single_result(path)
        with open(path) as f:
            ts = json.load(f).get("timestamp", "")
        if ts < cutoff:
            dropped += 1
            continue
        # run_maxsat writes "cmaes" where every other runner writes "cma_es";
        # normalized on read because 900+ existing result files use it.
        method = "cma_es" if r["method"] == "cmaes" else r["method"]
        runs[(r["benchmark"], method)].append(r)
    if dropped:
        print(f"dropped {dropped} result(s) dated before {cutoff}")
    return runs


def stat(runs, benchmark, method, key="test_accuracy_at_selected", scale=100.0, min_seeds=1):
    """Mean, std and n over seeds, or None when there is nothing to report."""
    rows = runs.get((benchmark, method), [])
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) < min_seeds or not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return mean * scale, var**0.5 * scale, n


README_START = "<!-- BENCH:START -->"
README_END = "<!-- BENCH:END -->"


README_NONDIFF = [
    ("snn", "SNN/LIF (MNIST)", "`threshold()`"),
    ("int8", "Int8 quantized", "`round()`"),
    ("argmax", "Argmax attention", "`argmax()`"),
    ("staircase", "Staircase activation", "`floor()`"),
    ("moe", "Hard MoE routing", "`argmax()`"),
]


BRIEF_METHODS = [("polystep", "PolyStep"), ("cma_es", "CMA-ES"), ("openai_es", "OpenAI-ES"), ("spsa", "SPSA")]


FULL_METHODS = BRIEF_METHODS[:1] + [
    ("openai_es", "OpenAI-ES"),
    ("cma_es", "CMA-ES"),
    ("eggroll", "EGGROLL"),
    ("mezo", "MeZO"),
    ("spsa", "SPSA"),
    ("random_search", "Random search"),
]


BRIEF_MAXSAT_SIZES = (100, 5000, 100000)


def _md_cell(s, places=1):
    """``mean +- std``, or a dash when nothing was measured."""
    if s is None:
        return "-"
    return f"{s[0]:.{places}f} ± {s[1]:.{places}f}" if s[2] > 1 else f"{s[0]:.{places}f}"


TIMEOUT_AWARE = ("rc2", "probsat", "sls")


def _maxsat_cell(runs, benchmark, method):
    """Format accuracy, timeout, or missing data."""
    rows = runs.get((benchmark, method), [])
    if not rows:
        return "-"
    if method in TIMEOUT_AWARE and all(r.get("timed_out") for r in rows):
        return "timeout"
    if any(not r.get("has_final_accuracy", True) for r in rows):
        return "no data"
    return _md_cell(stat(runs, benchmark, method, key="final_accuracy"))


def markdown_tables(runs, full):
    """Result tables as Markdown from saved experiment results.

    ``full`` adds every method measured and every MAX-SAT size; the short form is
    what the README carries.
    """
    methods = FULL_METHODS if full else BRIEF_METHODS
    md = ["### Non-differentiable tasks", "", "Test accuracy %. A dash means the result is not in this release.", ""]

    head = ["Task"] + [n for _, n in methods] + ["Adam (surrogate)", "Non-diff op"]
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "---|" * len(head))
    for bench, label, op in README_NONDIFF:
        cells = [_md_cell(stat(runs, bench, m)) for m, _ in methods]
        md.append("| " + " | ".join([label] + cells + [_md_cell(stat(runs, bench, "adam")), op]) + " |")

    sizes = sorted({int(b[len("maxsat_") : -1]) for (b, _) in runs if b.startswith("maxsat_")})
    if not full:
        sizes = [v for v in sizes if v in BRIEF_MAXSAT_SIZES]
    if sizes:
        md += ["", "### MAX-SAT (% clauses satisfied)", ""]
        maxsat_methods = [("polystep", "PolyStep"), ("cma_es", "CMA-ES"), ("openai_es", "OpenAI-ES")]
        head = ["Variables"] + [n for _, n in maxsat_methods] + ["probSAT", "RC2"]
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        for v in sizes:
            b = f"maxsat_{v}v"
            cells = [_maxsat_cell(runs, b, m) for m, _ in maxsat_methods]
            cells += [_maxsat_cell(runs, b, m) for m in ("probsat", "rc2")]
            md.append("| " + " | ".join([f"{v:,}"] + cells) + " |")

    md += ["", "### Differentiable sanity checks", "", "| Task | PolyStep | Adam |", "|---|---|---|"]
    md.append(
        f"| MNIST (2-layer MLP) | {_md_cell(stat(runs, 'mnist', 'polystep'))} | "
        f"{_md_cell(stat(runs, 'mnist', 'adam'))} |"
    )
    mse_ps = stat(runs, "timeseries", "polystep", key="final_mse", scale=1.0)
    mse_ad = stat(runs, "timeseries", "adam", key="final_mse", scale=1.0)
    md.append(f"| ETTh1 (LSTM, MSE; lower is better) | {_md_cell(mse_ps, 3)} | {_md_cell(mse_ad, 3)} |")
    return "\n".join(md)


def write_markdown(path, runs, full):
    """Splice the generated tables between the BENCH markers, leaving the rest alone."""
    with open(path) as f:
        text = f.read()
    # Require one marker pair to avoid leaving stale tables.
    starts, ends = text.count(README_START), text.count(README_END)
    if (starts, ends) != (1, 1):
        raise SystemExit(f"{path} needs exactly one {README_START} / {README_END} pair, found {starts}/{ends}")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    with open(path, "w") as f:
        f.write(head + README_START + "\n" + markdown_tables(runs, full) + "\n" + README_END + tail)
    print(f"wrote the benchmark tables in {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="experiments/results/revision")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="drop results whose timestamp precedes this date")
    ap.add_argument("--readme", help="regenerate the short tables between the BENCH markers here")
    ap.add_argument("--index", help="regenerate the full tables between the BENCH markers here")
    args = ap.parse_args()
    if not (args.readme or args.index):
        ap.error("provide --readme or --index as an output destination")
    runs = collect(args.results_dir, args.cutoff)
    if args.readme:
        write_markdown(args.readme, runs, full=False)
    if args.index:
        write_markdown(args.index, runs, full=True)


if __name__ == "__main__":
    main()
