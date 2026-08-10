r"""Emit the paper's result tables and numeric macros from the result JSONs.

Every cell is filled from ``experiments/results/revision`` or emitted as
``\placeholder``, which the paper renders as a red PENDING marker, so an
unmeasured number cannot appear in the PDF.

Usage:
    python experiments/scripts/generate_paper_tables.py \
        --results-dir experiments/results/revision \
        --paper-dir ../polystep_arxiv

Writes tables/snn_comparison.tex, tables/mnist_comparison.tex,
tables/rl_nondiff.tex and tables/macros.tex under --paper-dir.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# Results older than this cutoff carry the probe-scale unit error and are dropped.
# The timestamp is UTC and lexicographically ordered (%Y-%m-%dT%H:%M:%SZ), so a
# plain string comparison against the cutoff works.
DEFAULT_CUTOFF = "2026-08-01T11:02:28Z"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate_results import load_single_result  # noqa: E402

PLACEHOLDER = r"\placeholder"

# Table row order and display names.
GF_ORDER = [
    ("polystep", r"\sysname{}"),
    ("openai_es", "OpenAI-ES"),
    ("cma_es", "CMA-ES"),
    ("eggroll", "EGGROLL"),
    ("mezo", "MeZO"),
    ("spsa", "SPSA"),
    ("random_search", "Random search"),
    ("adam", r"Adam$^\dagger$"),
]

# Marker printed next to each cost figure, showing the axis the row was budgeted
# on. PolyStep and Adam set the budget; they do not receive one.
AXIS_MARK = {"evals": r"$^{e}$", "steps": r"$^{s}$", "wallclock": r"$^{w}$", None: ""}

# Legend text per marker, assembled from the markers a table actually prints.
# The $^{s}$ wording states the protocol: the step budget fixes an evaluation
# total at reference population 32, and each arm spends that total at its own
# population, so recorded step counts differ where populations do.
AXIS_LEGEND = {
    r"$^{e}$": r"$^{e}$matched on evaluations.",
    r"$^{s}$": (
        r"$^{s}$budgeted on optimizer steps: the arm spends the evaluation total "
        r"implied by \sysname{}'s step count at reference population $32$, at its "
        r"own population size, so recorded steps differ where the population does."
    ),
    r"$^{w}$": r"$^{w}$matched on wall-clock.",
}

RL_METHODS = [("polystep", r"\sysname{}"), ("es", "OpenAI-ES"), ("ppo", "PPO"), ("dqn", "DQN")]
RL_ENVS = [("cartpole", "CartPole-v1"), ("acrobot", "Acrobot-v1")]
RL_PRECISIONS = [("", "F32"), ("_nondiff_int8", "INT8"), ("_nondiff_binary", "Binary")]


def collect(results_dir, cutoff):
    """Group every non-leaked result at or after ``cutoff`` by (benchmark, method).

    load_single_result raises LeakedResultError on a test-selected run, and we do
    not catch it: a leaked file must break the build, not be skipped. Pre-cutoff
    runs carry the probe-scale defect and are dropped back to placeholders.
    """
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


def evals_of(runs, benchmark, method):
    """Mean evaluations actually spent, and the minimum, across seeds.

    Mean, not max: a max over seeds prints the one seed that spent the budget and
    hides the ones that stopped early. ``is not None``, not truthiness, because a
    genuine 0 is a measurement (two runners write it) and must not vanish.
    """
    rows = runs.get((benchmark, method), [])
    vals = [r.get("function_evals") for r in rows if r.get("function_evals") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals), min(vals)


def si_cell(s):
    """A siunitx S-column cell. \\placeholder cannot sit in an S column.

    At one seed there is no spread to report, so the mean is printed alone rather
    than with a fabricated 0.0 standard deviation.
    """
    if s is None:
        return "{" + PLACEHOLDER + "}"
    return f"{s[0]:.1f}" if s[2] < 2 else f"{s[0]:.1f} \\pm {s[1]:.1f}"


def _si(n):
    for div, suf in ((1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suf}".replace(".00", "")
    return f"{n:.0f}"


def human_evals(v):
    """Mean spend, and the minimum in parentheses when a seed stopped early.

    Printing the minimum is what stops a self-terminating method from reading as
    budget-matched.
    """
    if v is None:
        return PLACEHOLDER
    mean, lo = v
    if lo < 0.95 * mean:
        return f"{_si(mean)} (min {_si(lo)})"
    return _si(mean)


def mean_of(runs, benchmark, method, key):
    rows = runs.get((benchmark, method), [])
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def axis_of(runs, benchmark, method):
    """The single axis every seed of this cell was budgeted on.

    Raises when seeds disagree: a cell whose seeds were matched differently is not
    one number.
    """
    rows = runs.get((benchmark, method), [])
    axes = {r.get("match_axis") for r in rows}
    if len(axes) > 1:
        raise ValueError(f"{benchmark}/{method} mixes matching axes across seeds: {sorted(map(str, axes))}")
    return axes.pop() if axes else None


def _steps(v):
    if v is None:
        return PLACEHOLDER
    return f"{v:,.0f}".replace(",", r"{,}")


def gallery_table(runs, benchmark, *, dagger=False):
    """One row per method: the two budget axes, then accuracy.

    Wall-clock is not a column: it is a property of the implementation and the
    device rather than of the update rule. It is reported, with that caveat
    attached, in Appendix~\ref{app:wallclock-matched}.
    """
    best_gf = None
    # Uncertainty width is per table: a wide SNN std overflows the (2) format
    # the other galleries fit in, and siunitx misaligns the +- column when it does.
    unc_digits = (
        3
        if any(
            s is not None and s[2] >= 2 and s[1] >= 9.95 for s in (stat(runs, benchmark, key) for key, _ in GF_ORDER)
        )
        else 2
    )
    marks_used = set()
    lines = [
        rf"\begin{{tabular}}{{lrrS[table-format=2.1({unc_digits})]r}}",
        r"\toprule",
        r"{Method} & {Steps} & {Evals} & {Test accuracy (\%)} & {Seeds} \\",
        r"\midrule",
    ]
    for key, name in GF_ORDER:
        # No Adam row where the architecture has no differentiable twin; `dagger`
        # marks the benchmarks that have one.
        if key == "adam" and not dagger:
            continue
        s = stat(runs, benchmark, key)
        if s is None:
            lines.append(f"{name} & {PLACEHOLDER} & {PLACEHOLDER} & {{{PLACEHOLDER}}} & {PLACEHOLDER} \\\\")
            continue
        # "Best baseline" only ranks methods with a spread to report; a single
        # seed is not an estimate to compare against.
        if key not in ("polystep", "adam") and s[2] >= 2 and (best_gf is None or s[0] > best_gf[0]):
            best_gf = s
        mark = AXIS_MARK[axis_of(runs, benchmark, key)]
        if mark:
            marks_used.add(mark)
        lines.append(
            f"{name} & {_steps(mean_of(runs, benchmark, key, 'total_steps'))} & "
            f"{human_evals(evals_of(runs, benchmark, key))}{mark} & "
            f"{si_cell(s)} & {s[2]} \\\\"
        )
    # Legend entries only for markers actually printed, so no table names a row
    # that does not exist.
    foot = [AXIS_LEGEND[m] for m in (r"$^{e}$", r"$^{s}$", r"$^{w}$") if m in marks_used]
    foot.append(
        r"\sysname{} and Adam set the budget the others are matched to."
        if dagger
        else r"\sysname{} sets the budget the others are matched to."
    )
    if dagger:
        # On smooth benchmarks Adam trains the model itself; "surrogate
        # relaxation" is only true where the forward pass jumps.
        if benchmark in ("mnist", "timeseries", "etth1"):
            foot.append(
                r"$^\dagger$Adam trains the same differentiable model with exact "
                r"gradients, the information-advantaged reference."
            )
        else:
            foot.append(
                r"$^\dagger$Adam trains a differentiable surrogate relaxation of the same "
                r"architecture, not the hard model."
            )
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\parbox{0.97\linewidth}{\scriptsize " + " ".join(foot) + "}}",
        r"\end{tabular}",
    ]
    return "\n".join(lines), stat(runs, benchmark, "polystep"), best_gf, stat(runs, benchmark, "adam")


POPSIZE = {"openai_es": 32, "eggroll": 32, "mezo": 2, "spsa": 2, "random_search": 1}


def _prefix_best(run, key, limit):
    """Best validation seen up to ``limit`` on axis ``key``.

    Best-so-far, not the last point: every arm selects the best-on-validation
    iterate, so a truncated read-off has to select the same way.
    """
    best = None
    for point in run.get("step_logs", []):
        if point.get(key, 0) > limit:
            break
        v = point.get("val_accuracy")
        if v is not None and (best is None or v > best):
            best = v
    return best


def _read_off(runs, bench, method, key, limit):
    vals = [v for r in runs.get((bench, method), []) if (v := _prefix_best(r, key, limit)) is not None]
    return sum(vals) / len(vals) if vals else None


def axis_table(runs, benchmarks):
    r"""The same cells read on the step and evaluation axes, from one set of runs.

    The baselines on these benchmarks were matched on evaluations, so truncating
    their trajectories at \sysname{}'s step count recovers the step axis.
    Validation accuracy throughout: a trajectory carries no test score.
    """
    methods = [("openai_es", "OpenAI-ES"), ("cma_es", "CMA-ES"), ("eggroll", "EGGROLL")]
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Benchmark & Method & Steps & Evaluations \\",
        r"\midrule",
    ]
    for bench, label in benchmarks:
        ps_runs = runs.get((bench, "polystep"), [])
        if not ps_runs:
            continue
        pstep = mean_of(runs, bench, "polystep", "total_steps")
        pval = mean_of(runs, bench, "polystep", "best_accuracy")
        first = True
        for key, name in methods:
            st = _read_off(runs, bench, key, "evals", pstep * POPSIZE.get(key, 32))
            fu = mean_of(runs, bench, key, "best_accuracy")
            cells = [f"${100 * v:.1f}$" if v is not None else PLACEHOLDER for v in (st, fu)]
            lines.append(f"{label if first else ''} & {name} & " + " & ".join(cells) + r" \\")
            first = False
        best = f"$\\mathbf{{{100 * pval:.1f}}}$" if pval is not None else PLACEHOLDER
        lines.append(r" & \sysname{} & " + " & ".join([best] * 2) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


STEP_CAPPED = {"argmax", "moe", "timeseries"}


def step_sweep_table(runs, benchmarks, methods):
    r"""Validation accuracy for every method on every benchmark, at matched steps.

    Population methods on the evaluation-matched showcases are read off their own
    trajectory at \sysname{}'s step count; everywhere else the run was already
    budgeted on steps and its recorded selection stands. Validation throughout:
    a truncated trajectory carries no test score.
    """
    lines = [
        r"\begin{tabular}{l" + "c" * len(benchmarks) + "}",
        r"\toprule",
        "Method & " + " & ".join(lab for _, lab in benchmarks) + r" \\",
        r"\midrule",
    ]
    for key, name in methods:
        cells = []
        for bench, _ in benchmarks:
            pstep = mean_of(runs, bench, "polystep", "total_steps")
            if key == "polystep":
                v = mean_of(runs, bench, "polystep", "best_accuracy")
            elif bench in STEP_CAPPED or key in ("mezo", "spsa", "random_search"):
                v = mean_of(runs, bench, key, "best_accuracy")
            else:
                v = _read_off(runs, bench, key, "evals", pstep * POPSIZE.get(key, 32))
            cells.append(f"${100 * v:.1f}$" if v is not None else PLACEHOLDER)
        if key == "polystep":
            cells = [c.replace("$", "$\\mathbf{", 1)[:-1] + "}$" if c != PLACEHOLDER else c for c in cells]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rl_table(runs):
    seed_counts = set()
    body = []
    for mkey, mname in RL_METHODS:
        cells = []
        for env, _ in RL_ENVS:
            for suffix, _ in RL_PRECISIONS:
                s = stat(runs, env, mkey + suffix, key="final_accuracy", scale=1.0)
                if s is None:
                    cells.append(PLACEHOLDER)
                else:
                    seed_counts.add(s[2])
                    # A single seed has no spread; printing "\pm 0.000" fabricates one.
                    cells.append(f"${s[0]:.3f}$" if s[2] < 2 else f"${s[0]:.3f} \\pm {s[1]:.3f}$")
        body.append(f"{mname} & " + " & ".join(cells) + r" \\")

    seeds = (
        f"{min(seed_counts)}--{max(seed_counts)}"
        if len(seed_counts) > 1
        else (str(next(iter(seed_counts))) if seed_counts else PLACEHOLDER)
    )
    header = " & ".join(p for _, p in RL_PRECISIONS)
    return "\n".join(
        [
            r"% Auto-generated by experiments/scripts/generate_paper_tables.py",
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{RL policy search: 2 environments $\times$ 3 precision regimes "
            rf"({seeds} seeds, normalized return of the final policy averaged over the "
            rf"evaluation rollouts, mean $\pm$ std over training seeds). No RL runner "
            rf"holds a validation split or selects a checkpoint; the reported number is "
            rf"the last iterate. \sysname{{}} and OpenAI-ES take the same number of "
            rf"optimizer steps; the environment interactions that buys each method differ "
            rf"and are reported in Table~\ref{{tab:rl_cost}}, which is where to read the "
            rf"cost of this table. PPO and DQN are run with the quantization in the forward "
            rf"pass and no straight-through estimator, so they are a diagnostic of "
            rf"gradient flow under quantization, not a claim about quantization-aware "
            rf"training.}}",
            r"\label{tab:rl_nondiff}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Method & \multicolumn{3}{c}{CartPole-v1} & \multicolumn{3}{c}{Acrobot-v1} \\",
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            rf" & {header} & {header} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


#: The reference-gap ladder. Each cell flips one assumption of the analysed setting on or off,
#: starting from the tuned config or from theory mode, at seed 42 on the SNN.
LADDER_UP = [
    ("tuned", "tuned configuration", ""),
    ("tuned_plus_jitter", "$+$ smooth radius jitter", "(iv)"),
    ("tuned_plus_orthoplex", "$+$ orthoplex frame", "(i)"),
    ("tuned_minus_biased_rot", "$+$ independent rotations", "(ii)"),
    ("tuned_plus_decay", "$+$ decaying step radius", "(v)"),
]
LADDER_DOWN = [
    ("theory", "theory mode", ""),
    ("theory_nojitter", "$-$ smooth radius jitter", "(iv)"),
    ("theory_simplex", "$-$ orthoplex frame", "(i)"),
    ("theory_biased_rot", "$-$ independent rotations", "(ii)"),
    ("theory_flat_radius", "$-$ decaying step radius", "(v)"),
]


def _ablate_acc(ablate_dir, cell, theory_dir):
    if cell == "tuned":
        path = os.path.join(ablate_dir, "..", "revision", "snn_polystep_42.json")
    elif cell == "theory":
        path = os.path.join(theory_dir, "snn_polystep_42.json")
    else:
        path = os.path.join(ablate_dir, cell, "snn_polystep_42.json")
    if not os.path.exists(path):
        return None
    m = json.load(open(path))["metrics"]
    return m.get("test_accuracy_at_selected", m.get("final_accuracy"))


def theory_ladder_table(ablate_dir, theory_dir):
    """Which of the theorem's hypotheses actually costs accuracy.

    Two directions, so the answer is not an artifact of where you start.
    """
    rows = []
    for ladder in (LADDER_UP, LADDER_DOWN):
        base = _ablate_acc(ablate_dir, ladder[0][0], theory_dir)
        for cell, label, cond in ladder:
            acc = _ablate_acc(ablate_dir, cell, theory_dir)
            if acc is None:
                rows.append(f"{label} & {cond} & {PLACEHOLDER} & {PLACEHOLDER} \\\\")
                continue
            delta = "" if cell == ladder[0][0] else f"${100 * (acc - base):+.2f}$"
            rows.append(f"{label} & {cond} & ${100 * acc:.2f}$ & {delta} \\\\")
        rows.append(r"\midrule")
    rows.pop()
    return "\n".join(
        [
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Configuration & Condition & Test acc. (\%) & $\Delta$ vs.\ block head \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def rl_cost_table(runs):
    r"""Environment interactions per cell, the budget an RL reader cares about.

    Optimizer steps are matched between \sysname{} and OpenAI-ES, which buys the
    two methods different numbers of rollouts. PPO and DQN are on their own SB3
    schedule.
    """

    def env_steps(bench, method):
        vals = [r.get("rl_env_steps") for r in runs.get((bench, method), []) if r.get("rl_env_steps")]
        return sum(vals) / len(vals) if vals else None

    lines = [
        r"% Auto-generated by experiments/scripts/generate_paper_tables.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Environment interactions behind Table~\ref{tab:rl_nondiff}, in "
        r"millions, mean over seeds. \sysname{} and OpenAI-ES take the same number of "
        r"optimizer steps, which buys them different numbers of rollouts: we spend more "
        r"on CartPole and fewer on Acrobot. PPO and DQN run their own SB3 schedule at "
        r"$1.0$M throughout.}",
        r"\label{tab:rl_cost}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & \multicolumn{3}{c}{CartPole-v1} & \multicolumn{3}{c}{Acrobot-v1} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r" & F32 & INT8 & Binary & F32 & INT8 & Binary \\",
        r"\midrule",
    ]
    for mkey, mname in RL_METHODS:
        cells = []
        for env, _ in RL_ENVS:
            for suffix, _ in RL_PRECISIONS:
                v = env_steps(env, mkey + suffix)
                cells.append(PLACEHOLDER if v is None else f"${v / 1e6:.0f}$")
        lines.append(f"{mname} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def macros(snn_ps, snn_gf, snn_adam, mnist_ps, runs):
    """Numeric macros. Anything unmeasured stays \\placeholder."""

    def acc(s):
        if s is None:
            return PLACEHOLDER
        if s[2] < 2:
            return f"${s[0]:.1f}$\\%"
        return f"${s[0]:.1f} \\pm {s[1]:.1f}$\\%"

    out = [
        r"% Auto-generated by experiments/scripts/generate_paper_tables.py.",
        r"% Do not edit: regenerate after any re-run so a stale number cannot ship.",
        rf"\newcommand{{\snnBestAcc}}{{{acc(snn_ps)}}}",
        rf"\newcommand{{\snnBestBaseline}}{{{acc(snn_gf)}}}",
        rf"\newcommand{{\snnAdamAcc}}{{{acc(snn_adam)}}}",
        rf"\newcommand{{\mnistBestAcc}}{{{acc(mnist_ps)}}}",
    ]
    for macro, bench in (
        (r"\intEightBestAcc", "int8"),
        (r"\argmaxBestAcc", "argmax"),
        (r"\staircaseBestAcc", "staircase"),
        (r"\moeBestAcc", "moe"),
    ):
        out.append(rf"\newcommand{{{macro}}}{{{acc(stat(runs, bench, 'polystep'))}}}")

    # Comparative figures the prose quotes, generated so a re-run cannot leave a
    # stale ratio in the text.
    def ratio(bench, key, method="openai_es"):
        ours = mean_of(runs, bench, "polystep", key)
        theirs = mean_of(runs, bench, method, key)
        if not ours or not theirs:
            return PLACEHOLDER
        r = theirs / ours
        return f"${r:.0f}\\times$" if r >= 10 else f"${r:.1f}\\times$"

    best_name = None
    best = None
    for key, name in GF_ORDER:
        if key in ("polystep", "adam"):
            continue
        s = stat(runs, "snn", key)
        if s and s[2] >= 2 and (best is None or s[0] > best):
            best, best_name = s[0], name
    out += [
        rf"\newcommand{{\snnBestBaselineName}}{{{best_name or PLACEHOLDER}}}",
        rf"\newcommand{{\snnStepRatio}}{{{ratio('snn', 'total_steps')}}}",
        rf"\newcommand{{\mnistStepRatio}}{{{ratio('mnist', 'total_steps')}}}",
    ]

    mse = stat(runs, "timeseries", "polystep", key="final_mse", scale=1.0)
    mse_cell = PLACEHOLDER if mse is None else rf"${mse[0]:.3f} \pm {mse[1]:.3f}$"
    out.append(rf"\newcommand{{\timeseriesBestMse}}{{{mse_cell}}}")
    # No maxsat_1000000v macro: that row is footnoted in the table and must not
    # look generated.
    out.append(rf"\newcommand{{\maxsatHundredKAcc}}{{{acc(stat(runs, 'maxsat_100000v', 'polystep'))}}}")
    return "\n".join(out) + "\n"


README_START = "<!-- BENCH:START -->"
README_END = "<!-- BENCH:END -->"

# (benchmark key, row label, the op that breaks the gradient)
README_NONDIFF = [
    ("snn", "SNN/LIF (MNIST)", "`threshold()`"),
    ("int8", "Int8 quantized", "`round()`"),
    ("argmax", "Argmax attention", "`argmax()`"),
    ("staircase", "Staircase activation", "`floor()`"),
    ("moe", "Hard MoE routing", "`argmax()`"),
]
# Plain names: GF_ORDER's labels carry LaTeX macros that mean nothing in markdown.
# The README carries the short set; EXPERIMENT_INDEX carries every method measured.
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


#: Solvers that record ``hyperparameters.timed_out``. Only these can report a timeout;
#: elsewhere a zero is a measurement or a broken run and must not be dressed up as one.
TIMEOUT_AWARE = ("rc2", "probsat", "sls")


def _maxsat_cell(runs, benchmark, method):
    """As ``_md_cell``, but a solver that hit its wall clock reads timeout.

    Only the recorded flag counts. A zero from anything else is either a real
    measurement or a run that never wrote the metric, and both stay visible.
    """
    rows = runs.get((benchmark, method), [])
    if not rows:
        return "-"
    if method in TIMEOUT_AWARE and all(r.get("timed_out") for r in rows):
        return "timeout"
    if any(not r.get("has_final_accuracy", True) for r in rows):
        return "no data"
    return _md_cell(stat(runs, benchmark, method, key="final_accuracy"))


def markdown_tables(runs, full):
    """Result tables as markdown, from the same JSONs as the paper's.

    ``full`` adds every method measured and every MAX-SAT size; the short form is
    what the README carries.
    """
    methods = FULL_METHODS if full else BRIEF_METHODS
    md = ["### Non-differentiable tasks", "", "Test accuracy %. A dash means the cell is not in this release.", ""]

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
    # Exactly one pair: with two, the splice would rewrite the first and leave the
    # second holding stale numbers, so the file would carry two contradictory tables.
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
    ap.add_argument("--paper-dir", default="../polystep_arxiv")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="drop results whose timestamp precedes this date")
    ap.add_argument("--readme", help="regenerate the short tables between the BENCH markers here")
    ap.add_argument("--index", help="regenerate the full tables between the BENCH markers here")
    args = ap.parse_args()

    runs = collect(args.results_dir, args.cutoff)
    # The headline cells are matched on optimizer steps, which fixes every
    # population method at FAIR_POPSIZE. The wall-clock and evaluation arms live
    # in subdirectories at a different population and are reported separately.

    tables = os.path.join(args.paper_dir, "tables")
    os.makedirs(tables, exist_ok=True)

    snn, snn_ps, snn_gf, snn_adam = gallery_table(runs, "snn", dagger=True)
    mnist, mnist_ps, _, _ = gallery_table(runs, "mnist", dagger=True)
    step_axis = axis_table(
        runs,
        [("snn", "SNN (hard LIF)"), ("mnist", "MNIST"), ("int8", "INT8"), ("staircase", "Staircase")],
    )

    written = {
        "snn_comparison.tex": snn + "\n",
        "mnist_comparison.tex": mnist + "\n",
        "step_axis.tex": step_axis + "\n",
        "step_sweep.tex": step_sweep_table(
            runs,
            [
                ("snn", "SNN"),
                ("mnist", "MNIST"),
                ("int8", "INT8"),
                ("staircase", "Stair"),
                ("argmax", "Argmax"),
                ("moe", "MoE"),
            ],
            [(k, n) for k, n in GF_ORDER if k != "adam"],
        )
        + "\n",
        "rl_nondiff.tex": rl_table(runs) + "\n",
        "rl_cost.tex": rl_cost_table(runs) + "\n",
        "theory_ladder.tex": theory_ladder_table(
            os.path.join(os.path.dirname(args.results_dir), "ablate_theory"),
            os.path.join(args.results_dir, "theory"),
        )
        + "\n",
        "macros.tex": macros(snn_ps, snn_gf, snn_adam, mnist_ps, runs),
    }
    for bench in ("int8", "argmax", "staircase", "moe"):
        body, _, _, _ = gallery_table(runs, bench, dagger=bench != "moe")
        written[f"{bench}_comparison.tex"] = body + "\n"
    for name, text in written.items():
        with open(os.path.join(tables, name), "w") as f:
            f.write(text)
        print(f"wrote tables/{name}")

    if args.readme:
        write_markdown(args.readme, runs, full=False)
    if args.index:
        write_markdown(args.index, runs, full=True)

    pending = sum(text.count(PLACEHOLDER) for text in written.values())
    print(f"\n{pending} placeholder cells remain; the PDF renders each in red.")
    print(f"{len(runs)} (benchmark, method) groups found in {args.results_dir}")


if __name__ == "__main__":
    main()
