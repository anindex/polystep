"""Aggregate complete practical comparisons and retain all final seeds."""

import json
import numpy as np
from experiments.runners.run_controlled import FINAL_SEEDS, atomic_json
from experiments.runners.run_practical import OUT
from experiments.scripts.practical_campaign import groups, result_path
from experiments.scripts.aggregate_controlled import paired_statistics, holm


def main():
    records = {}
    summaries = {}
    missing = []
    for task, axis, method in groups():
        key = f"{task}_{axis}_{method}"
        selection = OUT / "selected" / f"{key}.json"
        if not selection.exists():
            missing.append(key)
            continue
        cfg = json.loads(selection.read_text())["config"]
        paths = [result_path(task, axis, method, cfg, seed, "final") for seed in FINAL_SEEDS]
        if not all(p.exists() for p in paths):
            missing.append(key)
            continue
        rows = [json.loads(p.read_text()) for p in paths]
        endpoint = "macro_accuracy" if task == "dvs" else "accuracy"
        expected_budget = 10_000_000 if axis == "evals" else 3600
        for seed, row in zip(FINAL_SEEDS, rows):
            if row["seed"] != seed or row["stage"] != "final" or row["budget"] != expected_budget:
                raise ValueError(f"inconsistent final record: {key}/{seed}")
            if row["config"] != dict(task=task, axis=axis, method=method, **cfg):
                raise ValueError(f"configuration mismatch: {key}/{seed}")
        scores = np.array([100 * r["test"][endpoint] for r in rows])
        if not np.isfinite(scores).all():
            raise ValueError(f"nonfinite outcome: {key}")
        summaries[key] = dict(
            task=task,
            axis=axis,
            method=method,
            config=cfg,
            endpoint=endpoint,
            seeds=list(FINAL_SEEDS),
            scores=scores.tolist(),
            mean=float(scores.mean()),
            sample_sd=float(scores.std(ddof=1)),
            failures=sum(r["failure"] is not None for r in rows),
            ordinary_accuracy=[100 * r["test"]["accuracy"] for r in rows],
            evaluations=[r["evals"] for r in rows],
            seconds=[r["wall_seconds"] for r in rows],
        )
        records[key] = rows
    contrasts = []
    for key, base in summaries.items():
        if base["method"] not in ("polystep", "polystep_tuned"):
            continue
        for other, comparison in summaries.items():
            if comparison["task"] != base["task"] or comparison["axis"] != base["axis"]:
                continue
            if comparison["method"] == "polystep_tuned" or comparison["method"] == base["method"]:
                continue
            contrasts.append(
                dict(base=key, comparator=other, **paired_statistics(np.array(base["scores"]) - comparison["scores"]))
            )
    if not missing:
        for row, p in zip(contrasts, holm([c["sign_flip_p"] for c in contrasts])):
            row["holm_p"] = p
    atomic_json(
        OUT / "summary.json",
        dict(completed=summaries, missing=missing, contrasts=contrasts, practical_family_complete=not missing),
    )
    for task, axis in sorted({(g[0], g[1]) for g in groups()}):
        keys = [f"{t}_{a}_{m}" for t, a, m in groups() if (t, a) == (task, axis)]
        if all(k in summaries for k in keys):
            plot(task, axis, keys, summaries, records, contrasts)
    print(f"Practical comparisons complete: {len(summaries)}/{len(summaries) + len(missing)}")


def plot(task, axis, keys, summaries, records, contrasts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    dest = OUT / "figures"
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    objective = "macro_accuracy" if task == "dvs" else "accuracy"
    for index, key in enumerate(keys):
        color = plt.cm.tab10(index)
        label = summaries[key]["method"].replace("_", " ")
        rows = records[key]
        for col, (field, xlabel) in enumerate(
            (("evals", "Training candidate evaluations"), ("wall_seconds", "Elapsed seconds"))
        ):
            if field == "evals" and summaries[key]["method"].startswith("adam"):
                continue
            xx = [np.array([p.get(field, 0) for p in r["logs"] if p.get("eligible", True)]) for r in rows]
            lo, hi = max(x[0] for x in xx), min(x[-1] for x in xx)
            support = np.linspace(lo, hi, 200) if hi > lo else np.array([lo])
            for row_index, (metric, scale, ylabel) in enumerate(
                ((objective, 100, "Validation accuracy (%)"), ("loss", 1, "Validation loss"))
            ):
                yy = [scale * np.array([p[metric] for p in r["logs"] if p.get("eligible", True)]) for r in rows]
                ax = axes[row_index, col]
                for x, y in zip(xx, yy):
                    ax.plot(x, y, color=color, alpha=0.13, lw=0.5)
                if hi > lo:
                    values = np.array([np.interp(support, x, y) for x, y in zip(xx, yy)])
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    ax.plot(support, mean, label=label, color=color)
                    ax.fill_between(support, mean - sd, mean + sd, color=color, alpha=0.09)
                ax.set(xlabel=xlabel, ylabel=ylabel)
                ax.grid(alpha=0.2)
        values = summaries[key]["scores"]
        y = len(keys) - index
        axes[0, 2].scatter(values, [y] * 10, color=color, s=12, alpha=0.65)
        axes[0, 2].scatter([np.mean(values)], [y], color=color, marker="|", s=120)
    axes[0, 2].set(
        yticks=range(len(keys), 0, -1),
        yticklabels=[summaries[k]["method"].replace("_", " ") for k in keys],
        xlabel="Selected-checkpoint test accuracy (%)",
    )
    relevant = [c for c in contrasts if c["base"] == f"{task}_{axis}_polystep_tuned"]
    for i, c in enumerate(relevant):
        mean = c["mean_difference"]
        low, high = c["confidence_interval"]
        axes[1, 2].plot([low, high], [i, i], color="C0")
        axes[1, 2].scatter(c["paired_differences"], [i] * 10, color="C0", alpha=0.35, s=10)
        axes[1, 2].scatter([mean], [i], color="black", s=20)
    axes[1, 2].axvline(0, color=".5", lw=0.8)
    axes[1, 2].set(
        yticks=range(len(relevant)),
        yticklabels=[summaries[c["comparator"]]["method"].replace("_", " ") for c in relevant],
        xlabel="Practical PolyStep minus comparator (percentage points)",
    )
    axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        f"{task.upper()}, {axis}: 10 final seeds; validation selection; curves: mean ± sample SD; differences: paired bootstrap 95% CI",
        fontsize=10,
    )
    fig.savefig(dest / f"{task}_{axis}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
