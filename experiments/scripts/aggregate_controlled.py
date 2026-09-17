"""Summarize completed final seeds; never fill missing outcomes."""

import itertools
import json

from experiments.runners.run_controlled import RESULTS, FINAL_SEEDS, RULES, atomic_json, config_id
import numpy as np


def paired_statistics(difference):
    difference = np.asarray(difference, dtype=float)
    if difference.shape != (10,) or not np.isfinite(difference).all():
        raise ValueError("ten finite paired outcomes are required")
    rng = np.random.default_rng(0)
    bootstrap = difference[rng.integers(0, 10, size=(10000, 10))].mean(1)
    signs = np.array(list(itertools.product((-1, 1), repeat=10)))
    observed = abs(difference.mean())
    p = float(np.mean(np.abs(signs @ difference / 10) >= observed - 1e-12))
    return dict(
        mean_difference=float(difference.mean()),
        confidence_interval=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        sign_flip_p=p,
        paired_differences=difference.tolist(),
    )


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main():
    summaries = {}
    records = {}
    missing = []
    arms = [
        f"{g}_{r}_{n}" for g, n, r in itertools.product(("orthoplex", "antipodal"), ("natural", "normalized"), RULES)
    ]
    arms += ["joint_softmax_natural", "schedule_constant", "schedule_decay"]
    for arm in arms:
        selected = RESULTS / "selected" / f"{arm}.json"
        if not selected.exists():
            missing.append(arm)
            continue
        cfg = json.loads(selected.read_text())["config"]
        paths = [RESULTS / "final" / arm / config_id(cfg) / f"{s}.json" for s in FINAL_SEEDS]
        if not all(p.exists() for p in paths):
            missing.append(arm)
            continue
        rows = [json.loads(p.read_text()) for p in paths]
        assert [r["seed"] for r in rows] == list(FINAL_SEEDS)
        if not all(r["stage"] == "final" and r["budget"] == 10_000_000 and r["config"] == cfg for r in rows):
            raise ValueError(f"inconsistent final records: {arm}")
        scores = np.array([100 * r["test"]["accuracy"] for r in rows])
        summaries[arm] = dict(
            config=cfg,
            seeds=list(FINAL_SEEDS),
            test_accuracy=scores.tolist(),
            mean=float(scores.mean()),
            sample_sd=float(scores.std(ddof=1)),
            failures=sum(r["failure"] is not None for r in rows),
        )
        records[arm] = rows
    contrasts = []
    for geometry, normalization in itertools.product(("orthoplex", "antipodal"), ("natural", "normalized")):
        base = f"{geometry}_softmax_{normalization}"
        for rule in RULES[1:]:
            other = f"{geometry}_{rule}_{normalization}"
            if base in summaries and other in summaries:
                difference = np.array(summaries[base]["test_accuracy"]) - summaries[other]["test_accuracy"]
                contrasts.append(dict(softmax=base, comparator=other, **paired_statistics(difference)))
    # Multiplicity correction is withheld until the entire prespecified family is present.
    if len(contrasts) == 16:
        for row, p in zip(contrasts, holm([r["sign_flip_p"] for r in contrasts])):
            row["holm_p"] = p
    atomic_json(
        RESULTS / "summary.json",
        dict(
            completed_arms=summaries,
            missing_arms=missing,
            weighting_family_complete=len(contrasts) == 16,
            weighting_contrasts=contrasts,
        ),
    )
    if all(arm in summaries for arm in arms[:20]):
        plot(records, summaries)
        endpoint_plot(summaries, contrasts)
    print(f"Final arms complete: {len(summaries)}/{len(arms)}")


def plot(records, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    destination = RESULTS / "figures"
    destination.mkdir(exist_ok=True)
    for geometry, normalization in itertools.product(("orthoplex", "antipodal"), ("natural", "normalized")):
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), layout="constrained")
        for rule, color in zip(RULES, plt.cm.tab10.colors):
            rows = records[f"{geometry}_{rule}_{normalization}"]
            for col, (axis, label) in enumerate(
                (
                    ("step", "Optimizer updates"),
                    ("evals", "Candidate evaluations"),
                    ("processed_examples", "Processed training examples"),
                    ("wall_seconds", "Elapsed seconds"),
                )
            ):
                traces = [r["logs"] for r in rows]
                xx = [np.array([p.get(axis, 0) for p in trace], float) for trace in traces]
                lo, hi = max(x[0] for x in xx), min(x[-1] for x in xx)
                support = np.linspace(lo, hi, 200) if hi > lo else np.array([])
                for row_index, (metric, ylabel, scale) in enumerate(
                    (("accuracy", "Validation accuracy (%)", 100), ("loss", "Validation loss", 1))
                ):
                    yy = [scale * np.array([p[metric] for p in trace]) for trace in traces]
                    interpolated = np.array([np.interp(support, x, y) for x, y in zip(xx, yy)])
                    mean, sd = interpolated.mean(0), interpolated.std(0, ddof=1)
                    ax = axes[row_index, col]
                    for x, y in zip(xx, yy):
                        ax.plot(x, y, color=color, alpha=0.1, lw=0.5)
                    ax.plot(support, mean, color=color, label=rule, lw=1.5)
                    ax.fill_between(support, mean - sd, mean + sd, color=color, alpha=0.08)
                    ax.set_xlabel(label)
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.2)
                    if axis in ("evals", "processed_examples"):
                        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.savefig(destination / f"{geometry}_{normalization}_curves.pdf")
        plt.close(fig)


def diagnostic_statistics(path):
    with np.load(path) as d:
        changes = d["loss_changes"]
        directions = d["directions"].reshape(len(changes), len(RULES), -1)
        norm = np.linalg.norm(directions, axis=-1)
        denominator = norm * norm[:, 0, None]
        alignment = np.divide(
            (directions * directions[:, 0, None, :]).sum(-1),
            denominator,
            out=np.full_like(denominator, np.nan),
            where=denominator > 0,
        )
        result = dict(
            mean_norm=d["norms"].mean(axis=(0, 2)).tolist(),
            zero_fraction=d["zero_rows"].mean(axis=(0, 2)).tolist(),
            mean_loss_change=changes.mean(0).tolist(),
            decrease_fraction=(changes < 0).mean(0).tolist(),
            direction_variance=directions.var(0, ddof=1).sum(-1).tolist(),
            cosine_to_softmax=[float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None for x in alignment.T],
            evaluations=int(d["snapshot_evals"]),
            step=int(d["snapshot_step"]),
        )
        if d["background_directions"].size:
            b = d["background_directions"].reshape(len(changes), -1)
            result.update(
                background_variance=float(b.var(0, ddof=1).sum()),
                background_mean_loss_change=float(d["background_loss_changes"].mean()),
            )
        return result


def aggregate_diagnostics():
    rows = []
    for geometry, normalization, seed in itertools.product(
        ("orthoplex", "antipodal"), ("natural", "normalized"), FINAL_SEEDS
    ):
        root = RESULTS / "diagnostics" / f"{geometry}_{normalization}" / str(seed)
        if not (root / "complete.json").exists():
            raise ValueError(f"incomplete diagnostics: {root}")
        for source, index in itertools.product(("softmax", "greedy"), range(20)):
            path = root / f"{source}_{index:02d}.npz"
            rows.append(
                dict(
                    geometry=geometry,
                    normalization=normalization,
                    seed=seed,
                    source=source,
                    snapshot=index,
                    **diagnostic_statistics(path),
                )
            )
    atomic_json(RESULTS / "diagnostics/summary.json", dict(rows=rows, independent_units="ten final training seeds"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = RESULTS / "figures"
    dest.mkdir(exist_ok=True)
    for geometry, normalization in itertools.product(("orthoplex", "antipodal"), ("natural", "normalized")):
        group = [r for r in rows if (r["geometry"], r["normalization"]) == (geometry, normalization)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
        for ax, field, label in zip(
            axes.flat,
            (
                "mean_norm",
                "zero_fraction",
                "mean_loss_change",
                "decrease_fraction",
                "direction_variance",
                "cosine_to_softmax",
            ),
            (
                "Mean raw block-direction norm",
                "Zero block fraction",
                "Joint diagnostic loss change",
                "Strict loss-decrease fraction",
                "Joint direction variance",
                "Cosine with softmax direction",
            ),
        ):
            # Reduce repeated probes/snapshots within a seed before showing seed variation.
            values = []
            for seed in FINAL_SEEDS:
                cells = [[np.nan if x is None else x for x in r[field]] for r in group if r["seed"] == seed]
                array = np.array(cells)
                values.append([float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else np.nan for v in array.T])
            values = np.array(values)
            for j, rule in enumerate(RULES):
                v = values[:, j]
                v = v[np.isfinite(v)]
                if len(v):
                    ax.scatter(np.full(len(v), j), v, color=f"C{j}", s=13, alpha=0.6)
                    ax.errorbar(j, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="_", color="black", capsize=3)
            ax.set_xticks(range(5), RULES, rotation=25)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)
            if field == "mean_loss_change":
                ax.axhline(0, color=".5", lw=0.7)
        fig.suptitle(
            f"{geometry}, {normalization}; 10 seeds, 40 snapshots/seed, 64 draws/snapshot; mean ± sample SD across seeds",
            fontsize=10,
        )
        fig.savefig(dest / f"{geometry}_{normalization}_diagnostics.pdf")
        plt.close(fig)


def endpoint_plot(summaries, contrasts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), layout="constrained")
    for col, (geometry, normalization) in enumerate(
        itertools.product(("orthoplex", "antipodal"), ("natural", "normalized"))
    ):
        for j, rule in enumerate(RULES):
            values = summaries[f"{geometry}_{rule}_{normalization}"]["test_accuracy"]
            axes[0, col].scatter(values, [j] * 10, color=f"C{j}", s=14, alpha=0.6)
            axes[0, col].scatter([np.mean(values)], [j], color="black", marker="|", s=100)
        axes[0, col].set(
            yticks=range(5),
            yticklabels=RULES,
            xlabel="Selected-checkpoint test accuracy (%)",
            title=f"{geometry}, {normalization}",
        )
        base = f"{geometry}_softmax_{normalization}"
        for j, row in enumerate(r for r in contrasts if r["softmax"] == base):
            low, high = row["confidence_interval"]
            axes[1, col].plot([low, high], [j, j], color=f"C{j + 1}")
            axes[1, col].scatter(row["paired_differences"], [j] * 10, color=f"C{j + 1}", alpha=0.5, s=12)
            axes[1, col].scatter(row["mean_difference"], j, color="black", s=20)
        axes[1, col].axvline(0, color=".5", lw=0.7)
        axes[1, col].set(yticks=range(4), yticklabels=RULES[1:], xlabel="Softmax minus comparator (percentage points)")
    fig.suptitle(
        "10 final seeds at 10 million candidate evaluations; validation selection; difference intervals: paired bootstrap 95%",
        fontsize=10,
    )
    fig.savefig(RESULTS / "figures/weighting_endpoints.pdf")
    plt.close(fig)


def self_check():
    assert paired_statistics(np.zeros(10))["sign_flip_p"] == 1
    assert paired_statistics(np.ones(10))["sign_flip_p"] == 2 / 1024
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    self_check()
    main()
    import sys

    if "--diagnostics" in sys.argv:
        aggregate_diagnostics()
