#!/usr/bin/env python
"""Exploratory figure 01: train-test distance for every splitter, one panel per dataset.

The paper reports this quantity pooled over all eight datasets
(``assets/figures/distance_distribution``). Pooling hides that the same splitter
can behave very differently from one dataset to the next, so this figure keeps the
paper's definition and its data and only stops pooling.

Definition (unchanged from the paper, ``notebooks/paper/01_figures.ipynb``): for every
out-of-distribution test molecule, the distance to each of its 5 nearest molecules in
the training set (72 % subset of the train file), over all 10 folds. Two distances:
Tanimoto (Jaccard on ECFP4, 2048 bits) and Tree Mover's Distance (TMD).

Input : datasets/TDC/all_splitters_nearest_distances.csv  (frozen, read-only)
Output: assets/exploratory/01_train_test_distance_per_dataset/
            tanimoto.{png,pdf}   box plots per dataset
            tmd.{png,pdf}        same for TMD
            summary.{png,pdf}    medians as a dataset x splitter grid
            summary_stats.csv    the numbers behind all three

Run: python scripts/exploratory/fig01_train_test_distance_per_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as st  # noqa: E402

SOURCE = "datasets/TDC/all_splitters_nearest_distances.csv"
FOLDER = "01_train_test_distance_per_dataset"
METRICS = {
    "tanimoto": "Tanimoto distance to the 5 nearest training molecules",
    "tmd": "Tree Mover's Distance to the 5 nearest training molecules",
}
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def load() -> pd.DataFrame:
    return pd.read_csv(
        st.REPO / SOURCE,
        usecols=["tanimoto", "tmd", "split", "dataset", "index"],
        dtype={"tanimoto": "float32", "tmd": "float32", "split": "category", "dataset": "category", "index": "int8"},
    ).rename(columns={"index": "fold"})


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset x splitter x metric: quantiles, fold-median spread, counts."""
    rows = []
    for metric in METRICS:
        grouped = df.groupby(["dataset", "split"], observed=True)[metric]
        quantiles = grouped.quantile(QUANTILES).unstack()
        quantiles.columns = ["p05", "p25", "median", "p75", "p95"]
        fold_medians = df.groupby(["dataset", "split", "fold"], observed=True)[metric].median()
        fold_stats = fold_medians.groupby(["dataset", "split"], observed=True).agg(
            fold_median_min="min", fold_median_max="max"
        )
        block = quantiles.join(fold_stats).join(grouped.size().rename("n_values")).reset_index()
        block.insert(2, "metric", metric)
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    random_median = out[out["split"] == "random"].set_index(["dataset", "metric"])["median"]
    out["median_minus_random"] = out["median"] - out.set_index(["dataset", "metric"]).index.map(random_median).values
    return out


def dataset_sizes() -> Dict[str, int]:
    sizes = {}
    for dataset in st.DATASETS:
        path = st.REPO / "datasets" / "TDC" / dataset / f"{dataset}_standardize.csv"
        with open(path, "rb") as handle:
            sizes[dataset] = sum(1 for _ in handle) - 1
    return sizes


def splitter_order(stats: pd.DataFrame, df: pd.DataFrame) -> List[str]:
    """Random first (the baseline), then by Tanimoto median pooled over datasets, as in the paper."""
    pooled = df.groupby("split", observed=True)["tanimoto"].median().sort_values()
    return ["random"] + [name for name in pooled.index if name != "random"]


def footer(fig, extra: str = "") -> None:
    digest = st.source_fingerprint(SOURCE)
    source = f"Source: {SOURCE}" + (f"  (sha256 {digest[:12]}…, provenance/paper-v1)" if digest else "")
    note = (
        "Same data and definition as the paper's pooled figure: each test molecule contributes its 5 nearest "
        "training molecules (72 % training subset), all 10 folds. " + extra
    )
    fig.text(0.008, 0.012, note + "\n" + source, fontsize=7.5, color=st.INK_SECONDARY, va="bottom", ha="left")


def box_figure(
    df: pd.DataFrame, stats: pd.DataFrame, metric: str, order: List[str], sizes: Dict[str, int]
) -> plt.Figure:
    fold_medians = df.groupby(["dataset", "split", "fold"], observed=True)[metric].median()
    table = stats[stats["metric"] == metric].set_index(["dataset", "split"])
    shared_x = metric == "tanimoto"

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 9.6), sharey=True)
    fig.subplots_adjust(left=0.118, right=0.992, top=0.80, bottom=0.105, wspace=0.11, hspace=0.29)
    y_positions = np.arange(len(order))[::-1]  # first splitter at the top

    for ax, dataset in zip(axes.flat, st.DATASETS):
        x_max = 1.0 if shared_x else float(table.loc[dataset]["p95"].max()) * 1.2
        random_median = float(table.loc[(dataset, "random"), "median"])
        ax.axvline(random_median, color=st.BASELINE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

        for y, splitter in zip(y_positions, order):
            row = table.loc[(dataset, splitter)]
            is_baseline = splitter == "random"
            face = st.BASELINE_LIGHT if is_baseline else st.SERIES_LIGHT
            edge = st.BASELINE if is_baseline else st.SERIES
            dark = st.BASELINE_DARK if is_baseline else st.SERIES_DARK
            box_y = y + 0.11
            ax.plot([row["p05"], row["p95"]], [box_y, box_y], color=edge, linewidth=1.1, zorder=2)
            ax.barh(
                box_y,
                row["p75"] - row["p25"],
                left=row["p25"],
                height=0.44,
                color=face,
                edgecolor=edge,
                linewidth=1.0,
                zorder=3,
            )
            ax.plot([row["median"]] * 2, [box_y - 0.22, box_y + 0.22], color=dark, linewidth=2.0, zorder=4)
            folds = fold_medians.loc[(dataset, splitter)].to_numpy()
            ax.scatter(
                folds,
                np.full(folds.shape, y - 0.3),
                s=13,
                color=dark,
                alpha=0.75,
                edgecolors=st.SURFACE,
                linewidths=0.4,
                zorder=5,
            )
            label = f"{row['median']:.2f}" if shared_x else f"{row['median']:.0f}"
            ax.text(
                1.0,
                y,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.5,
                color=st.INK,
                bbox={"facecolor": st.SURFACE, "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
                zorder=6,
            )

        ax.set_title(dataset, pad=16)
        ax.text(
            0.0,
            1.018,
            f"{sizes[dataset]:,} molecules",
            transform=ax.transAxes,
            fontsize=8.5,
            color=st.INK_SECONDARY,
            va="bottom",
        )
        ax.text(
            1.0, 1.018, "median", transform=ax.transAxes, fontsize=8, color=st.INK_SECONDARY, va="bottom", ha="right"
        )
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.62, len(order) - 0.45)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([st.SPLITTER_NAMES[name] for name in order])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        for label_obj, splitter in zip(ax.get_yticklabels(), order):
            label_obj.set_color(st.INK_SECONDARY if splitter == "random" else st.INK)
    for ax in axes[-1]:
        ax.set_xlabel("Tanimoto distance" if shared_x else "Tree Mover's Distance (scale differs per panel)")

    fig.suptitle(
        ("Tanimoto" if shared_x else "Tree Mover's") + " distance between test and training molecules, per dataset",
        x=0.008,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.008,
        0.94,
        "Further right = test molecules are less like the training set (a harder, more out-of-distribution split). "
        "Splitters are in the same order in every panel:\nrandom first, then by the median pooled over all datasets "
        "(the paper's view). A box out of step with its neighbours is a dataset where that splitter behaves differently.",
        fontsize=9.5,
        color=st.INK_SECONDARY,
        va="top",
        ha="left",
        linespacing=1.45,
    )
    handles = [
        Patch(
            facecolor=st.SERIES_LIGHT, edgecolor=st.SERIES, label="Middle 50 % of values; line = 5th–95th percentile"
        ),
        Line2D([], [], color=st.SERIES_DARK, linewidth=2, label="Median over all folds"),
        Line2D([], [], marker="o", linestyle="", color=st.SERIES_DARK, markersize=4, label="Median of each fold (10)"),
        Patch(facecolor=st.BASELINE_LIGHT, edgecolor=st.BASELINE, label="Random split (baseline)"),
        Line2D([], [], color=st.BASELINE, linestyle=(0, (4, 3)), linewidth=1, label="Random-split median"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.002, 0.888),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.6,
    )
    footer(fig)
    return fig


def _annotate(ax, values: np.ndarray, norm, cmap, fmt: str) -> None:
    for (i, j), value in np.ndenumerate(values):
        r, g, b, _ = cmap(norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ax.text(
            j,
            i,
            format(value, fmt),
            ha="center",
            va="center",
            fontsize=9,
            color=st.INK if luminance > 0.5 else "#ffffff",
        )


def summary_figure(stats: pd.DataFrame, order: List[str]) -> plt.Figure:
    table = stats[stats["metric"] == "tanimoto"]
    median = table.pivot(index="split", columns="dataset", values="median").loc[order, st.DATASETS]
    delta = table.pivot(index="split", columns="dataset", values="median_minus_random").loc[order, st.DATASETS]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharey=True)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.80, bottom=0.12, wspace=0.05)
    panels = [
        (axes[0], median, st.sequential_cmap(st.SEQUENTIAL_BLUE), "Median Tanimoto distance", ".2f"),
        (
            axes[1],
            delta,
            st.sequential_cmap(st.SEQUENTIAL_ORANGE),
            "Increase over the random split, same dataset",
            "+.2f",
        ),
    ]
    for ax, frame, cmap, title, fmt in panels:
        values = frame.to_numpy()
        norm = plt.Normalize(vmin=min(0.0, values.min()) if fmt.startswith("+") else values.min(), vmax=values.max())
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.grid(False)
        _annotate(ax, values, norm, cmap, fmt)
        ax.set_title(title, pad=10)
        ax.set_xticks(range(len(st.DATASETS)))
        ax.set_xticklabels(st.DATASETS)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([st.SPLITTER_NAMES[name] for name in order])
        ax.tick_params(length=0, labelcolor=st.INK)
        # 2px surface gaps between cells
        ax.set_xticks(np.arange(-0.5, len(st.DATASETS)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
        ax.grid(which="minor", color=st.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "How far test molecules are from the training set: every splitter on every dataset",
        x=0.008,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.008,
        0.915,
        "Darker = larger distance. Left: the median itself. Right: how much each splitter adds over a random split "
        "of the same dataset,\nwhich is the shift the splitter actually creates. Rows are ordered by the pooled median, "
        "as in the paper.",
        fontsize=9.5,
        color=st.INK_SECONDARY,
        va="top",
        ha="left",
        linespacing=1.45,
    )
    footer(fig)
    return fig


def main() -> None:
    st.apply_style()
    df = load()
    stats = summarise(df)
    order = splitter_order(stats, df)
    sizes = dataset_sizes()

    out_dir = st.OUTPUT_ROOT / FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_out = stats.copy()
    stats_out["dataset"] = pd.Categorical(stats_out["dataset"], st.DATASETS, ordered=True)
    stats_out["split"] = pd.Categorical(stats_out["split"], order, ordered=True)
    stats_out.sort_values(["metric", "dataset", "split"]).round(4).to_csv(out_dir / "summary_stats.csv", index=False)

    figures = {
        "tanimoto": box_figure(df, stats, "tanimoto", order, sizes),
        "tmd": box_figure(df, stats, "tmd", order, sizes),
        "summary": summary_figure(stats, order),
    }
    for name, fig in figures.items():
        for path in st.save(fig, FOLDER, name):
            print(f"wrote {path.relative_to(st.REPO)}")
        plt.close(fig)

    check = stats[(stats["metric"] == "tanimoto")].pivot(index="split", columns="dataset", values="median")
    print(check.loc[order, st.DATASETS].round(3).to_string())


if __name__ == "__main__":
    main()
