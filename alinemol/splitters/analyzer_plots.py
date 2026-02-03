"""
Visualization functions for Split Quality Analyzer.

This module provides plotting functions to visualize split quality metrics,
including chemical space plots, similarity distributions, property distributions,
and radar charts for splitter comparisons.

Example:
    >>> from alinemol.splitters import SplitAnalyzer, get_splitter
    >>> from alinemol.splitters.analyzer_plots import (
    ...     plot_chemical_space,
    ...     plot_similarity_distribution,
    ...     plot_splitter_comparison_radar,
    ... )
    >>>
    >>> analyzer = SplitAnalyzer(smiles)
    >>> train_idx, test_idx = next(get_splitter("scaffold").split(smiles))
    >>>
    >>> # Plot chemical space
    >>> fig = plot_chemical_space(analyzer, train_idx, test_idx)
    >>> fig.savefig("chemical_space.png")
    >>>
    >>> # Compare splitters with radar chart
    >>> comparison_df = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
    >>> fig = plot_splitter_comparison_radar(comparison_df)
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from alinemol.splitters.analyzer import SplitAnalyzer, SplitQualityReport


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def _check_seaborn():
    """Check if seaborn is available."""
    if not HAS_SEABORN:
        raise ImportError("seaborn is required for this plot. Install with: pip install seaborn")


def plot_chemical_space(
    analyzer: SplitAnalyzer,
    train_idx: Union[List[int], np.ndarray],
    test_idx: Union[List[int], np.ndarray],
    method: str = "umap",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    train_color: str = "#1f77b4",
    test_color: str = "#ff7f0e",
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure":
    """
    Plot train/test split in 2D chemical space.

    Reduces molecular fingerprints to 2D using the specified method and
    visualizes the train/test split distribution.

    Args:
        analyzer: SplitAnalyzer instance with precomputed fingerprints.
        train_idx: Indices of training samples.
        test_idx: Indices of test samples.
        method: Dimensionality reduction method ("umap", "tsne", or "pca").
        figsize: Figure size as (width, height).
        alpha: Point transparency.
        train_color: Color for training points.
        test_color: Color for test points.
        title: Plot title. If None, auto-generated.
        ax: Existing axes to plot on. If None, creates new figure.

    Returns:
        matplotlib Figure object.

    Example:
        >>> fig = plot_chemical_space(analyzer, train_idx, test_idx, method="umap")
        >>> fig.savefig("chemical_space.png", dpi=300, bbox_inches="tight")
    """
    _check_matplotlib()

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Get fingerprints
    fps = analyzer.fingerprints

    # Dimensionality reduction
    if method.lower() == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fps) - 1))
    elif method.lower() == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'.")

    embedding = reducer.fit_transform(fps)

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot points
    ax.scatter(
        embedding[train_idx, 0],
        embedding[train_idx, 1],
        c=train_color,
        alpha=alpha,
        label=f"Train (n={len(train_idx)})",
        s=30,
    )
    ax.scatter(
        embedding[test_idx, 0],
        embedding[test_idx, 1],
        c=test_color,
        alpha=alpha,
        label=f"Test (n={len(test_idx)})",
        s=30,
    )

    # Labels and legend
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(loc="upper right")

    if title is None:
        title = f"Chemical Space ({method.upper()})"
    ax.set_title(title)

    return fig


def plot_similarity_distribution(
    reports: Union[SplitQualityReport, List[SplitQualityReport]],
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure":
    """
    Plot the distribution of train-test similarities.

    Shows a box plot or violin plot of the similarity metrics across
    different splits or splitters.

    Args:
        reports: Single report or list of SplitQualityReport objects.
        figsize: Figure size as (width, height).
        colors: List of colors for each splitter.
        ax: Existing axes to plot on.

    Returns:
        matplotlib Figure object.

    Example:
        >>> reports = analyzer.analyze_splitter("scaffold", n_splits=5)
        >>> fig = plot_similarity_distribution(reports)
    """
    _check_matplotlib()
    _check_seaborn()

    if isinstance(reports, SplitQualityReport):
        reports = [reports]

    # Collect data for plotting
    data = []
    for report in reports:
        if report.similarity_metrics is not None:
            sim = report.similarity_metrics
            data.append(
                {
                    "splitter": report.splitter_name,
                    "split_index": report.split_index,
                    "mean_similarity": sim.mean_sim,
                    "min_similarity": sim.min_sim,
                    "max_similarity": sim.max_sim,
                    "p5": sim.percentile_5,
                    "p25": sim.percentile_25,
                    "p75": sim.percentile_75,
                    "p95": sim.percentile_95,
                }
            )

    if not data:
        raise ValueError("No similarity metrics found in reports")

    df = pd.DataFrame(data)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot
    if colors is not None:
        palette = dict(zip(df["splitter"].unique(), colors))
    else:
        palette = None

    sns.boxplot(
        data=df,
        x="splitter",
        y="mean_similarity",
        palette=palette,
        ax=ax,
    )

    ax.set_xlabel("Splitter")
    ax.set_ylabel("Mean Train-Test Similarity")
    ax.set_title("Similarity Distribution by Splitter")

    # Add reference line at 0.5 (moderate similarity)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Moderate similarity")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_property_distributions(
    analyzer: SplitAnalyzer,
    train_idx: Union[List[int], np.ndarray],
    test_idx: Union[List[int], np.ndarray],
    properties: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> "plt.Figure":
    """
    Plot property distributions for train and test sets.

    Creates histograms comparing the distribution of molecular properties
    between training and test sets.

    Args:
        analyzer: SplitAnalyzer instance.
        train_idx: Indices of training samples.
        test_idx: Indices of test samples.
        properties: List of properties to plot. If None, uses all computed properties.
        figsize: Figure size. If None, auto-calculated based on number of properties.

    Returns:
        matplotlib Figure object.

    Example:
        >>> fig = plot_property_distributions(analyzer, train_idx, test_idx,
        ...                                   properties=["MW", "LogP"])
    """
    _check_matplotlib()

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    props_df = analyzer.properties

    if properties is None:
        properties = list(props_df.columns)

    n_props = len(properties)
    if figsize is None:
        ncols = min(3, n_props)
        nrows = (n_props + ncols - 1) // ncols
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(
        nrows=(n_props + 2) // 3,
        ncols=min(3, n_props),
        figsize=figsize,
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, prop_name in enumerate(properties):
        if prop_name not in props_df.columns:
            continue

        ax = axes[idx]

        train_values = props_df.iloc[train_idx][prop_name].dropna()
        test_values = props_df.iloc[test_idx][prop_name].dropna()

        # Plot histograms
        bins = np.histogram_bin_edges(np.concatenate([train_values, test_values]), bins=30)

        ax.hist(
            train_values,
            bins=bins,
            alpha=0.5,
            label=f"Train (n={len(train_values)})",
            color="#1f77b4",
            density=True,
        )
        ax.hist(
            test_values,
            bins=bins,
            alpha=0.5,
            label=f"Test (n={len(test_values)})",
            color="#ff7f0e",
            density=True,
        )

        ax.set_xlabel(prop_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(prop_name)

    # Hide unused axes
    for idx in range(len(properties), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_splitter_comparison_radar(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    colors: Optional[List[str]] = None,
) -> "plt.Figure":
    """
    Create a radar chart comparing splitters across multiple metrics.

    Args:
        comparison_df: DataFrame from analyzer.compare_splitters() with aggregate=True.
        metrics: List of metric columns to include. If None, auto-selects key metrics.
        figsize: Figure size as (width, height).
        colors: List of colors for each splitter.

    Returns:
        matplotlib Figure object.

    Example:
        >>> comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
        >>> fig = plot_splitter_comparison_radar(comparison)
    """
    _check_matplotlib()

    if "splitter_name" not in comparison_df.columns:
        raise ValueError("comparison_df must have 'splitter_name' column")

    # Select metrics for radar chart
    if metrics is None:
        # Auto-select key metrics (look for mean columns)
        candidate_metrics = [
            "sim_mean_sim_mean",
            "scaffold_scaffold_overlap_percentage_mean",
            "size_test_ratio_mean",
        ]
        # Add property metrics if available
        for col in comparison_df.columns:
            if "prop_" in col and "_ks_stat_mean" in col:
                candidate_metrics.append(col)

        metrics = [m for m in candidate_metrics if m in comparison_df.columns]

    if not metrics:
        raise ValueError("No valid metrics found in comparison_df")

    # Get splitter names
    splitters = comparison_df["splitter_name"].tolist()
    n_splitters = len(splitters)

    # Prepare data - normalize to [0, 1]
    values_dict = {}
    for metric in metrics:
        col_values = comparison_df[metric].values
        min_val = col_values.min()
        max_val = col_values.max()
        if max_val > min_val:
            normalized = (col_values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(col_values) * 0.5
        values_dict[metric] = normalized

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Default colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_splitters))

    for idx, splitter in enumerate(splitters):
        values = [values_dict[m][idx] for m in metrics]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, "o-", linewidth=2, label=splitter, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    # Format labels
    labels = [m.replace("_mean", "").replace("_", " ").title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)

    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Splitter Comparison", size=14, y=1.1)

    plt.tight_layout()
    return fig


def plot_scaffold_overlap_heatmap(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "YlOrRd",
) -> "plt.Figure":
    """
    Create a heatmap showing scaffold overlap across splitters.

    Args:
        comparison_df: DataFrame from analyzer.compare_splitters().
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    _check_matplotlib()
    _check_seaborn()

    if "scaffold_scaffold_overlap_percentage_mean" not in comparison_df.columns:
        raise ValueError("Scaffold overlap data not found in comparison_df")

    fig, ax = plt.subplots(figsize=figsize)

    # Create single-row heatmap
    data = comparison_df[["splitter_name", "scaffold_scaffold_overlap_percentage_mean"]].copy()
    data = data.set_index("splitter_name").T

    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "Scaffold Overlap (%)"},
    )

    ax.set_ylabel("")
    ax.set_title("Scaffold Overlap by Splitter")

    plt.tight_layout()
    return fig


def plot_metrics_summary(
    reports: List[SplitQualityReport],
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """
    Create a summary plot of key metrics from multiple reports.

    Args:
        reports: List of SplitQualityReport objects.
        figsize: Figure size.

    Returns:
        matplotlib Figure object with multiple subplots.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Train/Test sizes
    ax = axes[0, 0]
    train_sizes = [r.size_metrics.train_size for r in reports]
    test_sizes = [r.size_metrics.test_size for r in reports]
    x = range(len(reports))
    width = 0.35
    ax.bar([i - width / 2 for i in x], train_sizes, width, label="Train", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], test_sizes, width, label="Test", color="#ff7f0e")
    ax.set_xlabel("Split Index")
    ax.set_ylabel("Sample Count")
    ax.set_title("Train/Test Sizes")
    ax.legend()

    # 2. Mean Similarity
    ax = axes[0, 1]
    sim_means = [r.similarity_metrics.mean_sim if r.similarity_metrics else 0 for r in reports]
    sim_stds = [r.similarity_metrics.std_sim if r.similarity_metrics else 0 for r in reports]
    ax.bar(x, sim_means, yerr=sim_stds, capsize=3, color="#2ca02c")
    ax.set_xlabel("Split Index")
    ax.set_ylabel("Mean Similarity")
    ax.set_title("Train-Test Similarity")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # 3. Scaffold Overlap
    ax = axes[1, 0]
    overlaps = [r.scaffold_metrics.scaffold_overlap_percentage if r.scaffold_metrics else 0 for r in reports]
    ax.bar(x, overlaps, color="#d62728")
    ax.set_xlabel("Split Index")
    ax.set_ylabel("Scaffold Overlap (%)")
    ax.set_title("Scaffold Overlap")

    # 4. Label Balance (if available)
    ax = axes[1, 1]
    balance_diffs = [
        r.size_metrics.label_balance_diff for r in reports if r.size_metrics.label_balance_diff is not None
    ]
    if balance_diffs:
        ax.bar(range(len(balance_diffs)), balance_diffs, color="#9467bd")
        ax.set_xlabel("Split Index")
        ax.set_ylabel("Label Balance Diff")
        ax.set_title("Label Distribution Shift")
    else:
        ax.text(
            0.5,
            0.5,
            "No label data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Label Distribution Shift")

    plt.tight_layout()
    return fig


def save_comparison_report(
    comparison_df: pd.DataFrame,
    reports: List[SplitQualityReport],
    output_path: str,
    format: str = "html",
) -> None:
    """
    Save a comprehensive comparison report.

    Args:
        comparison_df: DataFrame from analyzer.compare_splitters().
        reports: List of all SplitQualityReport objects.
        output_path: Path to save the report.
        format: Output format ("html", "csv", or "json").

    Example:
        >>> save_comparison_report(comparison_df, reports, "report.html", format="html")
    """
    import json

    if format.lower() == "csv":
        comparison_df.to_csv(output_path, index=False)
    elif format.lower() == "json":
        report_dicts = [r.to_dict() for r in reports]
        with open(output_path, "w") as f:
            json.dump(
                {"comparison": comparison_df.to_dict(orient="records"), "reports": report_dicts},
                f,
                indent=2,
            )
    elif format.lower() == "html":
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Split Quality Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Split Quality Analysis Report</h1>
    <p>Generated for {len(reports)} splits across {comparison_df["splitter_name"].nunique() if "splitter_name" in comparison_df.columns else "N/A"} splitters</p>

    <h2>Comparison Summary</h2>
    {comparison_df.to_html(index=False)}

    <h2>Individual Split Reports</h2>
"""
        for report in reports[:10]:  # Limit to first 10 for readability
            html_content += f"""
    <div class="metric-card">
        <h3>{report.splitter_name} - Split {report.split_index}</h3>
        <p><strong>Train Size:</strong> {report.size_metrics.train_size} |
           <strong>Test Size:</strong> {report.size_metrics.test_size}</p>
"""
            if report.similarity_metrics:
                html_content += f"""
        <p><strong>Mean Similarity:</strong> {report.similarity_metrics.mean_sim:.3f} |
           <strong>Min:</strong> {report.similarity_metrics.min_sim:.3f} |
           <strong>Max:</strong> {report.similarity_metrics.max_sim:.3f}</p>
"""
            if report.scaffold_metrics:
                html_content += f"""
        <p><strong>Scaffold Overlap:</strong> {report.scaffold_metrics.scaffold_overlap_percentage:.1f}%</p>
"""
            html_content += "    </div>\n"

        html_content += """
</body>
</html>
"""
        with open(output_path, "w") as f:
            f.write(html_content)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'html', 'csv', or 'json'.")
