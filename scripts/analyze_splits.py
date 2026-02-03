#!/usr/bin/env python3
"""
Split Quality Analysis Script
=============================

A command-line tool for analyzing and comparing the quality of molecular dataset
splits. Computes comprehensive metrics including similarity distributions, scaffold
overlap, and property distributions to help evaluate splitting strategies.

Features:
---------
- Analyze multiple splitting strategies on a dataset
- Compute similarity, scaffold, and property metrics
- Generate visualizations (chemical space, distributions, radar charts)
- Export reports in JSON, CSV, or HTML format

Usage Examples:
---------------
    # Analyze scaffold and kmeans splitters
    python scripts/analyze_splits.py -f data/molecules.csv --splitters scaffold kmeans

    # Analyze all available splitters
    python scripts/analyze_splits.py -f data/molecules.csv --splitters all

    # Generate visualizations and save report
    python scripts/analyze_splits.py -f data/molecules.csv --splitters scaffold kmeans \\
        --visualize --save-report -o results/analysis/

    # Quick analysis with fewer splits
    python scripts/analyze_splits.py -f data/molecules.csv --splitters scaffold --n-splits 3

    # List available splitters
    python scripts/analyze_splits.py --list-splitters

Input File Format:
------------------
CSV file with at minimum a 'smiles' column. Optionally include 'label' column
for balance metrics.

    smiles,label
    CCO,1
    CCCO,0
    ...

Author: ALineMol Team
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Repository path configuration
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_PATH)

from alinemol.splitters import get_splitter_names  # noqa: E402
from alinemol.splitters.analyzer import SplitAnalyzer, SplitQualityReport  # noqa: E402
from alinemol.utils.logger_utils import Logger  # noqa: E402

# Configure logging
logger_instance = Logger(__name__, level=logging.INFO)
logger = logger_instance.get_logger()

# Default values
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2


def list_available_splitters() -> None:
    """Print a formatted list of available splitters."""
    print("\n" + "=" * 60)
    print("Available Splitters for Analysis")
    print("=" * 60 + "\n")

    splitters = get_splitter_names()

    # Group by category
    categories = {
        "Structure-Based": ["scaffold", "scaffold_generic"],
        "Clustering-Based": ["kmeans", "perimeter", "max_dissimilarity", "umap", "butina"],
        "Property-Based": ["molecular_weight", "molecular_weight_reverse", "molecular_logp"],
        "Similarity-Based": ["hi", "lo"],
        "Information Leakage": ["datasail", "datasail_group"],
        "Baseline": ["random", "stratified_random"],
    }

    for category, names in categories.items():
        available = [n for n in names if n in splitters]
        if available:
            print(f"{category}:")
            print("-" * 40)
            for name in available:
                print(f"  {name}")
            print()

    print("=" * 60)
    print("\nUsage: python scripts/analyze_splits.py -f <file> --splitters <name1> <name2>")
    print("       python scripts/analyze_splits.py -f <file> --splitters all")
    print()


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load molecular data from CSV file.

    Args:
        file_path: Path to the input file.

    Returns:
        DataFrame with SMILES and optional labels.
    """
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix in [".tsv", ".txt"]:
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if "smiles" not in df.columns:
        raise ValueError("Input file must have a 'smiles' column")

    logger.info(f"Loaded {len(df)} molecules from {file_path}")
    return df


def get_splitters_to_analyze(splitter_arg: List[str]) -> List[str]:
    """
    Resolve splitter names from command line argument.

    Args:
        splitter_arg: List of splitter names or ["all"].

    Returns:
        List of splitter names to analyze.
    """
    available = get_splitter_names()

    if "all" in splitter_arg:
        # Exclude problematic splitters that may be slow or have dependencies
        exclude = {"datasail", "datasail_group"}  # These may require extra dependencies
        return [s for s in available if s not in exclude]

    # Validate requested splitters
    invalid = set(splitter_arg) - set(available)
    if invalid:
        raise ValueError(f"Unknown splitters: {invalid}. Use --list-splitters to see available options.")

    return splitter_arg


def run_analysis(
    smiles: List[str],
    splitter_names: List[str],
    n_splits: int = DEFAULT_N_SPLITS,
    test_size: float = DEFAULT_TEST_SIZE,
    labels: Optional[List[int]] = None,
    fingerprint_type: str = "ecfp",
) -> Dict[str, Any]:
    """
    Run split quality analysis on the dataset.

    Args:
        smiles: List of SMILES strings.
        splitter_names: List of splitter names to analyze.
        n_splits: Number of splits per splitter.
        test_size: Fraction of data for test set.
        labels: Optional labels for balance metrics.
        fingerprint_type: Fingerprint type for similarity computation.

    Returns:
        Dictionary containing comparison DataFrame and individual reports.
    """
    logger.info(f"Initializing analyzer with {len(smiles)} molecules")
    analyzer = SplitAnalyzer(
        smiles,
        fingerprint_type=fingerprint_type,
        n_jobs=1,  # Use single thread for stability
    )

    all_reports: List[SplitQualityReport] = []
    splitter_summaries: Dict[str, Dict[str, Any]] = {}

    for splitter_name in splitter_names:
        logger.info(f"\nAnalyzing splitter: {splitter_name}")
        logger.info("-" * 40)

        try:
            reports = analyzer.analyze_splitter(
                splitter_name=splitter_name,
                n_splits=n_splits,
                test_size=test_size,
                labels=labels,
            )
            all_reports.extend(reports)

            # Compute summary for this splitter
            summary = analyzer.get_summary_stats(reports)
            splitter_summaries[splitter_name] = {
                "n_splits": len(reports),
                "summary": summary,
            }

            # Print quick summary
            if reports and reports[0].similarity_metrics:
                sim_means = [r.similarity_metrics.mean_sim for r in reports if r.similarity_metrics]
                logger.info(f"  Mean similarity: {sum(sim_means) / len(sim_means):.3f}")

            if reports and reports[0].scaffold_metrics:
                overlaps = [r.scaffold_metrics.scaffold_overlap_percentage for r in reports if r.scaffold_metrics]
                logger.info(f"  Scaffold overlap: {sum(overlaps) / len(overlaps):.1f}%")

        except Exception as e:
            logger.error(f"Failed to analyze {splitter_name}: {e}")
            splitter_summaries[splitter_name] = {"error": str(e)}

    # Create comparison DataFrame
    comparison_df = analyzer.compare_splitters(
        splitter_names=[s for s in splitter_names if "error" not in splitter_summaries.get(s, {})],
        n_splits=n_splits,
        test_size=test_size,
        labels=labels,
        aggregate=True,
    )

    return {
        "comparison_df": comparison_df,
        "reports": all_reports,
        "summaries": splitter_summaries,
        "analyzer": analyzer,
    }


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_format: str = "json",
) -> None:
    """
    Save analysis results to disk.

    Args:
        results: Results dictionary from run_analysis().
        output_dir: Directory to save results.
        save_format: Output format ("json", "csv", or "html").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = results["comparison_df"]
    reports = results["reports"]

    if save_format == "csv":
        # Save comparison DataFrame
        csv_path = output_dir / "comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison to {csv_path}")

        # Save individual reports
        reports_df = pd.DataFrame([r.to_dataframe_row() for r in reports])
        reports_path = output_dir / "all_reports.csv"
        reports_df.to_csv(reports_path, index=False)
        logger.info(f"Saved all reports to {reports_path}")

    elif save_format == "json":
        json_path = output_dir / "analysis_report.json"
        report_data = {
            "comparison": comparison_df.to_dict(orient="records"),
            "reports": [r.to_dict() for r in reports],
            "summaries": results["summaries"],
        }
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        logger.info(f"Saved JSON report to {json_path}")

    elif save_format == "html":
        from alinemol.splitters.analyzer_plots import save_comparison_report

        html_path = output_dir / "analysis_report.html"
        save_comparison_report(comparison_df, reports, str(html_path), format="html")
        logger.info(f"Saved HTML report to {html_path}")


def generate_visualizations(
    results: Dict[str, Any],
    smiles: List[str],
    output_dir: Path,
    splitter_names: List[str],
) -> None:
    """
    Generate visualization plots.

    Args:
        results: Results dictionary from run_analysis().
        smiles: List of SMILES strings.
        output_dir: Directory to save plots.
        splitter_names: List of splitter names.
    """
    try:
        from alinemol.splitters.analyzer_plots import (
            plot_chemical_space,
            plot_similarity_distribution,
            plot_metrics_summary,
            plot_splitter_comparison_radar,
        )
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"Could not import plotting dependencies: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = results["analyzer"]
    reports = results["reports"]
    comparison_df = results["comparison_df"]

    # 1. Chemical space plot for first split of first splitter
    if reports:
        logger.info("Generating chemical space plot...")
        first_report = reports[0]
        splitter = get_splitter(first_report.splitter_name, n_splits=1)
        train_idx, test_idx = next(splitter.split(smiles))

        try:
            fig = plot_chemical_space(analyzer, train_idx, test_idx, method="pca")
            fig.savefig(output_dir / "chemical_space.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved chemical space plot")
        except Exception as e:
            logger.warning(f"Failed to generate chemical space plot: {e}")

    # 2. Similarity distribution
    if reports:
        logger.info("Generating similarity distribution plot...")
        try:
            fig = plot_similarity_distribution(reports)
            fig.savefig(output_dir / "similarity_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved similarity distribution plot")
        except Exception as e:
            logger.warning(f"Failed to generate similarity plot: {e}")

    # 3. Metrics summary for each splitter
    for splitter_name in splitter_names:
        splitter_reports = [r for r in reports if r.splitter_name == splitter_name]
        if splitter_reports:
            try:
                fig = plot_metrics_summary(splitter_reports)
                fig.savefig(
                    output_dir / f"metrics_summary_{splitter_name}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate metrics summary for {splitter_name}: {e}")

    # 4. Radar chart comparison
    if len(comparison_df) > 1:
        logger.info("Generating radar chart comparison...")
        try:
            fig = plot_splitter_comparison_radar(comparison_df)
            fig.savefig(output_dir / "splitter_comparison_radar.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved radar chart")
        except Exception as e:
            logger.warning(f"Failed to generate radar chart: {e}")

    logger.info(f"Visualizations saved to {output_dir}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the analysis to console."""
    comparison_df = results["comparison_df"]
    summaries = results["summaries"]

    print("\n" + "=" * 70)
    print("SPLIT QUALITY ANALYSIS SUMMARY")
    print("=" * 70)

    for splitter_name, summary in summaries.items():
        print(f"\n{splitter_name.upper()}")
        print("-" * 40)

        if "error" in summary:
            print(f"  Error: {summary['error']}")
            continue

        if "summary" in summary:
            stats = summary["summary"]
            if "similarity" in stats:
                sim = stats["similarity"]
                print(f"  Mean Similarity: {sim['mean']:.3f} (+/- {sim['std']:.3f})")

            if "scaffold_overlap" in stats:
                overlap = stats["scaffold_overlap"]
                print(f"  Scaffold Overlap: {overlap['mean']:.1f}% (+/- {overlap['std']:.1f}%)")

    # Print comparison table
    if len(comparison_df) > 0:
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)

        # Select key columns for display
        display_cols = ["splitter_name"]
        for col in comparison_df.columns:
            if any(key in col for key in ["sim_mean_sim_mean", "scaffold_overlap", "size_test_ratio"]):
                display_cols.append(col)

        display_cols = [c for c in display_cols if c in comparison_df.columns]
        print(comparison_df[display_cols].to_string(index=False))

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare molecular dataset split quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_splits.py -f molecules.csv --splitters scaffold kmeans
  python scripts/analyze_splits.py -f molecules.csv --splitters all --visualize
  python scripts/analyze_splits.py --list-splitters
        """,
    )

    parser.add_argument(
        "--list-splitters",
        action="store_true",
        help="List available splitters and exit",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to CSV file with SMILES column",
    )

    parser.add_argument(
        "--splitters",
        nargs="+",
        default=["scaffold", "random"],
        help="Splitters to analyze (or 'all' for all available)",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of splits per splitter (default: {DEFAULT_N_SPLITS})",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Test set fraction (default: {DEFAULT_TEST_SIZE})",
    )

    parser.add_argument(
        "--fingerprint",
        type=str,
        default="ecfp",
        choices=["ecfp", "fcfp", "maccs"],
        help="Fingerprint type for similarity computation (default: ecfp)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save analysis report to disk",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "html"],
        help="Report output format (default: json)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for reports and plots",
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column name for labels (default: 'label')",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle --list-splitters
    if args.list_splitters:
        list_available_splitters()
        return

    # Validate arguments
    if not args.file:
        print("Error: --file is required unless using --list-splitters")
        sys.exit(1)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Load data
    df = load_data(file_path)
    smiles = df["smiles"].tolist()

    # Get labels if available
    labels = None
    if args.label_column in df.columns:
        labels = df[args.label_column].tolist()
        logger.info(f"Using labels from column '{args.label_column}'")

    # Resolve splitters
    splitter_names = get_splitters_to_analyze(args.splitters)
    logger.info(f"Analyzing splitters: {', '.join(splitter_names)}")

    # Run analysis
    results = run_analysis(
        smiles=smiles,
        splitter_names=splitter_names,
        n_splits=args.n_splits,
        test_size=args.test_size,
        labels=labels,
        fingerprint_type=args.fingerprint,
    )

    # Print summary
    print_summary(results)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = file_path.parent / "split_analysis"

    # Save report if requested
    if args.save_report:
        save_results(results, output_dir, args.format)

    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(results, smiles, output_dir / "plots", splitter_names)

    logger.info("Analysis complete!")


# Import get_splitter at module level for visualization
try:
    from alinemol.splitters import get_splitter
except ImportError:
    get_splitter = None


if __name__ == "__main__":
    main()
