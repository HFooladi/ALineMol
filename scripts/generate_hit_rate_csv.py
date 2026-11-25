#!/usr/bin/env python3
"""
Script to generate hit_rate.csv with ROC-AUC and top-100 hit rate metrics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict
import sys

# Add the alinemol package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "alinemol"))
from alinemol.utils.frozen import assert_not_frozen
from alinemol.utils.metric_utils import calculate_top100_hit_rate


def load_config() -> Dict:
    """Load configuration from datasets/config.yml"""
    config_path = Path("datasets/config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_roc_auc_from_results(results_df: pd.DataFrame, dataset: str, splitter: str, model: str) -> float:
    """Extract ROC-AUC value from the results.csv file"""
    # Filter the results for the specific combination
    mask = (results_df["dataset"] == dataset) & (results_df["split"] == splitter) & (results_df["model"] == model)

    filtered_df = results_df[mask]

    if len(filtered_df) == 0:
        print(f"Warning: No results found for {dataset}/{splitter}/{model}")
        return np.nan

    # Get the OOD test ROC-AUC (assuming this is what we want)
    # The results seem to have multiple rows per model, so we'll take the mean
    roc_auc_values = filtered_df["OOD_test_roc_auc"].values

    if len(roc_auc_values) == 0:
        print(f"Warning: No ROC-AUC values found for {dataset}/{splitter}/{model}")
        return np.nan

    # Return the mean ROC-AUC across all folds
    return float(np.mean(roc_auc_values))


def calculate_hit_rate_for_fold(dataset: str, splitter: str, model: str, fold: int) -> float:
    """Calculate top-100 hit rate for a specific fold"""
    try:
        # Path to prediction file
        pred_path = Path(f"classification_results/TDC/{dataset}/{splitter}/{model}/{fold}/prediction.csv")

        # Path to ground truth file
        gt_path = Path(f"datasets/TDC/{dataset}/split/{splitter}/test_{fold - 1}.csv")

        if not pred_path.exists():
            print(f"Warning: Prediction file not found: {pred_path}")
            return np.nan

        if not gt_path.exists():
            print(f"Warning: Ground truth file not found: {gt_path}")
            return np.nan

        # Load prediction data
        pred_df = pd.read_csv(pred_path)

        # Load ground truth data
        gt_df = pd.read_csv(gt_path)

        # Check if we have the expected columns
        if "canonical_smiles" not in pred_df.columns or "label" not in pred_df.columns:
            print(f"Warning: Unexpected columns in prediction file: {pred_df.columns.tolist()}")
            return np.nan

        if "smiles" not in gt_df.columns or "label" not in gt_df.columns:
            print(f"Warning: Unexpected columns in ground truth file: {gt_df.columns.tolist()}")
            return np.nan

        pred_df = pred_df.rename(columns={"label": "prediction"})

        # Calculate hit rate
        results = calculate_top100_hit_rate(
            predictions_df=pred_df,
            labels_df=gt_df,
            smiles_col_pred="canonical_smiles",
            smiles_col_label="smiles",
            prediction_col="prediction",  # This contains the predicted probabilities
            label_col="label",  # This contains the true labels
            top_k=100,
        )

        return results["hit_rate"]

    except Exception as e:
        print(f"Error calculating hit rate for {dataset}/{splitter}/{model}/{fold}: {e}")
        return np.nan


def main():
    """Main function to generate hit_rate.csv"""
    print("Loading configuration...")
    config = load_config()

    # Load the existing results.csv
    results_path = Path("classification_results/TDC/results.csv")
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    print("Loading existing results...")
    results_df = pd.read_csv(results_path)

    # Get datasets, splitters, and models from config
    datasets = config["datasets"]["TDC"]
    splitters = config["splitting"]
    models = []

    # Add GNN models
    for gnn_type, gnn_models in config["models"]["GNN"].items():
        models.extend(gnn_models)

    # Add ML models
    models.extend(config["models"]["ML"])

    print(f"Processing {len(datasets)} datasets, {len(splitters)} splitters, {len(models)} models")

    # Initialize results list
    results_list = []

    # Process each combination
    total_combinations = len(datasets) * len(splitters) * len(models)
    current = 0

    for dataset in datasets:
        for splitter in splitters:
            for model in models:
                current += 1
                print(f"Processing {current}/{total_combinations}: {dataset}/{splitter}/{model}")

                # Get ROC-AUC from existing results
                roc_auc = get_roc_auc_from_results(results_df, dataset, splitter, model)

                # Calculate hit rates for each fold separately
                for fold in range(1, 11):  # Assuming 10 folds (fold 1-10)
                    hit_rate = calculate_hit_rate_for_fold(dataset, splitter, model, fold)

                    # Add to results for each fold
                    results_list.append(
                        {
                            "dataset": dataset,
                            "splitter": splitter,
                            "model": model,
                            "fold": fold,
                            "roc_auc": roc_auc,
                            "hit_rate": hit_rate if not np.isnan(hit_rate) else np.nan,
                        }
                    )

    # Create DataFrame and save
    results_df_new = pd.DataFrame(results_list)

    # Save to CSV
    output_path = Path("classification_results/TDC/hit_rate.csv")
    assert_not_frozen(output_path, what="hit-rate table")
    results_df_new.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    print(f"Total combinations processed: {len(results_df_new)}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total rows (dataset/splitter/model/fold combinations): {len(results_df_new)}")
    print(f"Rows with valid ROC-AUC: {results_df_new['roc_auc'].notna().sum()}")
    print(f"Rows with valid hit rate: {results_df_new['hit_rate'].notna().sum()}")
    print(f"Rows with both metrics: {(results_df_new['roc_auc'].notna() & results_df_new['hit_rate'].notna()).sum()}")
    print(
        f"Unique dataset/splitter/model combinations: {len(results_df_new.groupby(['dataset', 'splitter', 'model']))}"
    )


if __name__ == "__main__":
    main()
