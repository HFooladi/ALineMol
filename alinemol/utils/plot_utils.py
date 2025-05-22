"""
Utility functions for visualization
"""

# Import base packages
import json  # for JSON manipulation
import os  # for file manipulation
from typing import List, Optional  # for type hints

import datamol as dm  # for moelcule processing
import matplotlib  # for plotting
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation
import seaborn as sns  # for plotting
import umap  # for dimensionality reduction
import yaml  # for configuration
from sklearn.calibration import calibration_curve  # for calibration curve
from sklearn.decomposition import PCA  # for PCA
from sklearn.manifold import TSNE  # for t-SNE

# Path to the repository and datasets
REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(REPO_PATH, "datasets")


# Set the plotting style
# Set matplotlib parameters
rcparams = {
    # LaTeX setup
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    # Font settings
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    # Figure settings
    "figure.dpi": 300,  # Higher DPI for better quality
    "figure.figsize": [6.4, 4.8],  # Default figure size
    "figure.constrained_layout.use": True,  # Better layout handling
    # Axes settings
    "axes.linewidth": 1.0,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    # Legend settings
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.loc": "upper right",
    # Tick settings
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
matplotlib.rcParams.update(rcparams)

# Seaborn settings
sns.set_style("whitegrid", rc=rcparams)
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.5)

# Load the configuration file (wich contains datasets, models, and splitting)
CFG = yaml.safe_load(open(os.path.join(DATASET_PATH, "config.yml"), "r"))

ML_MODELS: List = CFG["models"]["ML"]
GNN_MODELS: List = CFG["models"]["GNN"]["scratch"]
PRETRAINED_GNN_MODELS: List = CFG["models"]["GNN"]["pretrained"]
ALL_MODELS: List = [ML_MODELS, GNN_MODELS, PRETRAINED_GNN_MODELS]
DATASET_NAMES: List = CFG["datasets"]["TDC"]
SPLIT_TYPES: List = CFG["splitting"]

metric_mapping = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
spitting_mapping = {
    "random": "Random",
    "scaffold": "Scaffold",
    "scaffold_generic": "Scaffold generic",
    "molecular_weight": "Molecular weight",
    "molecular_logp": "Molecular logp",
    "molecular_weight_reverse": "Molecular weight reverse",
    "kmeans": "K-means",
    "max_dissimilarity": "Max dissimilarity",
}


def reduce_dimensionality(data: np.ndarray, method="pca"):
    """
    Reduce the dimensionality of the data using PCA, t-SNE, or UMAP

    Args:
        data (np.ndarray): data to reduce (N * D)
        method (str): method to use for dimensionality reduction
            options: `pca`, `tsne`, `umap`

    Returns:
        np.ndarray: reduced data (N * 2)

    Raises:
        ValueError: if method is not 'pca', 'tsne', or 'umap'
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(random_state=42)
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'.")
    return reducer.fit_transform(data)


def plot_ID_OOD(
    ID_test_score: List,
    OOD_test_score: List,
    threshold: float = 0.0,
    dataset_category: str = "MoleculeNet",
    dataset_name: str = "HIV",
    metric: str = "ROC-AUC",
    save: bool = False,
):
    """
    Plot ID vs OOD test ROC-AUC scores

    Args:
        ID_test_score: list of ID test scores
        OOD_test_score: list of OOD test scores
        dataset_category (str): category of dataset
        dataset_name (str): name of dataset
        metric (str): name of metric
            options: "ROC-AUC", "PR-AUC", "Accuracy"
        save (bool): whether to save plot

    Returns:
        None
    """
    chosen_index = np.where(np.array(ID_test_score) > threshold)
    ID_test_score = np.array(ID_test_score)[chosen_index]
    OOD_test_score = np.array(OOD_test_score)[chosen_index]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(ID_test_score, OOD_test_score, s=40, linewidth=1)
    ax.axline((0.6, 0.6), slope=1, linestyle="--")
    ax.axline((0.5, 0.6), (1, 1.1), linestyle="--")
    ax.axline((0.5, 0.4), (1, 0.9), linestyle="--")

    ax.set_title(f"{dataset_name} Dataset ({dataset_category})")
    ax.set_xlabel(f"ID Test {metric}")
    ax.set_ylabel(f"OOD Test {metric}")

    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)

    ax.grid(False)

    if save:
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"{dataset_name}_{metric}_ID_OOD.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
    plt.show()


def plot_ID_OOD_sns(data: pd.DataFrame, dataset_category="TDC", dataset_name="CYP2C19", save: bool = False):
    """
    Plot ID vs OOD test ROC-AUC scores using seaborn

    Args:
        data (pd.DataFrame): DataFrame with the following columns:
            "model", "ID_test_accuracy", "OOD_test_accuracy", "ID_test_roc_auc", "OOD_test_roc_auc", "ID_test_pr_auc", "OOD_test_pr_auc"
        dataset_category (str): category of dataset
        dataset_name (str): name of dataset
        save (bool): whether to save plot

    Returns:
        None
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))
    sns.set_theme(
        style="whitegrid",
        rc={
            "text.usetex": True,
            "pgf.rcfonts": False,
            "font.serif": "Computer Modern Roman",
            "font.family": "serif",
        },
    )

    fig.suptitle(f"{dataset_name} Dataset ({dataset_category})", fontsize=26, y=0.95)
    sns.scatterplot(
        data=data,
        x="ID_test_accuracy",
        y="OOD_test_accuracy",
        hue="model",
        palette="plasma",
        s=40,
        ax=ax[0],
    )
    ax[0].axline((0.5, 0.5), slope=1, linestyle="--")

    ax[0].set_xlabel("ID Test Accuracy", fontsize=16)
    ax[0].set_ylabel("OOD Test Accuracy", fontsize=16)

    sns.scatterplot(
        data=data,
        x="ID_test_roc_auc",
        y="OOD_test_roc_auc",
        hue="model",
        palette="plasma",
        s=40,
        ax=ax[1],
    )
    ax[1].axline((0.5, 0.5), slope=1, linestyle="--")

    ax[1].set_xlabel("ID ROC-AUC", fontsize=16)
    ax[1].set_ylabel("OOD ROC-AUC", fontsize=16)

    sns.scatterplot(
        data=data,
        x="ID_test_pr_auc",
        y="OOD_test_pr_auc",
        hue="model",
        palette="plasma",
        s=40,
        ax=ax[2],
    )
    ax[2].axline((0.5, 0.5), slope=1, linestyle="--")

    ax[2].set_xlabel("ID PR-AUC", fontsize=16)
    ax[2].set_ylabel("OOD PR-AUC", fontsize=16)

    if save:
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"ID_vs_OOD_{dataset_category}_{dataset_name}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
    plt.show()


def plot_ID_OOD_bar(data: pd.DataFrame, metrics=["accuracy", "roc_auc", "pr_auc"], save: bool = False) -> None:
    """
    Plot ID vs OOD scores bar plot

    Args:
        data (pd.DataFrame): DataFrame with the following columns:
            "model", "ID_test_accuracy", "OOD_test_accuracy", "ID_test_roc_auc", "OOD_test_roc_auc", "ID_test_pr_auc", "OOD_test_pr_auc"
        save (bool): whether to save plot

    Returns:
        None
    """
    for metric in metrics:
        tidy = data.melt(
            id_vars=["model"],
            value_vars=[f"ID_test_{metric}", f"OOD_test_{metric}"],
            var_name="split",
            value_name=metric,
        )
        fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
        sns.barplot(y="model", x=metric, hue="split", data=tidy, ax=ax, orient="h")
        ax.grid(True, axis="x", linestyle="--")
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(metric, fontsize=20)
        ax.set_ylabel("Model", fontsize=20)
        ax.set_xlim(0.5, 1.0)
        ax.set_xticks(np.arange(0.5, 1.0, 0.05))
        ax.set_title(f"{metric} comparison between ID and OOD", fontsize=20)
        if save:
            fig.savefig(
                os.path.join(REPO_PATH, "assets", "dummy.pdf"),
                bbox_inches="tight",
                backend="pgf",
            )
        plt.show()


def visualize_chemspace(data: pd.DataFrame, split_names: List[str], mol_col: str = "smiles", size_col=None, size=10):
    """
    Visualize chemical space using UMAP

    Args:
        data (pd.DataFrame): pd.DataFrame with columns "smiles", "label", "split"
        split_names (list): list of split names
        mol_col (str): name of column containing SMILES
        size_col: name of column containing size information

    Returns:
        None
    """
    assert mol_col in data.columns, f"{mol_col} not found in data"
    figs = plt.figure(num=3)
    features = [dm.to_fp(mol) for mol in data[mol_col]]
    embedding = umap.UMAP().fit_transform(features)
    data["UMAP_0"], data["UMAP_1"] = embedding[:, 0], embedding[:, 1]
    for split_name in split_names:
        plt.figure(figsize=(12, 8))
        fig = sns.scatterplot(data=data, x="UMAP_0", y="UMAP_1", s=size, style=size_col, hue=split_name, alpha=0.7)
        fig.set_title(f"UMAP Embedding of compounds for {split_name} split")
        fig.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.show()
    return figs


def plot_ml_gnn_comparisson(dataset: str = "CYP2C19", split_type: str = "scaffold") -> None:
    """
    Plot difference between ML models, GNN models and pretrained GNN models

    Args:
        dataset (str): name of dataset
        split_type (str): type of split

    Returns:
        None
    """

    diff = []
    for models in ALL_MODELS:
        results = pd.read_csv(os.path.join("classification_results", "TDC", dataset, split_type, "results.csv"))
        results = results[results["model"].isin(models)]
        metrics = ["accuracy", "roc_auc", "pr_auc"]
        diff_models = []
        for metric in metrics:
            results[f"diff_{metric}"] = results[f"ID_test_{metric}"] - results[f"OOD_test_{metric}"]
            diff_models.append(results.groupby("model")[f"diff_{metric}"].mean())
        diff_models = pd.concat(diff_models, axis=1)
        diff.append(diff_models)
    mean_df = pd.DataFrame(
        [diff[0].mean(axis=0), diff[1].mean(axis=0), diff[2].mean(axis=0)],
        index=["ML_MODELS", "GNN_MODELS", "PRETRAINED_GNN_MODELS"],
    )
    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
    mean_df.plot(kind="bar", ax=ax)
    ax.set_xlabel("Model", fontsize=20)
    ax.set_ylabel("Difference", fontsize=20)
    ax.grid(True, axis="y", linestyle="--")
    plt.show()


def calibration_plot(dataset_category="TDC", dataset_names="CYP2C19", split_type="scaffold", model_name="GCN"):
    """
    Plot calibration curve for each split
    This function plots the calibration curve for each split of the dataset

    Args:
        dataset_category (str): category of dataset
        dataset_names (str): name of dataset
        split_type (str): type of split
        model_name (str): name of model

    Returns:
        None
    """

    SPLIT_PATH = os.path.join(DATASET_PATH, dataset_category, dataset_names, "split")
    # Load your predictions and labels
    indices = 10
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(indices):
        df = pd.read_csv(
            os.path.join(
                "classification_results",
                dataset_category,
                dataset_names,
                split_type,
                model_name,
                str(i + 1),
                "prediction.csv",
            )
        )
        y_pred_prob = df["label"].values
        df1 = pd.read_csv(os.path.join(SPLIT_PATH, split_type, f"test_{i}.csv"))
        y_true = df1["label"].values

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)

        # Plot calibration curve
        ax[i // 5, i % 5].plot(prob_pred, prob_true, marker="o", label=model_name)
        ax[i // 5, i % 5].plot([0, 1], [0, 1], linestyle="--", color="black")
        ax[i // 5, i % 5].set_title(f"Calibration plot for split {i + 1}")
        ax[i // 5, i % 5].set_xlabel("Predicted probability")
        ax[i // 5, i % 5].set_ylabel("True probability")
        ax[i // 5, i % 5].legend()
    plt.tight_layout()
    plt.show()


def heatmap_plot(results: pd.DataFrame = None, metric: str = "roc_auc", perc=False, save: bool = False) -> None:
    """
    We want to have a heatmap with one axis datasets and one axis splits. The values in the heatmap are the difference between ID and OOD
    for each dataset and split (averagd over all models and repeats). We will use the results.csv file to get the values.

    Args:
        results (pd.DataFrame): DataFrame with the results
        metric (str): name of metric
        perc (bool): whether to plot the percentage difference or absolute difference
        save (bool): whether to save plot

    Returns:
        None

    Options:
        metric: "accuracy", "roc_auc", "pr_auc"
    """

    if results is None:
        results = pd.read_csv(os.path.join("classification_results", "TDC", "results.csv"))  # read the results

    assert metric in ["accuracy", "roc_auc", "pr_auc"], "Invalid metric"  # check if the metric is valid
    assert "dataset" in results.columns, (
        "dataset column not found in results"
    )  # check if the dataset column is in the results
    assert "split" in results.columns, (
        "split column not found in results"
    )  # check if the split column is in the results

    dataset_names = results["dataset"].unique()  # get the unique dataset names
    # split_types = results["split"].unique()  # get the unique split types
    # reorder the split types to this order
    split_types = SPLIT_TYPES
    # create a dataframe to store the difference between ID and OOD for each dataset and split
    df = pd.DataFrame(index=dataset_names, columns=split_types)

    # fill the dataframe with the difference between ID and OOD for each dataset and split
    for dataset in dataset_names:
        for split in split_types:
            num = results[(results["dataset"] == dataset) & (results["split"] == split)][f"ID_test_{metric}"].mean()
            den = results[(results["dataset"] == dataset) & (results["split"] == split)][f"OOD_test_{metric}"].mean()
            if perc:
                df.loc[dataset, split] = (num - den) / num * 100
            else:
                df.loc[dataset, split] = num - den

    df = df.astype(float)
    # plot the heatmap
    vmin, vmax = 0.0, 0.2
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    a = sns.heatmap(
        df, ax=ax, cmap="coolwarm", annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cbar_kws={"label": f"{metric}"}
    )
    plt.xlabel("", fontsize=24)
    plt.ylabel("Dataset", fontsize=24)
    plt.title(f"Difference between ID and OOD {metric_mapping[metric]}", fontsize=24)
    a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=18)
    a.set_yticklabels(a.get_yticklabels(), rotation=0, horizontalalignment="right", fontsize=18)
    ax.set_xticklabels([spitting_mapping[split] for split in SPLIT_TYPES])

    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def heatmap_plot_id_ood(results: pd.DataFrame = None, metric: str = "roc_auc", perc=False, save: bool = False) -> None:
    """
    We want to have a heatmap (one ID and one OOD) with one axis datasets and one axis splits. The values in the heatmap are the ID and OOD
    for each dataset and split (averagd over all models and repeats). We will use the results.csv file to get the values.

    Args:
        results (pd.DataFrame): DataFrame with the results
        metric (str): name of metric
        perc (bool): whether to plot the percentage difference or absolute difference
        save (bool): whether to save plot

    Returns:
        None

    Options:
        metric: "accuracy", "roc_auc", "pr_auc"
    """
    metric_mapping = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    dataset_names = results["dataset"].unique()  # get the unique dataset names
    # split_types = results["split"].unique()  # get the unique split types
    # reorder the split types to this order
    split_types = SPLIT_TYPES
    # create a dataframe to store the difference between ID and OOD for each dataset and split
    id_df = pd.DataFrame(index=dataset_names, columns=split_types)
    ood_df = pd.DataFrame(index=dataset_names, columns=split_types)

    # fill the dataframe with the difference between ID and OOD for each dataset and split
    for dataset in dataset_names:
        for split in split_types:
            id = results[(results["dataset"] == dataset) & (results["split"] == split)][f"ID_test_{metric}"].mean()
            ood = results[(results["dataset"] == dataset) & (results["split"] == split)][f"OOD_test_{metric}"].mean()
            id_df.loc[dataset, split] = id
            ood_df.loc[dataset, split] = ood

    id_df = id_df.astype(float)
    ood_df = ood_df.astype(float)

    # plot the heatmap
    vmin, vmax = 0.5, 1.0
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # one plot for id and one plot for ood
    sns.heatmap(
        id_df, ax=ax[0], annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": f"{metric}"}, vmin=vmin, vmax=vmax
    )
    ax[0].set_title(f"ID {metric_mapping[metric]}", fontsize=18)
    ax[0].set_xlabel("", fontsize=18)
    ax[0].set_ylabel("Dataset", fontsize=18)
    ax[0].tick_params(axis="both", which="major", labelsize=14)
    ax[0].set_xticklabels([spitting_mapping[split] for split in SPLIT_TYPES])

    sns.heatmap(
        ood_df, ax=ax[1], annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": f"{metric}"}, vmin=vmin, vmax=vmax
    )
    ax[1].set_title(f"OOD {metric_mapping[metric]}", fontsize=18)
    ax[1].set_xlabel("", fontsize=18)
    ax[1].set_ylabel("Dataset", fontsize=18)
    ax[1].tick_params(axis="both", which="major", labelsize=14)
    ax[1].set_xticklabels([spitting_mapping[split] for split in SPLIT_TYPES])

    # plt.tight_layout()
    if save:
        # save as pdf
        fig.savefig(
            os.path.join("assets", f"heatmap_id_ood_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join("assets", f"heatmap_id_ood_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def heatmap_plot_dataset_fixed(
    results: pd.DataFrame = None, dataset="CYP2C19", metric: str = "roc_auc", perc=False, save: bool = False
) -> None:
    """
    We want to have a heatmap with one axis datasets and one axis splits. The values in the heatmap are the difference between ID and OOD
    for each dataset and split (averagd over all models and repeats). We will use the results.csv file to get the values.

    Args:
        results (pd.DataFrame): DataFrame with the results
        metric (str): name of metric
        perc (bool): whether to plot the percentage difference or absolute difference
        save (bool): whether to save plot

    Returns:
        None

    Options:
        metric: "accuracy", "roc_auc", "pr_auc"
    """
    if results is None:
        results = pd.read_csv(os.path.join("classification_results", "TDC", "results.csv"))  # read the results

    assert metric in ["accuracy", "roc_auc", "pr_auc"], "Invalid metric"  # check if the metric is valid
    assert "dataset" in results.columns, (
        "dataset column not found in results"
    )  # check if the dataset column is in the results
    assert "split" in results.columns, (
        "split column not found in results"
    )  # check if the split column is in the results
    assert "model" in results.columns, (
        "model column not found in results"
    )  # check if the model column is in the results

    # Just extract the dataset of interest
    results = results[results["dataset"] == dataset]

    split_types = results["split"].unique()  # get the unique split types
    models = results["model"].unique()  # get the unique models

    # create a dataframe to store the difference between ID and OOD for each model and split
    df = pd.DataFrame(index=models, columns=split_types)

    for model in models:
        for split in split_types:
            num = results[(results["model"] == model) & (results["split"] == split)][f"ID_test_{metric}"].mean()
            den = results[(results["model"] == model) & (results["split"] == split)][f"OOD_test_{metric}"].mean()
            if perc:
                df.loc[model, split] = (num - den) / num * 100
            else:
                df.loc[model, split] = num - den

    df = df.astype(float)
    # plot the heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.heatmap(df, ax=ax, cmap="coolwarm", annot=True, fmt=".2f")
    plt.xlabel("Split", fontsize=20)
    plt.ylabel("Model", fontsize=20)
    plt.title(f"Difference between ID and OOD {metric} for dataset {dataset}", fontsize=20)
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{dataset}_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{dataset}_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def heatmap_plot_all_dataset(
    results: pd.DataFrame = None, metric: str = "roc_auc", report="diff", perc=False, save: bool = False
) -> None:
    """
    We want to have a heatmap with one axis datasets and one axis splits for all the datassets. The values in the heatmap are the difference between ID and OOD
    for each dataset and split (averagd over all models and repeats). We will use the results.csv file to get the values.

    Args:
        results (pd.DataFrame): DataFrame with the results
        metric (str): name of metric
        perc (bool): whether to plot the percentage difference or absolute difference
        report (str): whether to report the difference or the ID and OOD values
        save (bool): whether to save plot

    Returns:
        None

    Options:
        metric: "accuracy", "roc_auc", "pr_auc"
        report: "diff", "ID", OOD"
    """
    if results is None:
        results = pd.read_csv(os.path.join("classification_results", "TDC", "results.csv"))  # read the results
    dataset_names = results["dataset"].unique()  # get the unique dataset names
    # split_types = results["split"].unique()  # get the unique split types
    split_types = SPLIT_TYPES
    models = results["model"].unique()  # get the unique models
    vmin, vmax = 0.5, 1.0
    # We want subplots for fixing each time one dataset, then plot the heatmap od difference between ID and OOd for all the modles and split types with
    # the same dataset
    fig, ax = plt.subplots(4, 2, figsize=(16, 16))
    for i, dataset in enumerate(dataset_names):
        result_subset = results[results["dataset"] == dataset]
        df = pd.DataFrame(index=models, columns=split_types)

        for model in models:
            for split in split_types:
                num = result_subset[(result_subset["model"] == model) & (result_subset["split"] == split)][
                    f"ID_test_{metric}"
                ].mean()
                den = result_subset[(result_subset["model"] == model) & (result_subset["split"] == split)][
                    f"OOD_test_{metric}"
                ].mean()
                if report == "diff":
                    vmin, vmax = 0.0, 0.2
                    if perc:
                        df.loc[model, split] = (num - den) / num * 100
                    else:
                        df.loc[model, split] = num - den
                elif report == "ID":
                    df.loc[model, split] = num
                elif report == "OOD":
                    df.loc[model, split] = den

        df = df.astype(float)
        sns.heatmap(
            df,
            ax=ax[i // 2, i % 2],
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            cbar_kws={"label": f"{metric} difference"},
            vmin=vmin,
            vmax=vmax,
        )
        ax[i // 2, i % 2].set_title(f"{dataset}", fontsize=18)
        ax[i // 2, i % 2].set_xlabel("Split", fontsize=18)
        ax[i // 2, i % 2].set_ylabel("Model", fontsize=18)
        # ax[i // 2, i % 2].tick_params(axis='both', which='major', labelsize=14)

        ax[i // 2, i % 2].set_xlabel("")
        ax[i // 2, i % 2].set_ylabel("")

        # just keep the xticks (ax.set_xticks) for left plots and yticks (ax.set_yticks) for bottom plots
        if i % 2 == 0:
            ax[i // 2, i % 2].set_yticks(np.arange(len(models)) + 0.5)
            ax[i // 2, i % 2].set_yticklabels(models, fontsize=18)
            ax[i // 2, i % 2].set_xticks([])
        else:
            ax[i // 2, i % 2].set_xticks([])
            ax[i // 2, i % 2].set_yticks([])
        # if i // 2 == 3:
        #    ax[i // 2, i % 2].set_yticks(np.arange(len(models)) + 0.5)
        #    ax[i // 2, i % 2].set_yticklabels(models)
        # else:
        #    ax[i // 2, i % 2].set_yticks([])

        ax[3, 0].set_xticks(np.arange(len(split_types)) + 0.5)
        ax[3, 0].set_xticklabels(split_types, rotation=90, fontsize=18)
        ax[3, 0].set_xticklabels([spitting_mapping[split] for split in SPLIT_TYPES])
        ax[3, 1].set_xticks(np.arange(len(split_types)) + 0.5)
        ax[3, 1].set_xticklabels(split_types, rotation=90, fontsize=18)
        ax[3, 1].set_xticklabels([spitting_mapping[split] for split in SPLIT_TYPES])

    # plt.tight_layout()
    # save the plot to pdf
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_all_datasets_{metric}_{report}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_all_datasets_{metric}_{report}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


def dataset_fixed_split_comparisson(
    results: Optional[pd.DataFrame] = None,
    dataset="CYP2C19",
    split_type1="scaffold",
    split_type2="molecular_weight",
    metric="roc_auc",
    save=False,
):
    """
    Scatter plot the ID and OOD for a particular dataset and two different splits.
    x-axis is the ID performance and y-axis is the OOD performance of two different split types.

    Args:
        results (pd.DataFrame): DataFrame with the results
        dataset (str): name of dataset
        split_type1 (str): name of split type 1
        split_type2 (str): name of split type 2
        metric (str): name of metric
        save (bool): whether to save plot

    Returns:
        None

    Options:
        metric: "accuracy", "roc_auc", "pr_auc"
    """
    if results is None:
        results = pd.read_csv(os.path.join("classification_results", "TDC", "results.csv"))

    assert metric in ["accuracy", "roc_auc", "pr_auc"], "Invalid metric"
    assert "dataset" in results.columns, "dataset column not found in results"
    assert "split" in results.columns, "split column not found in results"

    results = results[results["dataset"] == dataset]
    results1 = results[results["split"] == split_type1]
    results2 = results[results["split"] == split_type2]

    ## concat results1 and results2 on axis 1 and change the duplicate columns name
    results = pd.concat([results1, results2], axis=1)
    results.columns = [f"{col}_{split_type1}" if col != "model" else col for col in results1.columns] + [
        f"{col}_{split_type2}" if col != "model" else col for col in results2.columns
    ]
    # keep only one of the model column
    results = results.loc[:, ~results.columns.duplicated()]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(
        data=results, x=f"ID_test_{metric}_{split_type1}", y=f"OOD_test_{metric}_{split_type1}", label=split_type1
    )
    sns.scatterplot(
        data=results, x=f"ID_test_{metric}_{split_type2}", y=f"OOD_test_{metric}_{split_type2}", label=split_type2
    )

    ax.set_xlabel(f"ID {metric}")
    ax.set_ylabel(f"OOD {metric}")
    ax.set_title(f"{metric} comparison between ID and OOD for models on dataset {dataset}", fontsize=20)
    ax.grid(axis="both", linestyle="--", alpha=0.6)
    # put legend outside the plot
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if save:
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"{dataset}_{split_type1}_{split_type2}_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
    plt.show()


def test_size_statistics(dataset_names: Optional[List] = None, split_types: Optional[List] = None, save=False):
    """
    Plot the size ratio of test set (OOD and ID) to the size of the dataset for different split types.

    Args:
        dataset_names (List[str], optional): List of dataset names. If None, uses default DATASET_NAMES.
        split_types (List[str], optional): List of split types. If None, uses default SPLIT_TYPES.
        save (bool): Whether to save plot. Defaults to False.

    Returns:
        None

    Notes:
        - Loads data from config files in dataset folders
        - ID test set is assumed to be 20% of train set
        - Creates boxplots showing distribution of test set size ratios
        - Test set ratios are shown as percentages of total dataset size
        - Separate boxplots are created for ID and OOD test sets
        - Results are grouped by split type
        - If save=True, saves plot as PDF in assets folder
    """
    dataset_category = "TDC"

    # Use defaults if not provided
    if dataset_names is None:
        dataset_names = DATASET_NAMES
    if split_types is None:
        split_types = SPLIT_TYPES

    num_of_splits = 10
    dfs = []

    for dataset_name in dataset_names:
        for split_type in split_types:
            dataset_folder = os.path.join(DATASET_PATH, dataset_category, dataset_name, "split", split_type)

            # Load config
            with open(os.path.join(dataset_folder, "config.json"), "r") as f:
                data_config = json.load(f)

            # Extract sizes for each split
            train_sizes = [data_config[f"train_size_{i}"] for i in range(num_of_splits)]
            ood_test_sizes = [data_config[f"test_size_{i}"] for i in range(num_of_splits)]
            id_test_sizes = [train_size * 0.2 for train_size in train_sizes]  # 20% of train set

            # Calculate percentages
            total_sizes = [train_sizes[i] + ood_test_sizes[i] for i in range(num_of_splits)]
            ood_test_fracs = [ood_test_sizes[i] / total_sizes[i] * 100 for i in range(num_of_splits)]
            id_test_fracs = [id_test_sizes[i] / total_sizes[i] * 100 for i in range(num_of_splits)]

            # Create dataframe for this split type and dataset
            df = pd.DataFrame(
                {
                    "ood_test_size": ood_test_fracs,
                    "id_test_size": id_test_fracs,
                    "split_type": [split_type] * num_of_splits,
                    "dataset": [dataset_name] * num_of_splits,
                }
            )
            dfs.append(df)

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_melt = df.melt(id_vars=["split_type", "dataset"], var_name="test_type", value_name="size_ratio")

    sns.boxplot(x="split_type", y="size_ratio", hue="test_type", data=df_melt, ax=ax)

    # Customize plot
    ax.set_ylabel("Size ratio %", fontsize=18)
    ax.set_xlabel("Split type", fontsize=18)
    ax.set_title("Size ratio of test set to the size of the dataset", fontsize=20)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=["Test (OOD)", "Test (ID)"],
        title="Test set type",
        title_fontsize=14,
        fontsize=14,
        loc="upper right",
    )

    plt.xticks(rotation=45, fontsize=16)

    if save:
        fig.savefig(os.path.join(REPO_PATH, "assets", "test_size_ratio.pdf"), bbox_inches="tight", backend="pgf")

    plt.show()
