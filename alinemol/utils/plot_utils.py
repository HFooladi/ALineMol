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
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from math import pi

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
    "figure.dpi": 600,  # Higher DPI for better quality
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

MODEL_MAPPING = {
    "randomForest": "Random Forest",
    "XGB": "XGBoost",
    "SVM": "SVM",
    "GCN": "GCN",
    "GAT": "GAT",
    "MPNN": "MPNN",
    "AttentiveFP": "AttentiveFP",
    "Weave": "Weave",
    "gin_supervised_edgepred": "GIN + Edge",
    "gin_supervised_contextpred": "GIN + Context",
    "gin_supervised_infomax": "GIN + InfoMax",
    "gin_supervised_masking": "GIN + Masking",
    "gem": "GEM",
    "grover": "GROVER",
}

SPLIT_TYPPE_MAPPING = {
    "random": "Random",
    "scaffold": "Scaffold",
    "scaffold_generic": "Scaffold generic",
    "molecular_weight": "Molecular weight",
    "molecular_weight_reverse": "Molecular weight reverse",
    "molecular_logp": "Molecular logP",
    "kmeans": "K-means",
    "max_dissimilarity": "Max dissimilarity",
    "umap": "UMAP",
    "hi": "Lo-Hi",
    "datasail": "DataSAIL",
}

METRIC_MAPPING = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


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
            os.path.join(REPO_PATH, "assets", "figures", f"{dataset_name}_{metric}_ID_OOD.pdf"),
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
            os.path.join(REPO_PATH, "assets", "figures", f"ID_vs_OOD_{dataset_category}_{dataset_name}.pdf"),
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
                os.path.join(REPO_PATH, "assets", "figures", "dummy.pdf"),
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
        df,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".3f",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": r"\textbf{$\Delta$ " + METRIC_MAPPING[metric] + "}"},
    )
    plt.xlabel("", fontsize=24)
    plt.ylabel(r"\textbf{Dataset}", fontsize=18, labelpad=10)
    plt.title(rf"\textbf{{Difference between ID and OOD}} \textbf{{{METRIC_MAPPING[metric]}}}", fontsize=24, pad=15)
    a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=14)
    a.set_yticklabels(a.get_yticklabels(), rotation=0, horizontalalignment="right", fontsize=14)
    ax.set_xticklabels(
        [r"\textbf{" + SPLIT_TYPPE_MAPPING[split] + "}" for split in SPLIT_TYPES], rotation=45, ha="right"
    )
    ax.set_yticklabels([r"\textbf{" + dataset + "}" for dataset in dataset_names])
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_{metric}.png"),
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
        id_df,
        ax=ax[0],
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={"label": rf"\textbf{{{METRIC_MAPPING[metric]}}}"},
        vmin=vmin,
        vmax=vmax,
    )
    ax[0].set_title(rf"\textbf{{ID}} \textbf{{{METRIC_MAPPING[metric]}}}", fontsize=18, pad=10)
    ax[0].set_xlabel("", fontsize=18)
    ax[0].set_ylabel(r"\textbf{Dataset}", fontsize=18, labelpad=10)
    ax[0].tick_params(axis="both", which="major", labelsize=14)
    ax[0].set_xticklabels(
        [r"\textbf{" + SPLIT_TYPPE_MAPPING[split] + "}" for split in SPLIT_TYPES], rotation=45, ha="right"
    )
    ax[0].set_yticklabels([r"\textbf{" + dataset + "}" for dataset in dataset_names])

    sns.heatmap(
        ood_df,
        ax=ax[1],
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={"label": rf"\textbf{{{METRIC_MAPPING[metric]}}}"},
        vmin=vmin,
        vmax=vmax,
    )
    ax[1].set_title(rf"\textbf{{OOD}} \textbf{{{METRIC_MAPPING[metric]}}}", fontsize=18, pad=10)
    ax[1].set_xlabel("", fontsize=18)
    ax[1].set_ylabel(r"\textbf{Dataset}", fontsize=18, labelpad=10)
    ax[1].tick_params(axis="both", which="major", labelsize=14)
    ax[1].set_xticklabels(
        [r"\textbf{" + SPLIT_TYPPE_MAPPING[split] + "}" for split in SPLIT_TYPES], rotation=45, ha="right"
    )
    ax[1].set_yticklabels([r"\textbf{" + dataset + "}" for dataset in dataset_names])

    # plt.tight_layout()
    if save:
        # save as pdf
        fig.savefig(
            os.path.join("assets", "figures", f"heatmap_id_ood_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join("assets", "figures", f"heatmap_id_ood_{metric}.png"),
            bbox_inches="tight",
            dpi=600,
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
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_{dataset}_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_{dataset}_{metric}.png"),
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
            cbar_kws={"label": r"\textbf{$\Delta$ " + METRIC_MAPPING[metric] + "}"},
            vmin=vmin,
            vmax=vmax,
            annot_kws={"size": 12},
        )
        ax[i // 2, i % 2].set_title(r"\textbf{" + dataset + "}", fontsize=18, pad=10)
        # ax[i // 2, i % 2].tick_params(axis='both', which='major', labelsize=14)

        ax[i // 2, i % 2].set_xlabel("")
        ax[i // 2, i % 2].set_ylabel("")

        # just keep the xticks (ax.set_xticks) for left plots and yticks (ax.set_yticks) for bottom plots
        if i % 2 == 0:
            ax[i // 2, i % 2].set_yticks(np.arange(len(models)) + 0.5)
            ax[i // 2, i % 2].set_yticklabels(
                [r"\textbf{" + MODEL_MAPPING[model] + "}" for model in models], fontsize=16
            )
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
        ax[3, 0].set_xticklabels(
            [r"\textbf{" + SPLIT_TYPPE_MAPPING[split] + "}" for split in SPLIT_TYPES],
            rotation=45,
            fontsize=16,
            ha="right",
        )
        ax[3, 1].set_xticks(np.arange(len(split_types)) + 0.5)
        ax[3, 1].set_xticklabels(
            [r"\textbf{" + SPLIT_TYPPE_MAPPING[split] + "}" for split in SPLIT_TYPES],
            rotation=45,
            fontsize=16,
            ha="right",
        )

    # plt.tight_layout()
    # save the plot to pdf
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_all_datasets_{metric}_{report}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"heatmap_all_datasets_{metric}_{report}.png"),
            bbox_inches="tight",
            dpi=600,
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
            os.path.join(REPO_PATH, "assets", "figures", f"{dataset}_{split_type1}_{split_type2}_{metric}.pdf"),
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
        - If save=True, saves plot as PDF in assets/figures folder
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
        fig.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "test_size_ratio.pdf"), bbox_inches="tight", backend="pgf"
        )

    plt.show()


def regplot_with_model_categories(results: pd.DataFrame, metric: str = "roc_auc", save: bool = False):
    """
    Create a regression plot between ID and OOD performance of model categories (Classical ML, GNN, Pretrained GNN)
    all-in-one and individual split types.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        metric (str): Metric to use for the plot
        save (bool): Whether to save the plot

    Returns:
        None
    """
    # 3 model type groups: each group has 6 rows × 2 columns
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 3], hspace=2.5, figure=fig)

    fig.suptitle(rf"\textbf{{ID vs OOD Performance Comparison ({METRIC_MAPPING[metric]})}}", fontsize=24, y=1)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.13)
    ax1 = plt.subplot(gs1[0, 0])
    ax2 = plt.subplot(gs1[0, 1])
    ax3 = plt.subplot(gs1[0, 2])

    # Create 6×6 grid (6 rows, 3 groups × 2 columns each)
    gs2 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[1], hspace=0.04, wspace=0.25)

    # Get available splits
    available_splits = results["split"].unique()
    n_splits = len(available_splits)  # 11

    # Create axes for each model type group
    ml_axes = []
    gnn_axes = []
    pretrained_gnn_axes = []

    for i in range(n_splits):
        row = i // 2
        col_in_group = i % 2

        # ML group: columns 0-1
        ml_ax = plt.subplot(gs2[row, col_in_group])

        # GNN group: columns 2-3
        gnn_ax = plt.subplot(gs2[row, col_in_group + 2])

        # Pretrained GNN group: columns 4-5
        pretrained_gnn_ax = plt.subplot(gs2[row, col_in_group + 4])

        ml_axes.append(ml_ax)
        gnn_axes.append(gnn_ax)
        pretrained_gnn_axes.append(pretrained_gnn_ax)

    ML_result = results[results["model_type"] == "Classical ML"]
    GNN_result = results[results["model_type"] == "GNN"]
    Pretrained_GNN_result = results[results["model_type"] == "Pretrained GNN"]

    make_regplot(ML_result[f"ID_test_{metric}"], ML_result[f"OOD_test_{metric}"], ax1)
    make_regplot(GNN_result[f"ID_test_{metric}"], GNN_result[f"OOD_test_{metric}"], ax2, color="#8da0cb")
    make_regplot(
        Pretrained_GNN_result[f"ID_test_{metric}"], Pretrained_GNN_result[f"OOD_test_{metric}"], ax3, color="#4daf4a"
    )

    for ax, title in [(ax1, "Classical ML Models"), (ax2, "GNN Models"), (ax3, "Pretrained GNN Models")]:
        ax.set_title(rf"\textbf{{{title}}}", fontsize=20, pad=10)
        ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=15)
        # ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=15 if ax == ax1 else 0)
        if ax == ax1:
            ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=15, labelpad=10)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=15)
        format_axis(ax)

    for i, split in enumerate(available_splits):
        ML_split = ML_result[ML_result["split"] == split]
        GNN_split = GNN_result[GNN_result["split"] == split]
        Pretrained_GNN_split = Pretrained_GNN_result[Pretrained_GNN_result["split"] == split]

        make_regplot(ML_split[f"ID_test_{metric}"], ML_split[f"OOD_test_{metric}"], ml_axes[i], linewidth=1)
        make_regplot(
            GNN_split[f"ID_test_{metric}"], GNN_split[f"OOD_test_{metric}"], gnn_axes[i], color="#8da0cb", linewidth=1
        )
        make_regplot(
            Pretrained_GNN_split[f"ID_test_{metric}"],
            Pretrained_GNN_split[f"OOD_test_{metric}"],
            pretrained_gnn_axes[i],
            color="#4daf4a",
            linewidth=1,
        )

        for ax in [ml_axes[i], gnn_axes[i], pretrained_gnn_axes[i]]:
            ax.set_title(rf"\textbf{{{SPLIT_TYPPE_MAPPING[split]}}}", fontsize=12)
            format_axis(ax, linewidth=1, ticklabelsize=10)

            # Y-label only for leftmost column of each group
            if ax == ml_axes[i] and i % 2 == 0:
                ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=10, labelpad=10)
            else:
                ax.set_ylabel("")

            # X-label only for bottom row
            if i >= len(available_splits) - 2:  # Last row
                ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=10)
            else:
                ax.set_xlabel("")

    ax1.text(-0.2, 1.15, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=20)
    ml_axes[0].text(-1, 1.2, r"\textbf{(b)}", transform=ml_axes[0].transAxes, fontsize=20)

    if save:
        plt.savefig(os.path.join(REPO_PATH, "assets", "figures", "regplot_with_categories.pdf"), bbox_inches="tight")
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "regplot_with_categories.png"), bbox_inches="tight", dpi=300
        )
    plt.show()


def regplot_with_model_categories_fixed_split(
    results: pd.DataFrame, split: str, metric: str = "roc_auc", save: bool = False
):
    """
    Create a regression plot between ID and OOD performance of model categories (Classical ML, GNN, Pretrained GNN)
    for a specific split type, all-in-one and individual datasets.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        split (str): Split type to use
        metric (str): Metric to use
        save (bool): Whether to save the plot

    Returns:
        None
    """
    # fotmat
    alpha = 0.7  # for the data points
    ticklabelsize_mainplot = 19
    ticklabelsize_datasetplot = 16

    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.3, 2], hspace=4, figure=fig)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.15)
    ax1 = plt.subplot(gs1[0, 0])
    ax2 = plt.subplot(gs1[0, 1])
    ax3 = plt.subplot(gs1[0, 2])

    # Add super title
    fig.suptitle(rf"\textbf{{ID vs OOD Performance Comparison ({METRIC_MAPPING[metric]})}}", fontsize=26, y=0.98)

    gs2 = gridspec.GridSpecFromSubplotSpec(4, 6, subplot_spec=gs[1], hspace=0.08, wspace=0.22)
    # Second and third rows: 4x2 grid (8 subplots)
    ml_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(0, 2)]
    gnn_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(2, 4)]
    pretrained_gnn_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(4, 6)]

    ML_result = results[results["model_type"] == "Classical ML"]
    GNN_result = results[results["model_type"] == "GNN"]
    Pretrained_GNN_result = results[results["model_type"] == "Pretrained GNN"]

    ML_result = ML_result[ML_result["split"] == split]
    GNN_result = GNN_result[GNN_result["split"] == split]
    Pretrained_GNN_result = Pretrained_GNN_result[Pretrained_GNN_result["split"] == split]

    # Plot main comparisons
    make_regplot(
        ML_result[f"ID_test_{metric}"],
        ML_result[f"OOD_test_{metric}"],
        ax1,
        alpha=alpha,
        fontsize=ticklabelsize_mainplot,
    )
    make_regplot(
        GNN_result[f"ID_test_{metric}"],
        GNN_result[f"OOD_test_{metric}"],
        ax2,
        color="#8da0cb",
        alpha=alpha,
        fontsize=ticklabelsize_mainplot,
    )
    make_regplot(
        Pretrained_GNN_result[f"ID_test_{metric}"],
        Pretrained_GNN_result[f"OOD_test_{metric}"],
        ax3,
        color="#4daf4a",
        alpha=alpha,
        fontsize=ticklabelsize_mainplot,
    )

    # Format main plots
    for ax, title in [
        (ax1, f"Classical ML Models ({split})"),
        (ax2, f"GNN Models ({split})"),
        (ax3, f"Pretrained GNN Models ({split})"),
    ]:
        ax.set_title(rf"\textbf{{{title}}}", fontsize=19, pad=20)
        ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=18)
        ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=18)
        if ax == ax2 or ax == ax3:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=18)
        format_axis(ax, ticklabelsize=18)

    # For other axis, plot the same for ML and GNN for each datasets
    # Plot and format datasets comparisons
    for i, dataset in enumerate(DATASET_NAMES):
        ML_dataset = ML_result[ML_result["dataset"] == dataset]
        GNN_dataset = GNN_result[GNN_result["dataset"] == dataset]
        Pretrained_GNN_dataset = Pretrained_GNN_result[Pretrained_GNN_result["dataset"] == dataset]

        # plot for the whole dataset in the background
        sns.scatterplot(
            x=f"ID_test_{metric}", y=f"OOD_test_{metric}", data=ML_result, ax=ml_axes[i], alpha=0.15, s=40, color="gray"
        )
        make_regplot(
            ML_dataset[f"ID_test_{metric}"],
            ML_dataset[f"OOD_test_{metric}"],
            ml_axes[i],
            linewidth=1,
            alpha=alpha,
            fontsize=ticklabelsize_datasetplot,
        )

        sns.scatterplot(
            x=f"ID_test_{metric}",
            y=f"OOD_test_{metric}",
            data=GNN_result,
            ax=gnn_axes[i],
            alpha=0.15,
            s=40,
            color="gray",
        )
        make_regplot(
            GNN_dataset[f"ID_test_{metric}"],
            GNN_dataset[f"OOD_test_{metric}"],
            gnn_axes[i],
            color="#8da0cb",
            alpha=alpha,
            linewidth=1,
            fontsize=ticklabelsize_datasetplot,
        )

        sns.scatterplot(
            x=f"ID_test_{metric}",
            y=f"OOD_test_{metric}",
            data=Pretrained_GNN_result,
            ax=pretrained_gnn_axes[i],
            alpha=0.15,
            s=40,
            color="gray",
        )
        make_regplot(
            Pretrained_GNN_dataset[f"ID_test_{metric}"],
            Pretrained_GNN_dataset[f"OOD_test_{metric}"],
            pretrained_gnn_axes[i],
            color="#4daf4a",
            alpha=alpha,
            linewidth=1,
            fontsize=ticklabelsize_datasetplot,
        )

        # Format dataset plots
        for ax in [ml_axes[i], gnn_axes[i], pretrained_gnn_axes[i]]:
            ax.set_title(rf"\textbf{{{dataset}}}", fontsize=18, pad=12)
            ax.tick_params(axis="both", labelsize=ticklabelsize_datasetplot)
            format_axis(ax, linewidth=1, ticklabelsize=ticklabelsize_datasetplot)

            # Only show y-label for leftmost plots
            if ax == ml_axes[i] and i % 2 == 0:
                ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=16, labelpad=10)
            else:
                ax.set_ylabel("")

            # Only show x-label for bottom plots
            if i >= len(DATASET_NAMES) - 2:
                ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=16, labelpad=10)
            else:
                ax.set_xlabel("")

    # For the wole plots in row 1, add panel a, For the whole plots on next rows, add panel B
    ax1.text(-0.15, 1.2, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=20)
    ml_axes[0].text(-0.6, 1.3, r"\textbf{(b)}", transform=ml_axes[0].transAxes, fontsize=20)

    # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"regplot_{split}_specific_with_categories.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"regplot_{split}_specific_with_categories.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


def make_regplot(x, y, ax, color="#66c2a5", alpha=0.5, linewidth=2, fontsize=12):
    """
    Make a regression plot
    """
    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": alpha, "s": 40},
        line_kws={"color": "red", "linewidth": linewidth},
        ci=95,
        color=color,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.8, linestyle="-", linewidth=0.5)
    corr = pearsonr(x, y)[0]
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5)
    ax.text(
        0.05,
        0.95,
        rf"\textbf{{r = {corr:.2f}}}",
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=props,
    )


def format_axis(ax, linewidth=2, ticklabelsize=14):
    """
    Format the axis of the plot
    """
    lims = [0.5, 0.95]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "--", alpha=1, color="gray", linewidth=linewidth)
    ax.set_xticks(np.arange(0.5, 1.0, 0.1))
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))
    ax.tick_params(axis="both", which="major", labelsize=ticklabelsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticklabelsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=ticklabelsize)


def regplot_with_model_categories_old(results: pd.DataFrame, metric: str = "roc_auc", save: bool = False):
    """
    Create a regression plot between ID and OOD performance of model categories (Classical ML, GNN, Pretrained GNN)
    all-in-one and individual split types.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        metric (str): Metric to use for the plot
        save (bool): Whether to save the plot

    Returns:
        None
    """
    # 3 model type groups: each group has 6 rows × 2 columns
    fig = plt.figure(figsize=(17, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 6], hspace=0.4, wspace=0, figure=fig)

    fig.suptitle(rf"\textbf{{ID vs OOD Performance Comparison ({METRIC_MAPPING[metric]})}}", fontsize=24, y=0.96)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.2)
    ax1 = plt.subplot(gs1[0, 0])
    ax2 = plt.subplot(gs1[0, 1])
    ax3 = plt.subplot(gs1[0, 2])

    # Create 6×6 grid (6 rows, 3 groups × 2 columns each)
    gs2 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[1], hspace=1, wspace=0.5)

    # Get available splits
    available_splits = results["split"].unique()
    n_splits = len(available_splits)  # 11

    # Create axes for each model type group
    ml_axes = []
    gnn_axes = []
    pretrained_gnn_axes = []

    for i in range(n_splits):
        row = i // 2
        col_in_group = i % 2

        # ML group: columns 0-1
        ml_ax = plt.subplot(gs2[row, col_in_group])

        # GNN group: columns 2-3
        gnn_ax = plt.subplot(gs2[row, col_in_group + 2])

        # Pretrained GNN group: columns 4-5
        pretrained_gnn_ax = plt.subplot(gs2[row, col_in_group + 4])

        ml_axes.append(ml_ax)
        gnn_axes.append(gnn_ax)
        pretrained_gnn_axes.append(pretrained_gnn_ax)

    ML_result = results[results["model_type"] == "Classical ML"]
    GNN_result = results[results["model_type"] == "GNN"]
    Pretrained_GNN_result = results[results["model_type"] == "Pretrained GNN"]

    make_regplot(ML_result[f"ID_test_{metric}"], ML_result[f"OOD_test_{metric}"], ax1)
    make_regplot(GNN_result[f"ID_test_{metric}"], GNN_result[f"OOD_test_{metric}"], ax2, color="#8da0cb")
    make_regplot(
        Pretrained_GNN_result[f"ID_test_{metric}"], Pretrained_GNN_result[f"OOD_test_{metric}"], ax3, color="#4daf4a"
    )

    for ax, title in [(ax1, "Classical ML Models"), (ax2, "GNN Models"), (ax3, "Pretrained GNN Models")]:
        ax.set_title(rf"\textbf{{{title}}}", fontsize=20, pad=10)
        ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=15)
        # ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=15 if ax == ax1 else 0)
        if ax == ax1:
            ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=15, labelpad=10)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=15)
        format_axis(ax)

    for i, split in enumerate(available_splits):
        ML_split = ML_result[ML_result["split"] == split]
        GNN_split = GNN_result[GNN_result["split"] == split]
        Pretrained_GNN_split = Pretrained_GNN_result[Pretrained_GNN_result["split"] == split]

        make_regplot(ML_split[f"ID_test_{metric}"], ML_split[f"OOD_test_{metric}"], ml_axes[i], linewidth=1)
        make_regplot(
            GNN_split[f"ID_test_{metric}"], GNN_split[f"OOD_test_{metric}"], gnn_axes[i], color="#8da0cb", linewidth=1
        )
        make_regplot(
            Pretrained_GNN_split[f"ID_test_{metric}"],
            Pretrained_GNN_split[f"OOD_test_{metric}"],
            pretrained_gnn_axes[i],
            color="#4daf4a",
            linewidth=1,
        )

        for ax in [ml_axes[i], gnn_axes[i], pretrained_gnn_axes[i]]:
            ax.set_title(rf"\textbf{{{SPLIT_TYPPE_MAPPING[split]}}}", fontsize=12)
            format_axis(ax, linewidth=1, ticklabelsize=10)

            # Y-label only for leftmost column of each group
            if ax == ml_axes[i] and i % 2 == 0:
                ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=10, labelpad=10)
            else:
                ax.set_ylabel("")

            # X-label only for bottom row
            if i >= len(available_splits) - 2:  # Last row
                ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=10)
            else:
                ax.set_xlabel("")

    ax1.text(-0.2, 1.4, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=20)
    ml_axes[0].text(-0.5, 1.5, r"\textbf{(b)}", transform=ml_axes[0].transAxes, fontsize=20)

    if save:
        plt.savefig(os.path.join(REPO_PATH, "assets", "figures", "regplot_with_categories.pdf"), bbox_inches="tight")
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "regplot_with_categories.png"), bbox_inches="tight", dpi=600
        )
    plt.show()


def regplot_with_model_categories_fixed_split_old(
    results: pd.DataFrame, split: str, metric: str = "roc_auc", save: bool = False
):
    """
    Create a regression plot between ID and OOD performance of model categories (Classical ML, GNN, Pretrained GNN)
    for a specific split type, all-in-one and individual datasets.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        split (str): Split type to use
        metric (str): Metric to use
        save (bool): Whether to save the plot

    Returns:
        None
    """

    def make_regplot(x, y, ax, color="#66c2a5", linewidth=2):
        sns.regplot(
            x=x,
            y=y,
            ax=ax,
            scatter_kws={"alpha": 0.5, "s": 40},
            line_kws={"color": "red", "linewidth": linewidth},
            ci=95,
            color=color,
        )

        corr = pearsonr(x, y)[0]
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5)
        ax.text(
            0.05,
            0.95,
            rf"\textbf{{r = {corr:.2f}}}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    fig = plt.figure(figsize=(17, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.4, wspace=0)

    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.2)
    ax1 = plt.subplot(gs1[0, 0])
    ax2 = plt.subplot(gs1[0, 1])
    ax3 = plt.subplot(gs1[0, 2])

    # Add super title
    fig.suptitle(rf"\textbf{{ID vs OOD Performance Comparison ({METRIC_MAPPING[metric]})}}", fontsize=24, y=0.98)

    gs2 = gridspec.GridSpecFromSubplotSpec(4, 6, subplot_spec=gs[1], hspace=1, wspace=0.3)
    # Second and third rows: 4x2 grid (8 subplots)
    ml_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(0, 2)]
    gnn_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(2, 4)]
    pretrained_gnn_axes = [plt.subplot(gs2[i, j]) for i in range(0, 4) for j in range(4, 6)]

    ML_result = results[results["model_type"] == "Classical ML"]
    GNN_result = results[results["model_type"] == "GNN"]
    Pretrained_GNN_result = results[results["model_type"] == "Pretrained GNN"]

    ML_result = ML_result[ML_result["split"] == split]
    GNN_result = GNN_result[GNN_result["split"] == split]
    Pretrained_GNN_result = Pretrained_GNN_result[Pretrained_GNN_result["split"] == split]

    # Plot main comparisons
    make_regplot(ML_result[f"ID_test_{metric}"], ML_result[f"OOD_test_{metric}"], ax1)
    make_regplot(GNN_result[f"ID_test_{metric}"], GNN_result[f"OOD_test_{metric}"], ax2, color="#8da0cb")
    make_regplot(
        Pretrained_GNN_result[f"ID_test_{metric}"], Pretrained_GNN_result[f"OOD_test_{metric}"], ax3, color="#4daf4a"
    )

    # Format main plots
    for ax, title in [
        (ax1, f"Classical ML Models ({split})"),
        (ax2, f"GNN Models ({split})"),
        (ax3, f"Pretrained GNN Models ({split})"),
    ]:
        ax.set_title(rf"\textbf{{{title}}}", fontsize=16, pad=10)
        ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=16)
        ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=16)
        if ax == ax2 or ax == ax3:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=16)
        format_axis(ax)

    # For other axis, plot the same for ML and GNN for each datasets
    # Plot and format datasets comparisons
    for i, dataset in enumerate(DATASET_NAMES):
        ML_dataset = ML_result[ML_result["dataset"] == dataset]
        GNN_dataset = GNN_result[GNN_result["dataset"] == dataset]
        Pretrained_GNN_dataset = Pretrained_GNN_result[Pretrained_GNN_result["dataset"] == dataset]

        # plot for the whole dataset in the background
        sns.scatterplot(
            x=f"ID_test_{metric}", y=f"OOD_test_{metric}", data=ML_result, ax=ml_axes[i], alpha=0.15, s=40, color="gray"
        )
        make_regplot(ML_dataset[f"ID_test_{metric}"], ML_dataset[f"OOD_test_{metric}"], ml_axes[i], linewidth=1)
        sns.scatterplot(
            x=f"ID_test_{metric}",
            y=f"OOD_test_{metric}",
            data=GNN_result,
            ax=gnn_axes[i],
            alpha=0.15,
            s=40,
            color="gray",
        )
        make_regplot(
            GNN_dataset[f"ID_test_{metric}"],
            GNN_dataset[f"OOD_test_{metric}"],
            gnn_axes[i],
            color="#8da0cb",
            linewidth=1,
        )
        sns.scatterplot(
            x=f"ID_test_{metric}",
            y=f"OOD_test_{metric}",
            data=Pretrained_GNN_result,
            ax=pretrained_gnn_axes[i],
            alpha=0.15,
            s=40,
            color="gray",
        )
        make_regplot(
            Pretrained_GNN_dataset[f"ID_test_{metric}"],
            Pretrained_GNN_dataset[f"OOD_test_{metric}"],
            pretrained_gnn_axes[i],
            color="#4daf4a",
            linewidth=1,
        )

        # Format dataset plots
        for ax in [ml_axes[i], gnn_axes[i], pretrained_gnn_axes[i]]:
            ax.set_title(rf"\textbf{{{dataset}}}", fontsize=14, pad=10)
            ax.tick_params(axis="both", labelsize=14)
            format_axis(ax, linewidth=1, ticklabelsize=10)

            # Only show y-label for leftmost plots
            if ax == ml_axes[i] and i % 2 == 0:
                ax.set_ylabel(rf"\textbf{{OOD {METRIC_MAPPING[metric]}}}", fontsize=12, labelpad=10)
            else:
                ax.set_ylabel("")

            # Only show x-label for bottom plots
            if i >= len(DATASET_NAMES) - 2:
                ax.set_xlabel(rf"\textbf{{ID {METRIC_MAPPING[metric]}}}", fontsize=12, labelpad=10)
            else:
                ax.set_xlabel("")

    # For the wole plots in row 1, add panel a, For the whole plots on next rows, add panel B
    ax1.text(-0.15, 1.3, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=20)
    ml_axes[0].text(-0.35, 1.4, r"\textbf{(b)}", transform=ml_axes[0].transAxes, fontsize=20)

    # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"regplot_{split}_specific_with_categories.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", f"regplot_{split}_specific_with_categories.png"),
            bbox_inches="tight",
            dpi=600,
        )
    plt.show()


def boxplot_heatmap_performance_difference(
    results: pd.DataFrame, metric: str = "roc_auc", perc: bool = False, save: bool = False
):
    """
    Boxplot and heatmap of the performance difference between ID and OOD for each splitter.
    Plot all-in-one (boxplot) and individual datasets (heatmap).

    Args:
        results (pd.DataFrame): DataFrame containing the results
        metric (str): Metric to use
        perc (bool): Whether to use percentage difference instead of absolute difference
        save (bool): Whether to save the plot

    Returns:
        None
    """
    results_plot = results.copy()
    results_plot["difference"] = results_plot[f"ID_test_{metric}"] - results_plot[f"OOD_test_{metric}"]
    results_plot["split"] = results_plot["split"].map(SPLIT_TYPPE_MAPPING)

    # Create a single figure with two subplots with more horizontal space between them
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.1})

    # First subplot - Boxplot
    sns.boxplot(
        x="split",
        y="difference",
        data=results_plot,
        hue="split",
        ax=ax1,
        showfliers=True,
        medianprops={"linewidth": 1.5},
        width=0.7,
        capwidths=0.25,
        fliersize=1,
    )
    format_xticklabels(ax1)
    plt.setp(ax1.get_yticklabels(), fontsize=16)
    ax1.set_title(r"$\textbf{Difference between ID and OOD test " + METRIC_MAPPING[metric] + "}$", fontsize=20, pad=20)
    ax1.set_xlabel("", fontsize=24)
    ax1.set_ylabel(r"$\Delta$" + r"$\textbf{" + METRIC_MAPPING[metric] + "}$", fontsize=20)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Second subplot - Heatmap
    dataset_names = results_plot["dataset"].unique()
    split_types = [SPLIT_TYPPE_MAPPING[split] for split in SPLIT_TYPES]
    df = pd.DataFrame(index=dataset_names, columns=split_types)

    # Fill the dataframe
    for dataset in dataset_names:
        for split in split_types:
            num = results_plot[(results_plot["dataset"] == dataset) & (results_plot["split"] == split)][
                f"ID_test_{metric}"
            ].mean()
            den = results_plot[(results_plot["dataset"] == dataset) & (results_plot["split"] == split)][
                f"OOD_test_{metric}"
            ].mean()
            df.loc[dataset, split] = num - den if not perc else (num - den) / num * 100

    df = df.astype(float)

    # Plot heatmap
    vmin, vmax = 0.0, 0.2
    sns.heatmap(df, ax=ax2, cmap="coolwarm", annot=True, fmt=".3f", vmin=vmin, vmax=vmax, annot_kws={"size": 12})
    ax2.set_xlabel("", fontsize=24)
    ax2.set_ylabel(r"$\textbf{Data Sets}$", fontsize=20)
    ax2.set_title(r"$\textbf{Difference between ID and OOD test " + METRIC_MAPPING[metric] + "}$", fontsize=20, pad=20)

    format_xticklabels(ax2, rotation=45, ha="right", fontsize=16, is_heatmap=True)
    plt.setp(ax2.get_yticklabels(), fontsize=16, ha="right")
    # Format colorbar
    colorbar = ax2.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.ax.set_ylabel(r"$\Delta$" + r"$\textbf{" + METRIC_MAPPING[metric] + "}$", fontsize=16)

    # Add subplot labels
    ax1.text(-0.15, 1.2, r"$\textbf{(a)}$", transform=ax1.transAxes, fontsize=20)
    ax2.text(-0.23, 1.2, r"$\textbf{(b)}$", transform=ax2.transAxes, fontsize=20)

    if save:
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "box_heatmap_id_ood_comparison_roc_auc.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "box_heatmap_id_ood_comparison_roc_auc.png"),
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()


def barplot_performance_difference_model_categories(results: pd.DataFrame, metric: str = "roc_auc", save: bool = False):
    """
    Barplot of the performance difference between ID and OOD for each model category for each splitter.
    Plot all-in-one and individual datasets.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.8, figure=fig)  # Increased hspace for more gap

    # First row: one wide subplot
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
    ax1 = plt.subplot(gs1[0, 0])

    # Second and third rows: 4x2 grid (8 subplots)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1])
    axes = [plt.subplot(gs2[i, j]) for i in range(2) for j in range(4)]

    results_plot = results.copy()
    results_plot["difference"] = results_plot[f"ID_test_{metric}"] - results_plot[f"OOD_test_{metric}"]

    df = results_plot.groupby(["split", "model_type", "dataset"])["difference"].mean().reset_index()
    df["split"] = df["split"].map(SPLIT_TYPPE_MAPPING)
    # rearrange the order of the split
    split_order = [SPLIT_TYPPE_MAPPING[split] for split in SPLIT_TYPES]
    df["split"] = pd.Categorical(df["split"], split_order)

    # Main plot
    g = sns.barplot(x="split", y="difference", hue="model_type", data=df, ax=ax1, legend=True)
    ax1.set_xlabel("")  # No x-label on main plot
    ax1.set_ylabel(r"\textbf{$\Delta$ " + METRIC_MAPPING[metric] + "}", fontsize=20)
    ax1.set_title(r"\textbf{Difference between ID and OOD test " + METRIC_MAPPING[metric] + "}", fontsize=24, pad=20)

    # ticks formatting
    format_xticklabels(ax1, rotation=30, ha="right", fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    # Grid and legend
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    legend = ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14, title=r"\textbf{Model Type}")
    for text in legend.get_texts():
        text.set_fontsize(14)
        text.set_text(r"\textbf{" + text.get_text() + "}")

    # Subplots
    dataset_names = results_plot["dataset"].unique()
    for i, dataset in enumerate(dataset_names):
        df_dataset = df[df["dataset"] == dataset]
        g = sns.barplot(x="split", y="difference", hue="model_type", data=df_dataset, ax=axes[i])

        axes[i].set_title(r"\textbf{" + dataset + "}", fontsize=18, pad=10)
        axes[i].set_xlabel("")
        if i % 4 == 0:
            axes[i].set_ylabel(r"\textbf{$\Delta$ " + METRIC_MAPPING[metric] + "}", fontsize=14)
        else:
            axes[i].set_ylabel("")

        if i >= 4:
            format_xticklabels(axes[i])
        else:
            axes[i].set_xticklabels([])

        axes[i].tick_params(axis="both", labelsize=14)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
        g.legend().remove()
        axes[i].set_ylim(-0.10, 0.20)
        axes[i].set_yticks(np.arange(-0.10, 0.21, 0.05))

    # Subplot labels
    ax1.text(-0.08, 1.15, r"\textbf{(a)}", transform=ax1.transAxes, fontsize=20)
    axes[0].text(-0.27, 1.35, r"\textbf{(b)}", transform=axes[0].transAxes, fontsize=20)

    if save:
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "grouped_barplot_ml_gnn_difference.pdf"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "grouped_barplot_ml_gnn_difference.png"),
            bbox_inches="tight",
            dpi=600,
        )

    plt.show()


def format_xticklabels(ax, rotation=45, ha="right", fontsize=16, is_heatmap=False):
    """
    Format the x-tick labels of the plot: bold, rotaton, ha, fontsize
    """
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    if not is_heatmap:
        ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(
        [r"\textbf{" + label + "}" for label in xticklabels], rotation=rotation, ha=ha, fontsize=fontsize
    )


def plot_radar_subplots(datasets, save=False):
    """
    Plot radar subplots for each property
    Args:
        datasets (dict): Dictionary of datasets with property values
        save (bool): Whether to save the plot

    Returns:
        None
    """
    properties = ["Molecular_Weight", "LogP", "TPSA", "HBD", "HBA", "Rotatable_Bonds"]
    dataset_names = list(datasets.keys())
    _ = len(dataset_names)

    # Create figure with better spacing
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.1, wspace=0.2)

    # Use a more distinguishable color palette
    colors = sns.color_palette("Set2")

    for i, property_name in enumerate(properties):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection="polar")

        # Collect and normalize median values
        values = [datasets[name][property_name].median() for name in dataset_names]  # Changed to median
        values = normalize_property(np.array(values), property_name)

        # Calculate angles
        angles = np.linspace(0, 2 * pi, len(dataset_names), endpoint=False)

        # Close the radar chart
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # Plot with enhanced styling
        ax.plot(angles, values, "o-", color=colors[i], linewidth=2.5, markersize=8)
        ax.fill(angles, values, color=colors[i], alpha=0.25)

        # Enhance grid and labels
        ax.grid(True, color="gray", alpha=0.3, linewidth=0.5)
        ax.set_xticks(angles[:-1])

        # Rotate labels for better readability
        ax.set_xticklabels([r"\textbf{" + name + "}" for name in dataset_names], fontsize=16)

        # Add value markers at specific intervals
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=16)

        # Add property statistics (using median and IQR)
        median_val = np.median(values[:-1])
        q1 = np.percentile(values[:-1], 25)
        q3 = np.percentile(values[:-1], 75)
        iqr = q3 - q1
        stats_text = f"Median: {median_val:.2f}\nIQR: {iqr:.2f}"
        ax.text(
            1.0,
            1.1,
            stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            fontsize=16,
            ha="right",
            va="top",
        )

        # Enhance title
        title_text = rf"\textbf{{{property_name}}}" + "\n" + r"\textbf{Distribution (Median)}"
        ax.set_title(title_text, fontsize=20, pad=22)

    # Add main title
    plt.suptitle(
        r"\textbf{Physicochemical Properties Distribution Across Data Sets (Median Values)}",
        fontsize=24,
        fontweight="bold",
        y=1.05,
    )

    # Add legend with property ranges
    # For the legend text
    legend_text = {
        r"\textbf{Molecular Weight}": r"\textbf{Range: 160-480 Da}",
        r"\textbf{LogP}": r"\textbf{Range: -2 to 5}",
        r"\textbf{TPSA}": r"\textbf{Range: 0-140 Å²}",
        r"\textbf{HBD}": r"\textbf{Range: 0-5}",
        r"\textbf{HBA}": r"\textbf{Range: 0-10}",
        r"\textbf{Rotatable Bonds}": r"\textbf{Range: 0-10}",
    }

    fig.text(
        1.02,
        0.9,
        "\n".join([f"{k}: {v}" for k, v in legend_text.items()]),
        fontsize=18,
        transform=fig.transFigure,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=12),
    )

    if save:
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "physchem_props_radar_subplots.pdf"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(REPO_PATH, "assets", "figures", "physchem_props_radar_subplots.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


def normalize_property(values, property_name):
    """
    Normalize properties based on their characteristics
    """
    if property_name == "LogP":
        # Center around 0 with typical drug-like range (-2 to 5)
        return (values - (-2)) / (5 - (-2))
    elif property_name == "Molecular_Weight":
        # Normalize based on typical drug-like range (160-480)
        return (values - 160) / (480 - 160)
    elif property_name == "TPSA":
        # Normalize based on typical range (0-140)
        return values / 140
    elif property_name == "HBD":
        # Normalize based on Lipinski's rule (≤5)
        return values / 5
    elif property_name == "HBA":
        # Normalize based on Lipinski's rule (≤10)
        return values / 10
    elif property_name == "Rotatable_Bonds":
        # Normalize based on typical flexibility rule (≤10)
        return values / 10
    else:
        # Default min-max normalization
        return (values - values.min()) / (values.max() - values.min())


def boxplot_test_set_size(test_size_df: pd.DataFrame, save: bool = False):
    """
    Boxplot of the test set sizes (ID and OOD) for each split type and dataset.

    Args:
        test_size_df (pd.DataFrame): DataFrame ["ood_test_size", "id_test_size", "split_type", "dataset"]
        save (bool): Whether to save the plot

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Melt the dataframe
    df_melt = test_size_df.melt(id_vars=["split_type", "dataset"], var_name="test_type", value_name="size_ratio")

    # Create boxplot with publication-ready styling
    sns.boxplot(
        x="split_type",
        y="size_ratio",
        hue="test_type",
        data=df_melt,
        ax=ax,
        palette="Set2",
        fliersize=3,
        medianprops={"linewidth": 1.5},
        width=0.7,
        capwidths=0.15,
    )

    # Customize axes
    ax.set_ylabel(r"\textbf{Test Set Size (\%)}", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_xlabel(r"\textbf{Split Type}", fontsize=18, fontweight="bold", labelpad=10)
    ax.set_title(
        r"\textbf{Distribution of Test Set Sizes Across Different Split Types}", fontsize=20, fontweight="bold", pad=20
    )

    # format xticklabels bold
    ax.set_xticks(range(len(ax.get_xticklabels())))
    ax.set_xticklabels(
        [r"\textbf{" + SPLIT_TYPPE_MAPPING[split.get_text()] + "}" for split in ax.get_xticklabels()],
        rotation=45,
        ha="right",
        fontsize=14,
    )
    plt.yticks(fontsize=14)

    # Enhance grid
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    label_mapping = {"ood_test_size": "Out-of-Distribution", "id_test_size": "In-Distribution"}
    labels = [label_mapping[label] for label in labels]
    ax.legend(
        handles=handles,
        labels=labels,
        title="Test Set Type",
        title_fontsize=14,
        fontsize=12,
        loc="upper right",
        frameon=True,
        framealpha=0.7,
        edgecolor="gray",
    )

    if save:
        fig.savefig(os.path.join(REPO_PATH, "assets", "figures", "test_size_ratio.pdf"), bbox_inches="tight", dpi=300)
        fig.savefig(os.path.join(REPO_PATH, "assets", "figures", "test_size_ratio.png"), bbox_inches="tight", dpi=300)

    plt.show()
