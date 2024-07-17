# import visualization packages
import os
from typing import Dict, List, Tuple, Union

import datamol as dm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.manifold import TSNE

REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

light_color = plt.get_cmap("plasma").colors[170]
dark_color = "black"


matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "font.size": 16,
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


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
    ax.scatter(ID_test_score, OOD_test_score, color=light_color, s=40, edgecolor=dark_color, linewidth=1)
    ax.axline((0.6, 0.6), slope=1, linestyle="--")
    ax.axline((0.5, 0.6), (1, 1.1), color=dark_color, linestyle="--")
    ax.axline((0.5, 0.4), (1, 0.9), color=dark_color, linestyle="--")

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


def plot_ID_OOD_bar(
    data: pd.DataFrame, metrics=["accuracy", "roc_auc", "pr_auc"], save: bool = False
) -> None:
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
                os.path.join(REPO_PATH, "assets" "dummy.pdf"),
                bbox_inches="tight",
                backend="pgf",
            )
        plt.show()


def visualize_chemspace(
    data: pd.DataFrame, split_names: List[str], mol_col: str = "smiles", size_col=None, size=10
):
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
    figs = plt.figure(num=3)
    features = [dm.to_fp(mol) for mol in data[mol_col]]
    embedding = umap.UMAP().fit_transform(features)
    data["UMAP_0"], data["UMAP_1"] = embedding[:, 0], embedding[:, 1]
    for split_name in split_names:
        plt.figure(figsize=(12, 8))
        fig = sns.scatterplot(
            data=data, x="UMAP_0", y="UMAP_1", s=size, style=size_col, hue=split_name, alpha=0.7
        )
        fig.set_title(f"UMAP Embedding of compounds for {split_name} split")
    return figs
