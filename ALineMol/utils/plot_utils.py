# import visualization packages
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

from typing import List, Tuple, Dict, Union

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(
        ID_test_score, OOD_test_score, color=light_color, s=40, edgecolor=dark_color, linewidth=1
    )
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


def plot_ID_OOD_sns(
    data: pd.DataFrame, dataset_category="TDC", dataset_name="CYP2C19", save: bool = False
):
    """
    Plot ID vs OOD test ROC-AUC scores using seaborn

    Args:
        data: pd.DataFrame with columns "ID", "OOD"
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
