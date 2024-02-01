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
    dataset_name: str = "HIV",
    metric: str = "ROC-AUC",
    save: bool = False,
):
    """
    Plot ID vs OOD test ROC-AUC scores

    Args:
        ID_test_score: list of ID test scores
        OOD_test_score: list of OOD test scores
        dataset_name (str): name of dataset
        metric (str): name of metric
        save (bool): whether to save plot

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(ID_test_score, OOD_test_score, color=light_color, s=40)
    ax.axline((0.6, 0.6), slope=1, linestyle="--")
    ax.axline((0.5, 0.6), (1, 1.1), color=dark_color, linestyle="--")
    ax.axline((0.5, 0.4), (1, 0.9), color=dark_color, linestyle="--")

    ax.set_title(f"{dataset_name} Dataset (MoleculeNet)")
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
