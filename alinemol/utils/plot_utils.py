# import visualization packages
import os
from typing import Dict, List, Tuple, Union

import datamol as dm
import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(REPO_PATH, "datasets")

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
sns.set_palette("Set2")
CFG = yaml.safe_load(open(os.path.join(DATASET_PATH, "config.yml"), "r"))

ML_MODELS = CFG["models"]["ML"]
GNN_MODELS = CFG["models"]["GNN"]["scratch"]
PRETRAINED_GNN_MODELS = CFG["models"]["GNN"]["pretrained"]
ALL_MODELS = [ML_MODELS, GNN_MODELS, PRETRAINED_GNN_MODELS]


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
                os.path.join(REPO_PATH, "assets" "dummy.pdf"),
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
    assert (
        "dataset" in results.columns
    ), "dataset column not found in results"  # check if the dataset column is in the results
    assert (
        "split" in results.columns
    ), "split column not found in results"  # check if the split column is in the results

    dataset_names = results["dataset"].unique()  # get the unique dataset names
    # split_types = results["split"].unique()  # get the unique split types
    # reorder the split types to this order
    split_types = ["scaffold", "scaffold_generic", "molecular_weight", "molecular_weight_reverse", "molecular_logp", "kmeans", "max_dissimilarity"]
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
    a = sns.heatmap(df, ax=ax, cmap="coolwarm", annot=True, fmt=".2f", vmin=vmin, vmax=vmax)
    plt.xlabel("Split", fontsize=24)
    plt.ylabel("Dataset", fontsize=24)
    plt.title(f"Difference between ID and OOD {metric}", fontsize=24)
    a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=18)
    a.set_yticklabels(a.get_yticklabels(), rotation=0, horizontalalignment="right", fontsize=18)
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        #save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def heeatmap_plot_id_ood(results: pd.DataFrame = None, metric: str = "roc_auc", perc=False, save: bool = False) -> None:
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
    metric = "accuracy"
    save=False
    # split_types = results["split"].unique()  # get the unique split types
    # reorder the split types to this order
    split_types = ["scaffold", "scaffold_generic", "molecular_weight", "molecular_weight_reverse", "molecular_logp", "kmeans", "max_dissimilarity"]
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
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    # one plot for id and one plot for ood
    sns.heatmap(id_df, ax=ax[0], annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': f"{metric}"}, vmin=vmin, vmax=vmax)
    ax[0].set_title(f"ID {metric}", fontsize=18)
    ax[0].set_xlabel("Split", fontsize=18)
    ax[0].set_ylabel("Dataset", fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=14)

    sns.heatmap(ood_df, ax=ax[1], annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': f"{metric}"}, vmin=vmin, vmax=vmax)
    ax[1].set_title(f"OOD {metric}", fontsize=18)
    ax[1].set_xlabel("Split", fontsize=18)
    ax[1].set_ylabel("Dataset", fontsize=18)
    ax[1].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    if save:
        # save as pdf
        fig.savefig(
            os.path.join("assets", f"heatmap_id_ood_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        #save as png
        fig.savefig(
            os.path.join("assets", f"heatmap_id_ood_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def heatmap_plot_dataset_fixed(results: pd.DataFrame = None, dataset="CYP2C19", metric: str = "roc_auc", perc=False, save: bool = False) -> None:
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
    assert (
        "dataset" in results.columns
    ), "dataset column not found in results"  # check if the dataset column is in the results
    assert (
        "split" in results.columns
    ), "split column not found in results"  # check if the split column is in the results
    assert "model" in results.columns, "model column not found in results"  # check if the model column is in the results

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
        #save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{dataset}_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        #save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_{dataset}_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()



def heatmap_plot_all_dataset(results: pd.DataFrame = None, metric: str = "roc_auc", perc=False, save: bool = False) -> None:
    """
    We want to have a heatmap with one axis datasets and one axis splits for all the datassets. The values in the heatmap are the difference between ID and OOD
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
    dataset_names = results["dataset"].unique()  # get the unique dataset names
    #split_types = results["split"].unique()  # get the unique split types
    split_types = ["scaffold", "scaffold_generic", "molecular_weight", "molecular_weight_reverse", "molecular_logp", "kmeans", "max_dissimilarity"]
    models = results["model"].unique()  # get the unique models
    vmin, vnmax = 0.0, 0.2
    # We want subplots for fixing each time one dataset, then plot the heatmap od difference between ID and OOd for all the modles and split types with 
    # the same dataset
    fig, ax = plt.subplots(4, 2, figsize=(20, 20))
    for i, dataset in enumerate(dataset_names):
        result_subset = results[results["dataset"]==dataset]
        df = pd.DataFrame(index=models, columns=split_types)

        for model in models:
            for split in split_types:
                num = result_subset[(result_subset["model"] == model) & (result_subset["split"] == split)][f"ID_test_{metric}"].mean()
                den = result_subset[(result_subset["model"] == model) & (result_subset["split"] == split)][f"OOD_test_{metric}"].mean()
                if perc:
                    df.loc[model, split] = (num - den) / num * 100
                else:
                    df.loc[model, split] = num - den
        
        df = df.astype(float)
        sns.heatmap(df, ax=ax[i // 2, i % 2], annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': f"{metric} difference"}, vmin=vmin, vmax=vnmax)
        ax[i // 2, i % 2].set_title(f"{dataset}", fontsize=18)
        ax[i // 2, i % 2].set_xlabel("Split", fontsize=18)
        ax[i // 2, i % 2].set_ylabel("Model", fontsize=18)
        #ax[i // 2, i % 2].tick_params(axis='both', which='major', labelsize=14)

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
        #if i // 2 == 3:
        #    ax[i // 2, i % 2].set_yticks(np.arange(len(models)) + 0.5)
        #    ax[i // 2, i % 2].set_yticklabels(models)
        #else:
        #    ax[i // 2, i % 2].set_yticks([])
        
        
        ax[3, 0].set_xticks(np.arange(len(split_types)) + 0.5)
        ax[3, 0].set_xticklabels(split_types, rotation=90, fontsize=18)
        ax[3, 1].set_xticks(np.arange(len(split_types)) + 0.5)
        ax[3, 1].set_xticklabels(split_types, rotation=90, fontsize=18)

    plt.tight_layout()
    # save the plot to pdf
    if save:
        # save as pdf
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_all_datasets_{metric}.pdf"),
            bbox_inches="tight",
            backend="pgf",
        )
        # save as png
        fig.savefig(
            os.path.join(REPO_PATH, "assets", f"heatmap_all_datasets_{metric}.png"),
            bbox_inches="tight",
        )
    plt.show()


def dataset_fixed_split_comparisson(results=None, dataset="CYP2C19", split_type1="scaffold", split_type2="molecular_weight", metric="roc_auc", save=False):
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
