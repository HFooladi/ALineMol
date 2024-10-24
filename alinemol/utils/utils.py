# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import json
import os
import os.path as osp
from pathlib import Path
import glob
import re
import yaml
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import RandomSplitter, ScaffoldSplitter, SMILESToBigraph
from sklearn.metrics import brier_score_loss

from alinemol.utils.metric_utils import eval_roc_auc, eval_pr_auc, eval_acc

filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(repo_path, "datasets")

with open (osp.join(DATASET_PATH, "config.yml"), "r") as f:
    CONFIG = yaml.safe_load(f)

SPLITTING_METHODS = CONFIG["splitting"]
DATASET_NAMES = CONFIG["datasets"]["TDC"]
ML_MODELS = CONFIG["models"]["ML"]
GNN_MODELS = CONFIG["models"]["GNN"]["scratch"]
PRETRAINED_GNN_MODELS =  CONFIG["models"]["GNN"]["pretrained"]
ALL_MODELS = ML_MODELS + GNN_MODELS + PRETRAINED_GNN_MODELS


def init_featurizer(args: Dict) -> Dict:
    """Initialize node/edge featurizer

    Args:
        args (dict): Settings

    Returns:
        dict: Settings with featurizers updated
    
    Raises:
        ValueError: If the node_featurizer_type is not in ['canonical', 'attentivefp']

    Example:
    ```
    from alinemol.utils.utils import init_featurizer
    args = {
        "model": "GCN",
        "atom_featurizer_type": "canonical",
        "bond_featurizer_type": "canonical",
    }
    args = init_featurizer(args)
    print(args)
    ```

    """
    if args["model"] in [
        "gin_supervised_contextpred",
        "gin_supervised_infomax",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer

        args["atom_featurizer_type"] = "pre_train"
        args["bond_featurizer_type"] = "pre_train"
        args["node_featurizer"] = PretrainAtomFeaturizer()
        args["edge_featurizer"] = PretrainBondFeaturizer()
        return args

    if args["atom_featurizer_type"] == "canonical":
        from dgllife.utils import CanonicalAtomFeaturizer

        args["node_featurizer"] = CanonicalAtomFeaturizer()
    elif args["atom_featurizer_type"] == "attentivefp":
        from dgllife.utils import AttentiveFPAtomFeaturizer

        args["node_featurizer"] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], " "got {}".format(
                args["atom_featurizer_type"]
            )
        )

    if args["model"] in ["Weave", "MPNN", "AttentiveFP"]:
        if args["bond_featurizer_type"] == "canonical":
            from dgllife.utils import CanonicalBondFeaturizer

            args["edge_featurizer"] = CanonicalBondFeaturizer(self_loop=True)
        elif args["bond_featurizer_type"] == "attentivefp":
            from dgllife.utils import AttentiveFPBondFeaturizer

            args["edge_featurizer"] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args["edge_featurizer"] = None

    return args


def load_dataset(args: Dict, df):
    """Load the dataset

    Args:
        args (dict): Settings
        df: Dataframe
    
    Returns:
        MoleculeCSVDataset
    
    Example:
    ```
    from alinemol.utils.utils import load_dataset
    args = {
        "smiles_column": "smiles",
        "task_names": "task_1,task_2",
        "result_path": "results",
        "num_workers": 4,
    }
    df = pd.read_csv("data.csv")
    dataset = load_dataset(args, df)
    print(dataset)
    ```
    """
    smiles_to_g = SMILESToBigraph(
        add_self_loop=True,
        node_featurizer=args["node_featurizer"],
        edge_featurizer=args["edge_featurizer"],
    )
    dataset = MoleculeCSVDataset(
        df=df,
        smiles_to_graph=smiles_to_g,
        smiles_column=args["smiles_column"],
        cache_file_path=args["result_path"] + "/graph.bin",
        task_names=args["task_names"],
        n_jobs=args["num_workers"],
    )

    return dataset


def get_configure(model: str) -> Dict:
    """Query for the manually specified configuration

    Args:
        model (str): Model type

    Returns:
        dict: Returns the manually specified configuration
    """
    with open(f"{filepath}/models/model_configures/{model}.json", "r") as f:
        config = json.load(f)
    return config


def increment_path(path, exist_ok=True, sep="_"):
    """
    Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.

    Args:
        path (str): Original path.
        exist_ok (bool): Whether to increment path or not if path exists.
        sep (str): Separator between name and number.

    Returns:
        str: Incremented path.
    """
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def mkdir_p(path: str):
    """Create a folder for the given path.

    Args:
        path (str): Folder to create
    """
    try:
        os.makedirs(path)
        print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Directory {} already exists.".format(path))
        else:
            raise


def init_trial_path(args: Dict) -> Dict:
    """Initialize the path for a hyperparameter setting

    Args:
        args (dict): Settings

    Returns:
        dict: Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args["result_path"] + "/{:d}".format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args["trial_path"] = path_to_results
    mkdir_p(args["trial_path"])

    return args


def init_inference_trial_path(args: Dict) -> Dict:
    """Initialize the path for a hyperparameter setting

    Args:
        args (dict): Settings

    Returns:
        dict: Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args["inference_result_path"] + "/{:d}".format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args["trial_path"] = path_to_results
    mkdir_p(args["trial_path"])

    return args


def split_dataset(args, dataset) -> Tuple:
    """Split the dataset

    Args:
        args (dict): Settings
        dataset: Dataset instance

    Returns:
        train_set: Training subset
        val_set: Validation subset
        test_set: Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args["split_ratio"].split(","))
    if args["split"] == "scaffold_decompose":
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset,
            frac_train=train_ratio,
            frac_val=val_ratio,
            frac_test=test_ratio,
            scaffold_func="decompose",
        )
    elif args["split"] == "scaffold_smiles":
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset,
            frac_train=train_ratio,
            frac_val=val_ratio,
            frac_test=test_ratio,
            scaffold_func="smiles",
        )
    elif args["split"] == "random":
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio
        )
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args["split"]))

    return train_set, val_set, test_set


def collate_molgraphs(data: List) -> Tuple:
    """Batching a list of datapoints for dataloader.

    Args:
        data (list of 4-tuples): Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns:
        smiles (list): List of smiles
        bg (DGLGraph): The batched DGLGraph.
        labels: Tensor of dtype float32 and shape (B, T)
            Batched datapoint labels. B is len(data) and
            T is the number of total tasks.
        masks: Tensor of dtype float32 and shape (B, T)
            Batched datapoint binary mask, indicating the
            existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


def collate_molgraphs_unlabeled(data: List) -> Tuple:
    """Batching a list of datapoints without labels

    Args:
        data (list of 2-tuples): Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.

    Returns:
        smiles (list): List of smiles
        bg (DGLGraph): The batched DGLGraph.
    """
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg


def load_model(exp_configure: Dict):
    """
    Args:
        exp_configure (dict)

    Returns:
        dgllife.model
    """
    if exp_configure["model"] == "GCN":
        from dgllife.model import GCNPredictor

        model = GCNPredictor(
            in_feats=exp_configure["in_node_feats"],
            hidden_feats=[exp_configure["gnn_hidden_feats"]] * exp_configure["num_gnn_layers"],
            activation=[F.relu] * exp_configure["num_gnn_layers"],
            residual=[exp_configure["residual"]] * exp_configure["num_gnn_layers"],
            batchnorm=[exp_configure["batchnorm"]] * exp_configure["num_gnn_layers"],
            dropout=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
            predictor_hidden_feats=exp_configure["predictor_hidden_feats"],
            predictor_dropout=exp_configure["dropout"],
            n_tasks=exp_configure["n_tasks"],
        )
    elif exp_configure["model"] == "GAT":
        from dgllife.model import GATPredictor

        model = GATPredictor(
            in_feats=exp_configure["in_node_feats"],
            hidden_feats=[exp_configure["gnn_hidden_feats"]] * exp_configure["num_gnn_layers"],
            num_heads=[exp_configure["num_heads"]] * exp_configure["num_gnn_layers"],
            feat_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
            attn_drops=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
            alphas=[exp_configure["alpha"]] * exp_configure["num_gnn_layers"],
            residuals=[exp_configure["residual"]] * exp_configure["num_gnn_layers"],
            predictor_hidden_feats=exp_configure["predictor_hidden_feats"],
            predictor_dropout=exp_configure["dropout"],
            n_tasks=exp_configure["n_tasks"],
        )
    elif exp_configure["model"] == "Weave":
        from dgllife.model import WeavePredictor

        model = WeavePredictor(
            node_in_feats=exp_configure["in_node_feats"],
            edge_in_feats=exp_configure["in_edge_feats"],
            num_gnn_layers=exp_configure["num_gnn_layers"],
            gnn_hidden_feats=exp_configure["gnn_hidden_feats"],
            graph_feats=exp_configure["graph_feats"],
            gaussian_expand=exp_configure["gaussian_expand"],
            n_tasks=exp_configure["n_tasks"],
        )
    elif exp_configure["model"] == "MPNN":
        from dgllife.model import MPNNPredictor

        model = MPNNPredictor(
            node_in_feats=exp_configure["in_node_feats"],
            edge_in_feats=exp_configure["in_edge_feats"],
            node_out_feats=exp_configure["node_out_feats"],
            edge_hidden_feats=exp_configure["edge_hidden_feats"],
            num_step_message_passing=exp_configure["num_step_message_passing"],
            num_step_set2set=exp_configure["num_step_set2set"],
            num_layer_set2set=exp_configure["num_layer_set2set"],
            n_tasks=exp_configure["n_tasks"],
        )
    elif exp_configure["model"] == "AttentiveFP":
        from dgllife.model import AttentiveFPPredictor

        model = AttentiveFPPredictor(
            node_feat_size=exp_configure["in_node_feats"],
            edge_feat_size=exp_configure["in_edge_feats"],
            num_layers=exp_configure["num_layers"],
            num_timesteps=exp_configure["num_timesteps"],
            graph_feat_size=exp_configure["graph_feat_size"],
            dropout=exp_configure["dropout"],
            n_tasks=exp_configure["n_tasks"],
        )
    elif exp_configure["model"] in [
        "gin_supervised_contextpred",
        "gin_supervised_infomax",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]:
        from dgllife.model import GINPredictor, load_pretrained

        model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=exp_configure["jk"],
            dropout=0.5,
            readout=exp_configure["readout"],
            n_tasks=exp_configure["n_tasks"],
        )
        model.gnn = load_pretrained(exp_configure["model"])
        model.gnn.JK = exp_configure["jk"]
    elif exp_configure["model"] == "NF":
        from dgllife.model import NFPredictor

        model = NFPredictor(
            in_feats=exp_configure["in_node_feats"],
            n_tasks=exp_configure["n_tasks"],
            hidden_feats=[exp_configure["gnn_hidden_feats"]] * exp_configure["num_gnn_layers"],
            batchnorm=[exp_configure["batchnorm"]] * exp_configure["num_gnn_layers"],
            dropout=[exp_configure["dropout"]] * exp_configure["num_gnn_layers"],
            predictor_hidden_size=exp_configure["predictor_hidden_feats"],
            predictor_batchnorm=exp_configure["batchnorm"],
            predictor_dropout=exp_configure["dropout"],
        )
    else:
        return ValueError(
            "Expect model to be from ['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP', "
            "'gin_supervised_contextpred', 'gin_supervised_infomax', "
            "'gin_supervised_edgepred', 'gin_supervised_masking', 'NF'], "
            "got {}".format(exp_configure["model"])
        )

    return model


def predict(args: Dict, model, bg):
    """
    Predict the output of the models for the input batch graphs.

    Args:
        args (dict)
        model (nn.Module): The model to predict
        bg (DGLGraph): The input batch graphs

    Returns:
        Torch.Tesnor
    """
    bg = bg.to(args["device"])
    if args["edge_featurizer"] is None:
        node_feats = bg.ndata.pop("h").to(args["device"])
        return model(bg, node_feats)
    elif args["bond_featurizer_type"] == "pre_train":
        node_feats = [
            bg.ndata.pop("atomic_number").to(args["device"]),
            bg.ndata.pop("chirality_type").to(args["device"]),
        ]
        edge_feats = [
            bg.edata.pop("bond_type").to(args["device"]),
            bg.edata.pop("bond_direction_type").to(args["device"]),
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop("h").to(args["device"])
        edge_feats = bg.edata.pop("e").to(args["device"])
        return model(bg, node_feats, edge_feats)


def compute_ID_OOD(
    dataset_category: str = "TDC",
    dataset_names: str = "CYP2C19",
    split_type: str = "scaffold",
    num_of_splits: int = 10,
) -> pd.DataFrame:
    """
    compute ID and OOD metrics for the given external dataset and a trained model.
    Args:
        dataset_category (str): Dataset category
        dataset_names (str): Dataset names
        split_type (str): Split type
        num_of_splits (int): Number of splits

    Returns:
        pd.DataFrame
    
    Example:
    ```
    from alinemol.utils.utils import compute_ID_OOD
    results = compute_ID_OOD(dataset_category="TDC", dataset_names="CYP2C19", split_type="scaffold", num_of_splits=10)
    print(results)
    ```

    NOTE: NEEDS MORE WORK/POLISHING
    """
    filenames = [f"test_{i}.csv" for i in range(0, num_of_splits)]
    SPLIT_PATH = os.path.join(DATASET_PATH, dataset_category, dataset_names, "split")
    RESULTS_PATH = os.path.join(
        repo_path, "classification_results", dataset_category, dataset_names, split_type
    )

    model_names = ALL_MODELS
    ID_test_accuracy = []
    OOD_test_accuracy = []

    ID_test_roc_auc = []
    OOD_test_roc_auc = []

    ID_test_pr_auc = []
    OOD_test_pr_auc = []

    test_size = []

    for i in range(0, num_of_splits):
        for model_name in model_names:
            df = pd.read_csv(
                os.path.join(RESULTS_PATH, model_name, str(i + 1), "eval.txt"), sep=":", header=None
            )
            ID_test_accuracy.append(df.iloc[1, 1])
            ID_test_roc_auc.append(df.iloc[2, 1])
            ID_test_pr_auc.append(df.iloc[3, 1])

    for i, filename in enumerate(filenames):
        df1 = pd.read_csv(os.path.join(SPLIT_PATH, split_type, filename))
        print(df1.shape)
        for model_name in model_names:
            df = pd.read_csv(os.path.join(RESULTS_PATH, model_name, str(i + 1), "prediction.csv"))
            print(df.shape)
            OOD_test_accuracy.append(eval_acc(df1, df))
            OOD_test_roc_auc.append(eval_roc_auc(df1, df))
            OOD_test_pr_auc.append(eval_pr_auc(df1, df))
            test_size.append(df1.shape[0])

    result_df = pd.DataFrame(
        {
            "ID_test_accuracy": ID_test_accuracy,
            "OOD_test_accuracy": OOD_test_accuracy,
            "ID_test_roc_auc": ID_test_roc_auc,
            "OOD_test_roc_auc": OOD_test_roc_auc,
            "ID_test_pr_auc": ID_test_pr_auc,
            "OOD_test_pr_auc": OOD_test_pr_auc,
        }
    )

    result_df["model"] = [model_name for i in range(0, num_of_splits) for model_name in model_names]
    result_df["test_size"] = test_size
    result_df["split"] = split_type
    result_df["dataset"] = dataset_names

    return result_df


def compute_difference(results: pd.DataFrame, metrics=["accuracy", "roc_auc", "pr_auc"]) -> pd.DataFrame:
    """
    Compute the difference between ID and OOD metrics for the given external dataset and a trained model.
    Args:
        results (pd.DataFrame): Results dataframe
        metrics (list): List of metrics

    Returns:
        pd.DataFrame
    
    Example:
    ```
    import pandas as pd
    from alinemol.utils.utils import compute_difference
    results = pd.read_csv("results.csv")
    diff = compute_difference(results.copy(), metrics=["accuracy", "roc_auc", "pr_auc"])
    print(diff)
    ```
    """
    assert "model" in results.columns, "model column is missing in the results dataframe"
    diff = []
    for metric in metrics:
        assert (
            f"ID_test_{metric}" in results.columns
        ), f'{f"ID_test_{metric}"} column is missing in the results dataframe'
        assert (
            f"OOD_test_{metric}" in results.columns
        ), f'{f"OOD_test_{metric}"} column is missing in the results dataframe'
        results_copy = results.copy()
        results_copy[f"diff_{metric}"] = results_copy[f"ID_test_{metric}"] - results_copy[f"OOD_test_{metric}"]
        diff.append(results_copy.groupby("model")[f"diff_{metric}"].mean())

    diff = pd.concat(diff, axis=1)
    diff.rename_axis(None, inplace=True)
    return diff


def downsample_majority_class(df:pd.DataFrame, ratio:float=1.5) -> pd.DataFrame:
    """
    Downsample the majority class in the given dataframe
    Args:
        df (pd.DataFrame): Dataframe
        ratio (float): Ratio of majority to minority class
    
    Example:
    ```
    import pandas as pd
    from alinemol.utils.utils import downsample_majority_class
    df = pd.DataFrame({'label': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]})
    df_downsampled = downsample_majority_class(df, ratio=1.5)
    print(df_downsampled)
    ```

    Returns:
        pd.DataFrame
    """
    # Separate majority and minority classes (assuming class 1 is the minority class)
    df_majority = df[df.label==0]
    df_minority = df[df.label==1]

    assert len(df_majority) > int(len(df_minority)*ratio) , "Majority class should be greater than minority class * ratio"
     
    # Downsample majority class
    df_majority_downsampled = df_majority.sample(n=int(len(df_minority)*ratio), random_state=42)

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle the dataframe
    df_downsampled = df_downsampled.sample(frac=1).reset_index(drop=False)

    return df_downsampled


def concatanate_results(dataset_names: str, dataset_category: str= "TDC", split_types: List[str]=SPLITTING_METHODS) -> pd.DataFrame:
    """
    Concatanate the results of the model evaluation

    Note:
    - The results are stored in the classification_results folder for each dataset
    """
    results = []
    for split_type in split_types:
        results.append(pd.read_csv(os.path.join("classification_results", dataset_category, dataset_names, split_type, "results.csv")))
    results = pd.concat(results)
    results.to_csv(os.path.join("classification_results", dataset_category, dataset_names, "results.csv"), index=False)
    return results

def brier_score(y_true, y_pred_prob):
    """
    Compute the Brier score for the given true labels and predicted probabilities
    Args:
        y_true (np.array): True labels
        y_pred_prob (np.array): Predicted probabilities
    
    Example:
    ```
    import numpy as np
    from alinemol.utils.utils import brier_score
    y_true = np.array([0, 1, 0, 1])
    y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8])
    brier_score = brier_score(y_true, y_pred_prob)
    print(brier_score)
    ```
    """
    brier_score = brier_score_loss(y_true, y_pred_prob)
    return brier_score

def expected_calibration_error(y_true, y_pred_prob, n_bins=10):
    """
    Compute the expected calibration error for the given true labels and predicted probabilities
    Args:
        y_true (np.array): True labels
        y_pred_prob (np.array): Predicted probabilities
        n_bins (int): Number of bins
    
    Example:
    ```
    import numpy as np
    from alinemol.utils.utils import expected_calibration_error
    y_true = np.array([0, 1, 0, 1])
    y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8])
    ece = expected_calibration_error(y_true, y_pred_prob)
    print(ece)
    ```
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_prob >= bin_lower) & (y_pred_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_pred_prob_in_bin = np.mean(y_pred_prob[in_bin])
            ece += np.abs(avg_pred_prob_in_bin - accuracy_in_bin) * prop_in_bin

    return ece