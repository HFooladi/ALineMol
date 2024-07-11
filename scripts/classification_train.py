# This part of the code inspired from dgl-lifesci codabase
import json
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import EarlyStopping  # Meter
from hyperopt import fmin, tpe
from torch.optim import Adam
from torch.utils.data import DataLoader

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from alinemol.hyper import init_hyper_space
from alinemol.utils import (
    collate_molgraphs,
    get_configure,
    init_featurizer,
    init_trial_path,
    load_dataset,
    load_model,
    mkdir_p,
    split_dataset,
)
from alinemol.utils.training_utils import run_a_train_epoch, run_an_eval_epoch
from alinemol.utils.logger_utils import logger


def main(args, exp_config, train_set, val_set, test_set):
    # Record settings
    exp_config.update(
        {
            "model": args["model"],
            "n_tasks": args["n_tasks"],
            "atom_featurizer_type": args["atom_featurizer_type"],
            "bond_featurizer_type": args["bond_featurizer_type"],
        }
    )
    if args["atom_featurizer_type"] != "pre_train":
        exp_config["in_node_feats"] = args["node_featurizer"].feat_size()
    if args["edge_featurizer"] is not None and args["bond_featurizer_type"] != "pre_train":
        exp_config["in_edge_feats"] = args["edge_featurizer"].feat_size()

    # Set up directory for saving results
    args = init_trial_path(args)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=exp_config["batch_size"],
        shuffle=True,
        collate_fn=collate_molgraphs,
        num_workers=args["num_workers"],
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=exp_config["batch_size"],
        collate_fn=collate_molgraphs,
        num_workers=args["num_workers"],
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=exp_config["batch_size"],
        collate_fn=collate_molgraphs,
        num_workers=args["num_workers"],
    )
    model = load_model(exp_config).to(args["device"])
    logger.info("Model architecture: {}".format(args["model"]))
    logger.info(model)
    logger.info("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    loss_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = Adam(model.parameters(), lr=exp_config["lr"], weight_decay=exp_config["weight_decay"])
    stopper = EarlyStopping(
        patience=exp_config["patience"], filename=args["trial_path"] + "/model.pth", metric=args["metric"]
    )

    for epoch in range(args["num_epochs"]):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        logger.info(
            "epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}".format(
                epoch + 1, args["num_epochs"], args["metric"], val_score, args["metric"], stopper.best_score
            )
        )

        if early_stop:
            break

    stopper.load_checkpoint(model)
    ## For validation score
    valid_score = run_an_eval_epoch(args, model, val_loader, test=False)
    logger.info("best validation {} {:.4f}".format(args["metric"], valid_score))

    ## For test score
    test_score = run_an_eval_epoch(args, model, test_loader, test=True)

    with open(args["trial_path"] + "/eval.txt", "w") as f:
        f.write("Best val {}: {}\n".format(args["metric"], stopper.best_score))
        f.write("Test {}: {}\n".format("accuracy_score", test_score[0]))
        f.write("Test {}: {}\n".format("roc_auc_score", test_score[1]))
        f.write("Test {}: {}\n".format("pr_auc_score", test_score[2]))

    # save the metrics
    reports = {
        f"{args['metric']}_val": stopper.best_score.item(),
        "acc": test_score[0],
        "roc_auc": test_score[1],
        "pr_auc": test_score[2],
        "model": args["model"],
        "test_size": len(test_set),
        "val_size": len(val_set),
        "train_size": len(train_set),
    }
    df = pd.DataFrame(reports, index=[0])
    df.to_csv(args["trial_path"] + "/metrics.csv")

    exp_config.update({"filepath": args["csv_path"]})
    logger.info("experimetns_config: {}".format(exp_config))
    with open(args["trial_path"] + "/configure.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    return args["trial_path"], stopper.best_score


def bayesian_optimization(args, train_set, val_set, test_set):
    # Run grid search
    results = []

    candidate_hypers = init_hyper_space(args["model"])

    def objective(hyperparams):
        configure = deepcopy(args)
        trial_path, val_metric = main(configure, hyperparams, train_set, val_set, test_set)

        if args["metric"] in ["roc_auc_score", "pr_auc_score"]:
            # Maximize ROCAUC is equivalent to minimize the negative of it
            val_metric_to_minimize = -1 * val_metric
        else:
            val_metric_to_minimize = val_metric

        results.append((trial_path, val_metric_to_minimize))

        return val_metric_to_minimize

    fmin(objective, candidate_hypers, algo=tpe.suggest, max_evals=args["num_evals"])
    results.sort(key=lambda tup: tup[1])
    best_trial_path, best_val_metric = results[0]

    return best_trial_path


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Multi-label Binary Classification")
    parser.add_argument(
        "-c", "--csv-path", type=str, required=True, help="Path to a csv file for loading a dataset"
    )
    parser.add_argument(
        "-sc", "--smiles-column", type=str, required=True, help="Header for the SMILES column in the CSV file"
    )
    parser.add_argument(
        "-t",
        "--task-names",
        default=None,
        type=str,
        help="Header for the tasks to model. If None, we will model "
        "all the columns except for the smiles_column in the CSV file. "
        "(default: None)",
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=["scaffold_decompose", "scaffold_smiles", "random"],
        default="scaffold_smiles",
        help="Dataset splitting method (default: scaffold_smiles). For scaffold "
        "split based on rdkit.Chem.AllChem.MurckoDecompose, "
        "use scaffold_decompose. For scaffold split based on "
        "rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles, "
        "use scaffold_smiles.",
    )
    parser.add_argument(
        "-sr",
        "--split-ratio",
        default="0.7,0.1,0.2",
        type=str,
        help="Proportion of the dataset to use for training, validation and test, " "(default: 0.7,0.1,0.2)",
    )
    parser.add_argument(
        "-me",
        "--metric",
        choices=["accuracy_score", "roc_auc_score", "pr_auc_score"],
        default="roc_auc_score",
        help="Metric for evaluation (default: roc_auc_score)",
    )
    parser.add_argument(
        "-mo",
        "--model",
        choices=[
            "GCN",
            "GAT",
            "Weave",
            "MPNN",
            "AttentiveFP",
            "gin_supervised_contextpred",
            "gin_supervised_infomax",
            "gin_supervised_edgepred",
            "gin_supervised_masking",
            "NF",
        ],
        default="GCN",
        help="Model to use (default: GCN)",
    )
    parser.add_argument(
        "-a",
        "--atom-featurizer-type",
        choices=["canonical", "attentivefp"],
        default="canonical",
        help="Featurization for atoms (default: canonical)",
    )
    parser.add_argument(
        "-b",
        "--bond-featurizer-type",
        choices=["canonical", "attentivefp"],
        default="canonical",
        help="Featurization for bonds (default: canonical)",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs allowed for training. "
        "We set a large number by default as early stopping "
        "will be performed. (default: 1000)",
    )
    parser.add_argument(
        "-nw", "--num-workers", type=int, default=4, help="Number of processes for data loading (default: 1)"
    )
    parser.add_argument(
        "-pe", "--print-every", type=int, default=20, help="Print the training progress every X mini-batches"
    )
    parser.add_argument(
        "-p",
        "--result-path",
        type=str,
        default="classification_results",
        help="Path to save training results (default: classification_results)",
    )
    parser.add_argument(
        "-ne",
        "--num-evals",
        type=int,
        default=None,
        help="Number of trials for hyperparameter search (default: None)",
    )
    args = vars(parser.parse_args())

    if torch.cuda.is_available():
        args["device"] = torch.device("cuda:0")
    else:
        args["device"] = torch.device("cpu")

    if args["task_names"] is not None:
        args["task_names"] = args["task_names"].split(",")

    args = init_featurizer(args)
    df = pd.read_csv(args["csv_path"])
    mkdir_p(args["result_path"])
    dataset = load_dataset(args, df)
    args["n_tasks"] = dataset.n_tasks
    train_set, val_set, test_set = split_dataset(args, dataset)

    if args["num_evals"] is not None:
        assert args["num_evals"] > 0, (
            "Expect the number of hyperparameter search trials to " "be greater than 0, got {:d}".format(
                args["num_evals"]
            )
        )
        print(
            "Start hyperparameter search with Bayesian " "optimization for {:d} trials".format(
                args["num_evals"]
            )
        )
        trial_path = bayesian_optimization(args, train_set, val_set, test_set)
    else:
        print("Use the manually specified hyperparameters")
        exp_config = get_configure(args["model"])
        main(args, exp_config, train_set, val_set, test_set)
