# This part of the code inspired from dgl-lifesci codabase
# This file is used to perform inference on the trained model with the test data

# Receive the model and the test data for prediction
# The model is loaded from the saved model file
# The test data is loaded from the test data file
# The model is used to predict the labels (hard or soft) for the test data
# The predictions are saved to a CSV file.

# import the necessary libraries
import json
import os
import sys
from argparse import ArgumentParser
from typing import Dict

import datamol as dm  # for molecular operations
import joblib
import numpy as np  # for numerical operations
import pandas as pd  # for data operations

from alinemol.utils import mkdir_p

# set the path to the repository
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


def main(args):
    # Load the model
    model = joblib.load(args["train_result_path"] + "/model.joblib")

    # Load the data (SMILES strings)
    smiles_list = args["smiles"]
    # Convert SMILES to fingerprints
    fps = np.array([dm.to_fp(dm.to_mol(smi)) for smi in smiles_list])

    # Perform inference
    if args["soft_classification"]:
        predictions = model.predict_proba(fps)[:, 1]
    else:
        predictions = model.predict(fps)

    output_data = {"canonical_smiles": smiles_list}
    if args["task_names"] is None:
        args["task_names"] = ["task_{:d}".format(t) for t in range(1, args["n_tasks"] + 1)]
    else:
        args["task_names"] = args["task_names"].split(",")
    ## THIS LINE IS REDUNDANT AT THIS POINT
    for task_id, task_name in enumerate(args["task_names"]):
        output_data[task_name] = predictions[:]
        # print(roc_auc_score(predictions[:, task_id], predictions[:, task_id])

    df = pd.DataFrame(output_data)
    # args = init_inference_trial_path(args)
    # Save the predictions
    args["trial_path"] = args["inference_result_path"]
    df.to_csv(args["trial_path"] + "/prediction.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser("Inference for Binary Classification with ML Models")
    parser.add_argument("-f", "--file_path", type=str, required=True, help="Path to a .csv/.txt file of SMILES strings")
    parser.add_argument(
        "-sc",
        "--smiles_column",
        type=str,
        help="Header for the SMILES column in the CSV file, can be "
        "omitted if the input file is a .txt file or the .csv "
        "file only has one column of SMILES strings",
    )
    parser.add_argument(
        "-tp",
        "--train_result_path",
        type=str,
        default="classification_results",
        help="Path to the saved training results, which will be used for "
        "loading the trained model and related configurations",
    )
    parser.add_argument(
        "-ip",
        "--inference_result_path",
        type=str,
        default="classification_inference_results",
        help="Path to save the inference results",
    )
    parser.add_argument(
        "-t",
        "--task_names",
        default=None,
        type=str,
        help="Task names for saving model predictions in the CSV file to output, "
        "which should be the same as the ones used for training. If not "
        "specified, we will simply use task1, task2, ...",
    )
    parser.add_argument(
        "-s",
        "--soft_classification",
        action="store_true",
        default=False,
        help="By default we will perform hard classification with binary labels. "
        "This flag allows performing soft classification instead.",
    )
    parser.add_argument(
        "-nw", "--num_workers", type=int, default=4, help="Number of processes for data loading (default: 1)"
    )
    args: Dict = vars(parser.parse_args())

    # Load configuration
    with open(args["train_result_path"] + "/configure.json", "r") as f:
        args.update(json.load(f))

    if args["file_path"].endswith(".csv") or args["file_path"].endswith(".csv.gz"):
        df = pd.read_csv(args["file_path"])
        if args["smiles_column"] is not None:
            smiles = df[args["smiles_column"]].tolist()
        else:
            assert len(df.columns) == 1, (
                "The CSV file has more than 1 columns and -sc (smiles-column) needs to be specified."
            )
            smiles = df[df.columns[0]].tolist()
    elif args["file_path"].endswith(".txt"):
        from dgllife.utils import load_smiles_from_txt

        smiles = load_smiles_from_txt(args["file_path"])
    else:
        raise ValueError("Expect the input data file to be a .csv or a .txt file, got {}".format(args["file_path"]))
    args["smiles"] = smiles

    # Handle directories
    mkdir_p(args["inference_result_path"])  # create the directory if it does not exist
    assert os.path.exists(args["train_result_path"]), "The path to the saved training results does not exist."

    main(args)
