# This part of the code inspired from dgl-lifesci codabase

import json
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
from dgllife.data import UnlabeledSMILES
from dgllife.utils import MolToBigraph
from torch.utils.data import DataLoader
from tqdm import tqdm

from alinemol.utils import (
    collate_molgraphs_unlabeled,
    init_featurizer,
    load_model,
    mkdir_p,
    predict,
)

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


def main(args):
    mol_to_g = MolToBigraph(
        add_self_loop=True, node_featurizer=args["node_featurizer"], edge_featurizer=args["edge_featurizer"]
    )
    dataset = UnlabeledSMILES(args["smiles"], mol_to_graph=mol_to_g)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        collate_fn=collate_molgraphs_unlabeled,
        num_workers=args["num_workers"],
    )
    model = load_model(args).to(args["device"])
    checkpoint = torch.load(args["train_result_path"] + "/model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    smiles_list = []
    predictions = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(tqdm(dataloader, desc="Iteration")):
            batch_smiles, bg = batch_data
            smiles_list.extend(batch_smiles)
            batch_logit = predict(args, model, bg)
            batch_pred = torch.sigmoid(batch_logit)
            if not args["soft_classification"]:
                batch_pred = (batch_pred >= 0.5).float()
            predictions.append(batch_pred.detach().cpu())

    predictions = torch.cat(predictions, dim=0)

    output_data = {"canonical_smiles": smiles_list}
    if args["task_names"] is None:
        args["task_names"] = ["task_{:d}".format(t) for t in range(1, args["n_tasks"] + 1)]
    else:
        args["task_names"] = args["task_names"].split(",")
    for task_id, task_name in enumerate(args["task_names"]):
        output_data[task_name] = predictions[:, task_id]
        # print(roc_auc_score(predictions[:, task_id], predictions[:, task_id])

    df = pd.DataFrame(output_data)
    # args = init_inference_trial_path(args)
    args["trial_path"] = args["inference_result_path"]
    df.to_csv(args["trial_path"] + "/prediction.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser("Inference for Multi-label Binary Classification")
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
    args = vars(parser.parse_args())

    # Load configuration
    with open(args["train_result_path"] + "/configure.json", "r") as f:
        args.update(json.load(f))

    if torch.cuda.is_available():
        args["device"] = torch.device("cuda:0")
    else:
        args["device"] = torch.device("cpu")

    if args["file_path"].endswith(".csv") or args["file_path"].endswith(".csv.gz"):
        import pandas

        df = pandas.read_csv(args["file_path"])
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
    args = init_featurizer(args)

    # Handle directories
    mkdir_p(args["inference_result_path"])
    assert os.path.exists(args["train_result_path"]), "The path to the saved training results does not exist."

    main(args)
