import os
import sys
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm


from argparse import ArgumentParser
from typing import Any, Dict
from splito import MolecularWeightSplit, ScaffoldSplit, KMeansSplit, PerimeterSplit, MaxDissimilaritySplit

# Setting up local details:
# This should be the location of the checkout of the ALineMol repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from alinemol.splitters.splits import MolecularLogPSplit

from alinemol.utils.utils import increment_path
from alinemol.splitters.splitting_configures import (
    MolecularWeightSplitConfig,
    ScaffoldSplitConfig,
    KMeansSplitConfig,
    PerimeterSplitConfig,
    MaxDissimilaritySplitConfig,
    MolecularWeightReverseSplitConfig,
    MolecularLogPSplitConfig,
)
from alinemol.utils.logger_utils import logger

NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "scaffold": ScaffoldSplit,
    "kmeans": KMeansSplit,
    "molecular_weight": MolecularWeightSplit,
    "perimeter": PerimeterSplit,
    "max_dissimilarity": MaxDissimilaritySplit,
    "molecular_weight_reverse": MolecularWeightSplit,
    "molecular_logp": MolecularLogPSplit,
}


NAME_TO_MODEL_CONFIG: Dict[str, Any] = {
    "scaffold": ScaffoldSplitConfig,
    "kmeans": KMeansSplitConfig,
    "molecular_weight": MolecularWeightSplitConfig,
    "perimeter": PerimeterSplitConfig,
    "max_dissimilarity": MaxDissimilaritySplitConfig,
    "molecular_weight_reverse": MolecularWeightReverseSplitConfig,
    "molecular_logp": MolecularLogPSplitConfig,
}


def parse_args():
    parser = ArgumentParser("Splitting molecules into train and test sets")
    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="Path to a .csv/.txt file of SMILES strings"
    )
    parser.add_argument(
        "-sp", "--splitter", type=str, default="scaffold", help="The name of the splitter to use"
    )
    parser.add_argument("-te", "--test_size", type=float, default=0.2, help="The size of the test set")
    parser.add_argument("-nj", "--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel")
    parser.add_argument(
        "-ns",
        "--n_splits",
        type=int,
        default=10,
        help="Number of splits to make (Reapeating the splitting process)",
    )
    parser.add_argument(
        "-to", "--tolerance", type=float, default=0.1, help="Tolerance for the splitting process"
    )
    parser.add_argument(
        "-sv",
        "--save",
        action="store_true",
        help="Save the final results of the splitting process",
    )

    
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    config = {}
    args = parse_args()
    file_path = Path(args["file_path"])
    splitter = args["splitter"]
    test_size = args["test_size"]
    n_jobs = args["n_jobs"]
    n_splits = args["n_splits"]
    tol = args["tolerance"]
    save = args["save"]

    internal_n_splits = 100  # Number of splits to make for test set
    config.update(args)

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".txt":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError("File must be a .csv or .txt file")

    split_folder = file_path.parent / "split"
    split_folder.mkdir(parents=True, exist_ok=True)
    split_path = file_path.parent / "split" / splitter
    split_path.mkdir(parents=True, exist_ok=True)

    method = NAME_TO_MODEL_CLS[splitter]
    smiles = df["smiles"].values
    hopts = NAME_TO_MODEL_CONFIG[splitter]
    if method in [KMeansSplit, MaxDissimilaritySplit, PerimeterSplit]:
        splitter = method(n_splits=internal_n_splits, test_size=test_size, random_state=42, n_jobs=n_jobs, **hopts)
    elif method in  [MolecularWeightSplit, MolecularLogPSplit]:
        splitter = method(
            smiles=smiles, n_splits=internal_n_splits, test_size=test_size, random_state=42, **hopts
        )
    else:
        splitter = method(
            smiles=smiles,
            n_splits=internal_n_splits,
            n_jobs=n_jobs,
            test_size=test_size,
            random_state=42,
            **hopts,
        )

    logger.info(file_path)
    logger.info(splitter)
    logger.info(hopts)

    """
    for i, (train_ind, test_ind) in enumerate(splitter.split(smiles)):
        train = df.iloc[train_ind]
        test = df.iloc[test_ind]
        train.to_csv(increment_path(split_path / f"train_{i}.csv"), index=False)
        test.to_csv(increment_path(split_path / f"test_{i}.csv"), index=False)

        logger.info(
            "percentage of actives in the train set: {}".format(
                train["label"].sum() / train["label"].shape[0]
            )
        )
        logger.info(
            "percentage of actives in the external test set: {}".format(
                test["label"].sum() / test["label"].shape[0]
            )
        )
        logger.info("number of molecules in the train set: {}".format(train.shape[0]))
        logger.info("number of molecules in the external test set: {}".format(test.shape[0]))

        config["train_size_" + str(i)] = train.shape[0]
        config["test_size_" + str(i)] = test.shape[0]
        config["train_actives_" + str(i)] = int(train["label"].sum())
        config["test_actives_" + str(i)] = int(test["label"].sum())
        config["train_actives_percentage_" + str(i)] = train["label"].sum() / train["label"].shape[0]
        config["test_actives_percentage_" + str(i)] = test["label"].sum() / test["label"].shape[0]
    """
    i = 0
    for train_ind, test_ind in tqdm(splitter.split(smiles)):
        train = df.iloc[train_ind]
        test = df.iloc[test_ind]
        config["train_size_" + str(i)] = train.shape[0]
        config["test_size_" + str(i)] = test.shape[0]
        config["train_actives_" + str(i)] = int(train["label"].sum())
        config["test_actives_" + str(i)] = int(test["label"].sum())
        config["train_actives_percentage_" + str(i)] = train["label"].sum() / train["label"].shape[0]
        config["test_actives_percentage_" + str(i)] = test["label"].sum() / test["label"].shape[0]

        if (
            config["train_actives_percentage_" + str(i)] - config["test_actives_percentage_" + str(i)] < tol
            and config["train_actives_percentage_" + str(i)] - config["test_actives_percentage_" + str(i)]
            > -tol
        ):
            if save:
                train.to_csv(increment_path(split_path / f"train_{i}.csv"), index=False)
                test.to_csv(increment_path(split_path / f"test_{i}.csv"), index=False)

            logger.info(
                "percentage of actives in the train set: {}".format(
                    train["label"].sum() / train["label"].shape[0]
                )
            )
            logger.info(
                "percentage of actives in the external test set: {}".format(
                    test["label"].sum() / test["label"].shape[0]
                )
            )
            logger.info("number of molecules in the train set: {}".format(train.shape[0]))
            logger.info("number of molecules in the external test set: {}".format(test.shape[0]))
            i += 1
        else:
            continue

        if i == n_splits:
            break

    if save:
        with open(split_path / "config.json", "w") as f:
            json.dump(config, f)
