from pathlib import Path
import pandas


from argparse import ArgumentParser
from typing import Any, Dict
from splito import MolecularWeightSplit, ScaffoldSplit, KMeansSplit, PerimeterSplit

from alinemol.utils.utils import increment_path

NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "scaffold": ScaffoldSplit,
    "kmeans": KMeansSplit,
    "molecular_weight": MolecularWeightSplit,
    "perimeter": PerimeterSplit,
}


def parse_args():
    parser = ArgumentParser("Splitting molecules into train and test sets")
    parser.add_argument(
        "-f", "--file-path", type=str, required=True, help="Path to a .csv/.txt file of SMILES strings"
    )
    parser.add_argument(
        "-sp", "--splitter", type=str, default="scaffold", help="The name of the splitter to use"
    )
    parser.add_argument("-tr", "--test-size", type=float, default=0.2, help="The size of the train set")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument("--n_splits", type=int, default=1, help="Number of splits to make")
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = parse_args()
    file_path = Path(args["file_path"])
    splitter = args["splitter"]
    test_size = args["test_size"]
    n_jobs = args["n_jobs"]
    n_splits = args["n_splits"]

    if file_path.suffix == ".csv":
        df = pandas.read_csv(file_path)
    elif file_path.suffix == ".txt":
        df = pandas.read_csv(file_path, sep="\t")
    else:
        raise ValueError("File must be a .csv or .txt file")

    split_folder = (file_path.parent / "split").mkdir(parents=True, exist_ok=True)
    split_path = (split_folder / splitter).mkdir(parents=True, exist_ok=True)

    method = NAME_TO_MODEL_CLS[splitter]
    smiles = df["smiles"].values
    hopts = {"make_generic": False}
    splitter = method(smiles, n_splits=n_splits, n_jobs=n_jobs, **hopts)

    for i, (train_ind, test_ind) in enumerate(splitter.split(smiles)):
        train = df.iloc[train_ind]
        test = df.iloc[test_ind]
        train.to_csv(increment_path(split_path / f"train_{i}.csv"), index=False)
        test.to_csv(increment_path(split_path / f"test_{i}.csv"), index=False)

        print("percentage of actives in the train set:", train["label"].sum() / train["label"].shape[0])
        print("percentage of actives in the external test set:", test["label"].sum() / test["label"].shape[0])
        print("number of molecules in the train set:", train.shape[0])
        print("number of molecules in the external test set:", test.shape[0])
