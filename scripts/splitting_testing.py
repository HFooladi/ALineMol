"""
In this script, I want to test the splitted data to make sure that they wotk as expected and pass the tests
"""

import os
import os.path as osp
import sys
import yaml
import pandas as pd
from tqdm import tqdm
import datamol as dm
from argparse import ArgumentParser

from alinemol.utils.split_utils import get_scaffold

# Setting up local details:
# This should be the location of the checkout of the ALineMol repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


with open(osp.join(DATASET_PATH, "config.yml"), "r") as f:
    config = yaml.safe_load(f)

DATSET_CATEGORY = "TDC"
SPLITTERS = config["splitting"]

datasets = config["datasets"][DATSET_CATEGORY]


def parse_options():
    parser = ArgumentParser("Testing splitting molecules into train and test sets")
    parser.add_argument("-sp", "--splitter", nargs="+", help="splitters to test", required=True, default=SPLITTERS)

    return parser.parse_args()


def main():
    # read the dataset for scaffold split (train and test set). Next calculate the scaffold for each molecule in the dataset
    # Then make sure that the scaffold doesn't appear in both the train and test set (intersection is empty)
    args = parse_options()
    splitters = args.splitter

    # SCAFFODL SPLIT
    # if split_type is in the splitters, then test the split
    split_type = "scaffold"
    if split_type in splitters:
        NUM_PASSED_TESTS = 0
        indices = 10

        for dataset in tqdm(datasets):
            N = 0
            for i in range(indices):
                train = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"train_{i}.csv")
                )
                test = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"test_{i}.csv")
                )

                train_smiles = train["smiles"].values
                test_smiles = test["smiles"].values

                train_scfs = set([get_scaffold(smi) for smi in train_smiles])
                test_scfs = [get_scaffold(smi) for smi in test_smiles]
                assert not any(test_scf in train_scfs for test_scf in test_scfs)
                N += 1

            print(f"Passed {N} tests for {split_type} split and {dataset} dataset")
            NUM_PASSED_TESTS += N

        print(f"Passed {NUM_PASSED_TESTS} tests for {split_type} split")

    # MOLECULAR WEIGHT SPLIT
    split_type = "molecular_weight"
    if split_type in splitters:
        NUM_PASSED_TESTS = 0
        indices = 10

        for dataset in tqdm(datasets):
            N = 0
            for i in range(indices):
                train = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"train_{i}.csv")
                )
                test = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"test_{i}.csv")
                )

                train_smiles = train["smiles"].values
                test_smiles = test["smiles"].values

                train_mws = [dm.descriptors.mw(dm.to_mol(smi)) for smi in train_smiles]
                assert all(dm.descriptors.mw(dm.to_mol(smi)) >= max(train_mws) for smi in test_smiles), (
                    f"Test set in {dataset} should have molecular weight greater than the train set"
                )
                N += 1

            print(f"Passed {N} tests for {split_type} split and {dataset} dataset")
            NUM_PASSED_TESTS += N

        print(f"Passed {NUM_PASSED_TESTS} tests for {split_type} split")

    # KMEANS SPLIT
    split_type = "kmeans"
    if split_type in splitters:
        NUM_PASSED_TESTS = 0
        indices = 10

        for dataset in tqdm(datasets):
            N = 0
            for i in range(indices):
                train = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"train_{i}.csv")
                )
                test = pd.read_csv(
                    osp.join(DATASET_PATH, DATSET_CATEGORY, dataset, "split", split_type, f"test_{i}.csv")
                )

                train_smiles = train["smiles"].values
                test_smiles = test["smiles"].values

                assert len(set(train_smiles).intersection(set(test_smiles))) == 0
                N += 1

            print(f"Passed {N} tests for {split_type} split and {dataset} dataset")
            NUM_PASSED_TESTS += N

        print(f"Passed {NUM_PASSED_TESTS} tests for {split_type} split")


if __name__ == "__main__":
    main()
