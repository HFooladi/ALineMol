# This part of the code inspired from dgl-lifesci codabase
import os
import sys

if __name__ == "__main__":
    import pandas as pd

    from argparse import ArgumentParser
    from dgllife.utils import analyze_mols

    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKOUT_PATH = repo_path

    os.chdir(CHECKOUT_PATH)
    sys.path.insert(0, CHECKOUT_PATH)

    from alinemol.utils import mkdir_p

    parser = ArgumentParser("Dataset analysis")
    parser.add_argument("-c", "--csv_path", type=str, required=True, help="Path to a csv file for loading a dataset")
    parser.add_argument(
        "-sc", "--smiles_column", type=str, required=True, help="Header for the SMILES column in the CSV file"
    )
    parser.add_argument("-np", "--num_processes", type=int, default=1, help="Number of processes to use for analysis")
    parser.add_argument("-p", "--path", type=str, default="analysis_results", help="Path to export analysis results")
    args = vars(parser.parse_args())

    mkdir_p(args["path"])

    df = pd.read_csv(args["csv_path"])
    analyze_mols(
        smiles=df[args["smiles_column"]].tolist(),
        num_processes=args["num_processes"],
        path_to_export=args["path"],
    )
