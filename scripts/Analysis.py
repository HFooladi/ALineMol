# This part of the code inspired from dgl-lifesci codabase

if __name__ == "__main__":
    import pandas

    from argparse import ArgumentParser
    from dgllife.utils import analyze_mols

    from alinemol.utils import mkdir_p

    parser = ArgumentParser("Dataset analysis")
    parser.add_argument(
        "-c", "--csv-path", type=str, required=True, help="Path to a csv file for loading a dataset"
    )
    parser.add_argument(
        "-sc", "--smiles-column", type=str, required=True, help="Header for the SMILES column in the CSV file"
    )
    parser.add_argument(
        "-np", "--num-processes", type=int, default=1, help="Number of processes to use for analysis"
    )
    parser.add_argument(
        "-p", "--path", type=str, default="analysis_results", help="Path to export analysis results"
    )
    args = vars(parser.parse_args())

    mkdir_p(args["path"])

    df = pandas.read_csv(args["csv_path"])
    analyze_mols(
        smiles=df[args["smiles_column"]].tolist(),
        num_processes=args["num_processes"],
        path_to_export=args["path"],
    )
