import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setting up local details:
# This should be the location of the checkout of the ALineMol repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from alinemol.utils.graph_utils import create_pyg_graphs, pairwise_graph_distances


def parse_args():
    parser = ArgumentParser("Calculate pairwise graph distances")
    parser.add_argument(
        "-sp",
        "--source_path",
        type=str,
        required=True,
        help="Path to a .csv/.txt file of SMILES strings of source file",
    )
    parser.add_argument(
        "-tp", "--target_path", type=str, default=None, help="Path to a .csv/.txt file of SMILES strings of targte file"
    )
    parser.add_argument("-op", "--output_path", type=str, required=True, help="Path to save the output file")
    parser.add_argument("--w", default=0.5, type=float, help="Layer weighting term")
    parser.add_argument("--L", default=4, type=int, help="Depth of computational tree")
    parser.add_argument("-nj", "--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    return parser.parse_args()


def main():
    args = parse_args()
    source_df = pd.read_csv(args.source_path)
    assert "smiles" in source_df.columns, 'Input file must contain a column named "smiles"'
    source_smiles = source_df["smiles"].tolist()
    source_graphs = create_pyg_graphs(source_smiles, "GCN")

    if args.target_path is not None:
        target_df = pd.read_csv(args.target_path)
        assert "smiles" in target_df.columns, 'Input file must contain a column named "smiles"'
        target_smiles = target_df["smiles"].tolist()
        target_graphs = create_pyg_graphs(target_smiles, "GCN")
        symmetric = False
    else:
        target_graphs = source_graphs
        symmetric = True

    n = len(source_graphs)
    ## break the graphs to n chunks and run pairwise_graph_distances on each chunks and save the results
    n_per_idx = 20000
    idxs = n // n_per_idx

    kwds = {"w": args.w, "L": args.L}
    if symmetric and idxs == 0:
        distances = pairwise_graph_distances(src_pyg_graphs = source_graphs, n_jobs=args.n_jobs, **kwds)
        np.save(args.output_path, distances)
    else:
        for idx in tqdm(range(idxs + 1)):
            start = n_per_idx * idx
            end = min(n_per_idx * (idx + 1), n)

            source_chunk = source_graphs[start:end]
            distances = pairwise_graph_distances(src_pyg_graphs = source_chunk, tgt_pyg_graphs=target_graphs, n_jobs=args.n_jobs, **kwds)
            np.save(args.output_path + f"_{idx}", distances)


if __name__ == "__main__":
    main()
