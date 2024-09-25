import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm


from argparse import ArgumentParser
from typing import Any, Dict

# Setting up local details:
# This should be the location of the checkout of the ALineMol repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

from alinemol.utils.graph_utils import pairwise_graph_distances, create_pyg_graphs

def parse_args():
    parser = ArgumentParser("Calculate pairwise graph distances")
    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="Path to a .csv/.txt file of SMILES strings"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to save the output file"
    )
    parser.add_argument('--w', default=0.5, type=float, help='Layer weighting term')
    parser.add_argument('--L', default=4, type=int, help='Depth of computational tree')
    parser.add_argument('-nj', '--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.file_path)
    assert 'smiles' in df.columns, 'Input file must contain a column named "smiles"'
    smiles = df['smiles'].tolist()
    graphs = create_pyg_graphs(smiles, "GCN")

    n = len(graphs)
    ## break the graphs to n chunks and run pairwise_graph_distances on each chunks and save the results
    n_per_idx = 5000
    idxs = n // n_per_idx
    for idx in tqdm(range(idxs)):
        start = n_per_idx * idx
        end = min(n_per_idx * (idx + 1), n)

        graphs_chunk = graphs[start:end]
        kwds = {'w': args.w, 'L': args.L}
        distances = pairwise_graph_distances(graphs_chunk, graphs, n_jobs=args.n_jobs, **kwds)
        np.save(args.output_path+f'_{idx}', distances)


if __name__ == "__main__":
    main()

