"""
Script for calculating pairwise graph distances between molecular graphs.

This script processes molecular SMILES strings from source and target files,
converts them to PyTorch Geometric graphs, and calculates pairwise distances
between them. The distances can be calculated either between two different sets
of molecules or within a single set (symmetric case).

The script supports parallel processing and can handle large datasets by
processing them in chunks.
"""

from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from alinemol.utils.graph_utils import create_pyg_graphs, pairwise_graph_distances


def parse_args() -> ArgumentParser:
    """
    Parse command line arguments for the distance calculation script.

    Returns:
        ArgumentParser: Parsed command line arguments containing:
            - source_path: Path to source SMILES file
            - target_path: Optional path to target SMILES file
            - output_path: Path to save distance matrix
            - w: Layer weighting term (default: 0.5)
            - L: Depth of computational tree (default: 4)
            - n_jobs: Number of parallel jobs (default: 1)
    """
    parser = ArgumentParser("Calculate pairwise graph distances")
    parser.add_argument(
        "-sp",
        "--source_path",
        type=str,
        required=True,
        help="Path to a .csv/.txt file of SMILES strings of source file",
    )
    parser.add_argument(
        "-tp", "--target_path", type=str, default=None, help="Path to a .csv/.txt file of SMILES strings of target file"
    )
    parser.add_argument("-op", "--output_path", type=str, required=True, help="Path to save the output distance matrix")
    parser.add_argument("--w", default=0.5, type=float, help="Layer weighting term for distance calculation")
    parser.add_argument("--L", default=4, type=int, help="Depth of computational tree for graph comparison")
    parser.add_argument("-nj", "--n_jobs", type=int, default=1, help="Number of parallel jobs for distance calculation")
    return parser.parse_args()


def load_and_validate_smiles(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load SMILES strings from a file and validate the data format.

    Args:
        file_path (str): Path to the input file containing SMILES strings

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame containing the data and list of SMILES strings

    Raises:
        AssertionError: If the input file doesn't contain a 'smiles' column
    """
    df = pd.read_csv(file_path)
    assert "smiles" in df.columns, 'Input file must contain a column named "smiles"'
    return df, df["smiles"].tolist()


def main() -> None:
    """
    Main function to calculate pairwise graph distances between molecular graphs.

    The function:
    1. Loads source (and optionally target) SMILES strings
    2. Converts them to PyTorch Geometric graphs
    3. Calculates pairwise distances in chunks if necessary
    4. Saves the resulting distance matrix
    """
    # Parse command line arguments
    args = parse_args()

    # Load and process source data
    source_df, source_smiles = load_and_validate_smiles(args.source_path)
    source_graphs = create_pyg_graphs(source_smiles, "GCN")

    # Handle target data (if provided)
    if args.target_path is not None:
        target_df, target_smiles = load_and_validate_smiles(args.target_path)
        target_graphs = create_pyg_graphs(target_smiles, "GCN")
        symmetric = False
    else:
        target_graphs = source_graphs
        symmetric = True

    # Calculate total number of graphs and chunk size
    n_graphs = len(source_graphs)
    chunk_size = 20000  # Process 20,000 graphs at a time
    n_chunks = n_graphs // chunk_size

    # Set up distance calculation parameters
    distance_params = {"w": args.w, "L": args.L}

    # Calculate distances
    if symmetric and n_chunks == 0:
        # For small symmetric cases, calculate all distances at once
        distances = pairwise_graph_distances(src_pyg_graphs=source_graphs, n_jobs=args.n_jobs, **distance_params)
        np.save(args.output_path, distances)
    else:
        # For large or asymmetric cases, process in chunks
        for chunk_idx in tqdm(range(n_chunks + 1)):
            # Calculate chunk boundaries
            start_idx = chunk_size * chunk_idx
            end_idx = min(chunk_size * (chunk_idx + 1), n_graphs)

            # Process current chunk
            source_chunk = source_graphs[start_idx:end_idx]
            distances = pairwise_graph_distances(
                src_pyg_graphs=source_chunk, tgt_pyg_graphs=target_graphs, n_jobs=args.n_jobs, **distance_params
            )

            # Save chunk results
            np.save(f"{args.output_path}_{chunk_idx}", distances)


if __name__ == "__main__":
    main()
