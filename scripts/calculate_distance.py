"""
Script for calculating pairwise graph distances between molecular graphs.

This script processes molecular SMILES strings from source and target files,
converts them to PyTorch Geometric graphs, and calculates pairwise distances
between them. The distances can be calculated either between two different sets
of molecules or within a single set (symmetric case).

The script supports parallel processing and can handle large datasets by
processing them in chunks.
"""

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from alinemol.utils.graph_utils import create_pyg_graphs, pairwise_graph_distances
from alinemol.utils.logger_utils import Logger

# Constants
DEFAULT_CHUNK_SIZE = 20000
DEFAULT_W_PARAM = 0.5
DEFAULT_L_PARAM = 4
DEFAULT_N_JOBS = 1
REQUIRED_SMILES_COLUMN = "smiles"


def parse_args() -> Namespace:
    """
    Parse command line arguments for the distance calculation script.

    Returns:
        Namespace: Parsed command line arguments containing:
            - source_path: Path to source SMILES file
            - target_path: Optional path to target SMILES file
            - output_path: Path to save distance matrix
            - w: Layer weighting term (default: 0.5)
            - L: Depth of computational tree (default: 4)
            - n_jobs: Number of parallel jobs (default: 1)
            - chunk_size: Size of chunks for processing (default: 20000)
            - verbose: Enable verbose logging

    Raises:
        SystemExit: If required arguments are missing or invalid
    """
    parser = ArgumentParser(
        prog="calculate_distance",
        description="Calculate pairwise graph distances between molecular graphs",
        epilog="Example: python calculate_distance.py -sp source.csv -op distances.npy",
    )

    # Required arguments
    parser.add_argument(
        "-sp",
        "--source_path",
        type=str,
        required=True,
        help="Path to a .csv/.txt file containing SMILES strings (must have 'smiles' column)",
        metavar="FILE",
    )
    parser.add_argument(
        "-op",
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output distance matrix (.npy format)",
        metavar="FILE",
    )

    # Optional arguments
    parser.add_argument(
        "-tp",
        "--target_path",
        type=str,
        default=None,
        help="Path to a .csv/.txt file containing target SMILES strings (if different from source)",
        metavar="FILE",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=DEFAULT_W_PARAM,
        help=f"Layer weighting term for distance calculation (default: {DEFAULT_W_PARAM})",
        metavar="FLOAT",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=DEFAULT_L_PARAM,
        help=f"Depth of computational tree for graph comparison (default: {DEFAULT_L_PARAM})",
        metavar="INT",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Number of parallel jobs for distance calculation (default: {DEFAULT_N_JOBS})",
        metavar="INT",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of chunks for processing large datasets (default: {DEFAULT_CHUNK_SIZE})",
        metavar="INT",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def validate_input_arguments(args: Namespace) -> None:
    """
    Validate input arguments for consistency and correctness.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
        FileNotFoundError: If input files don't exist
    """
    # Validate file paths exist
    source_path = Path(args.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {args.source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {args.source_path}")

    if args.target_path:
        target_path = Path(args.target_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {args.target_path}")
        if not target_path.is_file():
            raise ValueError(f"Target path is not a file: {args.target_path}")

    # Validate output directory exists
    output_path = Path(args.output_path)
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path.parent}")

    # Validate numeric parameters
    if args.w < 0 or args.w > 1:
        raise ValueError(f"Parameter 'w' must be between 0 and 1, got: {args.w}")
    if args.L < 1:
        raise ValueError(f"Parameter 'L' must be positive, got: {args.L}")
    if args.n_jobs < 1:
        raise ValueError(f"Number of jobs must be positive, got: {args.n_jobs}")
    if args.chunk_size < 1:
        raise ValueError(f"Chunk size must be positive, got: {args.chunk_size}")


def load_and_validate_smiles(file_path: str, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load SMILES strings from a file and validate the data format.

    Args:
        file_path: Path to the input file containing SMILES strings
        logger: Optional logger for progress reporting

    Returns:
        Tuple containing DataFrame with the data and list of SMILES strings

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is invalid or missing required columns
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    try:
        if logger:
            logger.info(f"Loading SMILES data from: {file_path}")

        # Validate file extension
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError(f"Unsupported file format: {file_path_obj.suffix}. Only .csv and .txt are supported.")

        # Load the data
        if file_path_obj.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:  # .txt file
            df = pd.read_csv(file_path, delimiter="\t")

        # Validate data structure
        if df.empty:
            raise ValueError(f"Input file is empty: {file_path}")

        if REQUIRED_SMILES_COLUMN not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(
                f"Input file must contain a column named '{REQUIRED_SMILES_COLUMN}'. "
                f"Available columns: {available_cols}"
            )

        # Remove rows with missing SMILES
        initial_count = len(df)
        df = df.dropna(subset=[REQUIRED_SMILES_COLUMN])
        final_count = len(df)

        if final_count == 0:
            raise ValueError(f"No valid SMILES strings found in {file_path}")

        if initial_count != final_count and logger:
            logger.warning(f"Removed {initial_count - final_count} rows with missing SMILES")

        smiles_list = df[REQUIRED_SMILES_COLUMN].tolist()

        if logger:
            logger.info(f"Successfully loaded {len(smiles_list)} SMILES strings")

        return df, smiles_list

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse input file {file_path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading SMILES data from {file_path}: {str(e)}")


def calculate_distance_matrix(
    source_graphs: List, target_graphs: Optional[List], args: Namespace, logger: logging.Logger
) -> None:
    """
    Calculate and save pairwise graph distance matrix.

    Args:
        source_graphs: List of source molecular graphs
        target_graphs: Optional list of target molecular graphs
        args: Command line arguments
        logger: Logger instance
    """
    symmetric = target_graphs is None
    if symmetric:
        target_graphs = source_graphs
        logger.info("Calculating symmetric distance matrix")
    else:
        logger.info("Calculating asymmetric distance matrix")

    n_source_graphs = len(source_graphs)
    n_target_graphs = len(target_graphs) if target_graphs is not None else len(source_graphs)
    n_chunks = n_source_graphs // args.chunk_size

    logger.info(f"Source graphs: {n_source_graphs}, Target graphs: {n_target_graphs}")
    logger.info(f"Chunk size: {args.chunk_size}, Number of chunks: {n_chunks + 1}")

    # Set up distance calculation parameters
    distance_params = {"w": args.w, "L": args.L}
    logger.info(f"Distance parameters: w={args.w}, L={args.L}, n_jobs={args.n_jobs}")

    try:
        if symmetric and n_chunks == 0:
            # Small symmetric case - calculate all distances at once
            logger.info("Processing small dataset in single batch")
            distances = pairwise_graph_distances(src_pyg_graphs=source_graphs, n_jobs=args.n_jobs, **distance_params)

            logger.info(f"Saving distance matrix to: {args.output_path}")
            np.save(args.output_path, distances)
            logger.info(f"Successfully saved distance matrix of shape: {distances.shape}")

        else:
            # Large or asymmetric case - process in chunks
            logger.info("Processing dataset in chunks")
            total_chunks = n_chunks + 1 if n_source_graphs % args.chunk_size > 0 else n_chunks

            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for chunk_idx in range(total_chunks):
                    # Calculate chunk boundaries
                    start_idx = args.chunk_size * chunk_idx
                    end_idx = min(args.chunk_size * (chunk_idx + 1), n_source_graphs)

                    if start_idx >= n_source_graphs:
                        break

                    logger.debug(f"Processing chunk {chunk_idx + 1}/{total_chunks}: indices {start_idx}-{end_idx}")

                    # Process current chunk
                    source_chunk = source_graphs[start_idx:end_idx]
                    distances = pairwise_graph_distances(
                        src_pyg_graphs=source_chunk, tgt_pyg_graphs=target_graphs, n_jobs=args.n_jobs, **distance_params
                    )

                    # Save chunk results
                    chunk_output_path = f"{args.output_path}_chunk_{chunk_idx:04d}.npy"
                    np.save(chunk_output_path, distances)
                    logger.debug(f"Saved chunk {chunk_idx} to: {chunk_output_path}")

                    pbar.update(1)

            logger.info(f"Successfully processed all chunks. Output files: {args.output_path}_chunk_*.npy")

    except Exception as e:
        logger.error(f"Error during distance calculation: {str(e)}")
        raise


def main() -> int:
    """
    Main function to calculate pairwise graph distances between molecular graphs.

    The function:
    1. Parses and validates command line arguments
    2. Sets up logging
    3. Loads source (and optionally target) SMILES strings
    4. Converts them to PyTorch Geometric graphs
    5. Calculates pairwise distances in chunks if necessary
    6. Saves the resulting distance matrix

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse and validate arguments
        args = parse_args()
        logger_instance = Logger(__name__, level=logging.DEBUG if args.verbose else logging.INFO)
        logger = logger_instance.get_logger()

        logger.info("Starting pairwise graph distance calculation")
        logger.info(f"Source file: {args.source_path}")
        logger.info(f"Target file: {args.target_path or 'None (symmetric calculation)'}")
        logger.info(f"Output path: {args.output_path}")

        # Validate input arguments
        validate_input_arguments(args)
        logger.info("Input arguments validated successfully")

        # Load and process source data
        logger.info("Loading source data...")
        source_df, source_smiles = load_and_validate_smiles(args.source_path, logger)

        logger.info("Converting source SMILES to graphs...")
        source_graphs = create_pyg_graphs(source_smiles, "GCN")
        logger.info(f"Created {len(source_graphs)} source graphs")

        # Handle target data (if provided)
        target_graphs = None
        if args.target_path is not None:
            logger.info("Loading target data...")
            target_df, target_smiles = load_and_validate_smiles(args.target_path, logger)

            logger.info("Converting target SMILES to graphs...")
            target_graphs = create_pyg_graphs(target_smiles, "GCN")
            logger.info(f"Created {len(target_graphs)} target graphs")

        # Calculate distances
        calculate_distance_matrix(source_graphs, target_graphs, args, logger)

        logger.info("Distance calculation completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        if "logger" in locals():
            logger.error(f"Fatal error: {str(e)}")
        else:
            print(f"Fatal error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
