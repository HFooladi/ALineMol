#!/usr/bin/env python3
"""
Molecular Splitting Script

This script provides a command-line interface for splitting molecular datasets into
train and test sets using various splitting strategies. It supports multiple splitting
methods including scaffold-based, clustering-based, and property-based approaches.

The script is designed to handle molecular data in SMILES format and generates
reproducible train/test splits for evaluating machine learning models on
out-of-distribution scenarios.

Usage:
    python scripts/splitting.py -f data/molecules.csv -sp scaffold --save

Author: ALineMol Team
"""

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from splito import KMeansSplit, MaxDissimilaritySplit, MolecularWeightSplit, PerimeterSplit, ScaffoldSplit
from tqdm import tqdm

# Import ALineMol modules
from alinemol.splitters.splits import MolecularLogPSplit, RandomSplit
from alinemol.splitters.umap_split import UMAPSplit
from alinemol.splitters.lohi.hi import HiSplit
from alinemol.splitters.splitting_configures import (
    KMeansSplitConfig,
    MaxDissimilaritySplitConfig,
    MolecularLogPSplitConfig,
    MolecularWeightReverseSplitConfig,
    MolecularWeightSplitConfig,
    PerimeterSplitConfig,
    ScaffoldSplitConfig,
    ScaffoldSplitGenericConfig,
    RandomSplitConfig,
    UMapSplitConfig,
    HiSplitConfig,
)
from alinemol.utils.logger_utils import Logger
from alinemol.utils.utils import increment_path

# Repository path configuration
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = REPO_PATH

# Configure Python path for local imports
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


logger_instance = Logger(__name__, level=logging.INFO)
logger = logger_instance.get_logger()

# Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_SPLITS = 10
DEFAULT_N_JOBS = -1
DEFAULT_TOLERANCE = 0.1
DEFAULT_RANDOM_STATE = 42
INTERNAL_N_SPLITS = 100

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".txt"}

# Dictionary mapping splitter names to their corresponding classes
SPLITTER_CLASSES: Dict[str, Any] = {
    "scaffold": ScaffoldSplit,
    "scaffold_generic": ScaffoldSplit,
    "kmeans": KMeansSplit,
    "molecular_weight": MolecularWeightSplit,
    "molecular_weight_reverse": MolecularWeightSplit,
    "perimeter": PerimeterSplit,
    "max_dissimilarity": MaxDissimilaritySplit,
    "molecular_logp": MolecularLogPSplit,
    "random": RandomSplit,
    "umap": UMAPSplit,
    "hi": HiSplit,
}

# Dictionary mapping splitter names to their configuration classes
SPLITTER_CONFIGS: Dict[str, Any] = {
    "scaffold": ScaffoldSplitConfig,
    "scaffold_generic": ScaffoldSplitGenericConfig,
    "kmeans": KMeansSplitConfig,
    "molecular_weight": MolecularWeightSplitConfig,
    "molecular_weight_reverse": MolecularWeightReverseSplitConfig,
    "perimeter": PerimeterSplitConfig,
    "max_dissimilarity": MaxDissimilaritySplitConfig,
    "molecular_logp": MolecularLogPSplitConfig,
    "random": RandomSplitConfig,
    "umap": UMapSplitConfig,
    "hi": HiSplitConfig,
}

# Splitter categories for different initialization patterns
SIMPLE_SPLITTERS = {RandomSplit, KMeansSplit, MaxDissimilaritySplit, PerimeterSplit, UMAPSplit}
SMILES_DEPENDENT_SPLITTERS = {MolecularWeightSplit, MolecularLogPSplit}
SPECIAL_SPLITTERS = {HiSplit}


class MolecularSplitter:
    """
    A comprehensive molecular dataset splitting utility.

    This class provides methods for splitting molecular datasets using various
    strategies including scaffold-based, clustering-based, and property-based
    approaches. It handles data loading, validation, splitting, and result saving.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MolecularSplitter with configuration parameters.

        Args:
            config: Dictionary containing splitting configuration parameters
        """
        self.config = config
        self.file_path = Path(config["file_path"])
        self.splitter_name = config["splitter"]
        self.test_size = config["test_size"]
        self.n_jobs = config["n_jobs"]
        self.n_splits = config["n_splits"]
        self.tolerance = config["tolerance"]
        self.save = config["save"]

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        if self.file_path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension. Supported: {SUPPORTED_EXTENSIONS}")

        if self.splitter_name not in SPLITTER_CLASSES:
            raise ValueError(f"Unknown splitter: {self.splitter_name}. Available: {list(SPLITTER_CLASSES.keys())}")

        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got: {self.test_size}")

        if self.n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got: {self.n_splits}")

    def load_data(self) -> pd.DataFrame:
        """
        Load molecular data from the input file.

        Returns:
            DataFrame containing the molecular data

        Raises:
            ValueError: If the file format is unsupported or data is invalid
        """
        try:
            if self.file_path.suffix == ".csv":
                df = pd.read_csv(self.file_path)
            elif self.file_path.suffix == ".txt":
                df = pd.read_csv(self.file_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

            # Validate required columns
            required_columns = ["smiles"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Loaded dataset with {len(df)} molecules from {self.file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data from {self.file_path}: {str(e)}")
            raise

    def _create_splitter(self, smiles: List[str]) -> Any:
        """
        Create and configure the appropriate splitter instance.

        Args:
            smiles: List of SMILES strings

        Returns:
            Configured splitter instance
        """
        method_class = SPLITTER_CLASSES[self.splitter_name]
        config_class = SPLITTER_CONFIGS[self.splitter_name]

        # Get configuration options
        hopts = config_class() if callable(config_class) else config_class

        # Initialize splitter based on its type
        if method_class in SIMPLE_SPLITTERS:
            splitter = method_class(
                n_splits=INTERNAL_N_SPLITS,
                test_size=self.test_size,
                random_state=DEFAULT_RANDOM_STATE,
                n_jobs=self.n_jobs,
                **hopts,
            )
        elif method_class in SMILES_DEPENDENT_SPLITTERS:
            splitter = method_class(
                smiles=smiles,
                n_splits=INTERNAL_N_SPLITS,
                test_size=self.test_size,
                random_state=DEFAULT_RANDOM_STATE,
                **hopts,
            )
        elif method_class in SPECIAL_SPLITTERS:
            splitter = method_class(**hopts)
        else:
            # Default initialization pattern
            splitter = method_class(
                smiles=smiles,
                n_splits=INTERNAL_N_SPLITS,
                n_jobs=self.n_jobs,
                test_size=self.test_size,
                random_state=DEFAULT_RANDOM_STATE,
                **hopts,
            )

        logger.info(f"Initialized {self.splitter_name} splitter with configuration: {hopts}")
        return splitter

    def _setup_output_directories(self) -> Path:
        """
        Create necessary output directories for saving splits.

        Returns:
            Path to the splitter-specific output directory
        """
        split_folder = self.file_path.parent / "split"
        split_folder.mkdir(parents=True, exist_ok=True)

        splitter_path = split_folder / self.splitter_name
        splitter_path.mkdir(parents=True, exist_ok=True)

        return splitter_path

    def _is_split_acceptable(self, train_active_pct: float, test_active_pct: float) -> bool:
        """
        Check if a split meets the tolerance criteria for active percentage difference.

        Args:
            train_active_pct: Percentage of actives in training set
            test_active_pct: Percentage of actives in test set

        Returns:
            True if the split meets tolerance criteria
        """
        active_diff = abs(train_active_pct - test_active_pct)
        return active_diff <= self.tolerance

    def _log_split_statistics(self, train_df: pd.DataFrame, test_df: pd.DataFrame, split_idx: int) -> Dict[str, Any]:
        """
        Log and return statistics for a data split.

        Args:
            train_df: Training set DataFrame
            test_df: Test set DataFrame
            split_idx: Index of the current split

        Returns:
            Dictionary containing split statistics
        """
        stats = {}

        # Calculate statistics if 'label' column exists
        if "label" in train_df.columns and "label" in test_df.columns:
            train_actives = int(train_df["label"].sum())
            test_actives = int(test_df["label"].sum())
            train_active_pct = train_actives / len(train_df)
            test_active_pct = test_actives / len(test_df)

            stats.update(
                {
                    f"train_actives_{split_idx}": train_actives,
                    f"test_actives_{split_idx}": test_actives,
                    f"train_actives_percentage_{split_idx}": train_active_pct,
                    f"test_actives_percentage_{split_idx}": test_active_pct,
                }
            )

            logger.info(
                f"Split {split_idx}: Train actives: {train_active_pct:.3f}, Test actives: {test_active_pct:.3f}"
            )

        stats.update(
            {
                f"train_size_{split_idx}": len(train_df),
                f"test_size_{split_idx}": len(test_df),
            }
        )

        logger.info(f"Split {split_idx}: Train size: {len(train_df)}, Test size: {len(test_df)}")
        return stats

    def run_splitting(self) -> Dict[str, Any]:
        """
        Execute the molecular splitting process.

        Returns:
            Dictionary containing configuration and statistics for all splits
        """
        # Load data
        df = self.load_data()
        smiles = df["smiles"].values.tolist()

        # Create splitter
        splitter = self._create_splitter(smiles)

        # Setup output directories
        output_path = self._setup_output_directories()

        # Initialize results tracking
        results_config = self.config.copy()
        successful_splits = 0

        logger.info(f"Starting splitting process with {self.splitter_name} splitter")

        # Perform splits
        for train_indices, test_indices in tqdm(
            splitter.split(smiles), desc=f"Processing {self.splitter_name} splits", total=INTERNAL_N_SPLITS
        ):
            # Create train/test subsets
            train_df = df.iloc[train_indices]
            test_df = df.iloc[test_indices]

            # Calculate split statistics
            split_stats = self._log_split_statistics(train_df, test_df, successful_splits)
            results_config.update(split_stats)

            # Check tolerance if labels are available
            split_acceptable = True
            if "label" in df.columns:
                train_active_pct = split_stats[f"train_actives_percentage_{successful_splits}"]
                test_active_pct = split_stats[f"test_actives_percentage_{successful_splits}"]
                split_acceptable = self._is_split_acceptable(train_active_pct, test_active_pct)

            if split_acceptable:
                # Save split files if requested
                if self.save:
                    train_path = increment_path(output_path / f"train_{successful_splits}.csv")
                    test_path = increment_path(output_path / f"test_{successful_splits}.csv")

                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)

                successful_splits += 1

                # Stop if we have enough splits
                if successful_splits >= self.n_splits:
                    break

        logger.info(f"Successfully created {successful_splits} splits")

        # Save configuration
        if self.save:
            config_path = output_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(results_config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")

        return results_config


def parse_arguments() -> Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = ArgumentParser(
        description="Split molecular datasets into train/test sets using various strategies",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Available splitters:
  scaffold, scaffold_generic, kmeans, molecular_weight, molecular_weight_reverse,
  perimeter, max_dissimilarity, molecular_logp, random, umap, hi

Examples:
  python scripts/splitting.py -f molecules.csv -sp scaffold -te 0.2 -ns 10
  python scripts/splitting.py -f molecules.txt -sp kmeans -te 0.3 -ns 5 --save
        """,
    )

    parser.add_argument(
        "-f", "--file_path", type=str, required=True, help="Path to a .csv/.txt file containing SMILES strings"
    )
    parser.add_argument(
        "-sp",
        "--splitter",
        type=str,
        default="scaffold",
        choices=list(SPLITTER_CLASSES.keys()),
        help="Splitting strategy to use (default: scaffold)",
    )
    parser.add_argument(
        "-te",
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of data for test set (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "-nj", "--n_jobs", type=int, default=DEFAULT_N_JOBS, help=f"Number of parallel jobs (default: {DEFAULT_N_JOBS})"
    )
    parser.add_argument(
        "-ns",
        "--n_splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of splits to generate (default: {DEFAULT_N_SPLITS})",
    )
    parser.add_argument(
        "-to",
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Tolerance for active percentage difference (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument("--save", action="store_true", help="Save split files and configuration to disk")

    args = parser.parse_args()

    # Additional validation
    if not Path(args.file_path).exists():
        parser.error(f"Input file does not exist: {args.file_path}")

    if not 0 < args.test_size < 1:
        parser.error(f"test_size must be between 0 and 1, got: {args.test_size}")

    if args.n_splits <= 0:
        parser.error(f"n_splits must be positive, got: {args.n_splits}")

    return args


def main() -> None:
    """Main entry point for the molecular splitting script."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Convert arguments to configuration dictionary
        config = vars(args)

        # Initialize and run the molecular splitter
        splitter = MolecularSplitter(config)
        splitter.run_splitting()

        logger.info("Molecular splitting completed successfully")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during splitting: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
