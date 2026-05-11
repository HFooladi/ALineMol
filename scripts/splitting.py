#!/usr/bin/env python3
"""
Molecular Splitting Script
===========================

A command-line tool for splitting molecular datasets into train and test sets using any
splitter registered in the ``alinemol.splitters`` factory. Splitters appear here
automatically — new splitters need only be registered with ``@register_splitter`` to
become available via ``-sp <name>`` (and to show up under ``--list-splitters``).

Supported strategies include scaffold-based, clustering-based, property-based, and
similarity-based approaches. Two registered splitters cannot be driven by this CLI and
are explicitly skipped (see ``_SKIP_FROM_CLI``):

- ``lo``: requires a continuous ``values`` argument to ``split()``.
- ``datasail``: requires the optional ``datasail`` package.

Internal split generation
-------------------------
For each request of ``--n_splits N``, the script asks the underlying splitter for up to
``INTERNAL_N_SPLITS`` candidate splits and keeps the first ``N`` that pass the active-
percentage tolerance (see ``--tolerance``). This is why splitters are instantiated with
``n_splits=INTERNAL_N_SPLITS`` regardless of the user-supplied ``-ns`` value.

Input File Format
-----------------
CSV/TSV with at minimum a ``smiles`` column. Add a ``label`` column to apply the active-
percentage tolerance filter.

    smiles,label
    CCO,1
    CCCO,0
    ...

Output Structure
----------------
When saving, the script creates::

    {input_dir}/split/{splitter_name}/
        train_0.csv
        test_0.csv
        ...
        config.json

Usage Examples
--------------
    # Basic scaffold splitting
    python scripts/splitting.py -f data/molecules.csv -sp scaffold --save

    # Run all (CLI-supported) splitters at once
    python scripts/splitting.py -f data/molecules.csv -sp all --save

    # Use a config file
    python scripts/splitting.py -c config/split_config.yaml

    # Dry run to preview without saving
    python scripts/splitting.py -f data/molecules.csv -sp kmeans --dry-run

    # List available splitters (including those skipped from the CLI)
    python scripts/splitting.py --list-splitters

Author: ALineMol Team
"""

import inspect
import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from tqdm import tqdm

from alinemol.splitters import (
    get_splitter,
    get_splitter_names,
    is_splitter_registered,
    list_splitters,
)
from alinemol.splitters.splitting_configures import (
    BUTINASplitConfig,
    DataSAILSplitConfig,
    HiSplitConfig,
    KMeansSplitConfig,
    LoSplitConfig,
    MaxDissimilaritySplitConfig,
    MolecularLogPSplitConfig,
    MolecularWeightReverseSplitConfig,
    MolecularWeightSplitConfig,
    PerimeterSplitConfig,
    RandomSplitConfig,
    ScaffoldKMeansSplitConfig,
    ScaffoldSplitConfig,
    ScaffoldSplitGenericConfig,
    UMapSplitConfig,
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
INTERNAL_N_SPLITS = 100  # candidate splits requested from the underlying splitter

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".txt"}

# Per-splitter default kwargs. Splitters that need no extra defaults can be omitted.
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
    "butina": BUTINASplitConfig,
    "scaffold_kmeans": ScaffoldKMeansSplitConfig,
    "hi": HiSplitConfig,
    "lo": LoSplitConfig,
    "datasail": DataSAILSplitConfig,
}

# Optional one-line description overrides. Splitters not listed here fall back to the
# first line of their class docstring.
SPLITTER_DESCRIPTIONS: Dict[str, str] = {
    "scaffold": "Bemis-Murcko scaffold-based splitting (specific scaffolds)",
    "scaffold_generic": "Bemis-Murcko scaffold-based splitting (generic scaffolds)",
    "scaffold_kmeans": "Scaffold extraction + k-means on scaffold ECFP fingerprints",
    "kmeans": "K-means clustering on molecular fingerprints",
    "molecular_weight": "Split by molecular weight (test on larger molecules)",
    "molecular_weight_reverse": "Split by molecular weight (test on smaller molecules)",
    "perimeter": "Perimeter-based clustering split",
    "max_dissimilarity": "Maximum dissimilarity-based splitting",
    "molecular_logp": "Split by lipophilicity (LogP)",
    "random": "Random splitting (baseline)",
    "umap": "UMAP dimensionality reduction + hierarchical clustering",
    "butina": "Taylor-Butina clustering on Morgan fingerprints",
    "hi": "Hi-split: ensures low similarity between train/test sets",
}

# Splitters that are registered but cannot be driven by this CLI.
_SKIP_FROM_CLI: Dict[str, str] = {
    "lo": "requires a continuous `values` argument to split(); use the library API directly.",
    "datasail": "requires the optional `datasail` package (pip install datasail).",
}


def _splitter_accepts(name: str, kwarg: str) -> bool:
    """Return True if the registered splitter's ``__init__`` accepts ``kwarg``."""
    cls = list_splitters().get(name)
    if cls is None:
        return False
    params = inspect.signature(cls.__init__).parameters
    return kwarg in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _get_description(name: str) -> str:
    """Return a one-line description for a splitter, preferring overrides over docstrings."""
    if name in SPLITTER_DESCRIPTIONS:
        return SPLITTER_DESCRIPTIONS[name]
    cls = list_splitters().get(name)
    if cls is not None and cls.__doc__:
        return cls.__doc__.strip().splitlines()[0]
    return "No description available"


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file (.yaml, .yml, or .json)

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is unsupported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")

    logger.info(f"Loaded configuration from {config_path}")
    return config or {}


def print_available_splitters() -> None:
    """Print a formatted list of available splitters with descriptions.

    Splitters are pulled dynamically from the registry. CLI-skipped splitters are
    listed with a ``(skipped: <reason>)`` suffix so users know they exist but require
    the library API.
    """
    print("\n" + "=" * 70)
    print("Available Splitting Strategies (from registry)")
    print("=" * 70 + "\n")

    runnable: List[str] = []
    skipped: List[str] = []
    for name in get_splitter_names():
        (skipped if name in _SKIP_FROM_CLI else runnable).append(name)

    print("CLI-runnable:")
    print("-" * 40)
    for name in runnable:
        print(f"  {name:<25} {_get_description(name)}")

    if skipped:
        print("\nRegistered but skipped from CLI:")
        print("-" * 40)
        for name in skipped:
            print(f"  {name:<25} {_get_description(name)}")
            print(f"  {'':<25}   (skipped: {_SKIP_FROM_CLI[name]})")

    print("\n" + "=" * 70)
    print("\nUsage: python scripts/splitting.py -f <file> -sp <splitter> --save")
    print("       python scripts/splitting.py -f <file> -sp all --save  (run all CLI-runnable)")
    print()


class MolecularSplitter:
    """A comprehensive molecular dataset splitting utility.

    Generates train/test splits using any splitter registered in the
    ``alinemol.splitters`` factory. SMILES are passed to the splitter's
    ``split()`` method; per-splitter default kwargs live in
    ``alinemol.splitters.splitting_configures``.

    Attributes:
        config: Dictionary containing all configuration parameters
        file_path: Path to the input molecular data file
        splitter_name: Name of the splitting strategy to use
        test_size: Fraction of data to use for testing (0-1)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        n_splits: Number of train/test splits to keep (after tolerance filter)
        tolerance: Maximum allowed difference in active percentages
        save: Whether to save split files to disk
        output_dir: Directory for saving output files
        dry_run: If True, preview operations without saving

    Example:
        >>> config = {
        ...     "file_path": "data/molecules.csv",
        ...     "splitter": "scaffold",
        ...     "test_size": 0.2,
        ...     "n_splits": 10,
        ...     "n_jobs": -1,
        ...     "tolerance": 0.1,
        ...     "save": True,
        ...     "output_dir": None,
        ...     "dry_run": False,
        ... }
        >>> splitter = MolecularSplitter(config)
        >>> results = splitter.run_splitting()
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the MolecularSplitter with configuration parameters."""
        self.config = config
        self.file_path = Path(config["file_path"])
        self.splitter_name = config["splitter"]
        self.test_size = config["test_size"]
        self.n_jobs = config["n_jobs"]
        self.n_splits = config["n_splits"]
        self.tolerance = config["tolerance"]
        self.save = config["save"]
        self.output_dir = config.get("output_dir")
        self.dry_run = config.get("dry_run", False)

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")

        if self.file_path.suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension. Supported: {SUPPORTED_EXTENSIONS}")

        if not is_splitter_registered(self.splitter_name):
            raise ValueError(
                f"Unknown splitter: '{self.splitter_name}'. "
                f"Available: {sorted(get_splitter_names())}. Use --list-splitters."
            )

        if self.splitter_name in _SKIP_FROM_CLI:
            raise ValueError(
                f"Splitter '{self.splitter_name}' is not supported by this CLI: {_SKIP_FROM_CLI[self.splitter_name]}"
            )

        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got: {self.test_size}")

        if self.n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got: {self.n_splits}")

    def load_data(self) -> pd.DataFrame:
        """Load molecular data from the input file."""
        try:
            if self.file_path.suffix == ".csv":
                df = pd.read_csv(self.file_path)
            elif self.file_path.suffix == ".txt":
                df = pd.read_csv(self.file_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

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
        """Create a splitter via the factory, filtering kwargs to those it accepts."""
        # Per-splitter defaults (e.g. ``make_generic``, ``n_clusters``).
        cfg = SPLITTER_CONFIGS.get(self.splitter_name, {})
        hopts: Dict[str, Any] = dict(cfg() if callable(cfg) else cfg)

        # Standard run-level kwargs only forwarded when the splitter actually accepts them.
        standard: Dict[str, Any] = {
            "n_splits": INTERNAL_N_SPLITS,
            "test_size": self.test_size,
            "random_state": DEFAULT_RANDOM_STATE,
            "n_jobs": self.n_jobs,
        }
        for kw, value in standard.items():
            if _splitter_accepts(self.splitter_name, kw):
                hopts.setdefault(kw, value)

        splitter = get_splitter(self.splitter_name, **hopts)
        logger.info(f"Initialized {self.splitter_name} splitter with configuration: {hopts}")
        return splitter

    def _setup_output_directories(self) -> Path:
        """Create necessary output directories for saving splits."""
        if self.output_dir:
            base_path = Path(self.output_dir)
        else:
            base_path = self.file_path.parent / "split"

        base_path.mkdir(parents=True, exist_ok=True)

        splitter_path = base_path / self.splitter_name
        splitter_path.mkdir(parents=True, exist_ok=True)

        return splitter_path

    def _is_split_acceptable(self, train_active_pct: float, test_active_pct: float) -> bool:
        """Check if a split meets the tolerance criteria for active percentage difference."""
        active_diff = abs(train_active_pct - test_active_pct)
        return active_diff <= self.tolerance

    def _log_split_statistics(self, train_df: pd.DataFrame, test_df: pd.DataFrame, split_idx: int) -> Dict[str, Any]:
        """Log and return statistics for a data split."""
        stats: Dict[str, Any] = {}

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
        """Execute the molecular splitting process.

        Generates train/test splits according to the configured strategy.
        If save=True and dry_run=False, writes split files to disk.
        """
        df = self.load_data()
        smiles = df["smiles"].values.tolist()

        splitter = self._create_splitter(smiles)
        output_path = self._setup_output_directories()

        results_config = self.config.copy()
        successful_splits = 0

        if self.dry_run:
            logger.info(f"[DRY RUN] Would process {self.splitter_name} splitter")
            logger.info(f"[DRY RUN] Output directory: {output_path}")
            logger.info(f"[DRY RUN] Number of molecules: {len(df)}")
            logger.info(f"[DRY RUN] Test size: {self.test_size}")
            logger.info(f"[DRY RUN] Number of splits requested: {self.n_splits}")
            results_config["dry_run"] = True
            results_config["output_path"] = str(output_path)
            return results_config

        logger.info(f"Starting splitting process with {self.splitter_name} splitter")

        for train_indices, test_indices in tqdm(
            splitter.split(smiles), desc=f"Processing {self.splitter_name} splits", total=INTERNAL_N_SPLITS
        ):
            train_df = df.iloc[train_indices]
            test_df = df.iloc[test_indices]

            split_stats = self._log_split_statistics(train_df, test_df, successful_splits)
            results_config.update(split_stats)

            split_acceptable = True
            if "label" in df.columns:
                train_active_pct = split_stats[f"train_actives_percentage_{successful_splits}"]
                test_active_pct = split_stats[f"test_actives_percentage_{successful_splits}"]
                split_acceptable = self._is_split_acceptable(train_active_pct, test_active_pct)

            if split_acceptable:
                if self.save:
                    train_path = increment_path(output_path / f"train_{successful_splits}.csv")
                    test_path = increment_path(output_path / f"test_{successful_splits}.csv")

                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)

                successful_splits += 1

                if successful_splits >= self.n_splits:
                    break

        logger.info(f"Successfully created {successful_splits} splits")
        results_config["successful_splits"] = successful_splits
        results_config["output_path"] = str(output_path)

        if self.save:
            config_path = output_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(results_config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")

        return results_config


def run_all_splitters(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run every CLI-runnable splitter in the registry on the dataset."""
    all_results: Dict[str, Dict[str, Any]] = {}
    failed_splitters: List[str] = []

    runnable = [name for name in get_splitter_names() if name not in _SKIP_FROM_CLI]
    if _SKIP_FROM_CLI:
        logger.info(
            "Skipping (not supported by this CLI): "
            + ", ".join(f"{n} ({_SKIP_FROM_CLI[n]})" for n in sorted(_SKIP_FROM_CLI))
        )

    logger.info("=" * 60)
    logger.info(f"Running ALL splitting strategies ({len(runnable)} total)")
    logger.info("=" * 60)

    for splitter_name in runnable:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running splitter: {splitter_name}")
        logger.info(f"{'=' * 40}")

        try:
            splitter_config = config.copy()
            splitter_config["splitter"] = splitter_name

            splitter = MolecularSplitter(splitter_config)
            results = splitter.run_splitting()
            all_results[splitter_name] = results

            logger.info(f"Completed {splitter_name}: {results.get('successful_splits', 0)} splits")

        except Exception as e:
            logger.error(f"Failed to run {splitter_name}: {str(e)}")
            failed_splitters.append(splitter_name)
            all_results[splitter_name] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: All Splitters")
    logger.info("=" * 60)

    successful = [name for name in all_results if "error" not in all_results[name]]
    logger.info(f"Successful: {len(successful)}/{len(runnable)}")

    if failed_splitters:
        logger.warning(f"Failed splitters: {', '.join(failed_splitters)}")

    if config.get("save") and not config.get("dry_run"):
        if config.get("output_dir"):
            summary_dir = Path(config["output_dir"])
        else:
            summary_dir = Path(config["file_path"]).parent / "split"

        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "all_splitters_summary.json"

        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"Combined summary saved to {summary_path}")

    return all_results


def _validate_splitter_choice(name: str) -> str:
    """argparse ``type=`` callable: accept any registered splitter name or ``all``."""
    if name == "all" or is_splitter_registered(name):
        return name
    available = sorted(get_splitter_names())
    raise ValueError(f"unknown splitter '{name}'. Available: {available + ['all']}. Use --list-splitters.")


def parse_arguments() -> Namespace:
    """Parse and validate command line arguments."""
    parser = ArgumentParser(
        description="""
Molecular Dataset Splitting Tool
================================

Split molecular datasets into train/test sets using any registered splitter.
Splitters are pulled from the alinemol.splitters factory; ``--list-splitters``
shows the current set (including ones skipped from the CLI).

Input file must be CSV or TSV with a 'smiles' column (and optionally 'label').
        """,
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
--------
  # Basic scaffold splitting (most common)
  python scripts/splitting.py -f molecules.csv -sp scaffold --save

  # Run ALL CLI-runnable splitting strategies at once
  python scripts/splitting.py -f molecules.csv -sp all --save

  # Use a configuration file
  python scripts/splitting.py -c split_config.yaml

  # Preview what would happen (dry run)
  python scripts/splitting.py -f molecules.csv -sp kmeans --dry-run

  # Custom output directory
  python scripts/splitting.py -f molecules.csv -sp scaffold --save -o results/

  # List all available splitters
  python scripts/splitting.py --list-splitters

CONFIGURATION FILE
------------------
Create a YAML file (e.g., split_config.yaml):

    file_path: data/molecules.csv
    splitter: scaffold
    test_size: 0.2
    n_splits: 10
    n_jobs: -1
    tolerance: 0.1
    save: true
    output_dir: results/splits/

For more details, see: https://github.com/HFooladi/ALineMol
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="Path to YAML/JSON config file. CLI args override config values.",
    )

    parser.add_argument(
        "--list-splitters", action="store_true", help="List all available splitting strategies and exit"
    )

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        metavar="FILE",
        help="Path to .csv/.txt file containing SMILES strings (required unless --config)",
    )

    parser.add_argument(
        "-sp",
        "--splitter",
        type=_validate_splitter_choice,
        default="scaffold",
        metavar="METHOD",
        help="Splitting strategy name (see --list-splitters) or 'all'. (default: scaffold)",
    )

    parser.add_argument(
        "-te",
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        metavar="FRAC",
        help=f"Fraction of data for test set, 0 < value < 1 (default: {DEFAULT_TEST_SIZE})",
    )

    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        metavar="N",
        help=f"Number of parallel jobs, -1 for all CPUs (default: {DEFAULT_N_JOBS})",
    )

    parser.add_argument(
        "-ns",
        "--n_splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        metavar="N",
        help=(
            f"Number of accepted splits to keep (default: {DEFAULT_N_SPLITS}). "
            f"The script requests up to {INTERNAL_N_SPLITS} candidate splits and keeps "
            "the first --n_splits that pass the active-percentage tolerance."
        ),
    )

    parser.add_argument(
        "-to",
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        metavar="TOL",
        help=f"Max allowed difference in train/test active percentages (default: {DEFAULT_TOLERANCE})",
    )

    parser.add_argument(
        "-o", "--output_dir", type=str, metavar="DIR", help="Custom output directory (default: {input_dir}/split/)"
    )

    parser.add_argument("--save", action="store_true", help="Save split files and configuration to disk")

    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run", help="Preview operations without saving any files"
    )

    args = parser.parse_args()

    if args.list_splitters:
        print_available_splitters()
        sys.exit(0)

    config_values: Dict[str, Any] = {}
    if args.config:
        config_values = load_config_file(args.config)

    cli_defaults: Dict[str, Optional[Any]] = {
        "file_path": None,
        "splitter": "scaffold",
        "test_size": DEFAULT_TEST_SIZE,
        "n_jobs": DEFAULT_N_JOBS,
        "n_splits": DEFAULT_N_SPLITS,
        "tolerance": DEFAULT_TOLERANCE,
        "output_dir": None,
        "save": False,
        "dry_run": False,
    }

    for key, default_value in cli_defaults.items():
        cli_value = getattr(args, key, None)
        config_value = config_values.get(key)

        if cli_value == default_value and config_value is not None:
            setattr(args, key, config_value)

    if not args.file_path:
        parser.error("--file_path (-f) is required unless provided in config file")

    if not Path(args.file_path).exists():
        parser.error(f"Input file does not exist: {args.file_path}")

    if not 0 < args.test_size < 1:
        parser.error(f"test_size must be between 0 and 1, got: {args.test_size}")

    if args.n_splits <= 0:
        parser.error(f"n_splits must be positive, got: {args.n_splits}")

    if args.splitter != "all" and not is_splitter_registered(args.splitter):
        parser.error(f"Unknown splitter: {args.splitter}. Use --list-splitters to see available options.")

    return args


def main() -> None:
    """Main entry point for the molecular splitting script."""
    try:
        args = parse_arguments()
        config = vars(args)
        config.pop("config", None)
        config.pop("list_splitters", None)

        if config["splitter"] == "all":
            run_all_splitters(config)
        else:
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
