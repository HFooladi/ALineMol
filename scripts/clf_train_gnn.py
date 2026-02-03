#!/usr/bin/env python3
"""
Graph Neural Network Training Script for Molecular Property Prediction

This script provides a comprehensive training pipeline for Graph Neural Networks (GNNs)
on molecular property prediction tasks. It supports multiple GNN architectures including
GCN, GAT, MPNN, AttentiveFP, and various pre-trained models.

Key Features:
    - Multiple GNN architectures for molecular graphs
    - Bayesian hyperparameter optimization with Hyperopt
    - Early stopping and model checkpointing
    - Comprehensive metrics reporting (Accuracy, ROC-AUC, PR-AUC)
    - Flexible data splitting strategies
    - GPU/CPU training support
    - Configurable molecular featurization

Supported Models:
    - GCN: Graph Convolutional Network
    - GAT: Graph Attention Network
    - Weave: Weave convolution
    - MPNN: Message Passing Neural Network
    - AttentiveFP: Attentive FP
    - GIN: Graph Isomorphism Network variants
    - NF: Neural Fingerprints

Usage:
    python clf_train_gnn.py -c data.csv -sc smiles -t label1,label2 -mo GCN
    python clf_train_gnn.py -c molecules.csv -sc smiles -mo AttentiveFP -ne 50

Author: ALineMol Team
Inspired by: DGL-LifeSci codebase
"""

import json
import os
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import EarlyStopping
from hyperopt import fmin, tpe
from torch.optim import Adam
from torch.utils.data import DataLoader

from alinemol.hyper import init_hyper_space
from alinemol.utils import (
    collate_molgraphs,
    get_configure,
    init_featurizer,
    init_trial_path,
    load_dataset,
    load_model,
    mkdir_p,
    split_dataset,
)
from alinemol.utils.training_utils import run_a_train_epoch, run_an_eval_epoch
from alinemol.utils.logger_utils import logger

# Repository path configuration
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = REPO_PATH
DATASET_PATH = os.path.join(REPO_PATH, "datasets")

# Configure Python path for local imports
os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)

# Constants
DEFAULT_SPLIT_RATIO = "0.72,0.08,0.2"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_NUM_WORKERS = 4
DEFAULT_PRINT_EVERY = 20
DEFAULT_RESULT_PATH = "classification_results"
DEFAULT_DEVICE = "cpu"

# Available models
AVAILABLE_MODELS = [
    "GCN",
    "GAT",
    "Weave",
    "MPNN",
    "AttentiveFP",
    "gin_supervised_contextpred",
    "gin_supervised_infomax",
    "gin_supervised_edgepred",
    "gin_supervised_masking",
    "NF",
]

# Available metrics
AVAILABLE_METRICS = ["accuracy_score", "roc_auc_score", "pr_auc_score"]

# Available splitting strategies
AVAILABLE_SPLITS = ["scaffold_decompose", "scaffold_smiles", "stratified_random"]

# Featurizer types
AVAILABLE_FEATURIZERS = ["canonical", "attentivefp"]


class GNNTrainer:
    """
    A comprehensive trainer for Graph Neural Networks on molecular property prediction.

    This class encapsulates the entire training pipeline including data loading,
    model initialization, training loop, evaluation, and results saving.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GNN trainer with configuration parameters.

        Args:
            config: Dictionary containing training configuration
        """
        self.config = config.copy()
        self.device = config["device"]
        self.metric = config["metric"]
        self.num_epochs = config["num_epochs"]

        # Initialize paths and directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Set up directories for saving results."""
        self.config = init_trial_path(self.config)
        self.trial_path = Path(self.config["trial_path"])

    def _prepare_experiment_config(self, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the experiment configuration with model and featurization details.

        Args:
            exp_config: Base experiment configuration

        Returns:
            Updated experiment configuration
        """
        # Record model and task settings
        exp_config.update(
            {
                "model": self.config["model"],
                "n_tasks": self.config["n_tasks"],
                "atom_featurizer_type": self.config["atom_featurizer_type"],
                "bond_featurizer_type": self.config["bond_featurizer_type"],
            }
        )

        # Add featurization dimensions
        if self.config["atom_featurizer_type"] != "pre_train":
            exp_config["in_node_feats"] = self.config["node_featurizer"].feat_size()
        if self.config["edge_featurizer"] is not None and self.config["bond_featurizer_type"] != "pre_train":
            exp_config["in_edge_feats"] = self.config["edge_featurizer"].feat_size()

        return exp_config

    def _create_data_loaders(
        self, train_set, val_set, test_set, batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing.

        Args:
            train_set: Training dataset
            val_set: Validation dataset
            test_set: Test dataset
            batch_size: Batch size for data loaders

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        common_kwargs = {
            "batch_size": batch_size,
            "collate_fn": collate_molgraphs,
            "num_workers": self.config["num_workers"],
        }

        train_loader = DataLoader(dataset=train_set, shuffle=True, **common_kwargs)
        val_loader = DataLoader(dataset=val_set, shuffle=False, **common_kwargs)
        test_loader = DataLoader(dataset=test_set, shuffle=False, **common_kwargs)

        return train_loader, val_loader, test_loader

    def _initialize_model_and_training(
        self, exp_config: Dict[str, Any]
    ) -> Tuple[nn.Module, nn.Module, Any, EarlyStopping]:
        """
        Initialize the model, loss function, optimizer, and early stopping.

        Args:
            exp_config: Experiment configuration

        Returns:
            Tuple of (model, loss_criterion, optimizer, early_stopper)
        """
        # Load and setup model
        model = load_model(exp_config).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(f"Model architecture: {self.config['model']}")
        logger.info(f"Model: {model}")
        logger.info(f"Number of parameters: {total_params:,}")

        # Setup training components
        loss_criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = Adam(model.parameters(), lr=exp_config["lr"], weight_decay=exp_config["weight_decay"])
        early_stopper = EarlyStopping(
            patience=exp_config["patience"], filename=str(self.trial_path / "model.pth"), metric=self.metric
        )

        return model, loss_criterion, optimizer, early_stopper

    def _training_loop(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_criterion: nn.Module,
        optimizer: Any,
        early_stopper: EarlyStopping,
    ) -> None:
        """
        Execute the main training loop with validation and early stopping.

        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_criterion: Loss function
            optimizer: Optimizer
            early_stopper: Early stopping handler
        """
        logger.info(f"Starting training for up to {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            # Training phase
            run_a_train_epoch(self.config, epoch, model, train_loader, loss_criterion, optimizer)

            # Validation phase
            val_score = run_an_eval_epoch(self.config, model, val_loader)
            early_stop = early_stopper.step(val_score, model)

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"validation {self.metric}: {val_score:.4f}, "
                f"best validation {self.metric}: {early_stopper.best_score:.4f}"
            )

            if early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    def _evaluate_model(
        self, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, early_stopper: EarlyStopping
    ) -> Dict[str, float]:
        """
        Evaluate the trained model on validation and test sets.

        Args:
            model: Trained model
            val_loader: Validation data loader
            test_loader: Test data loader
            early_stopper: Early stopping handler with best model

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load best model
        early_stopper.load_checkpoint(model)

        # Validation score
        val_score = run_an_eval_epoch(self.config, model, val_loader, test=False)
        logger.info(f"Best validation {self.metric}: {val_score:.4f}")

        # Test scores
        test_scores = run_an_eval_epoch(self.config, model, test_loader, test=True)

        logger.info(f"Test accuracy: {test_scores[0]:.4f}")
        logger.info(f"Test ROC-AUC: {test_scores[1]:.4f}")
        logger.info(f"Test PR-AUC: {test_scores[2]:.4f}")

        return {
            "val_score": early_stopper.best_score.item(),
            "test_accuracy": test_scores[0],
            "test_roc_auc": test_scores[1],
            "test_pr_auc": test_scores[2],
        }

    def _save_results(
        self, metrics: Dict[str, float], exp_config: Dict[str, Any], train_set, val_set, test_set
    ) -> None:
        """
        Save training results including metrics and configuration.

        Args:
            metrics: Evaluation metrics
            exp_config: Experiment configuration
            train_set: Training dataset (for size info)
            val_set: Validation dataset (for size info)
            test_set: Test dataset (for size info)
        """
        # Save evaluation results to text file
        eval_path = self.trial_path / "eval.txt"
        with open(eval_path, "w") as f:
            f.write(f"Best val {self.metric}: {metrics['val_score']}\n")
            f.write(f"Test accuracy_score: {metrics['test_accuracy']}\n")
            f.write(f"Test roc_auc_score: {metrics['test_roc_auc']}\n")
            f.write(f"Test pr_auc_score: {metrics['test_pr_auc']}\n")

        # Save metrics to CSV
        results_df = pd.DataFrame(
            {
                f"{self.metric}_val": [metrics["val_score"]],
                "acc": [metrics["test_accuracy"]],
                "roc_auc": [metrics["test_roc_auc"]],
                "pr_auc": [metrics["test_pr_auc"]],
                "model": [self.config["model"]],
                "test_size": [len(test_set)],
                "val_size": [len(val_set)],
                "train_size": [len(train_set)],
            }
        )
        results_df.to_csv(self.trial_path / "metrics.csv", index=False)

        # Save configuration
        exp_config.update({"filepath": self.config["csv_path"]})
        config_path = self.trial_path / "configure.json"
        with open(config_path, "w") as f:
            json.dump(exp_config, f, indent=2)

        logger.info(f"Results saved to {self.trial_path}")
        logger.info(f"Experiment configuration: {exp_config}")

    def train(self, exp_config: Dict[str, Any], train_set, val_set, test_set) -> Tuple[str, float]:
        """
        Execute the complete training pipeline.

        Args:
            exp_config: Experiment configuration
            train_set: Training dataset
            val_set: Validation dataset
            test_set: Test dataset

        Returns:
            Tuple of (trial_path, best_validation_score)
        """
        try:
            # Prepare configuration
            exp_config = self._prepare_experiment_config(exp_config)

            # Create data loaders
            train_loader, val_loader, test_loader = self._create_data_loaders(
                train_set, val_set, test_set, exp_config["batch_size"]
            )

            # Initialize model and training components
            model, loss_criterion, optimizer, early_stopper = self._initialize_model_and_training(exp_config)

            # Execute training loop
            self._training_loop(model, train_loader, val_loader, loss_criterion, optimizer, early_stopper)

            # Evaluate model
            metrics = self._evaluate_model(model, val_loader, test_loader, early_stopper)

            # Save results
            self._save_results(metrics, exp_config, train_set, val_set, test_set)

            return str(self.trial_path), early_stopper.best_score

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


class BayesianOptimizer:
    """
    Bayesian hyperparameter optimization for GNN training.

    This class manages hyperparameter search using Tree-structured Parzen Estimator
    to find optimal hyperparameters for the GNN model.
    """

    def __init__(self, config: Dict[str, Any], num_evals: int):
        """
        Initialize the Bayesian optimizer.

        Args:
            config: Base configuration dictionary
            num_evals: Number of optimization trials to run
        """
        self.config = config
        self.num_evals = num_evals
        self.metric = config["metric"]
        self.results = []

    def _objective_function(self, hyperparams: Dict[str, Any], train_set, val_set, test_set) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            hyperparams: Hyperparameter configuration to evaluate
            train_set: Training dataset
            val_set: Validation dataset
            test_set: Test dataset

        Returns:
            Metric value to minimize (negative for metrics to maximize)
        """
        try:
            # Create a copy of configuration with new hyperparameters
            trial_config = deepcopy(self.config)

            # Initialize trainer with trial configuration
            trainer = GNNTrainer(trial_config)
            trial_path, val_metric = trainer.train(hyperparams, train_set, val_set, test_set)

            # Convert metric for minimization
            if self.metric in ["roc_auc_score", "pr_auc_score"]:
                # Maximize ROC-AUC/PR-AUC by minimizing negative
                metric_to_minimize = -1 * val_metric
            else:
                metric_to_minimize = val_metric

            self.results.append((trial_path, metric_to_minimize))
            logger.info(f"Trial completed: {trial_path}, metric: {val_metric:.4f}")

            return metric_to_minimize

        except Exception as e:
            logger.error(f"Trial failed with hyperparams {hyperparams}: {str(e)}")
            # Return a large value for failed trials
            return float("inf")

    def optimize(self, train_set, val_set, test_set) -> str:
        """
        Run Bayesian optimization to find best hyperparameters.

        Args:
            train_set: Training dataset
            val_set: Validation dataset
            test_set: Test dataset

        Returns:
            Path to the best trial results
        """
        logger.info(f"Starting Bayesian optimization with {self.num_evals} trials")

        # Get hyperparameter search space
        candidate_hypers = init_hyper_space(self.config["model"])

        # Define objective with fixed datasets
        def objective(hyperparams):
            return self._objective_function(hyperparams, train_set, val_set, test_set)

        # Run optimization
        try:
            best_hyperparams = fmin(
                fn=objective, space=candidate_hypers, algo=tpe.suggest, max_evals=self.num_evals, verbose=True
            )

            # Find best trial
            self.results.sort(key=lambda x: x[1])  # Sort by metric value
            best_trial_path, best_metric = self.results[0]

            logger.info(f"Optimization completed. Best trial: {best_trial_path}")
            logger.info(f"Best hyperparameters: {best_hyperparams}")
            logger.info(
                f"Best {self.metric}: {-best_metric if self.metric in ['roc_auc_score', 'pr_auc_score'] else best_metric:.4f}"
            )

            return best_trial_path

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            raise


def parse_arguments() -> Namespace:
    """
    Parse and validate command line arguments for GNN training.

    Returns:
        Parsed arguments namespace
    """
    parser = ArgumentParser(
        description="Train Graph Neural Networks for molecular property prediction",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=f"""
    Available Models:
    {", ".join(AVAILABLE_MODELS)}

    Available Metrics:
    {", ".join(AVAILABLE_METRICS)}

    Available Splitting Strategies:
    {", ".join(AVAILABLE_SPLITS)}

    Examples:
    # Basic GCN training
    python clf_train_gnn.py -c data.csv -sc smiles -t label1,label2 -mo GCN
    
    # AttentiveFP with hyperparameter optimization
    python clf_train_gnn.py -c molecules.csv -sc smiles -mo AttentiveFP -ne 50 -me roc_auc_score
    
    # Multi-GPU training with custom split ratio
    python clf_train_gnn.py -c data.csv -sc smiles -mo GAT -de cuda:0 -sr 0.8,0.1,0.1
    """,
    )

    # Required arguments
    parser.add_argument("-c", "--csv-path", type=str, required=True, help="Path to CSV file containing molecular data")
    parser.add_argument(
        "-sc", "--smiles-column", type=str, required=True, help="Column name for SMILES strings in the CSV file"
    )

    # Optional data arguments
    parser.add_argument(
        "-t",
        "--task-names",
        type=str,
        default=None,
        help="Comma-separated task column names. If None, uses all columns except SMILES column",
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=AVAILABLE_SPLITS,
        default="scaffold_smiles",
        help="Dataset splitting strategy (default: scaffold_smiles)",
    )
    parser.add_argument(
        "-sr",
        "--split-ratio",
        type=str,
        default=DEFAULT_SPLIT_RATIO,
        help=f"Train/validation/test split ratios (default: {DEFAULT_SPLIT_RATIO})",
    )

    # Model and training arguments
    parser.add_argument(
        "-mo", "--model", choices=AVAILABLE_MODELS, default="GCN", help="GNN model architecture (default: GCN)"
    )
    parser.add_argument(
        "-me",
        "--metric",
        choices=AVAILABLE_METRICS,
        default="roc_auc_score",
        help="Evaluation metric (default: roc_auc_score)",
    )
    parser.add_argument(
        "-a",
        "--atom-featurizer-type",
        choices=AVAILABLE_FEATURIZERS,
        default="canonical",
        help="Atom featurization method (default: canonical)",
    )
    parser.add_argument(
        "-b",
        "--bond-featurizer-type",
        choices=AVAILABLE_FEATURIZERS,
        default="canonical",
        help="Bond featurization method (default: canonical)",
    )

    # Training configuration
    parser.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f"Maximum training epochs with early stopping (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of data loading processes (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "-pe",
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
        help=f"Print training progress every N mini-batches (default: {DEFAULT_PRINT_EVERY})",
    )

    # Output and device arguments
    parser.add_argument(
        "-p",
        "--result-path",
        type=str,
        default=DEFAULT_RESULT_PATH,
        help=f"Directory to save results (default: {DEFAULT_RESULT_PATH})",
    )
    parser.add_argument(
        "-de",
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Computing device: 'cpu' or 'cuda:x' (default: {DEFAULT_DEVICE})",
    )

    # Hyperparameter optimization
    parser.add_argument(
        "-ne",
        "--num-evals",
        type=int,
        default=None,
        help="Number of Bayesian optimization trials. If None, uses default hyperparameters",
    )

    args = parser.parse_args()

    # Validation
    if not Path(args.csv_path).exists():
        parser.error(f"CSV file does not exist: {args.csv_path}")

    if args.num_evals is not None and args.num_evals <= 0:
        parser.error(f"Number of evaluation trials must be positive, got: {args.num_evals}")

    # Validate split ratios format
    try:
        split_ratios = [float(x.strip()) for x in args.split_ratio.split(",")]
        if len(split_ratios) != 3:
            parser.error(f"Split ratio must have exactly 3 values, got: {len(split_ratios)}")
        if not abs(sum(split_ratios) - 1.0) < 1e-6:
            parser.error(f"Split ratios must sum to 1.0, got sum: {sum(split_ratios)}")
        # Keep as string for compatibility with downstream functions
    except ValueError:
        parser.error(f"Invalid split ratio format: {args.split_ratio}")

    return args


def setup_device(device_str: str) -> torch.device:
    """
    Configure and validate the computing device.

    Args:
        device_str: Device specification string

    Returns:
        PyTorch device object
    """
    if device_str == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU for computation")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_str)
            logger.info(f"Using GPU: {device}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(device)}")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU")
            device = torch.device("cpu")
    else:
        logger.warning(f"Unknown device '{device_str}'. Using CPU")
        device = torch.device("cpu")

    return device


def prepare_datasets(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """
    Load and prepare datasets for training.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (dataset, train_set, val_set, test_set)
    """
    try:
        # Load data
        logger.info(f"Loading dataset from {config['csv_path']}")
        df = pd.read_csv(config["csv_path"])
        logger.info(f"Loaded {len(df)} molecules")

        # Validate required columns
        if config["smiles_column"] not in df.columns:
            raise ValueError(f"SMILES column '{config['smiles_column']}' not found in CSV")

        # Parse task names
        if config["task_names"] is not None:
            config["task_names"] = [name.strip() for name in config["task_names"].split(",")]
            missing_tasks = [task for task in config["task_names"] if task not in df.columns]
            if missing_tasks:
                raise ValueError(f"Task columns not found: {missing_tasks}")

        # Initialize featurizers
        config = init_featurizer(config)

        # Create output directory
        mkdir_p(config["result_path"])

        # Load and split dataset
        dataset = load_dataset(config, df)
        config["n_tasks"] = dataset.n_tasks

        logger.info(f"Dataset loaded with {config['n_tasks']} tasks")

        train_set, val_set, test_set = split_dataset(config, dataset)

        logger.info(f"Dataset split - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        return dataset, train_set, val_set, test_set

    except Exception as e:
        logger.error(f"Failed to prepare datasets: {str(e)}")
        raise


def main() -> None:
    """Main entry point for GNN training script."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Convert to dictionary for easier handling
        config = vars(args)

        # Setup device
        config["device"] = setup_device(config["device"])

        # Prepare datasets
        dataset, train_set, val_set, test_set = prepare_datasets(config)

        # Run training or optimization
        if config["num_evals"] is not None:
            # Bayesian optimization
            logger.info(f"Starting hyperparameter optimization with {config['num_evals']} trials")
            optimizer = BayesianOptimizer(config, config["num_evals"])
            best_trial_path = optimizer.optimize(train_set, val_set, test_set)
            logger.info(f"Optimization complete. Best results saved to: {best_trial_path}")
        else:
            # Single training run with default hyperparameters
            logger.info("Using default hyperparameters for training")
            exp_config = get_configure(config["model"])
            trainer = GNNTrainer(config)
            trial_path, best_score = trainer.train(exp_config, train_set, val_set, test_set)
            logger.info(f"Training complete. Results saved to: {trial_path}")
            logger.info(f"Best validation {config['metric']}: {best_score:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
