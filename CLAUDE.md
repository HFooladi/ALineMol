# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=alinemol --cov-report=term-missing

# Run tests excluding slow ones
pytest -m "not slow"
```

### Code Quality
```bash
# Check code style and lint
ruff check

# Format code
ruff format

# Run type checking
mypy alinemol/
```

### Building and Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using uv with test dependencies
uv pip install -e ".[test]"

# Using uv with development dependencies
uv pip install -e ".[dev,test]"

# Using pip (alternative)
pip install -e ".[dev,test]"

# Build documentation
mkdocs serve
```

### Environment Setup

#### Using install.sh (Recommended)
```bash
# CPU installation (default)
./install.sh
source .venv/bin/activate

# CUDA installation (choose your CUDA version)
./install.sh cu121  # CUDA 12.1
./install.sh cu118  # CUDA 11.8
./install.sh cu124  # CUDA 12.4
source .venv/bin/activate
```

#### Manual uv Installation
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# CPU PyTorch
uv pip install -e ".[dev,test]" -f https://download.pytorch.org/whl/cpu -f https://data.dgl.ai/wheels/repo.html

# Or CUDA 12.1 PyTorch
uv pip install -e ".[dev,test]" -f https://download.pytorch.org/whl/cu121 -f https://data.dgl.ai/wheels/repo.html
```

#### Using conda
```bash
# Create conda environment from file
conda env create -f environment.yml
conda activate alinemol
```

## Architecture Overview

ALineMol is a comprehensive framework for evaluating machine learning models on molecular property prediction with a focus on out-of-distribution (OOD) performance analysis.

### Core Components

**alinemol/models/**: Machine learning model implementations
- `cml.py`: Classical ML models (SVM, Random Forest, XGBoost, etc.)
- `fragGNN.py`: Graph neural network models for molecular data
- `layers.py`: Custom neural network layers
- `model_configures/`: JSON configuration files for different model architectures

**alinemol/splitters/**: Molecular data splitting strategies with unified API
- `base.py`: `BaseMolecularSplitter` abstract base class defining the unified interface
- `factory.py`: `get_splitter()` factory function and splitter registry
- `wrappers.py`: Wrapper classes for splito splitters (KMeans, Scaffold, MolecularWeight, etc.)
- `splits.py`: Native splitting implementations (RandomSplit, MolecularLogPSplit)
- `butina_split.py`: Butina clustering-based splits
- `umap_split.py`: UMAP dimensionality reduction + clustering splits
- `lohi/`: Hi/Lo similarity-based splitting strategies
- `datasail/`: DataSAIL integration for advanced splitting
- `splitting_configures.py`: Configuration management for split parameters

**Splitters Unified API:**
```python
# Factory function (recommended)
from alinemol.splitters import get_splitter, get_splitter_names
print(get_splitter_names())  # List all available splitters
splitter = get_splitter("scaffold", make_generic=True, n_splits=5)

# Direct class import
from alinemol.splitters import ScaffoldSplit, KMeansSplit
splitter = ScaffoldSplit(make_generic=True)

# All splitters accept SMILES in split() method
for train_idx, test_idx in splitter.split(smiles_list):
    train = [smiles_list[i] for i in train_idx]
```

**alinemol/preprocessing/**: Data preprocessing and standardization
- `standardizer.py`: SMILES standardization and molecular preprocessing pipeline

**alinemol/utils/**: Utility functions and helpers
- `utils.py`: General utility functions for dataset loading and evaluation
- `split_utils.py`: Splitting-related utility functions  
- `training_utils.py`: Model training and evaluation utilities
- `metric_utils.py`: Performance metrics calculation
- `plot_utils.py`: Visualization and plotting functions
- `graph_utils.py`: Graph construction for molecular data
- `logger_utils.py`: Logging configuration and utilities

**alinemol/hyper/**: Hyperparameter optimization
- `hyper.py`: Hyperparameter tuning utilities

### Key Workflows

1. **Data Preprocessing**: Use `standardization_pipeline()` from `preprocessing/standardizer.py` to clean and standardize molecular data
2. **Splitting**: Apply various splitting strategies from `splitters/` to create train/test splits that simulate distribution shift
3. **Model Training**: Use configurations from `model_configures/` with implementations in `models/` 
4. **Evaluation**: Compute ID vs OOD performance metrics using functions in `utils/`
5. **Visualization**: Generate plots and analysis using `plot_utils.py`

### Important Patterns

- Model configurations are stored as JSON files in `model_configures/` directory
- Splitting strategies implement a consistent interface via `BaseMolecularSplitter`
- Use `get_splitter()` factory function to create splitters by name
- All splitters accept SMILES in their `split()` method (SMILES-first API)
- All major functionality supports both classical ML and GNN approaches
- The framework emphasizes reproducibility through configurable parameters
- Extensive logging is integrated throughout the codebase

### Splitting Script

The `scripts/splitting.py` CLI tool supports all splitting strategies:

```bash
# List available splitters
python scripts/splitting.py --list-splitters

# Run a specific splitter (dry-run to preview)
python scripts/splitting.py -f data/molecules.csv -sp scaffold --dry-run

# Run and save splits
python scripts/splitting.py -f data/molecules.csv -sp scaffold --save

# Run all splitters at once
python scripts/splitting.py -f data/molecules.csv -sp all --save
```

### Dataset Integration

The framework integrates with multiple molecular datasets including:
- TDC (Therapeutics Data Commons) datasets
- Custom molecular property datasets
- Various molecular representation formats (SMILES, molecular graphs)

### Key Dependencies

- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **Molecular**: RDKit for chemical informatics
- **ML**: XGBoost, LightGBM for classical ML
- **DL**: PyTorch ecosystem for deep learning models
- **Graph**: DGL for graph neural networks