# `alinemol.splitters`

The splitters module provides a unified API for molecular dataset splitting with multiple strategies to simulate different types of distribution shift.

## Quick Start

### Using the Factory Function (Recommended)

The easiest way to create splitters is using the `get_splitter()` factory function:

```python
from alinemol.splitters import get_splitter, get_splitter_names

# List all available splitters
print(get_splitter_names())
# ['butina', 'datasail', 'hi', 'kmeans', 'lo', 'max_dissimilarity',
#  'molecular_logp', 'molecular_weight', 'molecular_weight_reverse',
#  'perimeter', 'random', 'scaffold', 'scaffold_generic', 'umap']

# Create a splitter
splitter = get_splitter("scaffold", make_generic=True, n_splits=5, test_size=0.2)

# Use with SMILES directly
smiles = ["CCO", "c1ccccc1", "CCN", ...]
for train_idx, test_idx in splitter.split(smiles):
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]
```

### Direct Class Import

You can also import splitter classes directly:

```python
from alinemol.splitters import ScaffoldSplit, KMeansSplit, MolecularWeightSplit

# Create splitter instance
splitter = ScaffoldSplit(make_generic=True, n_splits=5, test_size=0.2)

# Split your data
for train_idx, test_idx in splitter.split(smiles_list):
    # train_idx and test_idx are numpy arrays of indices
    pass
```

## Available Splitters

### Structure-Based Splitters

| Splitter | Description |
|----------|-------------|
| `scaffold` | Bemis-Murcko scaffold-based splitting |
| `scaffold_generic` | Generic scaffold-based splitting |
| `butina` | Taylor-Butina clustering algorithm |

### Property-Based Splitters

| Splitter | Description |
|----------|-------------|
| `molecular_weight` | Split by molecular weight (test on larger) |
| `molecular_weight_reverse` | Split by molecular weight (test on smaller) |
| `molecular_logp` | Split by lipophilicity (LogP) |

### Clustering-Based Splitters

| Splitter | Description |
|----------|-------------|
| `kmeans` | K-means clustering on fingerprints |
| `umap` | UMAP + hierarchical clustering |
| `max_dissimilarity` | Maximum dissimilarity selection |
| `perimeter` | Perimeter-based sampling |

### Similarity-Based Splitters

| Splitter | Description |
|----------|-------------|
| `hi` | Hi-split: ensures low train/test similarity |
| `lo` | Lo-split: for lead optimization scenarios |

### Other Splitters

| Splitter | Description |
|----------|-------------|
| `random` | Random baseline splitting |
| `datasail` | DataSAIL integration for advanced splitting |

## Factory Functions

::: alinemol.splitters.factory.get_splitter

::: alinemol.splitters.factory.get_splitter_names

::: alinemol.splitters.factory.list_splitters

::: alinemol.splitters.factory.register_splitter

## Base Class

::: alinemol.splitters.base.BaseMolecularSplitter

## Splitter Classes

### Wrapper Classes (splito-based)

::: alinemol.splitters.wrappers.ScaffoldSplit

::: alinemol.splitters.wrappers.KMeansSplit

::: alinemol.splitters.wrappers.MolecularWeightSplit

::: alinemol.splitters.wrappers.MaxDissimilaritySplit

::: alinemol.splitters.wrappers.PerimeterSplit

### Native Splitters

::: alinemol.splitters.splits.RandomSplit

::: alinemol.splitters.splits.MolecularLogPSplit

::: alinemol.splitters.umap_split.UMAPSplit

::: alinemol.splitters.butina_split.BUTINASplit

### Similarity-Based Splitters

::: alinemol.splitters.lohi.HiSplit

::: alinemol.splitters.lohi.LoSplit

### Advanced Splitters

::: alinemol.splitters.datasail.DataSAILSplit

## Command-Line Interface

The `scripts/splitting.py` tool provides a CLI for dataset splitting:

```bash
# List available splitters
python scripts/splitting.py --list-splitters

# Basic usage
python scripts/splitting.py -f data/molecules.csv -sp scaffold --save

# Run all splitters
python scripts/splitting.py -f data/molecules.csv -sp all --save

# Dry run (preview without saving)
python scripts/splitting.py -f data/molecules.csv -sp kmeans --dry-run

# Custom output directory
python scripts/splitting.py -f data/molecules.csv -sp scaffold --save -o results/
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-f, --file_path` | Path to CSV/TSV file with SMILES column |
| `-sp, --splitter` | Splitter name or "all" for all splitters |
| `-te, --test_size` | Test set fraction (default: 0.2) |
| `-ns, --n_splits` | Number of splits to generate (default: 10) |
| `-o, --output_dir` | Custom output directory |
| `--save` | Save split files to disk |
| `--dry-run` | Preview operations without saving |
| `--list-splitters` | List available splitters and exit |
