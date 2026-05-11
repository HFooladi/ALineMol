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
#  'perimeter', 'random', 'scaffold', 'scaffold_generic',
#  'scaffold_kmeans', 'umap']

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
| `scaffold_kmeans` | Scaffold extraction + k-means on scaffold ECFP |

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

::: alinemol.splitters.scaffold_kmeans_split.ScaffoldKMeansSplit

### Similarity-Based Splitters

::: alinemol.splitters.lohi.HiSplit

::: alinemol.splitters.lohi.LoSplit

### Advanced Splitters

::: alinemol.splitters.datasail.DataSAILSplit

## Split Quality Analysis

`SplitAnalyzer` complements the splitters by quantifying *how* a given split
behaves: train↔test Tanimoto similarity distribution, scaffold overlap,
property-distribution divergence, and basic size metrics. Use it to validate
that an "OOD" splitter actually produced a more dissimilar test set than the
random baseline.

### Basic usage

```python
from alinemol.splitters import SplitAnalyzer, get_splitter

analyzer = SplitAnalyzer(smiles_list)

# Analyze a single split
splitter = get_splitter("scaffold")
train_idx, test_idx = next(splitter.split(smiles_list))
report = analyzer.analyze_split(train_idx, test_idx, splitter_name="scaffold")

print(f"Mean train-test similarity: {report.similarity_metrics.mean_sim:.3f}")
print(f"Scaffold overlap:           {report.scaffold_metrics.scaffold_overlap_percentage:.1f}%")

# Compare multiple splitters in one DataFrame
comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
```

### Reusing a precomputed Jaccard distance matrix

For studies that analyze many splitter × seed combinations on the same dataset,
recomputing pairwise Tanimoto similarity inside every `analyze_split` call is
the dominant cost. `SplitAnalyzer` can consume a precomputed pairwise Jaccard
distance matrix and slice into it instead:

```python
import numpy as np
from alinemol.splitters import SplitAnalyzer

# Either pass a path to an .npy file (loaded with mmap)...
analyzer = SplitAnalyzer(
    smiles_list,
    precomputed_distance_matrix="datasets/TDC/CYP2C9/Jaccard_distance.npy",
)

# ...or pass the array directly.
distance = np.load("datasets/TDC/CYP2C9/Jaccard_distance.npy")
analyzer = SplitAnalyzer(smiles_list, precomputed_distance_matrix=distance)
```

!!! warning "SMILES order must match the matrix"
    The matrix rows/columns are indexed by the same integer indices as the SMILES
    list passed to `SplitAnalyzer`. Use `datasets/TDC/<NAME>/valid_canonical_smiles.txt`
    (the SMILES list the matrix was built from) and pass it through in the same
    order. The CLI helper `scripts/analyze_splits.py:resolve_precomputed_distance`
    auto-detects and validates this alignment.

The precomputed `Jaccard_distance.npy` files shipped with each TDC dataset are
computed from Morgan radius=2, 2048-bit fingerprints — which is also the default
`SplitAnalyzer` fingerprint configuration. (Splitter clustering, by contrast,
uses 1024-bit ECFP to preserve the cluster boundaries of the published splits.)

::: alinemol.splitters.analyzer.SplitAnalyzer

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
