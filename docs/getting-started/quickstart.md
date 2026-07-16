# Quickstart

This page walks through the core ALineMol loop: **create an OOD split →
verify it is a real distribution shift → compare against a baseline**. It uses
only the lean base install (no torch/DGL required).

!!! tip "Prefer a notebook?"
    Every step here is also available as a runnable, Colab-ready notebook in the
    [Tutorials](../tutorials/index.md) section.

## 1. Load your molecules

ALineMol's splitter API is **SMILES-first** — you pass a list of SMILES strings
directly to `split()`.

```python
import pandas as pd

df = pd.read_csv("my_dataset.csv")   # a column of SMILES + a label column
smiles = df["smiles"].tolist()
labels = df["label"].tolist()
```

## 2. Create an out-of-distribution split

Use the `get_splitter()` factory to build any of the available splitters by name.
A scaffold split holds out whole Bemis–Murcko scaffolds, simulating deployment on
novel chemical series.

```python
from alinemol.splitters import get_splitter, get_splitter_names

print(get_splitter_names())
# ['butina', 'datasail', 'hi', 'kmeans', 'lo', 'max_dissimilarity',
#  'molecular_logp', 'molecular_weight', 'molecular_weight_reverse',
#  'perimeter', 'random', 'scaffold', 'scaffold_generic',
#  'scaffold_kmeans', 'umap']

splitter = get_splitter("scaffold", n_splits=5, test_size=0.2)

for train_idx, test_idx in splitter.split(smiles):
    train_smiles = [smiles[i] for i in train_idx]
    test_smiles = [smiles[i] for i in test_idx]
    # ... train and evaluate your model on this fold
```

## 3. Verify the split is really OOD

A splitter *name* does not guarantee distribution shift. `SplitAnalyzer`
quantifies how dissimilar the test set is from the training set.

```python
from alinemol.splitters import SplitAnalyzer

analyzer = SplitAnalyzer(smiles)
train_idx, test_idx = next(splitter.split(smiles))

report = analyzer.analyze_split(train_idx, test_idx, splitter_name="scaffold")
print(f"Mean train-test similarity: {report.similarity_metrics.mean_sim:.3f}")
print(f"Scaffold overlap:           {report.scaffold_metrics.scaffold_overlap_percentage:.1f}%")
```

Lower train↔test similarity and lower scaffold overlap mean a stronger shift.

## 4. Compare splitters against the random baseline

```python
comparison = analyzer.compare_splitters(["random", "scaffold", "kmeans"])
print(comparison)   # a pandas DataFrame, one row per splitter
```

The `random` splitter is your ID reference point; structure- and
clustering-based splitters should show measurably lower similarity.

## From the command line

Everything above is also available through the `scripts/splitting.py` CLI:

```bash
# List splitters
python scripts/splitting.py --list-splitters

# Scaffold split, saved to disk
python scripts/splitting.py -f my_dataset.csv -sp scaffold --save

# Run every splitter at once
python scripts/splitting.py -f my_dataset.csv -sp all --save
```

## Where to go next

<div class="grid cards" markdown>

-   **Understand OOD evaluation** — the ID/OOD workflow and what "distribution
    shift" means for molecules.

    [:octicons-arrow-right-24: OOD evaluation](../guide/ood-evaluation.md)

-   **Choose a splitter** — a tour of all 16 strategies and when to use each.

    [:octicons-arrow-right-24: Splitting strategies](../guide/splitting-strategies.md)

-   **Analyze split quality** — the full `SplitAnalyzer` guide.

    [:octicons-arrow-right-24: Split analysis](../guide/split-analysis.md)

-   **API reference** — every class and function, auto-generated from source.

    [:octicons-arrow-right-24: Splitters API](../api/splitters.md)

</div>
