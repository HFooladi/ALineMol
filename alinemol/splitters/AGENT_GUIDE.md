# ALineMol splitters: guide for agents and scripts

Split a set of molecules into train/test in many ways and measure how far the test
set is from the training set. Input is SMILES; output is index arrays plus plain,
JSON-serialisable measurements.

Print this guide any time with `alinemol guide`.

## Install

```bash
pip install "git+https://github.com/HFooladi/ALineMol.git"
```

The default install is enough for everything in this guide (no torch, no DGL).
Python 3.9–3.11.

## The one call you usually want

```python
import pandas as pd
from alinemol.splitters import split_and_analyze

df = pd.read_csv("molecules.csv")            # needs a SMILES column
result = split_and_analyze(
    df["smiles"].tolist(),
    "scaffold",                              # any name from `alinemol list`
    labels=df["label"].to_numpy(),           # optional
    n_splits=5,
    test_size=0.2,
)

train_idx, test_idx = result.splits[0]       # numpy index arrays into df
train, test = df.iloc[train_idx], df.iloc[test_idx]

result.summary()        # DataFrame, one row per split
result.warnings         # read these: e.g. realised test size far from the request
result.to_json()        # everything, machine-readable
result.save("runs/splits/molecules/scaffold", data=df)   # result.json + train_i.csv / test_i.csv
```

Defaults: `random_state=42` (reproducible), `n_splits=5`, `test_size=0.2`, shift measured
with `k=5` nearest training molecules.

Several splitters at once, sharing fingerprints:

```python
from alinemol.splitters import compare_splits

comparison = compare_splits(smiles, ["random", "scaffold", "kmeans", "umap"], labels)
comparison.to_frame()   # one row per splitter (means over its splits)
comparison.errors       # {name: message} for splitters that failed; never silently dropped
```

## From the shell

```bash
alinemol list                       # splitters and how they treat test size
alinemol list --json                # plus parameters, defaults, aliases
alinemol split -f molecules.csv -s scaffold -s kmeans --json
alinemol split -f molecules.csv -s all -o runs/splits/molecules      # saves one folder per splitter
alinemol analyze -f molecules.csv --splits runs/splits/molecules/scaffold --json
```

- The SMILES column is auto-detected (`smiles`, `canonical_smiles`, ...); set `--smiles-column` otherwise.
- Nothing is written without `-o DIR` (or `--save`). Existing results are not replaced without `--overwrite`.
- Exit codes: 0 success, 1 at least one splitter failed (details on stderr / in `errors`), 2 bad input.
- `alinemol analyze` also reads folders that only contain `train_i.csv` / `test_i.csv`.

## Choosing a splitter

| Name | Based on | Test size | Notes |
|---|---|---|---|
| `random` | nothing (baseline) | exact | always include it as the reference |
| `scaffold`, `scaffold_generic` | Bemis-Murcko scaffolds | approximate | standard "new chemotype" split |
| `molecular_weight`, `molecular_weight_reverse`, `molecular_logp` | a property | exact | one split only (deterministic) |
| `kmeans`, `max_dissimilarity`, `perimeter`, `umap`, `butina`, `scaffold_kmeans` | clusters in fingerprint space | approximate | whole clusters go to one side |
| `hi` | similarity threshold (optimised) | ignored | one split; slow on large sets |
| `lo` | similarity clusters + activity | ignored | needs continuous activity values as `labels` |
| `datasail` | DataSAIL optimiser | approximate | needs `pip install datasail` |

`alinemol list --json` (or `describe_splitters()`) is the authoritative, always-current version of
this table, including every constructor parameter and default.

Extra constructor arguments go straight through:
`split_and_analyze(smiles, "kmeans", n_clusters=10)` or `--param n_clusters=10`.
`preset="paper"` (`--preset paper`) applies the settings used for the published ALineMol splits.

## Reading the shift numbers

All distances are **Tanimoto distances** on ECFP4 fingerprints (2048 bits), from each test
molecule to training molecules. 0 = identical molecule in train, 1 = nothing in common.
**Larger = the test set is further from the training set = a harder split.**

Per split, `result.shift[i]` holds:

- `nearest`: distance to the single closest training molecule (`1 - max similarity`).
- `k_nearest`: distances to the `k` closest training molecules, pooled (the paper's definition, k=5).

Each is a dict with `n, mean, std, min, p05, p25, median, p75, p95, max`.
`result.reports[i]` adds scaffold overlap, property shifts (KS tests for MW, LogP, TPSA, HBD, HBA)
and class balance; `result.reports[i].nearest_similarities` is the per-test-molecule vector.

Always compare against `random` **on the same dataset**: the baseline distance differs a lot
between datasets, so an absolute number alone says little.

Lower-level pieces if you already have splits:

```python
from alinemol.splitters import nearest_train_distances, summarize_distances, SplitAnalyzer

d = nearest_train_distances(train_idx, test_idx, smiles=smiles, k=5)       # (n_test, k), nearest first
d = nearest_train_distances(train_idx, test_idx, distance_matrix="J.npy")  # reuse a precomputed matrix
summarize_distances(d)
```

A precomputed square distance matrix (any metric, rows in the same order as the molecules)
can be passed as `distance_matrix=` to all of the above; it is memory-mapped.

## Things that will surprise you

- **Requested test size is not guaranteed.** Group-based splitters assign whole scaffolds or
  clusters, so `test_size=0.2` can come back as 0.05 or 0.6 on a small or lumpy dataset. The realised
  fraction is in `result.test_fractions` and flagged in `result.warnings`. Check it.
- **Deterministic splitters return one split** whatever `n_splits` says (noted in `warnings`).
- **`hi` and `lo` ignore `test_size`.**
- **Results are seeded by default** (`random_state=42`). Pass another seed for a different draw, or
  `None` for an unseeded one.
- **Low-level `get_splitter(name).split(smiles)` is unseeded by default** and yields a generator of
  `(train_idx, test_idx)`; prefer `split_and_analyze` unless you need the scikit-learn interface.
- **Unparseable SMILES** get an all-zero fingerprint (distance 1 to everything); clean the input first.
- **Class balance**: with labels, `max_label_imbalance=0.1` keeps only splits whose train/test mean
  label differ by at most 0.1 (up to 100 candidates are drawn).

## Saved result format (`result.json`, schema_version 1)

```json
{
  "schema_version": 1,
  "alinemol_version": "...",
  "splitter": "scaffold",
  "params": {"make_generic": false, "n_splits": 5, "test_size": 0.2, "random_state": 42, "...": "..."},
  "requested": {"n_splits": 5, "test_size": 0.2, "random_state": 42, "preset": null},
  "n_molecules": 1000,
  "n_splits": 5,
  "warnings": [],
  "splits": [
    {"split": 0, "train_size": 800, "test_size": 200, "test_fraction": 0.2,
     "train_idx": [0, 1, "..."], "test_idx": [5, 9, "..."],
     "shift": {"k": 5, "metric": "tanimoto_distance_ecfp4_2048", "nearest": {"median": 0.61, "...": "..."}, "k_nearest": {"...": "..."}},
     "report": {"size_metrics": {}, "similarity_metrics": {}, "scaffold_metrics": {}, "property_distributions": []}}
  ]
}
```

`train_i.csv` / `test_i.csv` are the input rows plus a leading `row_index` column pointing back to
the input file. Read indices back with `alinemol.splitters.load_splits(path)`.

## In the ALineMol repository itself

The published data (`datasets/`, `classification_results/`, ...) is frozen and read-only. Saving
into it is refused; write under `runs/` instead. See `provenance/README.md`.
