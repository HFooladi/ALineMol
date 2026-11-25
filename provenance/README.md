# Published data: frozen and hashed

The data in this checkout is the data the paper was published with
(J. Chem. Inf. Model., [10.1021/acs.jcim.5c00475](https://doi.org/10.1021/acs.jcim.5c00475)).
It is not in git, so it is protected in two ways:

1. **Hashed.** `provenance/paper-v1/` holds a SHA-256 manifest of every file. The
   manifest is in git, so any later change to the data can be detected.
2. **Locked.** The trees are read-only and carry a `.frozen` marker at their root.
   Scripts in this repo refuse to write into a tree that has the marker.

## What is frozen

| Tree | Contents |
|---|---|
| `datasets/` | raw and standardized datasets, all split files, distance matrices, nearest-distance tables |
| `classification_results/` | trained models, per-run metrics, predictions, aggregated `results.csv` / `hit_rate.csv` |
| `classification_inference_results/` | stand-alone inference predictions |
| `assets/figures/`, `assets/tables/` | the figures and tables as published |

Everything that was in these trees at freeze time is covered. Each file is tagged
`paper` (part of the dataset x splitter x model grid in `datasets/config.yml`) or
`other` (extra datasets, archived splits, earlier experiments).

Not frozen: `logs/`, `TMP/`, `assets/old_figures/`, `assets/exploratory/`, `runs/`.

## Manifest layout

```
provenance/paper-v1/
  manifest.json            summary: date, git commit, file and byte totals per set
  datasets.sha256          sha256sum format: "<hash>  <path>"
  datasets.meta.tsv        size, tag (paper/other), path
  results-core.*           metrics, predictions, configs, logs, figures, tables
  results-weights.*        model weights and graph caches (the bulk of the bytes)
```

## Everyday commands

```bash
python scripts/data_freeze.py status                 # manifest summary and lock state
python scripts/data_freeze.py verify                 # quick: every file exists with the right size
python scripts/data_freeze.py verify --full --set datasets results-core   # re-hash (minutes)
python scripts/data_freeze.py verify --full          # re-hash everything, weights included (hours)
make verify-data                                     # same as the quick check
```

`verify` exits non-zero and lists files that are changed, missing, or unexpected (present
on disk but not in the manifest). Plain `sha256sum -c provenance/paper-v1/datasets.sha256`
from the repo root works too.

## Where new work goes

Write new outputs under `runs/` (git-ignored), never into the frozen trees:

```bash
# splits of a published dataset land in runs/splits/<dataset>/<splitter>/ by default
python scripts/splitting.py -f datasets/TDC/HIV/HIV_standardize.csv -sp scaffold --save

# training / inference: point the result paths at runs/
python scripts/clf_train_ml.py ... -p runs/classification_results/TDC/HIV/scaffold/SVM
python scripts/clf_inference_ml.py ... -tp classification_results/TDC/HIV/scaffold/SVM/1 \
                                        -ip runs/inference/TDC/HIV/scaffold/SVM/1
```

Reading frozen data (loading models, splits, distance matrices) keeps working.
Code that writes should call `alinemol.utils.frozen.assert_not_frozen(path)` first.

## Changing frozen data

Don't, unless you are deliberately making a new release of the data.

```bash
python scripts/data_freeze.py unlock --yes-i-am-sure   # owner write access back, markers removed
# ... make the change ...
python scripts/data_freeze.py --label paper-v2 create  # new manifest; keep paper-v1 untouched
python scripts/data_freeze.py --label paper-v2 lock
```

`unlock` restores write access for the owner only (the original group-write bits are
not restored).

## Limits

This protects against accidental changes and makes any change detectable. It is not a
backup: the data still exists as a single copy on one disk.
