# AGENTS.md

Guidance for AI coding agents working with this repository or using it as a dependency.

## Using ALineMol to split data (the common case)

Read [`alinemol/splitters/AGENT_GUIDE.md`](alinemol/splitters/AGENT_GUIDE.md) (also printed by
`alinemol guide`). In short:

```python
from alinemol.splitters import split_and_analyze, compare_splits
result = split_and_analyze(smiles, "scaffold", labels, n_splits=5, test_size=0.2)
train_idx, test_idx = result.splits[0]
result.summary(); result.warnings; result.to_json()
```

```bash
alinemol list --json
alinemol split -f molecules.csv -s scaffold -s kmeans --json
```

## Working in this repository

- **Published data is frozen.** `datasets/`, `classification_results/`,
  `classification_inference_results/`, `assets/figures/` and `assets/tables/` are hashed and
  read-only (`provenance/README.md`). Never write there and never run
  `scripts/data_freeze.py unlock` unless the user explicitly asks. New outputs go under `runs/`.
- **Exploratory figures** (not part of the paper) live in `assets/exploratory/`, each made by a
  numbered script in `scripts/exploratory/`.
- Development commands, architecture and conventions: [`CLAUDE.md`](CLAUDE.md).
- Tests: `DGL_SKIP_GRAPHBOLT=1 pytest -m "not slow"`. Lint: `ruff check && ruff format`.
