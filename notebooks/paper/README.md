# Reproducing the paper

These three notebooks produce every figure and table in:

> Fooladi, H.; Vu, T. N. L.; Mathea, M.; Kirchmair, J.
> **Evaluating Machine Learning Models for Molecular Property Prediction:
> Performance and Robustness on Out-of-Distribution Data.**
> *Journal of Chemical Information and Modeling* **2025**, *65* (19), 9871–9891.
> [doi:10.1021/acs.jcim.5c00475](https://doi.org/10.1021/acs.jcim.5c00475)

Notebooks outside this directory (`../exploratory/`) are ongoing research and
are **not** part of the published results.

## Install

The paper notebooks need the full stack (GNNs + ML utilities), not the lean
splitter-only install:

```bash
./install.sh cu121          # or: ./install.sh cpu
source .venv/bin/activate
```

or manually:

```bash
uv pip install -e ".[all]" -f https://download.pytorch.org/whl/cu121 \
                           -f https://data.dgl.ai/wheels/repo.html
```

## ⚠️ Data availability — read this first

**A fresh clone cannot run these notebooks.** Their inputs are model training
and inference results, which are far too large for git and are excluded by
`.gitignore`:

| Path | Contents | In git? |
|---|---|---|
| `datasets/*` | TDC datasets and their splits | ❌ (only `config.yml`, `dev_config.yml`) |
| `classification_results/` | Trained models, per-split metrics | ❌ |
| `classification_inference_results/` | Held-out predictions | ❌ |
| `assets/figures/`, `assets/tables/` | Notebook output | ❌ |

You must run the pipeline below to regenerate these before any notebook here
will execute. There is no shortcut archive yet — see *Getting the data* at the
bottom.

## Pipeline

Run from the repository root. Steps 1–4 produce the notebooks' inputs; the
notebooks themselves are step 5.

**1. Obtain the datasets.** The eight TDC datasets used in the paper are
enumerated in [`datasets/config.yml`](../../datasets/config.yml): CYP1A2,
CYP2C9, CYP2C19, CYP2D6, CYP3A4, HIV, AMES, HERG. Each belongs at
`datasets/TDC/<NAME>/<NAME>.csv` with `smiles` and `label` columns, then
standardised to `<NAME>_standardize.csv` via
`alinemol.preprocessing.standardization_pipeline`.

**2. Generate the splits.** All eight splitting strategies from `config.yml`
(`random`, `scaffold`, `scaffold_generic`, `molecular_weight`,
`molecular_weight_reverse`, `molecular_logp`, `kmeans`, `max_dissimilarity`):

```bash
python scripts/splitting.py -f datasets/TDC/<NAME>/<NAME>_standardize.csv -sp all --save
```

**3. Train.** Classical ML, GNNs from scratch, and pretrained GNNs, over every
dataset × split × replicate. Writes `classification_results/`:

```bash
bash run.sh
```

> This is the expensive step — many GPU-hours across the full grid
> (8 datasets × 8 splits × 10 replicates × 12 models). `run.sh` requires a GPU
> and checks for `nvidia-smi` before starting.

**4. Run inference.** Writes `classification_inference_results/`:

```bash
bash run_inference.sh
```

**5. Run the notebooks**, in numbered order.

## What each notebook produces

| Notebook | Output |
|---|---|
| `01_figures.ipynb` | `assets/figures/` — `box_heatmap_id_ood_comparison_roc_auc`, `distance_distribution`, `grouped_barplot_hit_rate`, `grouped_barplot_ml_gnn_difference`, `hit_rate_vs_roc_auc_TDC`, `model_performance_by_splitter`, `radar_subplots`, `test_size_ratio`, `tSNE_visualization_TDC`, `activity_ratios`, `regplot_with_categories` (each as `.pdf` + `.png`) |
| `02_tables.ipynb` | `assets/tables/` — `Model_comparison.tex`, `ML_GNN_comparison.tex`, `ML_GNN_PREGNN_comparison.tex`, `ML_GNN_PREGNN_comparison_slope.tex` |
| `03_poster_figures.ipynb` | Conference-poster restyling of the same results — `assets/**/*_poster.{pdf,png}` |

## ⚠️ These notebooks overwrite their output

`01` and `02` write straight into `assets/figures/` and `assets/tables/`,
replacing whatever is already there. If you have figures you care about, back
up `assets/` before re-running.

Figure styling uses LaTeX. It is opt-in — export `ALINEMOL_USE_LATEX=1` for
paper-quality output, and leave it unset on machines without a LaTeX install.

## Notes

- Each notebook resolves the repository root by walking up to `pyproject.toml`
  and `chdir`s there, so it works regardless of where Jupyter was launched.
- Outputs are stripped from every committed notebook by the `nbstripout`
  pre-commit hook, so cells appear unexecuted on a fresh clone. That is
  expected.

## Getting the data

The intermediate results are not yet published as a downloadable archive, so
reproduction currently means running the full pipeline above. If you want the
trained models or result CSVs directly, please
[open an issue](https://github.com/HFooladi/ALineMol/issues).
