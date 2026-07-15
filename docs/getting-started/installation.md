# Installation

ALineMol supports **Python 3.9, 3.10, and 3.11**. The default install is lean
(splitter API + `SplitAnalyzer` only); heavier components are opt-in extras.

## Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

=== "Quick install (script)"

    ```bash
    git clone https://github.com/HFooladi/ALineMol.git
    cd ALineMol

    # CPU (default)
    ./install.sh

    # Or with CUDA support
    ./install.sh cu121   # CUDA 12.1
    ./install.sh cu118   # CUDA 11.8
    ./install.sh cu124   # CUDA 12.4

    source .venv/bin/activate
    ```

=== "Manual"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    git clone https://github.com/HFooladi/ALineMol.git
    cd ALineMol

    uv venv --python 3.11
    source .venv/bin/activate

    # CPU
    uv pip install -e ".[all]" \
      -f https://download.pytorch.org/whl/cpu \
      -f https://data.dgl.ai/wheels/repo.html

    # Or CUDA 12.1
    uv pip install -e ".[all]" \
      -f https://download.pytorch.org/whl/cu121 \
      -f https://data.dgl.ai/wheels/repo.html
    ```

## Using conda

```bash
git clone https://github.com/HFooladi/ALineMol.git
cd ALineMol

conda env create -f environment.yml
conda activate alinemol
pip install --no-deps -e .
```

## Optional extras

The lean base install ships only the splitter API and `SplitAnalyzer`. Pull in
heavier components on demand:

| Extra | Pulls in | When you need it |
|---|---|---|
| `[gnn]` | torch, dgl, dgllife, torch-geometric | Training/inference with GNN models |
| `[ml]` | statsmodels, POT, astartes | `alinemol.utils`, OT-based graph utilities, `astartes`-backed splitters |
| `[datasail]` | datasail | The `datasail` splitter |
| `[all]` | gnn + ml + datasail | Everything (used by `install.sh`) |
| `[docs]` | mkdocs-material, mkdocstrings, mike, … | Building this documentation |
| `[dev]` | ruff, pre-commit, mypy | Contributing |
| `[test]` | pytest, pytest-cov | Running the test suite |

```bash
# Splitters + GNN training only
uv pip install -e ".[gnn]"

# Everything, plus dev and test tooling
uv pip install -e ".[all,dev,test]"
```

!!! tip "GNN stack pin"
    DGL 2.1.0 ships prebuilt GraphBolt binaries only for torch 2.0.0–2.2.1, so
    the `[gnn]` extra caps torch at `<2.2.2`. Set `DGL_SKIP_GRAPHBOLT=1` if you
    hit a GraphBolt import error.

## Verify the install

```bash
python -c "from alinemol.splitters import get_splitter_names; print(get_splitter_names())"
```

You should see the list of available splitters. Next, head to the
[Quickstart](quickstart.md).
