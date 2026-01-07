# Overview

`ALineMol` is a Python library designed to assist in drug discovery by providing powerful methods for estimating the out-of-distribution (OOD) performance of molecular machine learning models, including both classical machine learning (RF, XGB, etc) and graph neural networks (GNNs). Built on top of the popular PyTorch library, `ALineMol` offers a simple, user-friendly API for assessing OOD performance. It is designed to be flexible and easy to integrate into existing workflows.

The library first generates OOD data based on various splitting strategies, then benchmarks and evaluates the performance of different models on this OOD data. This approach helps estimate the generalization power and robustness of models to OOD data.

## Installation

### Using uv (Recommended)

**Quick Install:**
```bash
git clone https://github.com/HFooladi/ALineMol.git
cd ALineMol

# CPU installation
./install.sh

# Or CUDA installation
./install.sh cu121  # CUDA 12.1
./install.sh cu118  # CUDA 11.8
./install.sh cu124  # CUDA 12.4

source .venv/bin/activate
```

**Manual Install:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

# CPU
uv pip install -e . -f https://download.pytorch.org/whl/cpu -f https://data.dgl.ai/wheels/repo.html

# Or CUDA 12.1
uv pip install -e . -f https://download.pytorch.org/whl/cu121 -f https://data.dgl.ai/wheels/repo.html
```

### Using conda

```bash
conda env create -f environment.yml
conda activate alinemol
pip install --no-deps -e .
```