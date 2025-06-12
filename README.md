# ALineMol: Evaluating Machine Learning Models for Molecular Property Prediction on OOD Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/HFooladi/ALineMol/actions/workflows/ci.yml/badge.svg)](https://github.com/HFooladi/ALineMol/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2025--g1vjf--v2-blue)](https://doi.org/10.26434/chemrxiv-2025-g1vjf-v2)

<p align="center">
  <img src="assets/banner/alinemol_banner.png" alt="ALineMol Banner" style="max-width:100%;">
</p>

## Overview

**ALineMol** is a comprehensive research framework for evaluating and quantitatively assessing the relationship between machine learning model performance on in-distribution (ID) and out-of-distribution (OOD) data in the molecular domain. This work addresses critical questions in AI-driven drug discovery about model generalization to novel chemical structures.

### Key Contributions

🔬 **Comprehensive Evaluation**: Systematic assessment of ML models (classical ML + GNNs) across multiple datasets using different splitting strategies

📊 **Distribution Shift Analysis**: Quantitative investigation of what constitutes "out-of-distribution" data in molecular property prediction

🎯 **ID-OOD Relationship**: Deep analysis of correlation between in-distribution and out-of-distribution performance across different scenarios

⚗️ **Drug Discovery Focus**: Practical insights for molecular property prediction and bioactivity classification in pharmaceutical research

### Setup

```bash
# Clone the repository
git clone https://github.com/HFooladi/ALineMol.git
cd ALineMol

# Create and activate conda environment
conda env create -f environment.yml
conda activate alinemol

# Install ALineMol package
pip install --no-deps -e .
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from alinemol.preprocessing import standardization_pipeline
from alinemol.splitters import ScaffoldSplit, MolecularWeightSplit
from alinemol.utils import compute_similarities

# Load and preprocess data
df = pd.read_csv("your_dataset.csv")  # Columns: 'smiles', 'label'
df_clean = standardization_pipeline(df)

# Create different types of splits
scaffold_splitter = ScaffoldSplit(test_size=0.2)
weight_splitter = MolecularWeightSplit(test_size=0.2, generalize_to_larger=True)

# Evaluate different splitting strategies
for train_idx, test_idx in scaffold_splitter.split(df_clean['smiles']):
    train_data = df_clean.iloc[train_idx]
    test_data = df_clean.iloc[test_idx]
    
    # Compute molecular similarities
    similarities = compute_similarities(
        train_data['smiles'], 
        test_data['smiles'],
        fingerprint='ecfp',
        fprints_hopts={'radius': 2, 'fpSize': 1024}
    )
    print(f"Average train-test similarity: {similarities.mean():.3f}")
```

### Comprehensive Evaluation Pipeline

```python
from alinemol.utils import load_dataset, split_dataset, compute_ID_OOD
from alinemol.utils.plot_utils import plot_ID_OOD_sns, heatmap_plot

# Evaluate multiple models across different split types
results = compute_ID_OOD(
    dataset_category="TDC",
    dataset_names="CYP2C19", 
    split_type="scaffold",
    num_of_splits=10
)

# Visualize ID vs OOD performance
plot_ID_OOD_sns(results, dataset_name="CYP2C19", save=True)

# Create performance heatmaps
heatmap_plot(results, metric="roc_auc", save=True)
```

## Splitting Strategies

ALineMol implements various molecular splitting strategies to simulate different types of distribution shift:

### 1. Structure-Based Splits
```python
from alinemol.splitters import ScaffoldSplit, PerimeterSplit

# Bemis-Murcko scaffold splitting
scaffold_split = ScaffoldSplit(make_generic=True)

# Perimeter-based clustering
perimeter_split = PerimeterSplit(n_clusters=10)
```

### 2. Property-Based Splits
```python
from alinemol.splitters import MolecularWeightSplit, MolecularLogPSplit

# Split by molecular weight (test on larger molecules)
mw_split = MolecularWeightSplit(generalize_to_larger=True)

# Split by lipophilicity
logp_split = MolecularLogPSplit(generalize_to_larger=True)
```

### 3. Similarity-Based Splits
```python
from alinemol.splitters.lohi import HiSplit, LoSplit

# Hi-split: ensures low similarity between train/test
hi_split = HiSplit(
    similarity_threshold=0.4,
    train_min_frac=0.7,
    test_min_frac=0.15
)

# Lo-split: for lead optimization scenarios
lo_split = LoSplit(
    threshold=0.4,
    min_cluster_size=5,
    std_threshold=0.6
)
```

### 4. Clustering-Based Splits
```python
from alinemol.splitters import UMAPSplit, KMeansSplit

# UMAP + clustering split
umap_split = UMAPSplit(
    n_clusters=20,
    n_neighbors=100,
    min_dist=0.1
)

# K-means clustering split  
kmeans_split = KMeansSplit(n_clusters=10, metric="jaccard")
```

## Development

### Tests

Run the test suite with pytest:

```bash
pytest
```

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code style
ruff check

# Format code
ruff format
```

### Documentation

Build and serve the documentation locally:

```bash
mkdocs serve
```

### Continuous Integration

This project uses GitHub Actions for continuous integration and deployment:

- **CI Workflow**: Automatically runs tests and linting on all pull requests and pushes to the main branch
- **Release Workflow**: Automatically builds and publishes the package to PyPI when a new release is created

To create a new release:

1. Update the version in `_version.py`
2. Create a new tag and GitHub release
3. The release workflow will automatically publish to PyPI

## Citation

If you find ALineMol useful in your research, please cite the following paper:

```bibtex
@article{fooladi2025evaluating,
  title={Evaluating Machine Learning Models for Molecular Property Prediction: Performance and Robustness on Out-of-Distribution Data},
  author={Fooladi, Hosein and Vu, Thi Ngoc Lan and Kirchmair, Johannes},
  year={2025},
  doi = {https://doi.org/10.26434/chemrxiv-2025-g1vjf-v2}
}
```

## Related Work

- **Splito**: Molecular splitting library - [GitHub](https://github.com/datamol-io/splito)
- **TDC**: Therapeutics Data Commons - [Website](https://tdcommons.ai/)
- **DGL-LifeSci**: Deep Graph Library for Life Sciences - [GitHub](https://github.com/awslabs/dgl-lifesci)

## Documentation

- 📖 [Full Documentation](docs/)
- 📝 [API Reference](docs/api/)
- 🎓 [Tutorials](docs/tutorials/)
- 📊 [Paper](https://doi.org/10.26434/chemrxiv-2025-g1vjf-v2)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Reporting bugs
- Suggesting enhancements  
- Submitting pull requests
- Code style guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
