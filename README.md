# ALineMol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/HFooladi/ALineMol/actions/workflows/ci.yml/badge.svg)](https://github.com/HFooladi/ALineMol/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2025--g1vjf--v2-blue)](https://doi.org/10.26434/chemrxiv-2025-g1vjf-v2)


## Overview

ALineMol is a research framework for exploring and quantitatively assessing the relationship between machine learning model performance on in-distribution (ID) and out-of-distribution (OOD) data in the chemical domain.

The project aims to provide robust evaluation methods to understand how well molecular property prediction models generalize to novel chemical structures beyond their training distributions.

## Key Features

- Systematic evaluation of machine learning models on OOD chemical data
- Benchmarking tools for molecular property prediction
- Customizable data splitting strategies for distribution shift analysis
- Preprocessing utilities for chemical structure representations

## Installation

ALineMol can be installed using pip. Follow these steps:

```bash
# Create and activate a new conda environment
conda env create -f environment.yml
conda activate alinemol

# Install the package
pip install --no-deps -e .
```

## Usage

Basic usage examples coming soon.

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
