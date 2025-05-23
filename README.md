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

🔬 **Comprehensive Evaluation**: Systematic assessment of 12 ML models (classical ML + GNNs) across 8 TDC datasets using 7 splitting strategies

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
