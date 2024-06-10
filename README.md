# ALineMol

Exploring performance of machine learning model on  out of distribution data in chemical domain.

The goal is to explore and assess quantitavely the relationship between the performance of machine learning models on out-of-distribution (OOD) and the in-distribution (ID) test data.

## Installation
`ALineMol` can be installed using pip. First, create a new conda environment with the required packages. Then, clone this repository, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate alinemol

pip install --no-deps -e .
```


## Development
### Tests

You can run tests locally with:

```bash
pytest
```

### Code style
We use `ruff` as a linter and formatter. 

```bash
ruff check
ruff format
```

### Documentation

You can build and run documentation server with:

```bash
mkdocs serve
```

## Citation <a name="citation"></a>
If you find the models useful in your research, we ask that you cite the following paper:
