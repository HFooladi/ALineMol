# ALineMol

Exploring performance of machine learning model on  out of distribution data in chemical domain.

The goal is to explore and assess quantitavely the relationship between the performance of machine learning models on out-of-distribution (OOD) and the in-distribution (ID) test data.

## Installation
`ALineMol` can be installed using pip. First, create a new conda environment with the required packages. Then, clone this repository, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate alinemol
git clone https://github.com/HFooladi/ALineMol.git
cd ALineMol 
pip install --no-deps -e .
```