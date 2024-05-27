import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def manual_smiles():
    return [
        "CCCCC",
        "C1=CC=CC=C1",
        "CCCCOC(C1=CC=CC=C1)OCCCC",
        "CC1=CC(=CC(=C1O)C)C(=O)C",
        "CCN(CC)S(=O)(=O)C1=CC=C(C=C1)C(=O)OCC",
        "C[Si](C)(C)CC1=CC=CC=C1",
        "CN1C=NC2=C1C(=O)NC(=O)N2C",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]


@pytest.fixture(scope="module")
def manual_smiles_for_scaffold():
    return [
    "CCOC1=CC=CC=C1C(=O)OCC(=O)NC1=CC=CC=C1",
    "NC(=O)C1=C(NC(=O)COC2=CC=CC=C2C(F)(F)F)SC=C1",
    "CC(C)NC(=O)CSCC1=CC=CC=C1Br",
    "CC1=CC=C(C(=O)NC(C)C)C=C1NC(=O)C1=CC=CO1",
    "O=C(CN1CCCCCC1=O)NCC1=CC=C(N2C=CN=C2)C(F)=C1",
]

@pytest.fixture(scope="module")
def manual_df_for_drop_duplicate():
    smiles  = ['CC', 'CC', 'CC', 'CCC', 'CCC', 'CCCC', 'CCCCC', 'CCCCC']
    label= [1, 1, 0, 1, 1, 1, 1, 0]
    df = pd.DataFrame({'smiles': smiles, 'label': label})
    return df


@pytest.fixture(scope="module")
def dataset_dili():
    # Load the dataset
    # The DILI dataset contains SMILES strings and binary labels
    # size of the datsaet: 475
    # number of unique scaffolds (Bemis-Murcko): 311
    # number of molecules with empty scaffolds (Bemis-Murcko): 35
    # number ofactive/inactive: 236/239
    dataset = pd.read_csv("tests/conftest/dili.csv")
    return dataset