import math

import pytest
from alinemol.splitters.splits import *
from alinemol.utils import split_molecules_train_test


@pytest.fixture
def molecules():
    return ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]


"""
def test_MolecularWeight(molecules, manual_smiles):
    mw = MolecularWeight(molecules)
    mw2 = MolecularWeight(manual_smiles)
    mw._before_sample()
    mw._sample()
    mw._after_sample()
"""


def test_random_splitting(test_dataset_dili):
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    split_type = "scaffold"
    train, external_test = split_molecules_train_test(
        test_dataset_dili, sampler=split_type, train_size=0.9, random_state=42
    )
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == test_dataset_dili.shape[0]
    assert math.ceil(test_dataset_dili.shape[0] * 0.9) == train.shape[0]


def test_scaffold_splitting(test_dataset_dili):
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    split_type = "scaffold"
    train, external_test = split_molecules_train_test(
        test_dataset_dili, sampler=split_type, train_size=0.9, random_state=42
    )
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == test_dataset_dili.shape[0]


def test_optisim_splitting(test_dataset_dili):
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    split_type = "optisim"
    train, external_test = split_molecules_train_test(
        test_dataset_dili, sampler=split_type, train_size=0.9, random_state=42
    )
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == test_dataset_dili.shape[0]


def test_sphere_exclusion_splitting(test_dataset_dili):
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    split_type = "optisim"
    train, external_test = split_molecules_train_test(
        test_dataset_dili, sampler=split_type, train_size=0.9, random_state=42
    )
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == test_dataset_dili.shape[0]
