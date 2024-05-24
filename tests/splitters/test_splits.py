import math

import pytest
from alinemol.splitters.splits import *
from alinemol.utils import split_molecules_train_test


@pytest.fixture
def molecules():
    return ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]

'''
def test_MolecularWeight(molecules, manual_smiles):
    mw = MolecularWeight(molecules)
    mw2 = MolecularWeight(manual_smiles)
    mw._before_sample()
    mw._sample()
    mw._after_sample()
'''

def test_random_splitting(dataset_dili):
    assert 'smiles' in dataset_dili.columns
    assert 'label' in dataset_dili.columns
    split_type="scaffold"
    train, external_test = split_molecules_train_test(dataset_dili, sampler=split_type, train_size=0.9, random_state=42)
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == dataset_dili.shape[0]
    assert math.ceil(dataset_dili.shape[0] * 0.9) == train.shape[0]



def test_scaffold_splitting(dataset_dili):
    assert 'smiles' in dataset_dili.columns
    assert 'label' in dataset_dili.columns
    split_type="scaffold"
    train, external_test = split_molecules_train_test(dataset_dili, sampler=split_type, train_size=0.9, random_state=42)
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == dataset_dili.shape[0]


def test_optisim_splitting(dataset_dili):
    assert 'smiles' in dataset_dili.columns
    assert 'label' in dataset_dili.columns
    split_type="optisim"
    train, external_test = split_molecules_train_test(dataset_dili, sampler=split_type, train_size=0.9, random_state=42)
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == dataset_dili.shape[0]


def test_sphere_exclusion_splitting(dataset_dili):
    assert 'smiles' in dataset_dili.columns
    assert 'label' in dataset_dili.columns
    split_type="optisim"
    train, external_test = split_molecules_train_test(dataset_dili, sampler=split_type, train_size=0.9, random_state=42)
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == dataset_dili.shape[0]