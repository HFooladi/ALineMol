import math

import pytest
from alinemol.utils import split_molecules_train_test

# Check if astartes[molecules] extra is available (requires aimsim)
try:
    from aimsim.chemical_datastructures import Molecule

    ASTARTES_MOLECULES_AVAILABLE = True
except ImportError:
    ASTARTES_MOLECULES_AVAILABLE = False


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


# Test 1: Test the split_molecules_train_test function
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


@pytest.mark.skipif(not ASTARTES_MOLECULES_AVAILABLE, reason="astartes[molecules] not installed")
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


@pytest.mark.skipif(not ASTARTES_MOLECULES_AVAILABLE, reason="astartes[molecules] not installed")
def test_sphere_exclusion_splitting(test_dataset_dili):
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    split_type = "sphere_exclusion"  # Fixed: was incorrectly set to "optisim"
    train, external_test = split_molecules_train_test(
        test_dataset_dili, sampler=split_type, train_size=0.9, random_state=42
    )
    assert train.shape[0] > 0
    assert external_test.shape[0] > 0
    assert train.shape[0] + external_test.shape[0] == test_dataset_dili.shape[0]
