from alinemol.utils.split_utils import *


def test_featurize():
    pass


def test_compute_similarities():
    pass


def test_split_hypers():
    pass


def test_split_molecules_train_test():
    pass


def test_split_molecules_train_val_test():
    pass


def test_get_scaffold(manual_smiles_for_scaffold):
    scaffold = [get_scaffold(smiles) for smiles in manual_smiles_for_scaffold]
    for i in range(len(manual_smiles_for_scaffold)):
        assert isinstance(scaffold[i], str)
        assert len(scaffold[i]) > 0
        assert len(scaffold[i]) <= len(manual_smiles_for_scaffold[i])
    
    