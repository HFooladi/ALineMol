import numpy as np
import datamol as dm
import pytest

from splito import MolecularWeightSplit, ScaffoldSplit, KMeansSplit

from alinemol.utils.split_utils import get_scaffold


@pytest.mark.parametrize("generalize_to_larger", [True, False])
def test_splits_molecular_weight(test_dataset_dili, generalize_to_larger):
    smiles = test_dataset_dili["smiles"].values
    splitter = MolecularWeightSplit(generalize_to_larger=generalize_to_larger, n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > len(test_ind)
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_mws = [dm.descriptors.mw(dm.to_mol(smi)) for smi in smiles[train_ind]]
        if generalize_to_larger:
            assert all(dm.descriptors.mw(dm.to_mol(smi)) >= max(train_mws) for smi in smiles[test_ind])
        else:
            assert all(dm.descriptors.mw(dm.to_mol(smi)) <= min(train_mws) for smi in smiles[test_ind])


@pytest.mark.parametrize("make_generic", [True, False])
def test_splits_scaffold(test_dataset_dili, make_generic):
    smiles = test_dataset_dili["smiles"].values
    splitter = ScaffoldSplit(smiles, n_splits=2, make_generic=make_generic)
    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_scfs = set([get_scaffold(smiles[i], make_generic=make_generic) for i in train_ind])
        test_scfs = [get_scaffold(smiles[i], make_generic=make_generic) for i in test_ind]
        assert not any(test_scf in train_scfs for test_scf in test_scfs)


def test_splits_kmeans_default_feats(test_dataset_dili):
    smiles = test_dataset_dili["smiles"].values
    splitter = KMeansSplit(n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "jaccard"


def test_splits_kmeans():
    X = np.random.random((100, 100))
    splitter = KMeansSplit(n_splits=2, metric="euclidean")

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(X)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "euclidean"
