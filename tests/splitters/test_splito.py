import numpy as np
import datamol as dm
import pytest

from splito import MolecularWeightSplit, ScaffoldSplit, KMeansSplit, MaxDissimilaritySplit, PerimeterSplit

from alinemol.utils.split_utils import get_scaffold
from alinemol.splitters.splits import MolecularLogPSplit


# Test 1: Test the MolecularWeightSplit class
@pytest.mark.parametrize("generalize_to_larger", [True, False])
def test_splits_molecular_weight(test_dataset_dili, generalize_to_larger):
    """Test that MolecularWeightSplit correctly splits molecules based on molecular weight,
    with test set having either larger or smaller weights than training set."""
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


# Test 2: Test the ScaffoldSplit class
@pytest.mark.parametrize("make_generic", [True, False])
def test_splits_scaffold(test_dataset_dili, make_generic):
    """Test that ScaffoldSplit correctly splits molecules based on scaffold similarity,
    ensuring test set scaffolds are not present in training set."""
    smiles = test_dataset_dili["smiles"].values
    splitter = ScaffoldSplit(smiles, n_splits=2, make_generic=make_generic)
    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_scfs = set([get_scaffold(smiles[i], make_generic=make_generic) for i in train_ind])
        test_scfs = [get_scaffold(smiles[i], make_generic=make_generic) for i in test_ind]
        assert not any(test_scf in train_scfs for test_scf in test_scfs)


# Test 3: Test the KMeansSplit class
def test_splits_kmeans_default_feats(test_dataset_dili):
    """Test that KMeansSplit works correctly with default molecular features and Jaccard metric."""
    smiles = test_dataset_dili["smiles"].values
    splitter = KMeansSplit(n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "jaccard"


def test_splits_kmeans():
    """Test that KMeansSplit works correctly with custom features and Euclidean metric."""
    X = np.random.random((100, 100))
    splitter = KMeansSplit(n_splits=2, metric="euclidean")

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(X)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._cluster_metric == "euclidean"


def test_splits_max_dissimilar_default_feats(test_dataset_dili):
    """Test that MaxDissimilaritySplit works correctly with default molecular features."""
    smiles = test_dataset_dili["smiles"].values
    splitter = MaxDissimilaritySplit(n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0


def test_splits_max_dissimilar():
    """Test that MaxDissimilaritySplit works correctly with custom features and Euclidean metric."""
    X = np.random.random((100, 100))
    splitter = MaxDissimilaritySplit(n_splits=2, metric="euclidean")

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(X)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0


def test_splits_perimeter(test_dataset_dili):
    """Test that PerimeterSplit works correctly with default molecular features and Jaccard metric."""
    smiles = test_dataset_dili["smiles"].values
    splitter = PerimeterSplit(n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._metric == "jaccard"


def test_splits_perimeter_euclidean():
    """Test that PerimeterSplit works correctly with custom features and Euclidean metric."""
    X = np.random.random((100, 100))
    splitter = PerimeterSplit(n_splits=2, metric="euclidean")

    for train_ind, test_ind in splitter.split(X):
        assert len(train_ind) + len(test_ind) == len(X)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > 0 and len(test_ind) > 0
        assert splitter._metric == "euclidean"


@pytest.mark.parametrize("generalize_to_larger", [True, False])
def test_splits_molecular_logp(test_dataset_dili, generalize_to_larger):
    """Test that MolecularLogPSplit correctly splits molecules based on LogP values,
    with test set having either larger or smaller LogP than training set."""
    smiles = test_dataset_dili["smiles"].values
    splitter = MolecularLogPSplit(generalize_to_larger=generalize_to_larger, n_splits=2)

    for train_ind, test_ind in splitter.split(smiles):
        assert len(train_ind) + len(test_ind) == len(smiles)
        assert len(set(train_ind).intersection(set(test_ind))) == 0
        assert len(train_ind) > len(test_ind)
        assert len(train_ind) > 0 and len(test_ind) > 0

        train_mlogps = [dm.descriptors.clogp(dm.to_mol(smi)) for smi in smiles[train_ind]]
        if generalize_to_larger:
            assert all(dm.descriptors.clogp(dm.to_mol(smi)) >= max(train_mlogps) for smi in smiles[test_ind])
        else:
            assert all(dm.descriptors.clogp(dm.to_mol(smi)) <= min(train_mlogps) for smi in smiles[test_ind])
