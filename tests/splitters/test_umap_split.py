import numpy as np
import pytest
from alinemol.splitters.umap_split import UMAPSplit, get_umap_clusters


@pytest.fixture
def sample_data():
    # Create a simple dataset with 20 samples and 10 features
    np.random.seed(42)
    return np.random.rand(20, 10)


@pytest.fixture
def sample_smiles():
    # Create a list of simple SMILES strings
    return [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1CC",
        "CC(=O)O",
        "CC(=O)OC",
        "CC(=O)OCC",
        "c1ccccc1C(=O)O",
        "c1ccccc1C(=O)OC",
        "c1ccccc1C(=O)OCC",
        "c1ccccc1C(=O)OCCC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OCC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OCCC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)OCCCC",
    ]


def test_umap_split_initialization():
    # Test default initialization
    splitter = UMAPSplit()
    assert splitter._n_clusters == 10
    assert splitter._n_neighbors == 100
    assert splitter._min_dist == 0.1
    assert splitter._n_components == 2
    assert splitter._umap_metric == "jaccard"
    assert splitter._linkage == "ward"

    # Test custom initialization
    splitter = UMAPSplit(
        n_clusters=5,
        n_neighbors=50,
        min_dist=0.2,
        n_components=3,
        umap_metric="euclidean",
        linkage="complete",
        n_splits=3,
        test_size=0.3,
    )
    assert splitter._n_clusters == 5
    assert splitter._n_neighbors == 50
    assert splitter._min_dist == 0.2
    assert splitter._n_components == 3
    assert splitter._umap_metric == "euclidean"
    assert splitter._linkage == "complete"


def test_umap_split_with_numpy_data(sample_data):
    splitter = UMAPSplit(n_clusters=3, n_splits=2, test_size=0.3)
    splits = list(splitter.split(sample_data))

    # Check number of splits
    assert len(splits) == 2

    # Check each split
    for train_idx, test_idx in splits:
        # Check indices are numpy arrays
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

        # Check no overlap between train and test
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # Check sizes
        assert len(train_idx) + len(test_idx) == len(sample_data)
        assert len(test_idx) == int(len(sample_data) * 0.3)


def test_umap_split_with_smiles(sample_smiles):
    splitter = UMAPSplit(n_clusters=3, n_splits=2, test_size=0.3)
    splits = list(splitter.split(sample_smiles))

    # Check number of splits
    assert len(splits) == 2

    # Check each split
    for train_idx, test_idx in splits:
        # Check indices are numpy arrays
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

        # Check no overlap between train and test
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # Check sizes
        assert len(train_idx) + len(test_idx) == len(sample_smiles)
        assert len(test_idx) == int(len(sample_smiles) * 0.3)


def test_umap_split_error_cases():
    splitter = UMAPSplit()

    # Test with None input
    with pytest.raises(ValueError):
        list(splitter.split(None))

    # Test with empty input
    with pytest.raises(ValueError):
        list(splitter.split([]))


def test_get_umap_clusters(sample_data):
    # Test with numpy array
    clusters = get_umap_clusters(
        sample_data, n_clusters=3, n_neighbors=5, min_dist=0.1, n_components=2, umap_metric="euclidean", linkage="ward"
    )

    assert isinstance(clusters, np.ndarray)
    assert len(clusters) == len(sample_data)
    assert len(np.unique(clusters)) == 3

    # Test with list of arrays
    data_list = [sample_data[i : i + 5] for i in range(0, len(sample_data), 5)]
    clusters = get_umap_clusters(
        data_list, n_clusters=3, n_neighbors=5, min_dist=0.1, n_components=2, umap_metric="euclidean", linkage="ward"
    )

    assert isinstance(clusters, np.ndarray)
    assert len(clusters) == len(sample_data)
    assert len(np.unique(clusters)) == 3


def test_umap_split_reproducibility(sample_data):
    # Test that same random state gives same splits
    splitter1 = UMAPSplit(n_clusters=3, n_splits=2, test_size=0.3, random_state=42)
    splitter2 = UMAPSplit(n_clusters=3, n_splits=2, test_size=0.3, random_state=42)

    splits1 = list(splitter1.split(sample_data))
    splits2 = list(splitter2.split(sample_data))

    for (train1, test1), (train2, test2) in zip(splits1, splits2):
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)
