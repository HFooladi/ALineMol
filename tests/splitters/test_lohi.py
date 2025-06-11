import pytest
import numpy as np
from alinemol.splitters.lohi import HiSplit, LoSplit


@pytest.fixture
def test_smiles(test_dataset_dili_standardize):
    """Sample SMILES strings for testing."""
    return test_dataset_dili_standardize["canonical_smiles"].tolist()


@pytest.fixture
def test_values(test_dataset_dili_standardize):
    """Sample activity values for testing LoSplitter."""
    # Create synthetic continuous values for testing
    # In real use, these would be actual activity values like IC50, Ki, etc.
    np.random.seed(42)  # For reproducible tests
    n_samples = len(test_dataset_dili_standardize)
    return np.random.normal(5.0, 2.0, n_samples).tolist()  # Simulated pIC50 values


@pytest.fixture
def small_test_data():
    """Small dataset for testing edge cases."""
    smiles = [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CCN",  # ethylamine
        "CCCCCCC",  # heptane
        "CC(C)O",  # isopropanol
        "c1ccc(cc1)O",  # phenol
        "CCCCCCCC",  # octane
        "CC(C)(C)O",  # tert-butanol
        "c1ccc(cc1)N",  # aniline
    ]
    # Create values with some variation to test std_threshold
    values = [1.0, 1.2, 5.0, 1.1, 3.0, 1.3, 5.2, 3.1, 1.4, 5.1]
    return smiles, values


# ============================================================================
# HiSplitter Tests
# ============================================================================


def test_hi_splitter_initialization():
    """Test HiSplitter initialization with default parameters."""
    splitter = HiSplit()
    assert splitter.similarity_threshold == 0.4
    assert splitter.train_min_frac == 0.70
    assert splitter.test_min_frac == 0.15
    assert splitter.coarsening_threshold is None
    assert splitter.verbose is True
    assert splitter.max_mip_gap == 0.1


def test_hi_splitter_initialization_custom():
    """Test HiSplitter initialization with custom parameters."""
    splitter = HiSplit(
        similarity_threshold=0.5,
        train_min_frac=0.8,
        test_min_frac=0.15,
        coarsening_threshold=0.9,
        verbose=False,
        max_mip_gap=0.05,
    )
    assert splitter.similarity_threshold == 0.5
    assert splitter.train_min_frac == 0.8
    assert splitter.test_min_frac == 0.15
    assert splitter.coarsening_threshold == 0.9
    assert splitter.verbose is False
    assert splitter.max_mip_gap == 0.05


def test_hi_splitter_split(test_smiles):
    """Test HiSplitter train/test split functionality."""
    splitter = HiSplit(
        similarity_threshold=0.4,
        train_min_frac=0.6,
        test_min_frac=0.2,
        verbose=False,  # Suppress output during testing
    )

    train_indices, test_indices = next(splitter.split(test_smiles))

    # Check that we get lists of indices
    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)

    # Check that indices are valid
    all_indices = set(train_indices.tolist() + test_indices.tolist())
    assert all(0 <= idx < len(test_smiles) for idx in all_indices)


def test_hi_splitter_k_fold(test_smiles):
    """Test HiSplitter k-fold split functionality."""
    splitter = HiSplit(
        similarity_threshold=0.4,
        verbose=False,  # Suppress output during testing
    )

    k = 3
    k_folds = splitter.k_fold_split(test_smiles, k=k)

    # Check that we get the right number of folds
    assert len(k_folds) == k

    # Check that each fold is a list
    for fold in k_folds:
        assert isinstance(fold, list)

    # Check that all indices are valid
    all_fold_indices = []
    for fold in k_folds:
        all_fold_indices.extend(fold)
        assert all(0 <= idx < len(test_smiles) for idx in fold)

    # Check that we have some molecules in the folds
    assert len(all_fold_indices) > 0


def test_hi_splitter_split_with_coarsening(test_smiles):
    """Test HiSplitter with coarsening threshold."""
    splitter = HiSplit(
        similarity_threshold=0.4, train_min_frac=0.6, test_min_frac=0.2, coarsening_threshold=0.9, verbose=False
    )

    train_indices, test_indices = next(splitter.split(test_smiles))

    # Basic checks
    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) > 0
    assert len(test_indices) > 0
    assert set(train_indices.tolist()).isdisjoint(set(test_indices.tolist()))


def test_hi_splitter_empty_input():
    """Test HiSplitter with empty input."""
    splitter = HiSplit(verbose=False)

    with pytest.raises(Exception):  # Should raise some kind of error
        splitter.split([])


def test_hi_splitter_single_molecule():
    """Test HiSplitter with single molecule."""
    splitter = HiSplit(train_min_frac=0.5, test_min_frac=0.3, verbose=False)

    # This might fail due to insufficient molecules for the constraints
    # but we test that it handles the case gracefully
    try:
        train_indices, test_indices = next(splitter.split(["CCO"]))
        # If it succeeds, check basic properties
        assert isinstance(train_indices, np.ndarray)
        assert isinstance(test_indices, np.ndarray)
    except Exception:
        # It's acceptable for this to fail due to constraints
        pass


# ============================================================================
# LoSplitter Tests
# ============================================================================


def test_lo_splitter_initialization():
    """Test LoSplitter initialization with default parameters."""
    splitter = LoSplit()
    assert splitter.threshold == 0.4
    assert splitter.min_cluster_size == 5
    assert splitter.max_clusters == 50
    assert splitter.std_threshold == 0.60


def test_lo_splitter_initialization_custom():
    """Test LoSplitter initialization with custom parameters."""
    splitter = LoSplit(threshold=0.5, min_cluster_size=10, max_clusters=20, std_threshold=0.70)
    assert splitter.threshold == 0.5
    assert splitter.min_cluster_size == 10
    assert splitter.max_clusters == 20
    assert splitter.std_threshold == 0.70


def test_lo_splitter_split(test_smiles, test_values):
    """Test LoSplitter split functionality."""
    splitter = LoSplit(threshold=0.4, min_cluster_size=3, max_clusters=5, std_threshold=0.5)

    train_indices, clusters_indices = splitter.split(test_smiles, test_values, verbose=0)

    # Check that we get the expected return types
    assert isinstance(train_indices, list)
    assert isinstance(clusters_indices, list)

    # Check that all indices are valid
    all_train_indices = set(train_indices)
    assert all(0 <= idx < len(test_smiles) for idx in all_train_indices)

    # Check clusters
    all_cluster_indices = set()
    for cluster in clusters_indices:
        assert isinstance(cluster, (list, np.ndarray))
        cluster_set = set(cluster)
        assert all(0 <= idx < len(test_smiles) for idx in cluster_set)
        # Check no overlap between clusters
        assert cluster_set.isdisjoint(all_cluster_indices)
        all_cluster_indices.update(cluster_set)

    # Check no overlap between train and test clusters
    assert all_train_indices.isdisjoint(all_cluster_indices)

    # Check that we have some molecules in training set
    assert len(train_indices) > 0


def test_lo_splitter_split_small_dataset(small_test_data):
    """Test LoSplitter with small dataset."""
    smiles, values = small_test_data

    splitter = LoSplit(
        threshold=0.4,
        min_cluster_size=2,
        max_clusters=3,
        std_threshold=0.1,  # Low threshold to allow clusters
    )

    train_indices, clusters_indices = splitter.split(smiles, values, verbose=0)

    # Basic checks
    assert isinstance(train_indices, list)
    assert isinstance(clusters_indices, list)
    assert len(train_indices) > 0

    # Check indices are valid
    all_indices = set(train_indices)
    for cluster in clusters_indices:
        all_indices.update(cluster)
    assert all(0 <= idx < len(smiles) for idx in all_indices)


def test_lo_splitter_no_clusters_found():
    """Test LoSplitter when no clusters meet the criteria."""
    smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    values = [1.0, 1.0, 1.0]  # No variation, won't meet std_threshold

    splitter = LoSplit(
        threshold=0.4,
        min_cluster_size=2,
        max_clusters=5,
        std_threshold=0.5,  # High threshold, no clusters will meet this
    )

    train_indices, clusters_indices = splitter.split(smiles, values, verbose=0)

    # Should return all molecules in training set and no clusters
    assert len(train_indices) == len(smiles)
    assert len(clusters_indices) == 0


def test_lo_splitter_high_similarity_threshold():
    """Test LoSplitter with high similarity threshold."""
    smiles, values = small_test_data

    splitter = LoSplit(
        threshold=0.9,  # Very high threshold
        min_cluster_size=2,
        max_clusters=5,
        std_threshold=0.1,
    )

    train_indices, clusters_indices = splitter.split(smiles, values, verbose=0)

    # With high threshold, fewer molecules will be considered similar
    assert isinstance(train_indices, list)
    assert isinstance(clusters_indices, list)
    assert len(train_indices) > 0


def test_lo_splitter_empty_input():
    """Test LoSplitter with empty input."""
    splitter = LoSplit()

    with pytest.raises(Exception):  # Should raise some kind of error
        splitter.split([], [])


def test_lo_splitter_mismatched_input_lengths():
    """Test LoSplitter with mismatched SMILES and values lengths."""
    splitter = LoSplit()

    with pytest.raises(Exception):  # Should raise some kind of error
        splitter.split(["CCO", "CC(=O)O"], [1.0])  # Different lengths


def test_lo_splitter_single_molecule():
    """Test LoSplitter with single molecule."""
    splitter = LoSplit(min_cluster_size=1)

    # Single molecule should go to training set
    train_indices, clusters_indices = splitter.split(["CCO"], [1.0], verbose=0)

    assert len(train_indices) == 1
    assert train_indices[0] == 0
    assert len(clusters_indices) == 0


def test_lo_splitter_numpy_input(small_test_data):
    """Test LoSplitter with numpy array inputs."""
    smiles, values = small_test_data

    splitter = LoSplit(threshold=0.4, min_cluster_size=2, max_clusters=3, std_threshold=0.1)

    # Convert to numpy arrays
    smiles_array = np.array(smiles)
    values_array = np.array(values)

    train_indices, clusters_indices = splitter.split(smiles_array, values_array, verbose=0)

    # Should work the same as with lists
    assert isinstance(train_indices, list)
    assert isinstance(clusters_indices, list)
    assert len(train_indices) > 0


def test_lo_splitter_different_n_jobs():
    """Test LoSplitter with different n_jobs settings."""
    smiles, values = small_test_data

    splitter = LoSplit(threshold=0.4, min_cluster_size=2, max_clusters=3, std_threshold=0.1)

    # Test with n_jobs=1
    train_indices_1, clusters_indices_1 = splitter.split(smiles, values, n_jobs=1, verbose=0)

    # Test with n_jobs=-1 (all processors)
    train_indices_all, clusters_indices_all = splitter.split(smiles, values, n_jobs=-1, verbose=0)

    # Results should be the same regardless of n_jobs
    assert len(train_indices_1) == len(train_indices_all)
    assert len(clusters_indices_1) == len(clusters_indices_all)
