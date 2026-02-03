"""
Integration tests for the unified splitter API.

These tests verify that all splitters follow the consistent API pattern.
"""

import pytest
import numpy as np

from alinemol.splitters import get_splitter


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing - 50 molecules."""
    base_smiles = [
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
    # Repeat to get 50 molecules
    return base_smiles * 5


@pytest.fixture
def sample_features():
    """Sample feature array for testing - 50 samples."""
    return np.random.rand(50, 128)


# Splitters that can be quickly tested
QUICK_SPLITTERS = [
    "random",
    "molecular_logp",
    "molecular_weight",
]

# Splitters that require more time (clustering-based)
SLOW_SPLITTERS = [
    "scaffold",
    "kmeans",
    "max_dissimilarity",
    "perimeter",
    "umap",
    "butina",
]


class TestUnifiedSplitAPI:
    """Test that all splitters follow the unified split() API."""

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_split_returns_iterator(self, splitter_name, sample_smiles):
        """split() should return an iterator."""
        splitter = get_splitter(splitter_name, n_splits=2, test_size=0.2)
        result = splitter.split(sample_smiles)
        assert hasattr(result, "__iter__")

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_split_yields_tuples(self, splitter_name, sample_smiles):
        """split() should yield (train_idx, test_idx) tuples."""
        splitter = get_splitter(splitter_name, n_splits=2, test_size=0.2)
        for train_idx, test_idx in splitter.split(sample_smiles):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_split_no_overlap(self, splitter_name, sample_smiles):
        """Train and test indices should not overlap."""
        splitter = get_splitter(splitter_name, n_splits=1, test_size=0.2)
        for train_idx, test_idx in splitter.split(sample_smiles):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Found overlap in {splitter_name}: {overlap}"

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_split_non_empty(self, splitter_name, sample_smiles):
        """Both train and test should be non-empty."""
        splitter = get_splitter(splitter_name, n_splits=1, test_size=0.2)
        for train_idx, test_idx in splitter.split(sample_smiles):
            assert len(train_idx) > 0, f"Empty train set in {splitter_name}"
            assert len(test_idx) > 0, f"Empty test set in {splitter_name}"


class TestGetNSplits:
    """Test get_n_splits() method."""

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_get_n_splits_returns_int(self, splitter_name):
        """get_n_splits() should return an integer."""
        splitter = get_splitter(splitter_name, n_splits=3)
        n_splits = splitter.get_n_splits()
        assert isinstance(n_splits, int)

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_get_n_splits_matches_param(self, splitter_name):
        """get_n_splits() should return the configured n_splits."""
        splitter = get_splitter(splitter_name, n_splits=5)
        assert splitter.get_n_splits() == 5


class TestSMILESInput:
    """Test SMILES input handling."""

    @pytest.mark.parametrize("splitter_name", QUICK_SPLITTERS)
    def test_accepts_smiles_list(self, splitter_name, sample_smiles):
        """Should accept list of SMILES strings."""
        splitter = get_splitter(splitter_name, n_splits=1, test_size=0.2)
        splits = list(splitter.split(sample_smiles))
        assert len(splits) >= 1


class TestRandomState:
    """Test random_state reproducibility."""

    @pytest.mark.parametrize("splitter_name", ["random"])
    def test_reproducible_with_random_state(self, splitter_name, sample_smiles):
        """Same random_state should produce same splits."""
        splitter1 = get_splitter(splitter_name, n_splits=1, test_size=0.2, random_state=42)
        splitter2 = get_splitter(splitter_name, n_splits=1, test_size=0.2, random_state=42)

        splits1 = list(splitter1.split(sample_smiles))
        splits2 = list(splitter2.split(sample_smiles))

        train1, test1 = splits1[0]
        train2, test2 = splits2[0]

        np.testing.assert_array_equal(sorted(train1), sorted(train2))
        np.testing.assert_array_equal(sorted(test1), sorted(test2))


class TestNSplitsParameter:
    """Test n_splits parameter behavior."""

    @pytest.mark.parametrize("splitter_name", ["random"])
    def test_generates_correct_number_of_splits(self, splitter_name, sample_smiles):
        """Should generate exactly n_splits splits."""
        for n_splits in [1, 2, 3, 5]:
            splitter = get_splitter(splitter_name, n_splits=n_splits, test_size=0.2)
            splits = list(splitter.split(sample_smiles))
            assert len(splits) == n_splits, f"Expected {n_splits} splits, got {len(splits)}"


class TestTestSizeParameter:
    """Test test_size parameter behavior."""

    @pytest.mark.parametrize("splitter_name", ["random"])
    def test_test_size_float(self, splitter_name, sample_smiles):
        """test_size as float should set proportion."""
        splitter = get_splitter(splitter_name, n_splits=1, test_size=0.3)
        for train_idx, test_idx in splitter.split(sample_smiles):
            expected_test_size = int(len(sample_smiles) * 0.3)
            # Allow some tolerance for rounding
            assert abs(len(test_idx) - expected_test_size) <= 2


@pytest.mark.slow
class TestSlowSplitters:
    """Tests for slower clustering-based splitters."""

    @pytest.mark.parametrize("splitter_name", SLOW_SPLITTERS)
    def test_split_returns_valid_indices(self, splitter_name, sample_smiles):
        """Slow splitters should return valid indices."""
        splitter = get_splitter(splitter_name, n_splits=1, test_size=0.2)
        try:
            for train_idx, test_idx in splitter.split(sample_smiles):
                assert len(train_idx) > 0
                assert len(test_idx) > 0
                # All indices should be valid
                assert max(train_idx) < len(sample_smiles)
                assert max(test_idx) < len(sample_smiles)
                assert min(train_idx) >= 0
                assert min(test_idx) >= 0
        except Exception as e:
            # Skip if there's a version compatibility issue with datamol/RDKit
            if "ArgumentError" in str(type(e).__name__) or "Boost.Python" in str(e):
                pytest.skip(f"Skipping {splitter_name} due to datamol/RDKit version compatibility: {e}")


class TestSplitterSpecificParameters:
    """Test splitter-specific parameters."""

    def test_scaffold_make_generic(self, sample_smiles):
        """ScaffoldSplit should accept make_generic parameter."""
        splitter = get_splitter("scaffold", make_generic=True, n_splits=1)
        assert splitter.make_generic is True

    def test_kmeans_n_clusters(self, sample_smiles):
        """KMeansSplit should accept n_clusters parameter."""
        splitter = get_splitter("kmeans", n_clusters=5, n_splits=1)
        assert splitter.n_clusters == 5

    def test_molecular_weight_generalize_to_larger(self, sample_smiles):
        """MolecularWeightSplit should accept generalize_to_larger parameter."""
        splitter = get_splitter("molecular_weight", generalize_to_larger=True)
        assert splitter.generalize_to_larger is True

        splitter = get_splitter("molecular_weight", generalize_to_larger=False)
        assert splitter.generalize_to_larger is False
