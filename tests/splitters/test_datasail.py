"""
Tests for the DataSAIL-based splitter.
"""

import pytest
import numpy as np

from alinemol.splitters import DataSAILSplit, get_splitter
from alinemol.splitters.datasail.utils import (
    compute_ecfp_fingerprints,
    compute_ecfp_similarity_matrix,
    compute_ecfp_distance_matrix,
    smiles_to_fingerprint_array,
)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # ethanol
        "CCCO",  # propanol
        "CCCCO",  # butanol
        "c1ccccc1",  # benzene
        "c1ccc(O)cc1",  # phenol
        "c1ccc(N)cc1",  # aniline
        "CC(=O)O",  # acetic acid
        "CCC(=O)O",  # propionic acid
        "CCCC(=O)O",  # butyric acid
        "CCN",  # ethylamine
    ] * 3  # 30 molecules


class TestDataSAILSplit:
    """Tests for DataSAILSplit class."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        splitter = DataSAILSplit()
        assert splitter.n_splits == 1
        assert splitter.test_size == 0.2
        assert splitter.technique == "C"
        assert splitter.cluster_method == "ECFP"

    def test_initialization_with_params(self):
        """Should initialize with custom parameters."""
        splitter = DataSAILSplit(
            technique="R",
            n_splits=3,
            test_size=0.3,
            random_state=42,
        )
        assert splitter.technique == "R"
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.3
        assert splitter.random_state == 42

    def test_get_n_splits(self):
        """get_n_splits should return configured value."""
        splitter = DataSAILSplit(n_splits=5)
        assert splitter.get_n_splits() == 5

    def test_get_splitter_factory(self):
        """Should be available via get_splitter factory."""
        splitter = get_splitter("datasail", n_splits=1, test_size=0.2)
        assert isinstance(splitter, DataSAILSplit)

    def test_get_splitter_aliases(self):
        """Should be available via aliases."""
        splitter1 = get_splitter("datasail")
        splitter2 = get_splitter("data_sail")
        assert type(splitter1) is type(splitter2)


class TestDataSAILUtils:
    """Tests for DataSAIL utility functions."""

    def test_compute_ecfp_fingerprints(self, sample_smiles):
        """Should compute fingerprints for SMILES list."""
        fps = compute_ecfp_fingerprints(sample_smiles[:5])
        assert len(fps) == 5
        # Each fingerprint should be a bitvector
        assert all(fp is not None for fp in fps)

    def test_compute_ecfp_fingerprints_invalid_smiles(self):
        """Should raise error for invalid SMILES."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            compute_ecfp_fingerprints(["invalid_smiles_xyz"])

    def test_compute_ecfp_similarity_matrix(self, sample_smiles):
        """Should compute symmetric similarity matrix."""
        sim_matrix = compute_ecfp_similarity_matrix(sample_smiles[:5])

        assert sim_matrix.shape == (5, 5)
        # Should be symmetric
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(5))
        # All values should be between 0 and 1
        assert np.all(sim_matrix >= 0)
        assert np.all(sim_matrix <= 1)

    def test_compute_ecfp_distance_matrix(self, sample_smiles):
        """Should compute distance matrix (1 - similarity)."""
        dist_matrix = compute_ecfp_distance_matrix(sample_smiles[:5])

        assert dist_matrix.shape == (5, 5)
        # Diagonal should be 0.0 (self-distance)
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(5))
        # All values should be between 0 and 1
        assert np.all(dist_matrix >= 0)
        assert np.all(dist_matrix <= 1)

    def test_smiles_to_fingerprint_array(self, sample_smiles):
        """Should convert SMILES to numpy array of fingerprints."""
        fp_array = smiles_to_fingerprint_array(sample_smiles[:5], n_bits=1024)

        assert fp_array.shape == (5, 1024)
        assert fp_array.dtype == np.int8
        # Values should be 0 or 1
        assert np.all((fp_array == 0) | (fp_array == 1))

    def test_smiles_to_fingerprint_array_custom_params(self, sample_smiles):
        """Should accept custom fingerprint parameters."""
        fp_array = smiles_to_fingerprint_array(
            sample_smiles[:5],
            radius=3,
            n_bits=2048,
        )

        assert fp_array.shape == (5, 2048)


class TestDataSAILSplitWithDatasail:
    """Tests that require the datasail package."""

    @pytest.mark.slow
    def test_split_with_datasail_installed(self, sample_smiles):
        """Should work if datasail is installed."""
        try:
            from datasail.sail import datasail  # noqa

            datasail_available = True
        except ImportError:
            datasail_available = False

        if not datasail_available:
            pytest.skip("datasail package not installed")

        splitter = DataSAILSplit(technique="R", n_splits=1, test_size=0.2)
        splits = list(splitter.split(sample_smiles))

        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_split_raises_without_datasail(self, sample_smiles):
        """Should raise ImportError if datasail not installed."""
        try:
            from datasail.sail import datasail  # noqa

            pytest.skip("datasail is installed, cannot test import error")
        except ImportError:
            pass

        splitter = DataSAILSplit(n_splits=1)
        with pytest.raises(ImportError, match="datasail is required"):
            list(splitter.split(sample_smiles))
