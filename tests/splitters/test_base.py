"""
Tests for the BaseMolecularSplitter abstract base class.
"""

import pytest
import numpy as np

from alinemol.splitters.base import BaseMolecularSplitter


class ConcreteSplitter(BaseMolecularSplitter):
    """Concrete implementation for testing."""

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = len(X)
        n_test = self._get_test_size(n_samples)
        indices = np.arange(n_samples)
        for _ in range(self.n_splits):
            yield indices[n_test:], indices[:n_test]


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
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


@pytest.fixture
def sample_features():
    """Sample feature array for testing."""
    return np.random.rand(10, 128)


class TestBaseMolecularSplitter:
    """Tests for BaseMolecularSplitter."""

    def test_is_abstract(self):
        """BaseMolecularSplitter should be abstract and not instantiable."""
        with pytest.raises(TypeError):
            BaseMolecularSplitter()

    def test_concrete_implementation(self):
        """Concrete implementation should be instantiable."""
        splitter = ConcreteSplitter(n_splits=3, test_size=0.2)
        assert splitter is not None
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.2

    def test_get_n_splits(self):
        """get_n_splits should return n_splits."""
        splitter = ConcreteSplitter(n_splits=5)
        assert splitter.get_n_splits() == 5

    def test_split_returns_iterator(self, sample_smiles):
        """split() should return an iterator of (train, test) tuples."""
        splitter = ConcreteSplitter(n_splits=3, test_size=0.2)
        splits = list(splitter.split(sample_smiles))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_no_overlap(self, sample_smiles):
        """Train and test indices should not overlap."""
        splitter = ConcreteSplitter(n_splits=1, test_size=0.3)
        for train_idx, test_idx in splitter.split(sample_smiles):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_split_covers_all_samples(self, sample_smiles):
        """Train and test should cover all samples."""
        splitter = ConcreteSplitter(n_splits=1, test_size=0.3)
        for train_idx, test_idx in splitter.split(sample_smiles):
            all_indices = set(train_idx) | set(test_idx)
            assert all_indices == set(range(len(sample_smiles)))


class TestSMILESDetection:
    """Tests for SMILES input detection."""

    def test_is_smiles_input_with_smiles(self, sample_smiles):
        """Should detect SMILES input correctly."""
        splitter = ConcreteSplitter()
        assert splitter._is_smiles_input(sample_smiles) is True

    def test_is_smiles_input_with_features(self, sample_features):
        """Should detect feature input correctly."""
        splitter = ConcreteSplitter()
        assert splitter._is_smiles_input(sample_features) is False

    def test_is_smiles_input_with_none(self):
        """Should handle None input."""
        splitter = ConcreteSplitter()
        assert splitter._is_smiles_input(None) is False

    def test_is_smiles_input_with_empty(self):
        """Should handle empty input."""
        splitter = ConcreteSplitter()
        assert splitter._is_smiles_input([]) is True  # Empty list is considered SMILES


class TestSMILESResolution:
    """Tests for SMILES resolution."""

    def test_resolve_smiles_from_input(self, sample_smiles):
        """Should resolve SMILES from input when input is SMILES."""
        splitter = ConcreteSplitter()
        resolved = splitter._resolve_smiles(sample_smiles)
        assert resolved == sample_smiles

    def test_resolve_smiles_from_stored(self, sample_smiles, sample_features):
        """Should resolve SMILES from stored _smiles when input is features."""
        splitter = ConcreteSplitter()
        splitter.set_smiles(sample_smiles)
        resolved = splitter._resolve_smiles(sample_features)
        assert resolved == sample_smiles

    def test_resolve_smiles_raises_without_smiles(self, sample_features):
        """Should raise error when no SMILES available."""
        splitter = ConcreteSplitter()
        with pytest.raises(ValueError, match="not SMILES strings"):
            splitter._resolve_smiles(sample_features)

    def test_set_smiles_returns_self(self, sample_smiles):
        """set_smiles should return self for chaining."""
        splitter = ConcreteSplitter()
        result = splitter.set_smiles(sample_smiles)
        assert result is splitter


class TestSizeCalculation:
    """Tests for train/test size calculation."""

    def test_get_test_size_with_float(self):
        """Should calculate test size from float proportion."""
        splitter = ConcreteSplitter(test_size=0.2)
        assert splitter._get_test_size(100) == 20

    def test_get_test_size_with_int(self):
        """Should use int test size directly."""
        splitter = ConcreteSplitter(test_size=15)
        assert splitter._get_test_size(100) == 15

    def test_get_test_size_default(self):
        """Should use default 20% when test_size is None."""
        splitter = ConcreteSplitter(test_size=None)
        assert splitter._get_test_size(100) == 20

    def test_get_train_size_with_float(self):
        """Should calculate train size from float proportion."""
        splitter = ConcreteSplitter(train_size=0.7)
        assert splitter._get_train_size(100, 20) == 70

    def test_get_train_size_default(self):
        """Should use complement of test size when train_size is None."""
        splitter = ConcreteSplitter(train_size=None)
        assert splitter._get_train_size(100, 20) == 80


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """__repr__ should return informative string."""
        splitter = ConcreteSplitter(n_splits=5, test_size=0.2, random_state=42)
        repr_str = repr(splitter)
        assert "ConcreteSplitter" in repr_str
        assert "n_splits=5" in repr_str
        assert "test_size=0.2" in repr_str
        assert "random_state=42" in repr_str
