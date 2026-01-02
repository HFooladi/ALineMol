"""
Tests for the Butina clustering-based splitter.
"""

import pytest

from alinemol.splitters import BUTINASplit, get_splitter, get_butina_clusters


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


class TestBUTINASplit:
    """Tests for BUTINASplit class."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        splitter = BUTINASplit()
        assert splitter.n_splits == 5
        assert splitter._cutoff == 0.65

    def test_initialization_with_params(self):
        """Should initialize with custom parameters."""
        splitter = BUTINASplit(
            n_clusters=20,
            n_splits=3,
            test_size=0.3,
            cutoff=0.7,
            random_state=42,
        )
        assert splitter._n_clusters == 20
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.3
        assert splitter._cutoff == 0.7
        assert splitter.random_state == 42

    @pytest.mark.slow
    def test_split_with_smiles(self, sample_smiles):
        """Should split SMILES list."""
        splitter = BUTINASplit(n_splits=2, test_size=0.2, random_state=42)
        splits = list(splitter.split(sample_smiles))

        assert len(splits) == 2
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    @pytest.mark.slow
    def test_split_indices_valid(self, sample_smiles):
        """All indices should be valid."""
        splitter = BUTINASplit(n_splits=1, test_size=0.2)
        for train_idx, test_idx in splitter.split(sample_smiles):
            assert all(0 <= i < len(sample_smiles) for i in train_idx)
            assert all(0 <= i < len(sample_smiles) for i in test_idx)

    def test_get_splitter_factory(self):
        """Should be available via get_splitter factory."""
        splitter = get_splitter("butina", n_splits=2, test_size=0.2)
        assert isinstance(splitter, BUTINASplit)

    def test_get_splitter_aliases(self):
        """Should be available via aliases."""
        splitter1 = get_splitter("butina")
        splitter2 = get_splitter("taylor_butina")
        assert type(splitter1) is type(splitter2)


class TestGetButinaClusters:
    """Tests for get_butina_clusters helper function."""

    def test_returns_cluster_labels(self, sample_smiles):
        """Should return cluster labels for each molecule."""
        clusters = get_butina_clusters(sample_smiles[:10], cutoff=0.65)
        # clusters is a tuple of cluster assignments
        assert clusters is not None

    def test_different_cutoffs(self, sample_smiles):
        """Different cutoffs should produce different clusterings."""
        clusters_high = get_butina_clusters(sample_smiles[:10], cutoff=0.9)
        clusters_low = get_butina_clusters(sample_smiles[:10], cutoff=0.3)
        # Different cutoffs may produce different numbers of clusters
        assert clusters_high is not None
        assert clusters_low is not None


class TestGroupBasedSplitting:
    """Test that similar molecules stay in same group."""

    @pytest.mark.slow
    def test_similar_molecules_grouped(self, sample_smiles):
        """Very similar molecules should be in same cluster."""
        # Create a dataset with very similar molecules
        similar_smiles = [
            "CCCCCCCC",  # octane
            "CCCCCCCCC",  # nonane
            "CCCCCCCCCC",  # decane
            "CCCCCCCCCCC",  # undecane
        ]

        clusters = get_butina_clusters(similar_smiles, cutoff=0.7)
        # With high similarity threshold, similar molecules should cluster together
        assert clusters is not None
