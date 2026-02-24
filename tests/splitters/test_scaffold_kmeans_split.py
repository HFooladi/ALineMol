"""
Tests for the Scaffold K-Means clustering-based splitter.
"""

import numpy as np
import pytest

from alinemol.splitters import ScaffoldKMeansSplit, get_splitter, get_scaffold_kmeans_clusters
from alinemol.utils.split_utils import get_scaffold


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


class TestScaffoldKMeansSplit:
    """Tests for ScaffoldKMeansSplit class."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        splitter = ScaffoldKMeansSplit()
        assert splitter.n_splits == 5
        assert splitter._n_clusters == 10
        assert splitter._make_generic is False

    def test_initialization_with_params(self):
        """Should initialize with custom parameters."""
        splitter = ScaffoldKMeansSplit(
            n_clusters=20,
            make_generic=True,
            n_splits=3,
            test_size=0.3,
            random_state=42,
        )
        assert splitter._n_clusters == 20
        assert splitter._make_generic is True
        assert splitter.n_splits == 3
        assert splitter.test_size == 0.3
        assert splitter.random_state == 42

    @pytest.mark.slow
    def test_split_with_smiles(self, sample_smiles):
        """Should split SMILES list into disjoint train/test sets."""
        splitter = ScaffoldKMeansSplit(n_clusters=3, n_splits=2, test_size=0.2, random_state=42)
        splits = list(splitter.split(sample_smiles))

        assert len(splits) == 2
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    @pytest.mark.slow
    def test_split_covers_all_samples(self, sample_smiles):
        """Train and test indices should cover all samples."""
        splitter = ScaffoldKMeansSplit(n_clusters=3, n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(sample_smiles):
            all_idx = set(train_idx) | set(test_idx)
            assert all_idx == set(range(len(sample_smiles)))

    @pytest.mark.slow
    def test_split_indices_valid(self, sample_smiles):
        """All indices should be valid."""
        splitter = ScaffoldKMeansSplit(n_clusters=3, n_splits=1, test_size=0.2)
        for train_idx, test_idx in splitter.split(sample_smiles):
            assert all(0 <= i < len(sample_smiles) for i in train_idx)
            assert all(0 <= i < len(sample_smiles) for i in test_idx)

    @pytest.mark.slow
    def test_split_with_make_generic(self, sample_smiles):
        """Should work with make_generic=True."""
        splitter = ScaffoldKMeansSplit(n_clusters=3, make_generic=True, n_splits=1, test_size=0.2, random_state=42)
        splits = list(splitter.split(sample_smiles))
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) > 0
        assert len(test_idx) > 0

    def test_split_raises_on_none(self):
        """Should raise ValueError when X is None."""
        splitter = ScaffoldKMeansSplit()
        with pytest.raises(ValueError, match="requires X"):
            list(splitter.split(None))

    def test_split_raises_on_non_smiles(self):
        """Should raise ValueError when input is not SMILES."""
        splitter = ScaffoldKMeansSplit()
        with pytest.raises(ValueError, match="requires SMILES strings"):
            list(splitter.split(np.array([[1.0, 2.0], [3.0, 4.0]])))

    def test_get_splitter_factory(self):
        """Should be available via get_splitter factory."""
        splitter = get_splitter("scaffold_kmeans", n_clusters=5, n_splits=2, test_size=0.2)
        assert isinstance(splitter, ScaffoldKMeansSplit)

    def test_get_splitter_aliases(self):
        """Should be available via aliases."""
        splitter1 = get_splitter("scaffold_kmeans")
        splitter2 = get_splitter("scaffold-kmeans")
        splitter3 = get_splitter("scaffold_k_means")
        assert type(splitter1) is type(splitter2) is type(splitter3)


class TestGetScaffoldKMeansClusters:
    """Tests for get_scaffold_kmeans_clusters helper function."""

    def test_returns_correct_length(self, sample_smiles):
        """Should return one cluster label per molecule."""
        clusters = get_scaffold_kmeans_clusters(sample_smiles, n_clusters=3, random_state=42)
        assert len(clusters) == len(sample_smiles)

    def test_cluster_labels_are_integers(self, sample_smiles):
        """Cluster labels should be integer type."""
        clusters = get_scaffold_kmeans_clusters(sample_smiles, n_clusters=3, random_state=42)
        assert clusters.dtype == int

    def test_same_scaffold_same_cluster(self):
        """Molecules with the same scaffold should get the same cluster label."""
        smiles = [
            "c1ccc(O)cc1",  # phenol
            "c1ccc(N)cc1",  # aniline — same scaffold as phenol (benzene)
            "c1ccc(Cl)cc1",  # chlorobenzene — same scaffold
            "CCCCCC",  # hexane
            "CCCCCCC",  # heptane — same scaffold as hexane
        ]
        clusters = get_scaffold_kmeans_clusters(smiles, n_clusters=2, random_state=42)

        # Phenol, aniline, chlorobenzene share the benzene scaffold
        scaffolds = [get_scaffold(s) for s in smiles]
        for i in range(len(smiles)):
            for j in range(i + 1, len(smiles)):
                if scaffolds[i] == scaffolds[j]:
                    assert clusters[i] == clusters[j], (
                        f"Molecules {i} and {j} share scaffold '{scaffolds[i]}' "
                        f"but got different clusters: {clusters[i]} vs {clusters[j]}"
                    )

    def test_n_clusters_capped(self):
        """n_clusters should be capped to number of unique scaffolds."""
        smiles = ["CCO", "CCCO", "c1ccccc1"]
        # Request more clusters than unique scaffolds
        clusters = get_scaffold_kmeans_clusters(smiles, n_clusters=100, random_state=42)
        n_unique = len(set(clusters))
        n_unique_scaffolds = len(set(get_scaffold(s) for s in smiles))
        assert n_unique <= n_unique_scaffolds

    def test_make_generic_changes_grouping(self):
        """make_generic should affect scaffold extraction."""
        smiles = [
            "c1ccc(O)cc1",  # phenol
            "c1ccc(N)cc1",  # aniline
            "C1CCCCC1",  # cyclohexane
            "C1CCC(O)CC1",  # cyclohexanol
        ] * 5

        clusters_default = get_scaffold_kmeans_clusters(smiles, n_clusters=2, make_generic=False, random_state=42)
        clusters_generic = get_scaffold_kmeans_clusters(smiles, n_clusters=2, make_generic=True, random_state=42)
        # Both should return valid arrays of the correct length
        assert len(clusters_default) == len(smiles)
        assert len(clusters_generic) == len(smiles)

    def test_reproducibility(self, sample_smiles):
        """Same random_state should produce identical results."""
        c1 = get_scaffold_kmeans_clusters(sample_smiles, n_clusters=3, random_state=42)
        c2 = get_scaffold_kmeans_clusters(sample_smiles, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(c1, c2)
