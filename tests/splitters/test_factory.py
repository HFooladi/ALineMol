"""
Tests for the splitter factory and registry.
"""

import pytest

from alinemol.splitters.factory import (
    get_splitter,
    list_splitters,
    get_splitter_names,
    get_splitter_aliases,
    is_splitter_registered,
)


class TestGetSplitter:
    """Tests for get_splitter factory function."""

    def test_get_scaffold_splitter(self):
        """Should create scaffold splitter."""
        splitter = get_splitter("scaffold")
        assert splitter is not None
        assert hasattr(splitter, "split")

    def test_get_kmeans_splitter(self):
        """Should create KMeans splitter."""
        splitter = get_splitter("kmeans", n_clusters=5)
        assert splitter is not None
        assert splitter.n_clusters == 5

    def test_get_splitter_with_params(self):
        """Should pass parameters to splitter constructor."""
        splitter = get_splitter("scaffold", make_generic=True, n_splits=3)
        assert splitter.make_generic is True
        assert splitter.n_splits == 3

    def test_get_splitter_invalid_name(self):
        """Should raise ValueError for unknown splitter."""
        with pytest.raises(ValueError, match="Unknown splitter"):
            get_splitter("nonexistent_splitter")

    def test_get_splitter_case_insensitive(self):
        """Should be case insensitive."""
        splitter1 = get_splitter("scaffold")
        splitter2 = get_splitter("SCAFFOLD")
        splitter3 = get_splitter("Scaffold")
        assert type(splitter1) is type(splitter2) is type(splitter3)

    def test_get_splitter_alias(self):
        """Should resolve aliases correctly."""
        # mw -> molecular_weight
        splitter = get_splitter("mw")
        assert splitter is not None

    def test_get_splitter_alias_kmeans(self):
        """Should resolve kmeans aliases."""
        splitter1 = get_splitter("kmeans")
        splitter2 = get_splitter("k-means")
        assert type(splitter1) is type(splitter2)


class TestListSplitters:
    """Tests for list_splitters function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        splitters = list_splitters()
        assert isinstance(splitters, dict)

    def test_contains_expected_splitters(self):
        """Should contain expected splitter names."""
        splitters = list_splitters()
        expected = ["scaffold", "kmeans", "molecular_weight", "random", "umap"]
        for name in expected:
            assert name in splitters, f"Missing splitter: {name}"

    def test_values_are_classes(self):
        """Values should be splitter classes."""
        splitters = list_splitters()
        for name, cls in splitters.items():
            assert isinstance(cls, type), f"{name} is not a class"


class TestGetSplitterNames:
    """Tests for get_splitter_names function."""

    def test_returns_sorted_list(self):
        """Should return a sorted list of names."""
        names = get_splitter_names()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_contains_expected_names(self):
        """Should contain expected names."""
        names = get_splitter_names()
        assert "scaffold" in names
        assert "kmeans" in names


class TestGetSplitterAliases:
    """Tests for get_splitter_aliases function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        aliases = get_splitter_aliases()
        assert isinstance(aliases, dict)

    def test_contains_expected_aliases(self):
        """Should contain expected aliases."""
        aliases = get_splitter_aliases()
        assert "mw" in aliases
        assert aliases["mw"] == "molecular_weight"

    def test_aliases_map_to_valid_splitters(self):
        """All aliases should map to registered splitters."""
        aliases = get_splitter_aliases()
        names = get_splitter_names()
        for alias, canonical in aliases.items():
            assert canonical in names, f"Alias '{alias}' maps to unknown splitter '{canonical}'"


class TestIsSplitterRegistered:
    """Tests for is_splitter_registered function."""

    def test_registered_splitter(self):
        """Should return True for registered splitters."""
        assert is_splitter_registered("scaffold") is True
        assert is_splitter_registered("kmeans") is True

    def test_registered_alias(self):
        """Should return True for registered aliases."""
        assert is_splitter_registered("mw") is True
        assert is_splitter_registered("k-means") is True

    def test_unregistered_splitter(self):
        """Should return False for unregistered splitters."""
        assert is_splitter_registered("nonexistent") is False

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert is_splitter_registered("SCAFFOLD") is True
        assert is_splitter_registered("Scaffold") is True


class TestRegisteredSplitters:
    """Tests to verify all expected splitters are registered."""

    @pytest.mark.parametrize(
        "splitter_name",
        [
            "scaffold",
            "scaffold_generic",
            "kmeans",
            "molecular_weight",
            "molecular_weight_reverse",
            "max_dissimilarity",
            "perimeter",
            "molecular_logp",
            "random",
            "umap",
            "butina",
            "hi",
            "lo",
            "datasail",
        ],
    )
    def test_splitter_registered(self, splitter_name):
        """Each expected splitter should be registered."""
        assert is_splitter_registered(splitter_name), f"Splitter '{splitter_name}' not registered"

    @pytest.mark.parametrize(
        "splitter_name",
        [
            "scaffold",
            "kmeans",
            "molecular_weight",
            "max_dissimilarity",
            "perimeter",
            "molecular_logp",
            "random",
            "umap",
            "butina",
        ],
    )
    def test_splitter_instantiable(self, splitter_name):
        """Each registered splitter should be instantiable."""
        splitter = get_splitter(splitter_name)
        assert splitter is not None
        assert hasattr(splitter, "split")
        assert hasattr(splitter, "get_n_splits")
