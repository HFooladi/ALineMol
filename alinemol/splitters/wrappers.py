"""
Wrapper classes for splito molecular splitters.

This module provides wrapper classes around splito splitters that conform
to the unified ALineMol API. Each wrapper inherits from BaseMolecularSplitter
and delegates to the underlying splito implementation.
"""

from typing import Optional, Union, Callable, Iterator, Tuple, List

import numpy as np

from alinemol.splitters.base import BaseMolecularSplitter
from alinemol.splitters.factory import register_splitter


@register_splitter("kmeans", aliases=["k-means", "k_means"])
class KMeansSplit(BaseMolecularSplitter):
    """
    K-Means clustering based molecular splitter.

    Groups molecules into clusters using K-Means clustering on molecular
    fingerprints, then assigns clusters to train/test sets.

    Args:
        n_clusters: Number of clusters to create. Default 10.
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        metric: Distance metric for clustering. Default "euclidean".
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs. Default None.

    Example:
        >>> splitter = KMeansSplit(n_clusters=10, n_splits=5, test_size=0.2)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     train = [smiles_list[i] for i in train_idx]
        ...     test = [smiles_list[i] for i in test_idx]
    """

    def __init__(
        self,
        n_clusters: int = 10,
        n_splits: int = 5,
        metric: Union[str, Callable] = "euclidean",
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_jobs = n_jobs
        self._splitter = None

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train, test) indices using K-Means clustering."""
        from splito import KMeansSplit as SplitoKMeansSplit

        # Convert SMILES to features if needed
        if self._is_smiles_input(X):
            X_features = self._convert_smiles_to_features(list(X), n_jobs=self.n_jobs)
        else:
            X_features = X

        # Create splitter
        self._splitter = SplitoKMeansSplit(
            n_clusters=self.n_clusters,
            n_splits=self.n_splits,
            metric=self.metric,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        yield from self._splitter.split(X_features, y, groups)


@register_splitter("scaffold", aliases=["murcko", "bemis_murcko"])
class ScaffoldSplit(BaseMolecularSplitter):
    """
    Bemis-Murcko scaffold-based molecular splitter.

    Groups molecules by their Bemis-Murcko scaffolds, ensuring molecules
    with the same scaffold are in the same split. This helps evaluate
    model generalization to novel scaffolds.

    Args:
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        make_generic: If True, use generic scaffolds (atoms replaced). Default False.
        n_jobs: Number of parallel jobs for scaffold extraction.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.

    Example:
        >>> splitter = ScaffoldSplit(make_generic=True, n_splits=5)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     # Molecules with same scaffold will be in same set
        ...     train = [smiles_list[i] for i in train_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        make_generic: bool = False,
        n_jobs: Optional[int] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.make_generic = make_generic
        self.n_jobs = n_jobs
        self._splitter = None

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train, test) indices based on scaffolds."""
        from splito import ScaffoldSplit as SplitoScaffoldSplit

        # Scaffold split requires SMILES
        smiles = self._resolve_smiles(X)

        # Create splitter
        self._splitter = SplitoScaffoldSplit(
            smiles=smiles,
            n_splits=self.n_splits,
            make_generic=self.make_generic,
            n_jobs=self.n_jobs,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        yield from self._splitter.split(smiles, y, groups)


@register_splitter("scaffold_generic", aliases=["generic_scaffold"])
class ScaffoldGenericSplit(ScaffoldSplit):
    """
    Generic scaffold-based molecular splitter.

    Same as ScaffoldSplit but with make_generic=True by default.
    Generic scaffolds replace specific atoms with generic placeholders.

    Args:
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        n_jobs: Number of parallel jobs for scaffold extraction.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_jobs: Optional[int] = None,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            make_generic=True,
            n_jobs=n_jobs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )


@register_splitter("molecular_weight", aliases=["mw", "molecular-weight"])
class MolecularWeightSplit(BaseMolecularSplitter):
    """
    Molecular weight-based splitter.

    Splits molecules based on molecular weight, allowing evaluation of
    model generalization to larger (or smaller) molecules.

    Args:
        generalize_to_larger: If True, test set contains heavier molecules.
            If False, test set contains lighter molecules. Default True.
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.

    Example:
        >>> splitter = MolecularWeightSplit(generalize_to_larger=True)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     # Test molecules will have higher MW than train
        ...     pass
    """

    def __init__(
        self,
        generalize_to_larger: bool = True,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.generalize_to_larger = generalize_to_larger
        self._splitter = None

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train, test) indices based on molecular weight."""
        from splito import MolecularWeightSplit as SplitoMolecularWeightSplit

        # Molecular weight split requires SMILES
        smiles = self._resolve_smiles(X)

        # Create splitter
        self._splitter = SplitoMolecularWeightSplit(
            generalize_to_larger=self.generalize_to_larger,
            n_splits=self.n_splits,
            smiles=smiles,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        yield from self._splitter.split(smiles, y, groups)


@register_splitter("molecular_weight_reverse", aliases=["mw_reverse"])
class MolecularWeightReverseSplit(MolecularWeightSplit):
    """
    Reverse molecular weight-based splitter.

    Same as MolecularWeightSplit but with generalize_to_larger=False,
    so test set contains lighter molecules.

    Args:
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            generalize_to_larger=False,
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )


@register_splitter("max_dissimilarity", aliases=["maxdiss", "maxdissimilarity"])
class MaxDissimilaritySplit(BaseMolecularSplitter):
    """
    Maximum dissimilarity-based splitter.

    Uses greedy maximum dissimilarity algorithm to select diverse test
    molecules, ensuring good coverage of chemical space.

    Args:
        n_clusters: Number of diverse molecules to select. Default 25.
        metric: Distance metric for diversity calculation. Default "euclidean".
        n_jobs: Number of parallel jobs.
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.

    Example:
        >>> splitter = MaxDissimilaritySplit(n_clusters=25, n_splits=5)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     # Test set contains maximally diverse molecules
        ...     pass
    """

    def __init__(
        self,
        n_clusters: int = 25,
        metric: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_jobs = n_jobs
        self._splitter = None

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train, test) indices using maximum dissimilarity."""
        from splito import MaxDissimilaritySplit as SplitoMaxDissimilaritySplit

        # Convert SMILES to features if needed
        if self._is_smiles_input(X):
            X_features = self._convert_smiles_to_features(list(X), n_jobs=self.n_jobs)
        else:
            X_features = X

        # Create splitter
        self._splitter = SplitoMaxDissimilaritySplit(
            n_clusters=self.n_clusters,
            metric=self.metric,
            n_jobs=self.n_jobs,
            n_splits=self.n_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        yield from self._splitter.split(X_features, y, groups)


@register_splitter("perimeter", aliases=["perimeter_split"])
class PerimeterSplit(BaseMolecularSplitter):
    """
    Perimeter-based molecular splitter.

    Uses perimeter-based sampling to create diverse train/test splits
    by selecting molecules from the perimeter of the chemical space.

    Args:
        n_clusters: Number of perimeter points to consider. Default 25.
        metric: Distance metric for perimeter calculation. Default "euclidean".
        n_jobs: Number of parallel jobs.
        n_splits: Number of re-shuffling & splitting iterations. Default 5.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.

    Example:
        >>> splitter = PerimeterSplit(n_clusters=25, n_splits=5)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     pass
    """

    def __init__(
        self,
        n_clusters: int = 25,
        metric: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_jobs = n_jobs
        self._splitter = None

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train, test) indices using perimeter sampling."""
        from splito import PerimeterSplit as SplitoPerimeterSplit

        # Convert SMILES to features if needed
        if self._is_smiles_input(X):
            X_features = self._convert_smiles_to_features(list(X), n_jobs=self.n_jobs)
        else:
            X_features = X

        # Create splitter
        self._splitter = SplitoPerimeterSplit(
            n_clusters=self.n_clusters,
            metric=self.metric,
            n_jobs=self.n_jobs,
            n_splits=self.n_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        yield from self._splitter.split(X_features, y, groups)
