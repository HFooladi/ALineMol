import numpy as np
from numpy.random import RandomState
from typing import Union, Optional, List
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans

import datamol as dm

from alinemol.splitters.factory import register_splitter
from alinemol.utils.split_utils import get_scaffold


@register_splitter("scaffold_kmeans", aliases=["scaffold-kmeans", "scaffold_k_means"])
class ScaffoldKMeansSplit(GroupShuffleSplit):
    """Group-based split that extracts Bemis-Murcko scaffolds, clusters them
    with k-means on ECFP fingerprints, and assigns each molecule to its
    scaffold's cluster.

    This creates a middle ground between ScaffoldSplit (exact scaffold matching,
    many small groups) and KMeansSplit (clusters on whole-molecule fingerprints,
    ignores scaffold structure).

    Args:
        n_clusters: Number of k-means clusters for scaffolds.
        make_generic: Whether to use generic Bemis-Murcko scaffolds.
        n_splits: Number of splits to generate.
        test_size: Size of the test set.
        train_size: Size of the train set.
        random_state: Random state for reproducibility.
        n_jobs: Number of jobs for parallelized scaffold extraction.

    Examples:
        >>> from alinemol.splitters import ScaffoldKMeansSplit
        >>> splitter = ScaffoldKMeansSplit(n_clusters=5, n_splits=2, make_generic=True)
        >>> smiles = ["CCO", "CCCO", "c1ccccc1", "c1ccc(O)cc1"] * 10
        >>> for train_idx, test_idx in splitter.split(smiles):
        ...     print(len(train_idx), len(test_idx))
        ...     break
    """

    def __init__(
        self,
        n_clusters: int = 10,
        make_generic: bool = False,
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._make_generic = make_generic
        self._n_jobs = n_jobs

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices."""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        # Verify input is SMILES strings
        if isinstance(X, (list, np.ndarray)) and len(X) > 0:
            first_elem = X[0] if isinstance(X, list) else X.flat[0]
            if not isinstance(first_elem, str):
                raise ValueError(f"{self.__class__.__name__} requires SMILES strings as input, got {type(first_elem)}.")
        else:
            raise ValueError("X must be a non-empty list or array of SMILES strings.")

        smiles_list = list(X) if isinstance(X, np.ndarray) else X
        groups = get_scaffold_kmeans_clusters(
            smiles_list,
            n_clusters=self._n_clusters,
            make_generic=self._make_generic,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )

        yield from super()._iter_indices(X, y, groups)


def get_scaffold_kmeans_clusters(
    smiles_list: List[str],
    n_clusters: int = 10,
    make_generic: bool = False,
    random_state: Optional[Union[int, RandomState]] = None,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Cluster molecules by their Bemis-Murcko scaffold fingerprints using k-means.

    Steps:
        1. Extract Bemis-Murcko scaffolds for each molecule.
        2. Compute ECFP fingerprints for unique scaffolds.
        3. Run k-means clustering on the scaffold fingerprints.
        4. Map each molecule to its scaffold's cluster label.

    Args:
        smiles_list: List of SMILES strings.
        n_clusters: Number of k-means clusters.
        make_generic: Whether to use generic Bemis-Murcko scaffolds.
        random_state: Random state for k-means reproducibility.
        n_jobs: Number of jobs for parallelized scaffold extraction.

    Returns:
        Array of cluster labels, one per molecule.

    Examples:
        >>> from alinemol.splitters import get_scaffold_kmeans_clusters
        >>> smiles = ["CCO", "CCCO", "c1ccccc1", "c1ccc(O)cc1"]
        >>> clusters = get_scaffold_kmeans_clusters(smiles, n_clusters=2)
        >>> len(clusters) == len(smiles)
        True
    """

    # Step 1: Extract scaffolds for each molecule
    def _extract_scaffold(smi: str) -> str:
        return get_scaffold(smi, make_generic=make_generic)

    scaffolds = dm.utils.parallelized(_extract_scaffold, smiles_list, n_jobs=n_jobs)

    # Step 2: Get unique scaffolds and compute fingerprints
    unique_scaffolds = list(set(scaffolds))
    scaffold_to_idx = {s: i for i, s in enumerate(unique_scaffolds)}

    def _scaffold_to_fp(scaffold_smi: str) -> np.ndarray:
        mol = dm.to_mol(scaffold_smi)
        if mol is None:
            return np.zeros(2048, dtype=np.float32)
        return dm.to_fp(mol, fp_type="ecfp", radius=2, fpSize=2048).astype(np.float32)

    scaffold_fps = np.array(dm.utils.parallelized(_scaffold_to_fp, unique_scaffolds, n_jobs=n_jobs))

    # Step 3: Cluster scaffold fingerprints with k-means
    effective_n_clusters = min(n_clusters, len(unique_scaffolds))
    kmeans = KMeans(n_clusters=effective_n_clusters, random_state=random_state, n_init=10)
    scaffold_cluster_labels = kmeans.fit_predict(scaffold_fps)

    # Step 4: Map each molecule to its scaffold's cluster label
    molecule_labels = np.array(
        [scaffold_cluster_labels[scaffold_to_idx[s]] for s in scaffolds],
        dtype=int,
    )

    return molecule_labels
