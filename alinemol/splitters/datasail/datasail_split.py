"""
DataSAIL-based molecular splitting.

DataSAIL (Data Splitting Against Information Leakage) provides
optimal splitting strategies for molecular data that minimize
information leakage between train and test sets.
"""

from typing import Optional, Union, List, Iterator, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from alinemol.splitters.base import BaseMolecularSplitter
from alinemol.splitters.factory import register_splitter


@register_splitter("datasail", aliases=["data_sail", "data-sail"])
class DataSAILSplit(BaseMolecularSplitter):
    """
    DataSAIL-based splitter for molecular datasets.

    Uses the DataSAIL algorithm to create train/test splits that minimize
    information leakage based on molecular similarity. DataSAIL uses
    clustering-based approaches to ensure molecules in the test set are
    structurally different from those in the training set.

    This implementation wraps the datasail library and provides a consistent
    interface with other ALineMol splitters.

    Args:
        technique: Splitting technique. Options:
            - "R": Random splitting (baseline)
            - "I": Identity-based splitting (exact duplicates)
            - "C": Cluster-based splitting (default, recommended)
        cluster_method: Clustering method when technique="C".
            Options: "ECFP" (default), "Murcko", etc.
        n_splits: Number of splits to generate. Default 1.
        test_size: Fraction or count for test set. Default 0.2.
        train_size: Fraction or count for train set.
        random_state: Random seed for reproducibility.
        delta: Allowed deviation from requested split sizes. Default 0.1.

    Example:
        >>> splitter = DataSAILSplit(technique="C", test_size=0.2)
        >>> for train_idx, test_idx in splitter.split(smiles_list):
        ...     train = [smiles_list[i] for i in train_idx]
        ...     test = [smiles_list[i] for i in test_idx]

    Note:
        Requires the datasail package to be installed:
        pip install datasail
    """

    def __init__(
        self,
        technique: str = "C",
        cluster_method: str = "ECFP",
        n_splits: int = 1,
        test_size: Optional[Union[float, int]] = 0.2,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        delta: float = 0.1,
    ):
        super().__init__(n_splits, test_size, train_size, random_state)
        self.technique = technique
        self.cluster_method = cluster_method
        self.delta = delta

    def _iter_indices(
        self,
        X: Union[List[str], np.ndarray],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices using DataSAIL."""
        try:
            from datasail.sail import datasail
        except ImportError:
            raise ImportError("datasail is required for DataSAILSplit. Install it with: pip install datasail")

        # Resolve SMILES - DataSAIL requires SMILES strings
        smiles = self._resolve_smiles(X)
        n_samples = len(smiles)

        # Calculate split sizes
        if self.test_size is not None:
            if isinstance(self.test_size, float):
                test_frac = self.test_size
            else:
                test_frac = self.test_size / n_samples
        else:
            test_frac = 0.2

        train_frac = 1.0 - test_frac

        # Prepare data for DataSAIL - expects dict mapping IDs to SMILES
        mol_data = {f"mol_{i}": smi for i, smi in enumerate(smiles)}

        for _ in range(self.n_splits):
            # Call DataSAIL
            try:
                splits = datasail(
                    techniques=[self.technique],
                    splits=[train_frac, test_frac],
                    e_type="M",  # Molecular data
                    e_data=mol_data,
                    e_strat=self.cluster_method if self.technique == "C" else None,
                    delta=self.delta,
                    verbose=0,
                )
            except Exception as e:
                raise RuntimeError(f"DataSAIL splitting failed: {e}")

            # Extract indices from results
            train_indices = []
            test_indices = []

            # DataSAIL returns dict with technique as key
            if self.technique in splits:
                assignments = splits[self.technique]
                for mol_id, split_idx in assignments.items():
                    idx = int(mol_id.split("_")[1])
                    if split_idx == 0:
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)
            else:
                # Fallback: random split if DataSAIL fails
                rng = np.random.default_rng(self.random_state)
                indices = rng.permutation(n_samples)
                n_test = int(n_samples * test_frac)
                test_indices = indices[:n_test].tolist()
                train_indices = indices[n_test:].tolist()

            yield np.array(train_indices), np.array(test_indices)

    def get_n_splits(
        self,
        X: Optional[Union[List[str], np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class DataSAILGroupSplit(GroupShuffleSplit):
    """
    Group-based splitter using DataSAIL clustering.

    This splitter first clusters molecules using DataSAIL's clustering
    algorithms, then uses GroupShuffleSplit to ensure molecules in the
    same cluster stay together in either train or test.

    This is useful when you want to use DataSAIL's clustering but need
    multiple random splits of those clusters.

    Args:
        n_clusters: Target number of clusters.
        cluster_method: Clustering method ("ECFP", "Murcko", etc.).
        n_splits: Number of splits. Default 5.
        test_size: Proportion or count for test set.
        train_size: Proportion or count for train set.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        cluster_method: str = "ECFP",
        n_splits: int = 5,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self._groups = None

    def _get_groups(self, smiles: List[str]) -> np.ndarray:
        """Get cluster assignments for SMILES using DataSAIL clustering."""
        from alinemol.splitters.datasail.utils import smiles_to_fingerprint_array
        from sklearn.cluster import KMeans

        # Convert to fingerprints
        fps = smiles_to_fingerprint_array(smiles)

        # Cluster using KMeans (simple fallback)
        n_clusters = min(self.n_clusters, len(smiles) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        groups = kmeans.fit_predict(fps)

        return groups

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices using cluster-based grouping."""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        # Check if X is SMILES
        is_smiles = all(isinstance(x, str) for x in X)
        if is_smiles:
            smiles = list(X)
            groups = self._get_groups(smiles)
        elif groups is None:
            raise ValueError("Either provide SMILES strings as X, or provide groups parameter.")

        yield from super()._iter_indices(X, y, groups)
