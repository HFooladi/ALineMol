import numpy as np
from numpy.random import RandomState
from typing import Callable, Union, Optional
from sklearn.model_selection import GroupShuffleSplit
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit.Chem.Fingerprints import taylor_butina_clustering
from alinemol.utils.split_utils import convert_to_default_feats_if_smiles

from typing import List


class BUTINASplit(GroupShuffleSplit):
    """Group-based split that uses the BUTINA clustering in the input space for splitting.
    From "BUTINA: A New Method for the Clustering of Chemical Compounds"
    https://doi.org/10.1021/ci00005a000

    Args:
        n_clusters: The number of clusters to use for clustering
        n_splits: The number of splits to generate
        metric: The metric to use for clustering
        test_size: The size of the test set
        cutoff: The cutoff value to use for clustering

    Example:
        >>> from alinemol.splitters import BUTINASplit
        >>> splitter = BUTINASplit(n_clusters=10, n_splits=5, cutoff=0.65)
        >>> train_idx, test_idx = splitter.split(X, y, groups)
    """

    def __init__(
        self,
        n_clusters: int = 10,
        n_splits: int = 5,
        metric: Union[str, Callable] = "euclidean",
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        cutoff: float = 0.65,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._cluster_metric = metric
        self._cutoff = cutoff

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        X, self._cluster_metric = convert_to_default_feats_if_smiles(X, self._cluster_metric)
        groups = get_butina_clusters(
            X=X,
            n_clusters=self._n_clusters,
            random_state=self.random_state,
            base_metric=self._cluster_metric,
        )
        yield from super()._iter_indices(X, y, groups)


def get_butina_clusters(smiles_list: List[str], cutoff: float = 0.65) -> List[int]:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    Args:
        smiles_list: List of SMILES strings
        cutoff: The cutoff value to use for clustering

    Returns:
        List of cluster labels corresponding to each SMILES string in the input list.

    Example:
        >>> from alinemol.splitters import get_butina_clusters
        >>> clusters = get_butina_clusters(smiles_list, cutoff=0.65)
        >>> print(clusters)
    """
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]
    return taylor_butina_clustering(fp_list, cutoff=cutoff)
