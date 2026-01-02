import numpy as np
from numpy.random import RandomState
from typing import Callable, Union, Optional, List
from sklearn.model_selection import GroupShuffleSplit
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from alinemol.utils.split_utils import convert_to_default_feats_if_smiles

from alinemol.splitters.factory import register_splitter


@register_splitter("butina", aliases=["taylor_butina", "butina_cluster"])
class BUTINASplit(GroupShuffleSplit):
    """Group-based split that uses the BUTINA clustering in the input space for splitting.
    From "BUTINA: A New Method for the Clustering of Chemical Compounds"
    https://doi.org/10.1021/ci9803381

    Args:
        n_clusters: The number of clusters to use for clustering
        n_splits: The number of splits to generate
        metric: The metric to use for clustering
        test_size: The size of the test set
        cutoff: The cutoff value to use for clustering

    Examples:
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

        # Check if input is SMILES strings
        if isinstance(X, (list, np.ndarray)) and len(X) > 0:
            first_elem = X[0] if isinstance(X, list) else X.flat[0]
            if isinstance(first_elem, str):
                # Input is SMILES - use Butina clustering directly
                smiles_list = list(X) if isinstance(X, np.ndarray) else X
                groups = get_butina_clusters(smiles_list, cutoff=self._cutoff)
            else:
                # Input is already features - convert using legacy method
                X_features, self._cluster_metric = convert_to_default_feats_if_smiles(X, self._cluster_metric)
                # For features, we can't use Butina directly - use simple group assignment
                # This is a fallback; prefer passing SMILES
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=min(self._n_clusters, len(X_features)), random_state=self.random_state)
                groups = kmeans.fit_predict(X_features)
        else:
            raise ValueError("X must be a list or array of SMILES strings or feature vectors")

        yield from super()._iter_indices(X, y, groups)


def get_butina_clusters(smiles_list: List[str], cutoff: float = 0.65) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    The Taylor-Butina clustering algorithm clusters molecules based on
    their structural similarity using fingerprints and a distance threshold.

    Args:
        smiles_list: List of SMILES strings
        cutoff: The distance cutoff (1 - Tanimoto similarity threshold).
            Molecules with distance < cutoff are considered similar.
            Default 0.65 means molecules with Tanimoto > 0.35 are clustered.

    Returns:
        Array of cluster labels corresponding to each SMILES string.

    Examples:
        >>> from alinemol.splitters import get_butina_clusters
        >>> clusters = get_butina_clusters(smiles_list, cutoff=0.65)
        >>> print(clusters)
    """
    # Generate fingerprints
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]

    # Compute distance matrix (lower triangular)
    n_fps = len(fp_list)
    dists = []
    for i in range(1, n_fps):
        # Tanimoto distance = 1 - Tanimoto similarity
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])

    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)

    # Convert cluster tuples to array of cluster labels
    cluster_labels = np.zeros(n_fps, dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for mol_idx in cluster:
            cluster_labels[mol_idx] = cluster_idx

    return cluster_labels
