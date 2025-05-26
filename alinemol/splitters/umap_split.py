import numpy as np
from numpy.random import RandomState
from typing import Callable, Union, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from alinemol.utils.split_utils import convert_to_default_feats_if_smiles

from typing import List


class UMAPSplit(GroupShuffleSplit):
    """Group-based split that uses the UMAP clustering in the input space for splitting.
    From "UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines"
    https://doi.org/10.26434/chemrxiv-2024-f1v2v-v2

    Args:
        n_clusters: The number of clusters to use for clustering
        n_splits: The number of splits to generate
        metric: The metric to use for clustering
        test_size: The size of the test set
    """

    def __init__(
        self,
        n_clusters: int = 10,
        n_splits: int = 5,
        metric: Union[str, Callable] = "euclidean",
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._cluster_metric = metric

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
        groups = get_umap_clusters(
            X=X,
            n_clusters=self._n_clusters,
            random_state=self.random_state,
            base_metric=self._cluster_metric,
        )
        yield from super()._iter_indices(X, y, groups)


def get_umap_clusters(smiles_list: List[str], n_clusters: int = 7) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the umap clustering algorithm.
    From "UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines"
    https://doi.org/10.26434/chemrxiv-2024-f1v2v-v2

    Args:
        smiles_list: List of SMILES strings
        n_clusters: The number of clusters to use for clustering

    Returns:
        Array of cluster labels corresponding to each SMILES string in the input list.
    """
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fp_list = [fp_gen.GetFingerprintAsNumPy(x) for x in mol_list]
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(np.stack(fp_list))
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pcs)
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(embedding)
    return ac.labels_
