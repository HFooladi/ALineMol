import numpy as np
from numpy.random import RandomState
from typing import Callable, Union, Optional, List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from alinemol.utils.split_utils import convert_to_default_feats_if_smiles

from alinemol.splitters.factory import register_splitter


@register_splitter("umap", aliases=["umap_cluster", "umap_split"])
class UMAPSplit(GroupShuffleSplit):
    """Group-based split that uses the UMAP clustering in the input space for splitting.

    From the following papers:
    1. "UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines"
        https://doi.org/10.26434/chemrxiv-2024-f1v2v-v2
    2. "On the Best Way to Cluster NCI-60 Molecules"
        https://doi.org/10.3390/biom13030498

    Args:
        n_clusters: The number of clusters to use for clustering
        n_neighbors: The number of neighbors to use for the UMAP algorithm
        min_dist: The minimum distance between points in the UMAP embedding
        n_components: The number of components to use for the PCA algorithm
        umap_metric: The metric to use for the UMAP algorithm
        linkage: The linkage to use for the AgglomerativeClustering algorithm
        n_splits: The number of splits to use for the split
        test_size: The size of the test set
        train_size: The size of the train set
        random_state: The random state to use for the split

    Examples:
        >>> from alinemol.splitters import UMAPSplit
        >>> splitter = UMAPSplit(n_clusters=2, linkage="ward", n_neighbors=3, min_dist=0.1, n_components=2, n_splits=5)
        >>> smiles = ["c1ccccc1", "CCC", "CCCC(CCC)C(=O)O", "NC1CCCCC1N","COc1cc(CNC(=O)CCCCC=CC(C)C)ccc1O", "Cc1cc(Br)c(O)c2ncccc12", "OCC(O)c1oc(O)c(O)c1O"]
        >>> for train_idx, test_idx in splitter.split(smiles):
        ...     print(train_idx)
        ...     print(test_idx)
        ...     break  # Just show the first split
    """

    def __init__(
        self,
        n_clusters: int = 10,
        n_neighbors: int = 100,
        min_dist: float = 0.1,
        n_components: int = 2,
        umap_metric: Union[str, Callable] = "jaccard",
        linkage: str = "ward",
        n_splits: int = 5,
        n_jobs: int = -1,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        **kwargs,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._n_clusters = n_clusters
        self._umap_metric = umap_metric
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._n_components = n_components
        self._linkage = linkage
        self._n_jobs = n_jobs
        self._kwargs = kwargs

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        X, self._umap_metric = convert_to_default_feats_if_smiles(X, self._umap_metric)
        groups = get_umap_clusters(
            X=X,
            n_clusters=self._n_clusters,
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            n_components=self._n_components,
            umap_metric=self._umap_metric,
            linkage=self._linkage,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
            **self._kwargs,
        )
        yield from super()._iter_indices(X, y, groups)


def get_umap_clusters(
    X: Union[np.ndarray, List[np.ndarray]],
    n_clusters: int = 10,
    n_neighbors: int = 100,
    min_dist: float = 0.1,
    n_components: int = 2,
    umap_metric: str = "euclidean",
    linkage: str = "ward",
    random_state: Optional[Union[int, RandomState]] = None,
    n_jobs: int = -1,
    return_embedding: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Cluster a list of SMILES strings using the umap clustering algorithm.

    Args:
        X: The input data (N * D)
        n_clusters: The number of clusters to use for clustering
        n_neighbors: The number of neighbors to use for the UMAP algorithm
        min_dist: The minimum distance between points in the UMAP embedding
        n_components: The number of components to use for the PCA algorithm
        umap_metric: The metric to use for the UMAP algorithm
        linkage: The linkage to use for the AgglomerativeClustering algorithm
        random_state: The random state to use for the PCA algorithm and the Empirical Kernel Map
        n_jobs: The number of jobs to use for the UMAP algorithm
        return_embedding: Whether to return the UMAP embedding

    Returns:
        Array of cluster labels corresponding to each SMILES string in the input list. If return_embedding is True, returns a tuple of the cluster labels and the UMAP embedding.

    Examples:
        >>> from alinemol.splitters import get_umap_clusters
        >>> X = np.random.rand(100, 128)
        >>> clusters_indices, embedding = get_umap_clusters(X, n_clusters=10, n_jobs=1, return_embedding=True)
        >>> print(clusters_indices)
    """
    if isinstance(X, list):
        X = np.stack(X)

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric=umap_metric,
        n_jobs=n_jobs,
        **kwargs,
    )
    embedding = reducer.fit_transform(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit_predict(embedding)
    indices = model.labels_
    if return_embedding:
        return indices, embedding
    else:
        return indices
