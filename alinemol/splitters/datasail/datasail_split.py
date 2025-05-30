from typing import Optional, Union, Sequence

import numpy as np
import networkx as nx
import datamol as dm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from datasail.cluster.vectors import run, SIM_OPTIONS
from datasail.reader.utils import DataSet
from datasail.settings import LOGGER

from sklearn.model_selection import GroupShuffleSplit

from alinemol.splitters.datasail.utils import get_linear_problem_k_fold, get_rdkit_fct

# In case users provide a list of SMILES instead of features, we rely on ECFP4 and the Tanimoto (Jaccard) distance by default
MOLECULE_DATA_SAIL_FEATURIZER = dict(name="ecfp", kwargs=dict(radius=2, fpSize=1024))
MOLECULE_DATA_SAIL_DISTANCE_METRIC = "jaccard"


class DataSAILSplit(GroupShuffleSplit):
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        train_size: Optional[float] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state, **kwargs
        )

    pass

    def _iter_indices(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        """Generate (train, test) indices"""
        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X to be provided.")

        X, self._umap_metric = convert_to_datasail_feats_if_smiles(X, self._umap_metric)
        groups = get_linear_problem_k_fold(
            fold_min_frac=self._fold_min_frac,
            k=self._k,
            verbose=self._verbose,
            max_mip_gap=self._max_mip_gap,
        )
        groups = get_datasail_clusters(
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


def run_ecfp(dataset: DataSet, method: SIM_OPTIONS = "tanimoto") -> None:
    """
    Compute 1024Bit-ECPFs for every molecule in the dataset and then compute pairwise Tanimoto-Scores of them.

    Args:
        dataset: The dataset to compute pairwise, elementwise similarities for
        method: The similarity measure to use. Default is "Tanimoto".
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    if dataset.type != "M":
        raise ValueError("ECFP with Tanimoto-scores can only be applied to molecular data.")

    LOGGER.info("Start ECFP clustering")

    invalid_mols = []
    scaffolds = {}
    for name in dataset.names:
        try:
            mol = Chem.MolFromSmiles(dataset.data[name])
        except Exception:
            mol = None
        # scaffold = read_molecule_encoding(dataset.data[name])
        # if scaffold is None:
        if mol is None:
            LOGGER.warning(f"RDKit cannot parse {name} >{dataset.data[name]}< as a molecule. Skipping.")
            invalid_mols.append(name)
            continue
        scaffolds[name] = mol

    for invalid_name in invalid_mols:
        dataset.names.remove(invalid_name)
        dataset.data.pop(invalid_name)
        poppable = []
        for key, value in dataset.id_map.items():
            if value == invalid_name:
                poppable.append(key)
        for pop in poppable:
            dataset.id_map.pop(pop)

    fps = [AllChem.GetMorganFingerprintAsBitVect(scaffolds[name], 2, nBits=1024) for name in dataset.names]
    dataset.cluster_names = dataset.names

    LOGGER.info(f"Reduced {len(dataset.names)} molecules to {len(dataset.cluster_names)}")
    LOGGER.info("Compute Tanimoto Coefficients")

    run(dataset, fps, method)

    dataset.cluster_map = {name: name for name in dataset.names}


def get_datasail_clusters(
    S: nx.Graph, fold_min_frac: float, k: int, verbose: bool = True, max_mip_gap: float = 0.1
) -> np.ndarray:
    """
    Get DataSAIL clusters.
    """
    return get_linear_problem_k_fold(S, fold_min_frac, k, verbose, max_mip_gap)


def rdkit_sim(fps, method: SIM_OPTIONS) -> np.ndarray:
    """
    Compute the similarity between elements of a list of rdkit vectors.

    Args:
        fps: List of RDKit vectors to fastly compute the similarity matrix
        method: Name of the method to use for calculation

    Returns:

    """
    fct = get_rdkit_fct(method)
    matrix = np.zeros((len(fps), len(fps)))
    for i in range(len(fps)):
        matrix[i, i] = 1
        matrix[i, :i] = fct(fps[i], fps[:i])
        matrix[:i, i] = matrix[i, :i]
    return matrix


def convert_to_datasail_feats_if_smiles(X: Union[Sequence[str], np.ndarray], metric: str, n_jobs: Optional[int] = None):
    """
    If the input is a sequence of strings, assumes this is a list of SMILES and converts it
    to a default set of ECFP4 features with the default Tanimoto distance metric.

    Reference:
        https://github.com/datamol-io/splito/blob/main/splito/_distance_split_base.py
    """

    def _to_feats(smi: str):
        mol = dm.to_mol(smi)
        feats = dm.to_fp(
            mol=mol, fp_type=MOLECULE_DATA_SAIL_FEATURIZER["name"], **MOLECULE_DATA_SAIL_FEATURIZER["kwargs"]
        )
        return feats

    if all(isinstance(x, str) for x in X):
        X = dm.utils.parallelized(_to_feats, X, n_jobs=n_jobs)
        metric = MOLECULE_DATA_SAIL_DISTANCE_METRIC
    return X, metric
