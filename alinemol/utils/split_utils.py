"""
Utility functions for splitting datasets and distance calculation.
"""

# importing required libraries
import warnings # for ignoring warnings
from typing import Dict, List, Tuple, Union, Optional, Sequence # for type hinting


import datamol as dm # for molecule processing
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import rdkit # for molecule processing
from rdkit import Chem # for molecule processing
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.model_selection import StratifiedShuffleSplit # for stratified splitting
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules 
from astartes.utils.exceptions import MoleculesNotInstalledError
from scipy.spatial import distance


# In case users provide a list of SMILES instead of features, we rely on ECFP4 and the Tanimoto (Jaccard) distance by default
MOLECULE_DEFAULT_FEATURIZER = dict(name="ecfp", kwargs=dict(radius=2, fpSize=2048))
MOLECULE_DEFAULT_DISTANCE_METRIC = "jaccard"

try:
    """
    aimsim depends on sklearn_extra, which uses a version checking technique that is due to
    be deprecated in a version of Python after 3.11, so it is throwing a deprecation warning
    We ignore this warning since we can't do anything about it (sklearn_extra seems to be
    abandonware) and in the future it will become an error that we can deal with.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        from aimsim.chemical_datastructures import Molecule
        from aimsim.exceptions import LoadingError
except ImportError:  # pragma: no cover
    raise MoleculesNotInstalledError(
        """To use molecule featurizer, install astartes with pip install astartes[molecules]."""
    )

AVAILABLE_SPLITTERS = [
    "random",
    "scaffold",
    "kmeans",
    "dbscan",
    "sphere_exclusion",
    "optisim",
    "target_property",
    "molecular_weight",
]

# url: https://github.com/JacksonBurns/astartes/tree/main/astartes/samplers/extrapolation


def featurize(molecules: Union[List, np.ndarray], fingerprint: str, fprints_hopts: Dict) -> np.ndarray:
    """Call AIMSim's Molecule to featurize the molecules according to the arguments.

    Args:
        molecules (np.array or list): SMILES strings or RDKit molecule objects.
        fingerprint (str): The molecular fingerprint to be used.
        fprints_hopts (dict): Hyperparameters for AIMSim.

    Returns:
        np.ndarray: X array (featurized molecules)
    """
    X = []
    for molecule in molecules:
        try:
            if type(molecule) in (np.str_, str):
                mol = Molecule(mol_smiles=molecule)
            else:
                mol = Molecule(mol_graph=molecule)
        except LoadingError as le:
            raise RuntimeError(
                "Unable to featurize molecules using '{:s}' with this configuration: fprint_hopts={:s}"
                "\nCheck terminal output for messages from the RDkit logger. ".format(fingerprint, repr(fprints_hopts))
            ) from le
        mol.descriptor.make_fingerprint(
            mol.mol_graph,
            fingerprint,
            fingerprint_params=fprints_hopts,
        )
        X.append(mol.descriptor.to_numpy())
    return np.array(X)


def compute_similarities(
    source_molecules: Union[List, np.ndarray],
    target_molecules: Union[List, np.ndarray],
    fingerprint: str,
    fprints_hopts: Dict,
) -> np.ndarray:
    """
    Compute similarities between two lists of molecules. It receives two lists of
    smiles or RDKit molecule objects, extracts their fingerprints and computes the similarities
    between them.

    Args:
        source_molecules (np.array or list): SMILES strings or RDKit molecule objects.
        target_molecules (np.array or list): SMILES strings or RDKit molecule objects.
        fingerprint (str): The molecular fingerprint to be used.
        fprints_hopts (dict): Hyperparameters for AIMSim.

    Returns:
        np.ndarray: Matrix of similarities between the two lists of molecules
    """
    fps1 = featurize(source_molecules, fingerprint, fprints_hopts)  # assumed train set
    fps2 = featurize(target_molecules, fingerprint, fprints_hopts)  # assumed test set
    sims = 1 - distance.cdist(fps1, fps2, metric="jaccard")
    return sims.astype(np.float32)


def get_scaffold(mol: Union[str, rdkit.Chem.Mol], make_generic: bool = False):
    """
    Computes the Bemis-Murcko scaffold of a compound.
    If make_generic is True, the scaffold is made generic by replacing all side chains with R groups.

    Args:
        mol (str or rdkit.Chem.Mol): SMILES string or RDKit molecule object.
        make_generic (bool): Whether to make the scaffold generic.

    Returns:
        str: The scaffold of the molecule.
    """
    mol = dm.to_mol(mol)
    scaffold = dm.to_scaffold_murcko(mol, make_generic=make_generic)
    scaffold = dm.to_smiles(scaffold)
    return scaffold


def split_molecules_train_test(
    mol_df: pd.DataFrame,
    sampler: str,
    train_size: float = 0.9,
    random_state: int = 42,
    hopts: dict = {},
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split molecules into train and test sets.


    Args:
        df (pd.DataFrame): Dataframe of moleucles. It must have two columns: 'smiles' and 'label'.
        sampler (str): Sampler to use.
            Options: random, scaffold, kmeans, dbscan, sphere_exclusion, optisim.
        train_size (float): Size of the train set.
        random_state (int): Random state for reproducibility.
        hops (dict): Hyperparameters for the sampler.

    Returns:
        tuple: Tuple containing the train and test sets.
    """
    assert "smiles" in mol_df.columns, "Dataframe must have a 'smiles' column."
    assert "label" in mol_df.columns, "Dataframe must have a 'label' column."

    X = np.array(mol_df["smiles"])
    y = np.array(mol_df["label"])

    hopts = {"shuffle": True}
    assert sampler in AVAILABLE_SPLITTERS, f"Sampler must be one of {AVAILABLE_SPLITTERS}"
    hopts.update(hopts)

    *others, train_ind, test_ind = train_test_split_molecules(
        molecules=X,
        y=y,
        train_size=train_size,
        fingerprint="morgan_fingerprint",
        fprints_hopts={
            "radius": 2,
            "n_bits": 2048,
        },
        sampler=sampler,
        random_state=random_state,
        hopts=hopts,
        return_indices=True,
    )

    train = mol_df.iloc[train_ind]
    test = mol_df.iloc[test_ind]

    return train, test


def split_molecules_train_val_test(
    mol_df: pd.DataFrame,
    sampler: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
    hopts: dict = {},
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split molecules into train and test sets.


    Args:
        df (pd.DataFrame): Dataframe of moleucles. It must have two columns: 'smiles' and 'label'.
        sampler (str): Sampler to use.
            Options: RandomSplit, ScaffoldSplit, KMeansSplit, DBScanSplit, SphereExclusionSplit, OptiSimSplit.
        train_size (float): Size of the train set.
        val_size (float): Size of the validation set.
        random_state (int): Random state for reproducibility.
        hops (dict): Hyperparameters for the sampler.

    Returns:
        tuple: Tuple containing the train and test sets.
    """
    assert "smiles" in mol_df.columns, "Dataframe must have a 'smiles' column."
    assert "label" in mol_df.columns, "Dataframe must have a 'label' column."

    X = np.array(mol_df["smiles"])
    y = np.array(mol_df["label"])

    test_size = 1 - train_size - val_size

    hopts = {"shuffle": True}
    hopts.update(hopts)

    *others, train_ind, val_ind, test_ind = train_val_test_split_molecules(
        molecules=X,
        y=y,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        fingerprint="morgan_fingerprint",
        fprints_hopts={
            "radius": 2,
            "n_bits": 2048,
        },
        sampler=sampler,
        random_state=random_state,
        hopts=hopts,
        return_indices=True,
    )

    train = mol_df.iloc[train_ind]
    val = mol_df.iloc[val_ind]
    test = mol_df.iloc[test_ind]

    return train, val, test


def sklearn_random_split(X, y, split_ratio, random_state=1234):
    """create random train/val/test split in sklearn
    Args:
        X (np.array): features
        y (np.array): labels
        split_ratio (tuple): train, val, test split ratio
        random_state (int): random seed

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert sum(split_ratio) == 1, "split ratio must sum to 1"
    train_ratio, val_ratio, test_ratio = split_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def sklearn_stratified_random_split(X, y, split_ratio, random_state=1234):
    """
    Create stratified random train/val/test splits using scikit-learn.
    
    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        split_ratio (tuple of float): Ratios for train, val, and test splits (must sum to 1).
        random_state (int): Random seed.
    
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert len(split_ratio) == 3, "split_ratio must have exactly three elements (train, val, test)."
    assert sum(split_ratio) == 1, "split_ratio must sum to 1."
    assert all(r > 0 for r in split_ratio), "split_ratio elements must be positive."
    
    train_ratio, val_ratio, test_ratio = split_ratio
    
    # First split: separate test set
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    for train_index, test_index in split.split(X, y):
        X_train_val, X_test = X[train_index], X[test_index]
        y_train_val, y_test = y[train_index], y[test_index]
    
    # Second split: separate train and validation sets
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=random_state)
    for train_index, val_index in split.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    return X_train, X_val, X_test, y_train, y_val, y_test



class EmpiricalKernelMapTransformer:
    """
    Transforms a dataset using the Empirical Kernel Map method.
    In this, a point is defined by its distance to a set of reference points.
    After this transformation, one can use the euclidean metric even if the original space was not euclidean compatible.

    Reference:
        https://github.com/datamol-io/splito/blob/main/splito/utils.py
    """

    def __init__(self, n_samples: int, metric: str, random_state: Optional[int] = None):
        self._n_samples = n_samples
        self._random_state = random_state
        self._samples = None
        self._metric = metric

    def __call__(self, X):
        """Transforms a list of datapoints"""
        return self.transform(X)

    def transform(self, X):
        """Transforms a single datapoint"""
        if self._samples is None:
            # Select the reference set
            rng = np.random.default_rng(self._random_state)
            self._samples = X[rng.choice(np.arange(len(X)), self._n_samples)]
        # Compute the distance to the reference set
        X = distance.cdist(X, self._samples, metric=self._metric)
        return X


def convert_to_default_feats_if_smiles(X: Union[Sequence[str], np.ndarray], metric: str, n_jobs: Optional[int] = None):
    """
    If the input is a sequence of strings, assumes this is a list of SMILES and converts it
    to a default set of ECFP4 features with the default Tanimoto distance metric.

    Reference:
        https://github.com/datamol-io/splito/blob/main/splito/_distance_split_base.py
    """

    def _to_feats(smi: str):
        mol = dm.to_mol(smi)
        feats = dm.to_fp(mol=mol, fp_type=MOLECULE_DEFAULT_FEATURIZER["name"], **MOLECULE_DEFAULT_FEATURIZER["kwargs"])
        return feats

    if all(isinstance(x, str) for x in X):
        X = dm.utils.parallelized(_to_feats, X, n_jobs=n_jobs)
        metric = MOLECULE_DEFAULT_DISTANCE_METRIC
    return X, metric


def pairwise_dataset_distance(
    X: Union[Sequence[str], np.ndarray, pd.DataFrame], metric: str, n_jobs: Optional[int] = None
):
    """
    Calculate the Tanimoto distance between a list of SMILES strings or a list of RDKit molecules.

    Args:
        X (np.array or smiles): Features or smiles strings (N * D if array, N if list)
        metric (str): Distance metric
        n_jobs (int): Number of jobs to run in parallel

    Returns:
        np.array: Distance matrix (N * N)

    Note:
        If the input is a sequence of strings, assumes this is a list of SMILES and converts it
        to a default set of ECFP4 features with the default Tanimoto (Jaccard) distance metric.
    """
    if isinstance(X, pd.DataFrame):
        assert "smiles" in X.columns, "Dataframe must have a 'smiles' column."
        X = X["smiles"]
    X, metric = convert_to_default_feats_if_smiles(X, metric, n_jobs=n_jobs)
    return distance.squareform(distance.pdist(X=np.array(X), metric=metric))


def retrive_index(original_df, splitted_df):
    """
    Retrieve the index of the splitted dataframe in the original dataframe.
    Args:
        original_df (pd.DataFrame): Original dataframe
        splitted_df (pd.DataFrame): Splitted dataframe

    Returns:
        np.array: Index of the splitted dataframe in the original dataframe
    """
    assert "smiles" in original_df.columns, "Dataframe must have a 'smiles' column."
    assert "smiles" in splitted_df.columns, "Dataframe must have a 'smiles' column."
    original_df["smiles"] = original_df["smiles"].astype(str)
    splitted_df["smiles"] = splitted_df["smiles"].astype(str)
    return original_df[original_df["smiles"].isin(splitted_df["smiles"])].index


def train_test_dataset_distance_retrieve(
    original_df: Union[str, pd.DataFrame],
    train_df: Union[str, pd.DataFrame],
    test_df: Union[str, pd.DataFrame],
    pairwise_distance: Union[str, np.array],
):
    """
    Compute the pairwise distance between the train and test set.

    Args:
        original_df (pd.DataFrame): Original dataframe
        train_df (pd.DataFrame): Train dataframe
        test_df (pd.DataFrame): Test dataframe
        pairwise_distance (str or np.array): Pairwise distance matrix

    Returns:
        np.array: Pairwise distance between the train and test set

    Note:
        If the pairwise distance is a string, it will be loaded from the file
        If the original_df, train_df, test_df are strings, they will be loaded from the file
    """

    if isinstance(pairwise_distance, str):
        pairwise_distance = np.load(pairwise_distance)

    if isinstance(original_df, str):
        original_df = pd.read_csv(original_df)

    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)

    if isinstance(test_df, str):
        test_df = pd.read_csv(test_df)

    assert "smiles" in original_df.columns, "Dataframe must have a 'smiles' column."
    assert "smiles" in train_df.columns, "Dataframe must have a 'smiles' column."

    assert (
        original_df.shape[0] == pairwise_distance.shape[0]
    ), "Pairwise distance matrix must have the same number of rows as the original dataframe."
    assert pairwise_distance.shape[0] == pairwise_distance.shape[1], "Pairwise distance matrix must be a square matrix"

    train_index = retrive_index(original_df, train_df)
    test_index = retrive_index(original_df, test_df)
    dist = pairwise_distance[np.ix_(train_index, test_index)]
    return dist


def retrieve_k_nearest_neighbors(
    pairwise_distance: Union[str, np.array],
    original_df: Union[str, pd.DataFrame],
    train_df: Union[str, pd.DataFrame],
    test_df: Union[str, pd.DataFrame],
    k=5,
):
    """
    Retrieve the k nearest neighbors from the distance matrix (full N*N square distance matrix).
    Firt, we retrieve the distance matrix between the train and test set (M * L matrix).

    For each test sample, retrieve the k nearest neighbors from the train set.
    determine similarty as 1 - distance. Then flatten the matrix to a vector.
    (N, N) -> (M, L) -> (L * k, )


    Args:
        pairwise_distance (np.array): Pairwise distance matrix
        original_df (pd.DataFrame): Original dataframe
        train_df (pd.DataFrame): Train dataframe
        test_df (pd.DataFrame): Test dataframe
        k (int): Number of neighbors to retrieve

    Returns:
        np.array: Tanimoto similarity of k nearest neighbors for each test set

    Reference:
        Similarity to Molecules in the Training Set Is a Good Discriminator for Prediction Accuracy in QSAR
        URL: https://pubs.acs.org/doi/full/10.1021/ci049782w
    """

    if isinstance(pairwise_distance, str):
        pairwise_distance = np.load(pairwise_distance)

    if isinstance(original_df, str):
        original_df = pd.read_csv(original_df)

    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)

    if isinstance(test_df, str):
        test_df = pd.read_csv(test_df)

    distance_matrix = train_test_dataset_distance_retrieve(original_df, train_df, test_df, pairwise_distance)

    # Then doing argparse along the axis 1 and pick the last k elements
    indices = np.argpartition(distance_matrix, k, axis=0)[:k, :]
    # Just retrive the elemns in each column based on indices
    best_n_distance = np.take_along_axis(distance_matrix, indices, axis=0)
    # since we want similarity, we use 1- distance
    best_n_similarity = 1 - best_n_distance
    # We want to flatten this matrix and just have a vector for each distance matrix
    best_n_similarity = best_n_similarity.flatten()

    return best_n_similarity
