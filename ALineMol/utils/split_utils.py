import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
from astartes.utils.exceptions import MoleculesNotInstalledError

from typing import Dict, List, Tuple, Union

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


AVAILABLE_SPLITTERS = ['random', 'scaffold', 'kmeans', 'dbscan', 'sphere_exclusion', 'optisim']


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
                "\nCheck terminal output for messages from the RDkit logger. ".format(
                    fingerprint, repr(fprints_hopts)
                )
            ) from le
        mol.descriptor.make_fingerprint(
            mol.mol_graph,
            fingerprint,
            fingerprint_params=fprints_hopts,
        )
        X.append(mol.descriptor.to_numpy())
    return np.array(X)


def compute_similarities(source_molecules: Union[List, np.ndarray], target_molecules: Union[List, np.ndarray], fingerprint: str, fprints_hopts: Dict) -> np.ndarray:
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


def split_hypers(sampler: str = "random") -> Dict:
    """
    Get the hyperparameters for the sampler.

    Args:
        sampler (str): Sampler to use.
            Options: 'random', 'scaffold', 'kmeans', 'dbscan'.

    Returns:
        dict: Hyperparameters for the sampler.

    Notes:
        url: https://github.com/JacksonBurns/astartes/tree/main/astartes/samplers/extrapolation
    """
    # ToDO: Check the parameters for the smaplers
    if sampler == "random":
        hopts = {}
    elif sampler == "scaffold":
        hopts = {"include_chirality": False}
    elif sampler == "kmeans":
        hopts = {
            "n_clusters": 100,
            "n_init": 10,
        }
    elif sampler == "dbscan":
        hopts = {
            "eps": 0.5,
            "metric": "euclidean",
        }
    elif sampler == "sphere_exclusion":
        hopts = {
            "metric": "euclidean",
            "distance_cutoff": 0.5,
        }
    elif sampler == "optisim":
        hopts = {
            "n_clusters": 10,
            "max_subsample_size": 1000,
            "distance_cutoff": 0.1,
        }
    return hopts


def split_molecules_train_test(
    mol_df: pd.DataFrame, train_size: float = 0.9, sampler: str = "random", random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split molecules into train and test sets.


    Args:
        df (pd.DataFrame): Dataframe of moleucles. It must have two columns: 'smiles' and 'label'.
        train_size (float): Size of the train set.
        sampler (str): Sampler to use.
            Options: 'random', 'scaffold', 'kmeans', 'dbscan'.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Tuple containing the train and test sets.
    """
    assert "smiles" in mol_df.columns, "Dataframe must have a 'smiles' column."
    assert "label" in mol_df.columns, "Dataframe must have a 'label' column."

    X = np.array(mol_df["smiles"])
    y = np.array(mol_df["label"])


    hopts = {"shuffle": True}
    hopts.update(split_hypers(sampler))

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
    train_size: float = 0.8,
    val_size: float = 0.1,
    sampler: str = "random",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split molecules into train and test sets.


    Args:
        df (pd.DataFrame): Dataframe of moleucles. It must have two columns: 'smiles' and 'label'.
        train_size (float): Size of the train set.
        val_size (float): Size of the validation set.
        sampler (str): Sampler to use.
            Options: 'random', 'scaffold', 'kmeans', 'dbscan'.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Tuple containing the train and test sets.
    """
    assert "smiles" in mol_df.columns, "Dataframe must have a 'smiles' column."
    assert "label" in mol_df.columns, "Dataframe must have a 'label' column."

    X = np.array(mol_df["smiles"])
    y = np.array(mol_df["label"])

    test_size = 1 - train_size - val_size

    hopts = {"shuffle": True}
    hopts.update(split_hypers(sampler))

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
