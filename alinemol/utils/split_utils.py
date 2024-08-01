import warnings
from typing import Dict, List, Tuple, Union

import datamol as dm
import numpy as np
import pandas as pd
import rdkit
from sklearn.model_selection import train_test_split
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules
from astartes.utils.exceptions import MoleculesNotInstalledError
from rdkit import Chem
from scipy.spatial import distance

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


def sklearn_random_split(X, y, split_ratio, random_state=42):
    """create random train/val/test split in sklearn
    Args:
        X (np.array): features
        y (np.array): labels
        split_ratio (tuple): train, val, test split ratio
    """
    assert sum(split_ratio) == 1, "split ratio must sum to 1"
    train_ratio, val_ratio, test_ratio = split_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
    