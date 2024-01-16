import numpy as np
import pandas as pd
from astartes.molecules import train_test_split_molecules, train_val_test_split_molecules

from typing import Dict, Tuple



def split_hypers(sampler: str="random") -> Dict:
    """
    Get the hyperparameters for the sampler.

    Args:
        sampler (str): Sampler to use. 
            Options: 'random', 'scaffold', 'kmeans', 'dbscan'.
    
    Returns:
        dict: Hyperparameters for the sampler.
    """
    #ToDO: Check the parameters for the smaplers
    if sampler == "random":
        hopts = {}
    elif sampler == "scaffold":
        hopts = {'include_chirality': False}
    elif sampler == "kmeans":
        hopts = {
            "n_clusters": 100,
            "n_init": 10,
        }
    return hopts



def split_molecules_train_test(
    mol_df: pd.DataFrame, train_size: float=0.9, sampler: str="random", random_state: int=42
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
    mol_df: pd.DataFrame, train_size: float=0.8, val_size: float=0.1, sampler: str="random", random_state: int=42
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