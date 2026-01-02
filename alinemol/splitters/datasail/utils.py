"""
Utility functions for DataSAIL integration.

These utilities provide helper functions for molecular similarity
calculations and fingerprint generation used by DataSAIL splitting.
"""

from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_ecfp_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> List:
    """
    Compute ECFP fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan fingerprint radius. Default 2 (ECFP4).
        n_bits: Fingerprint bit size. Default 1024.

    Returns:
        List of RDKit fingerprint objects.

    Raises:
        ValueError: If any SMILES string is invalid.
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)
    return fps


def compute_ecfp_similarity_matrix(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix for ECFP fingerprints.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan fingerprint radius. Default 2 (ECFP4).
        n_bits: Fingerprint bit size. Default 1024.

    Returns:
        Symmetric similarity matrix (n_samples x n_samples).
    """
    fps = compute_ecfp_fingerprints(smiles_list, radius, n_bits)

    n = len(fps)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


def compute_ecfp_distance_matrix(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """
    Compute pairwise Tanimoto distance matrix for ECFP fingerprints.

    Distance is computed as 1 - similarity.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan fingerprint radius. Default 2 (ECFP4).
        n_bits: Fingerprint bit size. Default 1024.

    Returns:
        Symmetric distance matrix (n_samples x n_samples).
    """
    return 1.0 - compute_ecfp_similarity_matrix(smiles_list, radius, n_bits)


def smiles_to_fingerprint_array(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 1024,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Convert SMILES strings to a numpy array of fingerprints.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan fingerprint radius. Default 2 (ECFP4).
        n_bits: Fingerprint bit size. Default 1024.
        n_jobs: Number of parallel jobs (currently not used).

    Returns:
        2D numpy array of shape (n_samples, n_bits).
    """
    fps = compute_ecfp_fingerprints(smiles_list, radius, n_bits)

    # Convert to numpy array
    fp_array = np.zeros((len(fps), n_bits), dtype=np.int8)
    for i, fp in enumerate(fps):
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[i] = arr

    return fp_array
