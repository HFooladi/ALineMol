from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from typing import List, Tuple, Union


def get_similar_mols(
    lhs: List[str], rhs: List[str], return_idx: bool = False
) -> Union[List[float], Tuple[List[float], List[int]]]:
    """
    Calculate maximal similarity between two sets of molecules for each molecule.

    This function computes the maximum Tanimoto similarity between each molecule in the left-hand set
    and all molecules in the right-hand set using ECFP4 fingerprints.

    Args:
        lhs: List of SMILES strings representing the left-hand set of molecules
        rhs: List of SMILES strings representing the right-hand set of molecules
        return_idx: If True, also returns indices of the most similar molecules

    Returns:
        If return_idx is False:
            List[float]: List of maximum similarities, where i-th element is the max similarity
                        between lhs[i] and any molecule in rhs
        If return_idx is True:
            Tuple[List[float], List[int]]: Tuple containing:
                - List of maximum similarities
                - List of indices in rhs corresponding to the most similar molecules
    """
    # Initialize ECFP4 fingerprint generator
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    # Convert SMILES to RDKit molecules and generate fingerprints for left-hand set
    lhs_mols = [Chem.MolFromSmiles(smiles) for smiles in lhs]
    lhs_fps = [fp_generator.GetFingerprint(x) for x in lhs_mols]

    # Convert SMILES to RDKit molecules and generate fingerprints for right-hand set
    rhs_mols = [Chem.MolFromSmiles(smiles) for smiles in rhs]
    rhs_fps = [fp_generator.GetFingerprint(x) for x in rhs_mols]

    # Calculate similarities and find maximum for each molecule
    nearest_sim = []
    nearest_idx = []
    for lhs_fp in lhs_fps:
        # Calculate Tanimoto similarity with all molecules in right-hand set
        sims = DataStructs.BulkTanimotoSimilarity(lhs_fp, rhs_fps)
        nearest_sim.append(max(sims))
        nearest_idx.append(np.argmax(sims))

    # Return appropriate result based on return_idx flag
    return (nearest_sim, nearest_idx) if return_idx else nearest_sim
