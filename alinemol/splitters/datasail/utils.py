from rdkit import Chem
import networkx as nx
import numpy as np


def get_rdkit_fct(smiles: str):
    """Get RDKit fingerprint from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.RDKFingerprint(mol)


def get_linear_problem_k_fold(
    S: nx.Graph, fold_min_frac: float, k: int, verbose: bool = True, max_mip_gap: float = 0.1
):
    """Get linear problem k-fold."""
    return get_linear_problem_k_fold(S, fold_min_frac, k, verbose, max_mip_gap)


def get_datasail_clusters(
    X: np.ndarray, n_clusters: int, n_neighbors: int, verbose: bool = True, max_mip_gap: float = 0.1
):
    """Get DataSail clusters."""
    return get_datasail_clusters(X, n_clusters, n_neighbors, verbose, max_mip_gap)
