import copy
from typing import Dict, Tuple, Union, List, Optional

import numpy as np
import ot
import torch
from tqdm import tqdm
from dgllife.data import UnlabeledSMILES
from dgllife.utils import MolToBigraph
from torch_geometric.data import Data
from torch_geometric.utils import from_dgl

from functools import partial
from joblib import Parallel, delayed, effective_n_jobs
from joblib_progress import joblib_progress


def choose_featurizer(
    model: str, atom_featurizer_type: str = "canonical", bond_featurizer_type: str = "canonical"
) -> Tuple:
    """Initialize node and edge featurizers based on model and featurizer types.

    This function selects appropriate atom (node) and bond (edge) featurizers
    based on the requested model and featurizer types. For pre-trained GIN models,
    it automatically uses the corresponding pre-trained featurizers.

    Args:
        model (str): Model name to determine appropriate featurizers.
            Special handling for GIN models and models requiring edge features.
        atom_featurizer_type (str, optional): Type of atom featurizer to use.
            Options: 'canonical', 'attentivefp'. Defaults to "canonical".
        bond_featurizer_type (str, optional): Type of bond featurizer to use.
            Options: 'canonical', 'attentivefp'. Defaults to "canonical".

    Returns:
        Tuple[Featurizer, Featurizer or None]: A tuple containing:
            - The node (atom) featurizer
            - The edge (bond) featurizer, or None if not needed for the model

    Raises:
        ValueError: If the atom_featurizer_type is not one of ['canonical', 'attentivefp']

    Notes:
        - Pre-trained GIN models require specific featurizers
        - Weave, MPNN, and AttentiveFP models require edge featurizers
        - Other models typically only need node featurizers

    Example:
        >>> from alinemol.utils.graph_utils import choose_featurizer
        >>> # For a GCN model with canonical featurizers
        >>> node_feat, edge_feat = choose_featurizer("GCN")
        >>> # For an AttentiveFP model with specialized featurizers
        >>> node_feat, edge_feat = choose_featurizer(
        ...    "AttentiveFP",
        ...    atom_featurizer_type="attentivefp",
        ...    bond_featurizer_type="attentivefp"
        ... )
        >>> print(node_feat)
        >>> print(edge_feat)
    """
    if model in [
        "gin_supervised_contextpred",
        "gin_supervised_infomax",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer

        atom_featurizer_type = "pre_train"
        bond_featurizer_type = "pre_train"
        node_featurizer = PretrainAtomFeaturizer()
        edge_featurizer = PretrainBondFeaturizer()
        return node_featurizer, edge_featurizer

    if atom_featurizer_type == "canonical":
        from dgllife.utils import CanonicalAtomFeaturizer

        node_featurizer = CanonicalAtomFeaturizer()
    elif atom_featurizer_type == "attentivefp":
        from dgllife.utils import AttentiveFPAtomFeaturizer

        node_featurizer = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], got {}".format(atom_featurizer_type)
        )

    if model in ["Weave", "MPNN", "AttentiveFP"]:
        if bond_featurizer_type == "canonical":
            from dgllife.utils import CanonicalBondFeaturizer

            edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        elif bond_featurizer_type == "attentivefp":
            from dgllife.utils import AttentiveFPBondFeaturizer

            edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        edge_featurizer = None

    return node_featurizer, edge_featurizer


def create_dgl_graphs(
    smiles: list[str], node_featurizer=None, edge_featurizer=None, add_self_loop=True
) -> UnlabeledSMILES:
    """Create DGL graph from SMILES string

    Args:
        smiles (list[str]): List of SMILES strings
        node_featurizer (Callable): Node featurizer
        edge_featurizer (Callable): Edge featurizer
        add_self_loop (bool): Whether to add self loop

    Returns:
        UnlabeledSMILES: UnlabeledSMILES object

    Note:
        UnlabeledSMILES contains the SMILES strings and the corresponding DGL graph
    """
    mol_to_g = MolToBigraph(
        add_self_loop=add_self_loop, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer
    )
    dataset = UnlabeledSMILES(smiles, mol_to_graph=mol_to_g)
    return dataset


def convert_dgl_pyg(dgl_graph) -> Data:
    """Convert DGL graph to PyG Data object

    Args:
        dgl_graph (DGLGraph): DGL graph object

    Returns:
        PyG Data: PyG Data object
    """
    data = from_dgl(dgl_graph)
    if "e" in data and "h" in data and "edge_index" in data:
        return Data(x=data["h"], edge_index=data.edge_index, edge_attr=data["e"])
    elif "h" in data and "edge_index" in data:
        return Data(x=data["h"], edge_index=data.edge_index)
    elif "edge_index" in data:
        return Data(edge_index=data.edge_index)


def create_pyg_graphs(
    smiles: list[str],
    model: str,
    atom_featurizer_type: str = "canonical",
    bond_featurizer_type: str = "canonical",
    add_self_loop=True,
) -> list[Data]:
    """
    Create PyG graph from SMILES string.

    Args:
        smiles (list[str]): List of SMILES strings
        model (str): Model name
        add_self_loop (bool): Whether to add self loop

    Returns:
        PyG Data: PyG Data object
    """
    node_featurizer, edge_featurizer = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    dgl_graphs = create_dgl_graphs(smiles, node_featurizer, edge_featurizer, add_self_loop=add_self_loop)
    pyg_graphs = [convert_dgl_pyg(dgl_graph[1]) for dgl_graph in dgl_graphs]
    return pyg_graphs


def get_neighbors(g: Data) -> Dict:
    """
    get neighbor indexes for each node

    Args:
        g : input torch_geometric graph

    Returns:
        adj: a dictionary that store the neighbor indexes

    """
    adj = {}
    for i in range(len(g.edge_index[0])):
        node1 = g.edge_index[0][i].item()
        node2 = g.edge_index[1][i].item()
        if node1 in adj.keys():
            adj[node1].append(node2)
        else:
            adj[node1] = [node2]
    return adj


# Tree Mover's Distance solver. Credit goes to the original authors.
# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License
# URL: https://github.com/chingyaoc/TMD
def TMD(g1: Data, g2: Data = None, w: Union[float, List[float]] = 1.0, L: int = 4) -> float:
    """Calculate the Tree Mover's Distance (TMD) between two molecular graphs.

    Tree Mover's Distance quantifies the structural similarity between graphs by measuring
    the optimal transport cost between their computation trees. This is particularly
    useful for comparing molecular structures in a way that aligns with how Graph Neural
    Networks process the structures.

    Args:
        g1 (Data): First PyTorch Geometric graph object
        g2 (Data, optional): Second PyTorch Geometric graph object. If None,
            calculates self-distance (should be 0). Defaults to None.
        w (float or List[float]): Weighting constant(s) for each depth.
            - If float: Same weight used for all depths
            - If List[float]: w[l] is the weight for depth-(l+1) tree
            Defaults to 1.0.
        L (int): Maximum depth of computation trees. Defaults to 4.

    Returns:
        float: The Tree Mover's Distance between g1 and g2, rounded to 2 decimal places.

    Notes:
        - Computation runtime scales with the depth L
        - Higher L values capture more global structural information
        - For comparing many molecules, use the pairwise_graph_distances function

    Reference:
        Chuang et al., Tree Mover's Distance: Bridging Graph Metrics and
        Stability of Graph Neural Networks, NeurIPS 2022

    Example:
        >>> from alinemol.utils.graph_utils import create_pyg_graphs, TMD
        >>> # Create PyG graphs from SMILES strings
        >>> smiles = ["CCO", "c1ccccc1"]
        >>> graphs = create_pyg_graphs(smiles, model="GCN")
        >>> # Calculate distance between two molecules
        >>> distance = TMD(graphs[0], graphs[1], L=3)
        >>> print(f"Distance between ethanol and benzene: {distance}")
    """
    # check if g2 is None
    if g2 is None:
        g2 = g1

    if isinstance(w, list):
        assert len(w) == L - 1
    else:
        w = [w] * (L - 1)

    # get attributes
    n1, n2 = len(g1.x), len(g2.x)
    feat1, feat2 = g1.x, g2.x
    adj1 = get_neighbors(g1)
    adj2 = get_neighbors(g2)

    _ = np.zeros(len(feat1[0]))
    D = np.zeros((n1, n2))

    # level 1 (pair wise distance)
    M = np.zeros((n1 + 1, n2 + 1))
    for i in range(n1):
        for j in range(n2):
            D[i, j] = torch.norm(feat1[i] - feat2[j])
            M[i, j] = D[i, j]
    # distance w.r.t. blank node
    M[:n1, n2] = torch.norm(feat1, dim=1)
    M[n1, :n2] = torch.norm(feat2, dim=1)

    # level l (tree OT)
    for layer_idx in range(L - 1):
        M1 = copy.deepcopy(M)
        M = np.zeros((n1 + 1, n2 + 1))

        # calculate pairwise cost between tree i and tree j
        for i in range(n1):
            for j in range(n2):
                try:
                    degree_i = len(adj1[i])
                except KeyError:
                    degree_i = 0
                try:
                    degree_j = len(adj2[j])
                except KeyError:
                    degree_j = 0

                if degree_i == 0 and degree_j == 0:
                    M[i, j] = D[i, j]
                # if degree of node is zero, calculate TD w.r.t. blank node
                elif degree_i == 0:
                    wass = 0.0
                    for jj in range(degree_j):
                        wass += M1[n1, adj2[j][jj]]
                    M[i, j] = D[i, j] + w[layer_idx] * wass
                elif degree_j == 0:
                    wass = 0.0
                    for ii in range(degree_i):
                        wass += M1[adj1[i][ii], n2]
                    M[i, j] = D[i, j] + w[layer_idx] * wass
                # otherwise, calculate the tree distance
                else:
                    max_degree = max(degree_i, degree_j)
                    if degree_i < max_degree:
                        cost = np.zeros((degree_i + 1, degree_j))
                        cost[degree_i] = M1[n1, adj2[j]]
                        dist_1, dist_2 = np.ones(degree_i + 1), np.ones(degree_j)
                        dist_1[degree_i] = max_degree - float(degree_i)
                    else:
                        cost = np.zeros((degree_i, degree_j + 1))
                        cost[:, degree_j] = M1[adj1[i], n2]
                        dist_1, dist_2 = np.ones(degree_i), np.ones(degree_j + 1)
                        dist_2[degree_j] = max_degree - float(degree_j)
                    for ii in range(degree_i):
                        for jj in range(degree_j):
                            cost[ii, jj] = M1[adj1[i][ii], adj2[j][jj]]
                    wass = ot.emd2(dist_1, dist_2, cost)

                    # summarize TMD at level l
                    M[i, j] = D[i, j] + w[layer_idx] * wass

        # fill in dist w.r.t. blank node
        for i in range(n1):
            try:
                degree_i = len(adj1[i])
            except KeyError:
                degree_i = 0

            if degree_i == 0:
                M[i, n2] = torch.norm(feat1[i])
            else:
                wass = 0.0
                for ii in range(degree_i):
                    wass += M1[adj1[i][ii], n2]
                M[i, n2] = torch.norm(feat1[i]) + w[layer_idx] * wass

        for j in range(n2):
            try:
                degree_j = len(adj2[j])
            except KeyError:
                degree_j = 0
            if degree_j == 0:
                M[n1, j] = torch.norm(feat2[j])
            else:
                wass = 0.0
                for jj in range(degree_j):
                    wass += M1[n1, adj2[j][jj]]
                M[n1, j] = torch.norm(feat2[j]) + w[layer_idx] * wass

    # final OT cost
    max_n = max(n1, n2)
    dist_1, dist_2 = np.ones(n1 + 1), np.ones(n2 + 1)
    if n1 < max_n:
        dist_1[n1] = max_n - float(n1)
        dist_2[n2] = 0.0
    else:
        dist_1[n1] = 0.0
        dist_2[n2] = max_n - float(n2)

    wass = ot.emd2(dist_1, dist_2, M)
    return round(wass, 2)


PAIRWISE_DISTANCE_FUNCTIONS = {
    "TMD": TMD,
}


def pairwise_graph_distances(
    src_pyg_graphs: List[Data],
    tgt_pyg_graphs: Optional[List[Data]] = None,
    metric: str = "TMD",
    n_jobs: int = 1,
    **kwds,
) -> np.ndarray:
    """Calculate pairwise distances between collections of molecular graphs.

    This function computes all pairwise distances between two sets of graphs
    (or within a single set if tgt_pyg_graphs=None). It supports parallel
    computation and various distance metrics.

    Args:
        src_pyg_graphs (List[Data]): Source collection of PyG graph objects.
        tgt_pyg_graphs (List[Data], optional): Target collection of PyG graph objects.
            If None, calculates distances within src_pyg_graphs. Defaults to None.
        metric (str): The metric to use for distance calculation.
            Options: "TMD" (Tree Mover's Distance). Defaults to "TMD".
        n_jobs (int): Number of parallel jobs for computation.
            Set to -1 to use all available cores. Defaults to 1.
        **kwds: Additional keyword arguments passed to the distance function.
            For TMD, these can include 'w' and 'L' parameters.

    Returns:
        np.ndarray: Matrix of pairwise distances with shape:
            - (len(src_pyg_graphs), len(tgt_pyg_graphs)) if tgt_pyg_graphs is provided
            - (len(src_pyg_graphs), len(src_pyg_graphs)) if tgt_pyg_graphs is None

    Notes:
        - For self-comparison (tgt_pyg_graphs=None), only computes the upper
          triangular part of the matrix and mirrors it for efficiency
        - Shows progress bars during computation
        - For large datasets, increasing n_jobs can significantly speed up computation

    Example:
        >>> from alinemol.utils.graph_utils import create_pyg_graphs, pairwise_graph_distances
        >>> import numpy as np

        >>> # Create PyG graphs from SMILES
        >>> smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCCCCC"]
        >>> graphs = create_pyg_graphs(smiles, model="GCN")

        >>> # Calculate all pairwise distances
        >>> dist_matrix = pairwise_graph_distances(graphs, metric="TMD", n_jobs=4, L=3)

        >>> # Find most similar pair
        >>> i, j = np.unravel_index(
        ...     np.argmin(dist_matrix + np.eye(len(graphs)) * 999),
        ...     dist_matrix.shape
        ... )
        >>> print(f"Most similar molecules: {smiles[i]} and {smiles[j]}")
    """
    symmetric = False
    if tgt_pyg_graphs is None:
        tgt_pyg_graphs = src_pyg_graphs
        symmetric = True

    func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    func = partial(func, **kwds)
    n_jobs = effective_n_jobs(n_jobs)
    distance_array = np.zeros((len(src_pyg_graphs), len(tgt_pyg_graphs)))

    if symmetric:
        number_of_comparisons = len(src_pyg_graphs) * (len(src_pyg_graphs) + 1) // 2
        with joblib_progress("computing the distance matrix ...", total=number_of_comparisons) as _:
            distances = Parallel(n_jobs=n_jobs)(
                delayed(func)(src_pyg_graphs[i], tgt_pyg_graphs[j])
                for i in range(len(src_pyg_graphs))
                for j in range(i, len(tgt_pyg_graphs))
            )
        k = 0
        for i in tqdm(range(len(src_pyg_graphs))):
            for j in range(i, len(tgt_pyg_graphs)):
                if k < len(distances):
                    distance_array[i, j] = distances[k]
                    distance_array[j, i] = distance_array[i, j]
                    k += 1

    else:
        number_of_comparisons = len(src_pyg_graphs) * len(tgt_pyg_graphs)
        with joblib_progress("computing the distance matrix ...", total=number_of_comparisons) as _:
            distances = Parallel(n_jobs=n_jobs)(
                delayed(func)(src_pyg_graphs[i], tgt_pyg_graphs[j])
                for i in range(len(src_pyg_graphs))
                for j in range(len(tgt_pyg_graphs))
            )
        k = 0
        for i in tqdm(range(len(src_pyg_graphs))):
            for j in range(len(tgt_pyg_graphs)):
                distance_array[i, j] = distances[k]
                k += 1

    return distance_array
