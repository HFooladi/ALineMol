import copy
from typing import Dict, Tuple

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
    """Initialize node/edge featurizer

    Args:
        model (str): Model name

    Returns:
        dict: Settings with featurizers updated

    Raises:
        ValueError: If the node_featurizer_type is not in ['canonical', 'attentivefp']

    Example:
    ```
    from alinemol.utils.graph_utils import choose_featurizer
    model = "GCN"
    atom_featurizer_type = "canonical"
    bond_featurizer_type = "canonical"
    result = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    print(result)
    ```

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
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], " "got {}".format(atom_featurizer_type)
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
def TMD(g1: Data, g2: Data = None, w=1.0, L: int = 4) -> float:
    """
    return the Tree Mover’s Distance (TMD) between g1 and g2

    Args:
        g1: First torch_geometric graph Data object
        g2: Second torch_geometric graph Data object (default is None)
        w: weighting constant for each depth
            if it is a list, then w[l] is the weight for depth-(l+1) tree
            if it is a constant, then every layer shares the same weight
        L: Depth of computation trees for calculating TMD

    Returns:
        wass : The TMD between g1 and g2

    Reference:
        Chuang et al., Tree Mover’s Distance: Bridging Graph Metrics and
        Stability of Graph Neural Networks, NeurIPS 2022
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

    blank = np.zeros(len(feat1[0]))
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
    for l in range(L - 1):
        M1 = copy.deepcopy(M)
        M = np.zeros((n1 + 1, n2 + 1))

        # calculate pairwise cost between tree i and tree j
        for i in range(n1):
            for j in range(n2):
                try:
                    degree_i = len(adj1[i])
                except:
                    degree_i = 0
                try:
                    degree_j = len(adj2[j])
                except:
                    degree_j = 0

                if degree_i == 0 and degree_j == 0:
                    M[i, j] = D[i, j]
                # if degree of node is zero, calculate TD w.r.t. blank node
                elif degree_i == 0:
                    wass = 0.0
                    for jj in range(degree_j):
                        wass += M1[n1, adj2[j][jj]]
                    M[i, j] = D[i, j] + w[l] * wass
                elif degree_j == 0:
                    wass = 0.0
                    for ii in range(degree_i):
                        wass += M1[adj1[i][ii], n2]
                    M[i, j] = D[i, j] + w[l] * wass
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
                    M[i, j] = D[i, j] + w[l] * wass

        # fill in dist w.r.t. blank node
        for i in range(n1):
            try:
                degree_i = len(adj1[i])
            except:
                degree_i = 0

            if degree_i == 0:
                M[i, n2] = torch.norm(feat1[i])
            else:
                wass = 0.0
                for ii in range(degree_i):
                    wass += M1[adj1[i][ii], n2]
                M[i, n2] = torch.norm(feat1[i]) + w[l] * wass

        for j in range(n2):
            try:
                degree_j = len(adj2[j])
            except:
                degree_j = 0
            if degree_j == 0:
                M[n1, j] = torch.norm(feat2[j])
            else:
                wass = 0.0
                for jj in range(degree_j):
                    wass += M1[n1, adj2[j][jj]]
                M[n1, j] = torch.norm(feat2[j]) + w[l] * wass

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
    src_pyg_graphs: list[Data], tgt_pyg_graphs=None, metric="TMD", n_jobs=1, **kwds
) -> np.ndarray:
    """
    Calculate pairwise Tree Mover's Distance (TMD) between graphs

    If tgt_pyg_graphs is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both src_pyg_graphs and tgt_pyg_graphs.

    Args:
        src_pyg_graphs (list[Data]): List of PyG Data objects
        tgt_pyg_graphs (list[Data]): List of PyG Data objects (default is None)
        metric (str): The metric to use when calculating distance
        n_jobs (int): The number of jobs to run in parallel (default is 1)
        **kwrds: Additional keyword arguments

    Returns:
        np.ndarray: Pairwise TMD matrix
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
        with joblib_progress("computing the distance matrix ...", total=number_of_comparisons) as progress:
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
        with joblib_progress("computing the distance matrix ...", total=number_of_comparisons) as progress:
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
