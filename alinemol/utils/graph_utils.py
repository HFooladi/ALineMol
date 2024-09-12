import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_dgl

from dgllife.data import UnlabeledSMILES
from dgllife.utils import MolToBigraph

from typing import Dict, Tuple



def choose_featurizer(model:str, atom_featurizer_type:str = "canonical", bond_featurizer_type:str = "canonical") -> Tuple:
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
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], " "got {}".format(
                atom_featurizer_type)
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


def create_dgl_graphs(smiles: list[str], node_featurizer=None, edge_featurizer=None, add_self_loop=True) -> UnlabeledSMILES:
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
    mol_to_g = MolToBigraph(add_self_loop=add_self_loop, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
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
    if "e" in data:
        return Data(x=data["h"], edge_index=data.edge_index, edge_attr=data["e"])
    else:
        return Data(x=data["h"], edge_index=data.edge_index)
def create_pyg_graphs(smiles: str, model: str, add_self_loop=True):
    """
    Create PyG graph from SMILES string

    Args:
        smiles (str): SMILES string
        model (str): Model name
        add_self_loop (bool): Whether to add self loop
    
    Returns:
        PyG Data: PyG Data object
    """
    node_featurizer, edge_featurizer = choose_featurizer(model)
    dgl_graphs = create_dgl_graphs(smiles, node_featurizer, edge_featurizer, add_self_loop=add_self_loop)
    pyg_graphs = [convert_dgl_pyg(dgl_graph[1]) for dgl_graph in dgl_graphs]
    return pyg_graphs
