# We will test all the functions available in alinemol/utils/graph_utils.py
import numpy as np
import pytest
from alinemol.utils.graph_utils import (
    choose_featurizer,
    create_dgl_graphs,
    convert_dgl_pyg,
    create_pyg_graphs,
    get_neighbors,
    TMD,
)
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
)


# Test 1: test_choose_featurizer
def test_choose_featurizer_1():
    model = "GCN"
    atom_featurizer_type = "canonical"
    bond_featurizer_type = "canonical"
    result = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    assert result[0].__class__ == CanonicalAtomFeaturizer
    assert result[1] == None


# Test 2: test_choose_featurizer
def test_choose_featurizer_2():
    model = "Weave"
    atom_featurizer_type = "canonical"
    bond_featurizer_type = "canonical"
    result = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    assert result[0].__class__ == CanonicalAtomFeaturizer
    assert result[1].__class__ == CanonicalBondFeaturizer


# Test 3: test_choose_featurizer
def test_choose_featurizer_3():
    model = "AttentiveFP"
    atom_featurizer_type = "attentivefp"
    bond_featurizer_type = "attentivefp"
    result = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    assert result[0].__class__ == AttentiveFPAtomFeaturizer
    assert result[1].__class__ == AttentiveFPBondFeaturizer


# Test 4: test_choose_featurizer
def test_choose_featurizer_4():
    model = "gin_supervised_contextpred"
    atom_featurizer_type = "canonical"
    bond_featurizer_type = "canonical"
    result = choose_featurizer(model, atom_featurizer_type, bond_featurizer_type)
    assert result[0].__class__ == PretrainAtomFeaturizer
    assert result[1].__class__ == PretrainBondFeaturizer


# Test 5: test_create_dgl_graphs
def test_create_dgl_graphs_1():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = CanonicalAtomFeaturizer()
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    result = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    assert len(result) == 3


# Test 6: test_create_dgl_graphs
def test_create_dgl_graphs_2():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = AttentiveFPAtomFeaturizer()
    edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    result = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    assert len(result) == 3


# Test 7: test_create_dgl_graphs
def test_create_dgl_graphs_3():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = PretrainAtomFeaturizer()
    edge_featurizer = PretrainBondFeaturizer()
    result = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    assert len(result) == 3


# Test 8: test_convert_dgl_pyg
def test_convert_dgl_pyg_1():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = CanonicalAtomFeaturizer()
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    dgl_graphs = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    for dgl_graph in dgl_graphs:
        pyg_graph = convert_dgl_pyg(dgl_graph[1])
        assert pyg_graph.__class__.__name__ == "Data"


# Test 9: test_convert_dgl_pyg
def test_convert_dgl_pyg_2():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = AttentiveFPAtomFeaturizer()
    edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    dgl_graphs = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    for dgl_graph in dgl_graphs:
        pyg_graph = convert_dgl_pyg(dgl_graph[1])
        assert pyg_graph.__class__.__name__ == "Data"


# Test 10: test_convert_dgl_pyg
def test_convert_dgl_pyg_3():
    smiles = ["C", "CC", "CCC"]
    node_featurizer = PretrainAtomFeaturizer()
    edge_featurizer = PretrainBondFeaturizer()
    dgl_graphs = create_dgl_graphs(smiles, node_featurizer, edge_featurizer)
    for dgl_graph in dgl_graphs:
        pyg_graph = convert_dgl_pyg(dgl_graph[1])
        assert pyg_graph.__class__.__name__ == "Data"


# Test 11: test_create_pyg_graphs
def test_create_pyg_graphs_1():
    smiles = ["C", "CC", "CCC"]
    model = "GCN"
    result = create_pyg_graphs(smiles, model)
    assert len(result) == 3
    assert result[0].__class__.__name__ == "Data"
    assert result[0].x.shape[1] == 74


# Test 12: test_create_pyg_graphs
def test_create_pyg_graphs_2():
    smiles = ["C", "CC", "CCC"]
    model = "Weave"
    result = create_pyg_graphs(smiles, model)
    assert len(result) == 3
    assert result[0].__class__.__name__ == "Data"
    assert result[0].x.shape[1] == 74
    assert result[0].edge_attr.shape[1] == 13


# Test 13: test_create_pyg_graphs
def test_create_pyg_graphs_3():
    smiles = ["C", "CC", "CCC"]
    model = "AttentiveFP"
    result = create_pyg_graphs(smiles, model, atom_featurizer_type="attentivefp", bond_featurizer_type="attentivefp")
    assert len(result) == 3
    assert result[0].__class__.__name__ == "Data"
    assert result[0].x.shape[1] == 39
    assert result[0].edge_attr.shape[1] == 11


# Test 14: test_create_pyg_graphs
def test_create_pyg_graphs_4():
    smiles = ["C", "CC", "CCC"]
    model = "gin_supervised_contextpred"
    result = create_pyg_graphs(smiles, model)
    assert len(result) == 3
    assert result[0].__class__.__name__ == "Data"


# Test 15: test_get_neighbors
def test_get_neighbors_1():
    smiles = ["C", "CC", "CCC"]
    model = "GCN"
    pyg_graphs = create_pyg_graphs(smiles, model)
    for pyg_graph in pyg_graphs:
        neighbors = get_neighbors(pyg_graph)
        assert neighbors.__class__ == dict


# Test 16: test_get_neighbors
def test_get_neighbors_2():
    smiles = ["C", "CC", "CCC"]
    model = "Weave"
    pyg_graphs = create_pyg_graphs(smiles, model)
    for pyg_graph in pyg_graphs:
        neighbors = get_neighbors(pyg_graph)
        assert neighbors.__class__ == dict


# Test 17: test_TMD
def test_TMD_1():
    smiles = ["C", "CC", "CCC"]
    model = "GCN"
    pyg_graphs = create_pyg_graphs(smiles, model)
    distance = np.zeros((len(pyg_graphs), len(pyg_graphs)))
    for i, src_pyg_graph in enumerate(pyg_graphs):
        for j, tgt_pyg_graph in enumerate(pyg_graphs):
            tmd = TMD(src_pyg_graph, tgt_pyg_graph, w=1.0)
            distance[i, j] = tmd
    assert distance.shape == (3, 3)
    assert distance[0, 0] == 0
    assert distance[1, 1] == 0
    assert distance[2, 2] == 0
    assert distance[0, 1] > 0
    assert distance[0, 2] > 0
    assert distance[1, 2] > 0


# Test 18: test_TMD
def test_TMD_2():
    smiles = ["C", "CC", "CCC"]
    model = "AttentiveFP"
    pyg_graphs = create_pyg_graphs(
        smiles, model, atom_featurizer_type="attentivefp", bond_featurizer_type="attentivefp"
    )
    distance = np.zeros((len(pyg_graphs), len(pyg_graphs)))
    for i, src_pyg_graph in enumerate(pyg_graphs):
        for j, tgt_pyg_graph in enumerate(pyg_graphs):
            tmd = TMD(src_pyg_graph, tgt_pyg_graph, w=1.0)
            distance[i, j] = tmd
    assert distance.shape == (3, 3)
    assert distance[0, 0] == 0
    assert distance[1, 1] == 0
    assert distance[2, 2] == 0
    assert distance[0, 1] > 0
    assert distance[0, 2] > 0
    assert distance[1, 2] > 0


# Test 19: test_TMD
def test_TMD_3():
    smiles = ["C", "CC", "CCC"]
    model = "gin_supervised_contextpred"
    pyg_graphs = create_pyg_graphs(smiles, model)
    distance = np.zeros((len(pyg_graphs), len(pyg_graphs)))
    for i, src_pyg_graph in enumerate(pyg_graphs):
        for j, tgt_pyg_graph in enumerate(pyg_graphs):
            with pytest.raises(TypeError):
                tmd = TMD(src_pyg_graph, tgt_pyg_graph, w=1.0)
