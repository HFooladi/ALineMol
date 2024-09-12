# We will test all the functions available in alinemol/utils/graph_utils.py

from alinemol.utils.graph_utils import choose_featurizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer

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

