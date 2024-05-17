import pytest
from alinemol.splitters.splits import *

@pytest.fixture
def molecules():
    return ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC"]


def Test_Scaffold():
    pass

def test_TargetProperty():
    pass


def test_MolecularWeight():
    smiles = molecules()
    mw = MolecularWeight(smiles)
    mw._before_sample()
    mw._sample()
    mw._after_sample()

