"""
Custom type definitions for the ALineMol codebase.
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable, TypeVar, Protocol
import numpy as np
from dgl.data.utils import Subset


# Define a type for datasets that have a labels attribute
class LabeledDataset(Protocol):
    """Protocol for datasets that have a labels attribute."""

    labels: np.ndarray

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Any: ...


# Type for dataset subsets
DatasetSubset = Subset

# Type for a list of dataset subsets (train, val, test)
DatasetSplit = List[DatasetSubset]

# Type for a k-fold split result
KFoldSplit = List[Tuple[DatasetSubset, DatasetSubset]]

# Type for configuration dictionaries
ConfigDict = Dict[str, Any]

# Type for SMILES strings
SMILES = str
SMILESList = List[SMILES]

# Type for random state
RandomStateType = Optional[Union[int, np.random.RandomState]]

# Type for model metrics
MetricDict = Dict[str, float]

# Generic type variable for any dataset
T = TypeVar("T")

# Type for a function that can be applied to molecules
MolFunction = Callable[[Any], Any]
