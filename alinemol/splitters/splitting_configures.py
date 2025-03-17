# Description: This file contains the configuration for the splitters.
# It is imported by splitting.py and used to configure the splitters.
# The configuration is set in the dictionary variables.
# The splitting.py script uses these configurations to split the molecules.

from typing import Union
from alinemol.utils.typing import ConfigDict

RandomSplitConfig: ConfigDict = {}

ScaffoldSplitConfig: ConfigDict = {"make_generic": False}

ScaffoldSplitGenericConfig: ConfigDict = {"make_generic": True}

MolecularWeightSplitConfig: ConfigDict = {"generalize_to_larger": True}

MolecularWeightReverseSplitConfig: ConfigDict = {"generalize_to_larger": False}

KMeansSplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

PerimeterSplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

MaxDissimilaritySplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

MoodSplitConfig: ConfigDict = {}

MolecularLogPSplitConfig: ConfigDict = {"generalize_to_larger": True}


