# Description: This file contains the configuration for the splitters.
# It is imported by splitting.py and used to configure the splitters.
# The configuration is set in the dictionary variables.
# The splitting.py script uses these configurations to split the molecules.

RandomSplitConfig = {}

ScaffoldSplitConfig = {"make_generic": False}

MolecularWeightSplitConfig = {"generalize_to_larger": True}

MolecularWeightReverseSplitConfig = {"generalize_to_larger": False}

KMeansSplitConfig = {"n_clusters": 10, "metric": "euclidean"}

PerimeterSplitConfig = {"n_clusters": 25, "metric": "euclidean"}

MaxDissimilaritySplitConfig = {"n_clusters": 10, "metric": "euclidean"}

MoodSplitConfig = {}

MolecularLogPSplitConfig = {"generalize_to_larger": True}
