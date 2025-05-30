# Description: This file contains the configuration for the splitters.
# It is imported by splitting.py and used to configure the splitters.
# The configuration is set in the dictionary variables.
# The splitting.py script uses these configurations to split the molecules.

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

UMapSplitConfig: ConfigDict = {
    "n_clusters": 20,
    "umap_metric": "jaccard",
    "n_neighbors": 100,
    "min_dist": 0.1,
    "n_components": 2,
    "linkage": "ward",
}

HiSplitConfig: ConfigDict = {
    "similarity_threshold": 0.4,
    "train_min_frac": 0.70,
    "test_min_frac": 0.15,
    "coarsening_threshold": None,
    "verbose": True,
    "max_mip_gap": 0.1,
}
