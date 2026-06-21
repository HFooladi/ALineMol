"""Per-splitter default kwargs consumed by ``scripts/splitting.py``.

Each ``*Config`` dict is merged into the splitter's constructor kwargs, after
filtering through the splitter's signature. Adding a new splitter? Add a
config dict here (use an empty ``{}`` if no defaults beyond the standard
``n_splits``/``test_size``/``random_state``).
"""

from alinemol.utils.typing import ConfigDict

RandomSplitConfig: ConfigDict = {}

ScaffoldSplitConfig: ConfigDict = {"make_generic": False}

ScaffoldSplitGenericConfig: ConfigDict = {"make_generic": True}

MolecularWeightSplitConfig: ConfigDict = {"generalize_to_larger": True}

MolecularWeightReverseSplitConfig: ConfigDict = {"generalize_to_larger": False}

KMeansSplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

PerimeterSplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

MaxDissimilaritySplitConfig: ConfigDict = {"n_clusters": 10, "metric": "euclidean"}

MolecularLogPSplitConfig: ConfigDict = {"generalize_to_larger": True}

UMapSplitConfig: ConfigDict = {
    "n_clusters": 20,
    "umap_metric": "jaccard",
    "n_neighbors": 100,
    "min_dist": 0.1,
    "n_components": 2,
    "linkage": "ward",
}

BUTINASplitConfig: ConfigDict = {
    "n_clusters": 10,
    "cutoff": 0.65,
    "metric": "euclidean",
}

ScaffoldKMeansSplitConfig: ConfigDict = {
    "n_clusters": 10,
    "make_generic": False,
}

HiSplitConfig: ConfigDict = {
    "similarity_threshold": 0.4,
    "train_min_frac": 0.70,
    "test_min_frac": 0.15,
    "coarsening_threshold": None,
    "verbose": True,
    "max_mip_gap": 0.1,
}

# LoSplit now follows the unified ``split(X, y, groups)`` contract (values are
# passed via ``y``). It still cannot be driven by ``scripts/splitting.py`` unless
# that script supplies a continuous label/values column to pass as ``y`` — the Lo
# algorithm needs continuous activity values, not binary labels. This config is
# kept for downstream callers using the library API directly.
LoSplitConfig: ConfigDict = {
    "threshold": 0.4,
    "min_cluster_size": 5,
    "max_clusters": 50,
    "std_threshold": 0.60,
}

DataSAILSplitConfig: ConfigDict = {
    "technique": "C",
    "cluster_method": "ECFP",
    "delta": 0.1,
}
