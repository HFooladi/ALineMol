from dataclasses import dataclass


@dataclass
class ScaffoldSplitConfig:
    make_generic: bool = False


@dataclass
class MolecularWeightSplitConfig:
    generalize_to_larger: bool = False


@dataclass
class KMeansSplitConfig:
    n_clusters: int = 10
