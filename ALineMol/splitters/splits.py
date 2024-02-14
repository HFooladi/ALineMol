from dataclasses import dataclass


@dataclass
class RandomSplit():
    """A random split."""
    shuffle: bool = True


@dataclass
class ScaffoldSplit():
    """A scaffold split."""
    include_chirality: bool = False


@dataclass
class KMeansSplit():
    """A k-means split."""
    n_clusters: int = 100
    n_init: int = 10


@dataclass
class DBScanSplit():
    """A DBScan split."""
    eps: float = 0.5
    metric: str = 'euclidean'


@dataclass
class SphereExclusionSplit():
    """A sphere exclusion split."""
    metrics: str = 'euclidean'
    distance_cutoff: float = 0.5


@dataclass
class OptiSimSplit():
    """An OptiSim split."""
    n_clusters: int = 10
    max_subsample_size: int = 1000
    distance_cutoff: float = 0.1
