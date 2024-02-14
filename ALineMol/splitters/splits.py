from dataclasses import dataclass


@dataclass
class RandomSplitter():
    """A random split."""
    name: str = 'random'


@dataclass
class ScaffoldSplitter():
    """A scaffold split."""
    include_chirality: bool = False
    name: str = 'scaffold'


@dataclass
class KMeansSplitter():
    """A k-means split."""
    n_clusters: int = 100
    n_init: int = 10
    name: str = 'kmeans'


@dataclass
class DBScanSplitter():
    """A DBScan split."""
    eps: float = 0.5
    metric: str = 'euclidean'
    name: str = 'dbscan'


@dataclass
class SphereExclusionSplitter():
    """A sphere exclusion split."""
    metrics: str
    distance_cutoff: float
    name: str = 'sphere_exclusion'


@dataclass
class OptiSimSplitter():
    """An OptiSim split."""
    n_clusters: int = 10
    max_subsample_size: int = 1000
    distance_cutoff: float = 0.1
    name: str = 'optisim'