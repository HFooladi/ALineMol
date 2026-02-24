"""
ALineMol Splitters Module

Provides molecular dataset splitting strategies for ML evaluation with
a focus on out-of-distribution (OOD) performance analysis.

Main API:
    get_splitter(name, **kwargs) - Factory function to create splitters by name
    list_splitters() - List all available splitters
    SplitAnalyzer - Analyze and compare split quality

Available Splitters:
    Structure-based: scaffold, scaffold_generic
    Clustering-based: kmeans, perimeter, max_dissimilarity, umap, butina, scaffold_kmeans
    Property-based: molecular_weight, molecular_logp
    Similarity-based: hi, lo
    Information leakage: datasail
    Baseline: random

Example - Creating splits:
    >>> from alinemol.splitters import get_splitter, list_splitters
    >>>
    >>> # List available splitters
    >>> print(list_splitters().keys())
    >>>
    >>> # Create a scaffold splitter
    >>> splitter = get_splitter("scaffold", make_generic=True, n_splits=5)
    >>> for train_idx, test_idx in splitter.split(smiles_list):
    ...     train = [smiles_list[i] for i in train_idx]
    ...     test = [smiles_list[i] for i in test_idx]
    >>>
    >>> # Direct class import also works
    >>> from alinemol.splitters import ScaffoldSplit
    >>> splitter = ScaffoldSplit(make_generic=True)

Example - Analyzing split quality:
    >>> from alinemol.splitters import SplitAnalyzer, get_splitter
    >>>
    >>> # Create analyzer
    >>> analyzer = SplitAnalyzer(smiles_list)
    >>>
    >>> # Analyze a single split
    >>> splitter = get_splitter("scaffold")
    >>> train_idx, test_idx = next(splitter.split(smiles_list))
    >>> report = analyzer.analyze_split(train_idx, test_idx, "scaffold")
    >>>
    >>> print(f"Mean similarity: {report.similarity_metrics.mean_sim:.3f}")
    >>> print(f"Scaffold overlap: {report.scaffold_metrics.scaffold_overlap_percentage:.1f}%")
    >>>
    >>> # Compare multiple splitters
    >>> comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
    >>> print(comparison)  # DataFrame with aggregated metrics
"""

from typing import List

# Base class
from alinemol.splitters.base import BaseMolecularSplitter

# Factory and registry
from alinemol.splitters.factory import (
    get_splitter,
    list_splitters,
    get_splitter_names,
    get_splitter_aliases,
    register_splitter,
    is_splitter_registered,
)

# Wrapper classes for splito (these auto-register via decorator)
from alinemol.splitters.wrappers import (
    KMeansSplit,
    ScaffoldSplit,
    ScaffoldGenericSplit,
    MolecularWeightSplit,
    MolecularWeightReverseSplit,
    MaxDissimilaritySplit,
    PerimeterSplit,
)

# Native ALineMol splitters (these auto-register via decorator)
from alinemol.splitters.splits import (
    MolecularLogPSplit,
    RandomSplit,
    StratifiedRandomSplit,
    stratified_split_dataset,
)

# UMAP-based splitter
from alinemol.splitters.umap_split import UMAPSplit, get_umap_clusters

# Butina clustering-based splitter
from alinemol.splitters.butina_split import BUTINASplit, get_butina_clusters

# Scaffold K-Means splitter
from alinemol.splitters.scaffold_kmeans_split import ScaffoldKMeansSplit, get_scaffold_kmeans_clusters

# Lo-Hi splitters for similarity-based splitting
from alinemol.splitters.lohi import LoSplit, HiSplit

# DataSAIL integration
from alinemol.splitters.datasail import DataSAILSplit, DataSAILGroupSplit

# Split Quality Analyzer
from alinemol.splitters.analyzer import (
    SplitAnalyzer,
    SplitQualityReport,
    SimilarityMetrics,
    ScaffoldMetrics,
    PropertyDistribution,
    SizeMetrics,
)


__all__: List[str] = [
    # Base class
    "BaseMolecularSplitter",
    # Factory and registry
    "get_splitter",
    "list_splitters",
    "get_splitter_names",
    "get_splitter_aliases",
    "register_splitter",
    "is_splitter_registered",
    # Wrapper classes (splito)
    "KMeansSplit",
    "ScaffoldSplit",
    "ScaffoldGenericSplit",
    "MolecularWeightSplit",
    "MolecularWeightReverseSplit",
    "MaxDissimilaritySplit",
    "PerimeterSplit",
    # Native splitters
    "MolecularLogPSplit",
    "RandomSplit",
    "StratifiedRandomSplit",
    "stratified_split_dataset",
    # UMAP
    "UMAPSplit",
    "get_umap_clusters",
    # Butina
    "BUTINASplit",
    "get_butina_clusters",
    # Scaffold K-Means
    "ScaffoldKMeansSplit",
    "get_scaffold_kmeans_clusters",
    # Lo-Hi
    "LoSplit",
    "HiSplit",
    # DataSAIL
    "DataSAILSplit",
    "DataSAILGroupSplit",
    # Split Quality Analyzer
    "SplitAnalyzer",
    "SplitQualityReport",
    "SimilarityMetrics",
    "ScaffoldMetrics",
    "PropertyDistribution",
    "SizeMetrics",
]
