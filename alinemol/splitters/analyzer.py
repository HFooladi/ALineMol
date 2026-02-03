"""
Split Quality Analyzer Module.

This module provides tools for evaluating and comparing the quality of
train/test splits in molecular machine learning. It computes comprehensive
metrics including similarity distributions, scaffold overlap, and property
distributions to help users understand the characteristics of their splits.

Example:
    >>> from alinemol.splitters import get_splitter, SplitAnalyzer
    >>>
    >>> # Create analyzer
    >>> analyzer = SplitAnalyzer(smiles)
    >>>
    >>> # Analyze a single split
    >>> splitter = get_splitter("scaffold")
    >>> train_idx, test_idx = next(splitter.split(smiles))
    >>> report = analyzer.analyze_split(train_idx, test_idx, splitter_name="scaffold")
    >>>
    >>> print(f"Mean train-test similarity: {report.similarity_metrics.mean_sim:.3f}")
    >>> print(f"Scaffold overlap: {report.scaffold_metrics.scaffold_overlap_percentage:.1f}%")
    >>>
    >>> # Compare multiple splitters
    >>> comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
    >>> print(comparison)  # DataFrame with aggregated metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Type aliases
ArrayLike = Union[List[float], np.ndarray]


@dataclass(frozen=True)
class SimilarityMetrics:
    """
    Metrics describing the similarity distribution between train and test sets.

    These metrics are computed by finding, for each test molecule, the maximum
    Tanimoto similarity to any training molecule. This helps validate OOD assumptions.

    Attributes:
        min_sim: Minimum similarity (most dissimilar test molecule).
        max_sim: Maximum similarity (most similar test molecule).
        mean_sim: Mean similarity across all test molecules.
        median_sim: Median similarity.
        std_sim: Standard deviation of similarities.
        percentile_5: 5th percentile (most dissimilar 5%).
        percentile_25: 25th percentile (first quartile).
        percentile_75: 75th percentile (third quartile).
        percentile_95: 95th percentile (most similar 5%).
        n_test_samples: Number of test samples analyzed.
    """

    min_sim: float
    max_sim: float
    mean_sim: float
    median_sim: float
    std_sim: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    n_test_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "min_sim": self.min_sim,
            "max_sim": self.max_sim,
            "mean_sim": self.mean_sim,
            "median_sim": self.median_sim,
            "std_sim": self.std_sim,
            "percentile_5": self.percentile_5,
            "percentile_25": self.percentile_25,
            "percentile_75": self.percentile_75,
            "percentile_95": self.percentile_95,
            "n_test_samples": self.n_test_samples,
        }


@dataclass(frozen=True)
class ScaffoldMetrics:
    """
    Metrics describing scaffold distribution and overlap between train and test sets.

    Scaffolds are computed using the Bemis-Murcko framework, which extracts the
    core ring systems and linkers from molecules.

    Attributes:
        train_unique_scaffolds: Number of unique scaffolds in training set.
        test_unique_scaffolds: Number of unique scaffolds in test set.
        shared_scaffolds: Number of scaffolds present in both sets.
        scaffold_overlap_percentage: Percentage of test scaffolds also in train.
        train_scaffold_coverage: Percentage of train molecules with valid scaffolds.
        test_scaffold_coverage: Percentage of test molecules with valid scaffolds.
    """

    train_unique_scaffolds: int
    test_unique_scaffolds: int
    shared_scaffolds: int
    scaffold_overlap_percentage: float
    train_scaffold_coverage: float
    test_scaffold_coverage: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary for serialization."""
        return {
            "train_unique_scaffolds": self.train_unique_scaffolds,
            "test_unique_scaffolds": self.test_unique_scaffolds,
            "shared_scaffolds": self.shared_scaffolds,
            "scaffold_overlap_percentage": self.scaffold_overlap_percentage,
            "train_scaffold_coverage": self.train_scaffold_coverage,
            "test_scaffold_coverage": self.test_scaffold_coverage,
        }


@dataclass(frozen=True)
class PropertyDistribution:
    """
    Statistics for a single molecular property distribution comparison.

    Compares the distribution of a property between train and test sets
    using statistical tests and descriptive statistics.

    Attributes:
        property_name: Name of the property (e.g., "MW", "LogP").
        train_mean: Mean value in training set.
        train_std: Standard deviation in training set.
        train_median: Median value in training set.
        test_mean: Mean value in test set.
        test_std: Standard deviation in test set.
        test_median: Median value in test set.
        ks_statistic: Kolmogorov-Smirnov test statistic.
        ks_pvalue: P-value from KS test (low = distributions differ).
        mean_diff: Absolute difference in means.
        median_diff: Absolute difference in medians.
    """

    property_name: str
    train_mean: float
    train_std: float
    train_median: float
    test_mean: float
    test_std: float
    test_median: float
    ks_statistic: float
    ks_pvalue: float
    mean_diff: float
    median_diff: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert to dictionary for serialization."""
        return {
            "property_name": self.property_name,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "train_median": self.train_median,
            "test_mean": self.test_mean,
            "test_std": self.test_std,
            "test_median": self.test_median,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "mean_diff": self.mean_diff,
            "median_diff": self.median_diff,
        }


@dataclass(frozen=True)
class SizeMetrics:
    """
    Metrics describing the sizes and balance of train/test sets.

    Attributes:
        total_samples: Total number of samples.
        train_size: Number of training samples.
        test_size: Number of test samples.
        train_ratio: Fraction of samples in training set.
        test_ratio: Fraction of samples in test set.
        train_positive_ratio: Ratio of positive labels in train (if labels provided).
        test_positive_ratio: Ratio of positive labels in test (if labels provided).
        label_balance_diff: Difference in positive ratios between train and test.
    """

    total_samples: int
    train_size: int
    test_size: int
    train_ratio: float
    test_ratio: float
    train_positive_ratio: Optional[float] = None
    test_positive_ratio: Optional[float] = None
    label_balance_diff: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        """Convert to dictionary for serialization."""
        return {
            "total_samples": self.total_samples,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "train_positive_ratio": self.train_positive_ratio,
            "test_positive_ratio": self.test_positive_ratio,
            "label_balance_diff": self.label_balance_diff,
        }


@dataclass
class SplitQualityReport:
    """
    Comprehensive quality report for a single train/test split.

    Aggregates all metrics (similarity, scaffold, property, size) for a split
    and provides serialization methods for export.

    Attributes:
        splitter_name: Name of the splitting strategy used.
        split_index: Index of this split (for cross-validation).
        size_metrics: Size and balance metrics.
        similarity_metrics: Similarity distribution metrics (optional).
        scaffold_metrics: Scaffold overlap metrics (optional).
        property_distributions: List of property distribution comparisons.

    Example:
        >>> report = analyzer.analyze_split(train_idx, test_idx, "scaffold")
        >>> print(report.similarity_metrics.mean_sim)
        >>> report_dict = report.to_dict()
        >>> row = report.to_dataframe_row()  # For comparison tables
    """

    splitter_name: str
    split_index: int
    size_metrics: SizeMetrics
    similarity_metrics: Optional[SimilarityMetrics] = None
    scaffold_metrics: Optional[ScaffoldMetrics] = None
    property_distributions: List[PropertyDistribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire report to a nested dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: Dict[str, Any] = {
            "splitter_name": self.splitter_name,
            "split_index": self.split_index,
            "size_metrics": self.size_metrics.to_dict(),
        }

        if self.similarity_metrics is not None:
            result["similarity_metrics"] = self.similarity_metrics.to_dict()

        if self.scaffold_metrics is not None:
            result["scaffold_metrics"] = self.scaffold_metrics.to_dict()

        if self.property_distributions:
            result["property_distributions"] = [p.to_dict() for p in self.property_distributions]

        return result

    def to_dataframe_row(self) -> Dict[str, Any]:
        """
        Convert to a flat dictionary suitable for DataFrame row.

        Returns:
            Flat dictionary with prefixed keys for easy tabular comparison.
        """
        row: Dict[str, Any] = {
            "splitter_name": self.splitter_name,
            "split_index": self.split_index,
        }

        # Add size metrics
        for key, value in self.size_metrics.to_dict().items():
            row[f"size_{key}"] = value

        # Add similarity metrics
        if self.similarity_metrics is not None:
            for key, value in self.similarity_metrics.to_dict().items():
                row[f"sim_{key}"] = value

        # Add scaffold metrics
        if self.scaffold_metrics is not None:
            for key, value in self.scaffold_metrics.to_dict().items():
                row[f"scaffold_{key}"] = value

        # Add property distributions (flattened)
        for prop_dist in self.property_distributions:
            prop_name = prop_dist.property_name.lower().replace(" ", "_")
            row[f"prop_{prop_name}_ks_stat"] = prop_dist.ks_statistic
            row[f"prop_{prop_name}_ks_pvalue"] = prop_dist.ks_pvalue
            row[f"prop_{prop_name}_mean_diff"] = prop_dist.mean_diff

        return row


class SplitAnalyzer:
    """
    Analyzer for evaluating and comparing train/test split quality.

    This class provides methods to compute comprehensive quality metrics for
    molecular dataset splits, including similarity distributions, scaffold
    overlap, and property distributions.

    Args:
        smiles: List of SMILES strings for the full dataset.
        fingerprint_type: Type of fingerprint to use for similarity computation.
            Options: "ecfp", "fcfp", "maccs", etc. Default: "ecfp".
        fingerprint_radius: Radius for circular fingerprints. Default: 2.
        fingerprint_nbits: Number of bits for fingerprints. Default: 2048.
        compute_properties: List of properties to compute. Default includes
            MW, LogP, TPSA, HBD, HBA.
        n_jobs: Number of parallel jobs. Default: 1.

    Attributes:
        smiles: The input SMILES strings.
        n_molecules: Number of molecules in the dataset.

    Example:
        >>> analyzer = SplitAnalyzer(smiles)
        >>>
        >>> # Analyze a single split
        >>> report = analyzer.analyze_split(train_idx, test_idx, "scaffold")
        >>> print(f"Mean similarity: {report.similarity_metrics.mean_sim:.3f}")
        >>>
        >>> # Compare multiple splitters
        >>> comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
        >>> print(comparison)
    """

    # Default properties to compute
    DEFAULT_PROPERTIES = ["MW", "LogP", "TPSA", "HBD", "HBA"]

    def __init__(
        self,
        smiles: List[str],
        fingerprint_type: str = "ecfp",
        fingerprint_radius: int = 2,
        fingerprint_nbits: int = 2048,
        compute_properties: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        self.smiles = list(smiles)
        self.n_molecules = len(smiles)
        self.fingerprint_type = fingerprint_type
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_nbits = fingerprint_nbits
        self.compute_properties = compute_properties or self.DEFAULT_PROPERTIES
        self.n_jobs = n_jobs

        # Lazy-loaded cached data
        self._fingerprints: Optional[np.ndarray] = None
        self._scaffolds: Optional[List[Optional[str]]] = None
        self._properties: Optional[pd.DataFrame] = None
        self._mols: Optional[List] = None

    @property
    def fingerprints(self) -> np.ndarray:
        """Lazily compute and cache molecular fingerprints."""
        if self._fingerprints is None:
            self._fingerprints = self._compute_fingerprints()
        return self._fingerprints

    @property
    def scaffolds(self) -> List[Optional[str]]:
        """Lazily compute and cache Bemis-Murcko scaffolds."""
        if self._scaffolds is None:
            self._scaffolds = self._compute_scaffolds()
        return self._scaffolds

    @property
    def properties(self) -> pd.DataFrame:
        """Lazily compute and cache molecular properties."""
        if self._properties is None:
            self._properties = self._compute_properties()
        return self._properties

    def _get_mols(self) -> List:
        """Get RDKit molecule objects, caching for reuse."""
        if self._mols is None:
            try:
                from rdkit import Chem
            except ImportError:
                raise ImportError("RDKit is required for molecular analysis. Install with: pip install rdkit")

            self._mols = []
            for smi in self.smiles:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    self._mols.append(mol)
                except Exception:
                    self._mols.append(None)
        return self._mols

    def _compute_fingerprints(self) -> np.ndarray:
        """Compute molecular fingerprints for all molecules."""
        try:
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit is required for fingerprint computation. Install with: pip install rdkit")

        fps = []
        mols = self._get_mols()

        for mol in mols:
            if mol is not None:
                if self.fingerprint_type.lower() in ["ecfp", "morgan"]:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=self.fingerprint_radius, nBits=self.fingerprint_nbits
                    )
                elif self.fingerprint_type.lower() == "fcfp":
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=self.fingerprint_radius, nBits=self.fingerprint_nbits, useFeatures=True
                    )
                elif self.fingerprint_type.lower() == "maccs":
                    from rdkit.Chem import MACCSkeys

                    fp = MACCSkeys.GenMACCSKeys(mol)
                else:
                    # Default to ECFP
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=self.fingerprint_radius, nBits=self.fingerprint_nbits
                    )
                fps.append(np.array(fp))
            else:
                # Invalid molecule - use zero vector
                fps.append(np.zeros(self.fingerprint_nbits))

        return np.array(fps)

    def _compute_scaffolds(self) -> List[Optional[str]]:
        """Compute Bemis-Murcko scaffolds for all molecules."""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError:
            raise ImportError("RDKit is required for scaffold computation. Install with: pip install rdkit")

        scaffolds = []
        mols = self._get_mols()

        for mol in mols:
            if mol is not None:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    from rdkit import Chem

                    scaffold_smi = Chem.MolToSmiles(scaffold)
                    scaffolds.append(scaffold_smi)
                except Exception:
                    scaffolds.append(None)
            else:
                scaffolds.append(None)

        return scaffolds

    def _compute_properties(self) -> pd.DataFrame:
        """Compute molecular properties for all molecules."""
        try:
            from rdkit.Chem import Descriptors
        except ImportError:
            raise ImportError("RDKit is required for property computation. Install with: pip install rdkit")

        mols = self._get_mols()
        data: Dict[str, List[Optional[float]]] = {prop: [] for prop in self.compute_properties}

        property_funcs = {
            "MW": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP,
            "TPSA": Descriptors.TPSA,
            "HBD": Descriptors.NumHDonors,
            "HBA": Descriptors.NumHAcceptors,
            "RotatableBonds": Descriptors.NumRotatableBonds,
            "Rings": Descriptors.RingCount,
            "AromaticRings": Descriptors.NumAromaticRings,
            "FractionCSP3": Descriptors.FractionCSP3,
            "HeavyAtoms": Descriptors.HeavyAtomCount,
        }

        for mol in mols:
            for prop in self.compute_properties:
                if prop in property_funcs and mol is not None:
                    try:
                        value = property_funcs[prop](mol)
                        data[prop].append(value)
                    except Exception:
                        data[prop].append(None)
                else:
                    data[prop].append(None)

        return pd.DataFrame(data)

    def _compute_similarity_metrics(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Optional[SimilarityMetrics]:
        """
        Compute similarity metrics between train and test sets.

        For each test molecule, finds the maximum Tanimoto similarity to
        any training molecule.
        """
        if len(test_idx) == 0 or len(train_idx) == 0:
            return None

        train_fps = self.fingerprints[train_idx]
        test_fps = self.fingerprints[test_idx]

        # Compute max similarity for each test molecule to training set
        max_similarities = []

        for test_fp in test_fps:
            # Tanimoto similarity
            intersections = np.sum(np.logical_and(train_fps, test_fp), axis=1)
            unions = np.sum(np.logical_or(train_fps, test_fp), axis=1)

            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                similarities = np.where(unions > 0, intersections / unions, 0.0)

            max_sim = np.max(similarities) if len(similarities) > 0 else 0.0
            max_similarities.append(max_sim)

        max_similarities = np.array(max_similarities)

        return SimilarityMetrics(
            min_sim=float(np.min(max_similarities)),
            max_sim=float(np.max(max_similarities)),
            mean_sim=float(np.mean(max_similarities)),
            median_sim=float(np.median(max_similarities)),
            std_sim=float(np.std(max_similarities)),
            percentile_5=float(np.percentile(max_similarities, 5)),
            percentile_25=float(np.percentile(max_similarities, 25)),
            percentile_75=float(np.percentile(max_similarities, 75)),
            percentile_95=float(np.percentile(max_similarities, 95)),
            n_test_samples=len(test_idx),
        )

    def _compute_scaffold_metrics(self, train_idx: np.ndarray, test_idx: np.ndarray) -> Optional[ScaffoldMetrics]:
        """Compute scaffold-related metrics for the split."""
        if len(test_idx) == 0 or len(train_idx) == 0:
            return None

        train_scaffolds = [self.scaffolds[i] for i in train_idx]
        test_scaffolds = [self.scaffolds[i] for i in test_idx]

        # Filter out None scaffolds
        train_valid = [s for s in train_scaffolds if s is not None]
        test_valid = [s for s in test_scaffolds if s is not None]

        train_scaffold_set = set(train_valid)
        test_scaffold_set = set(test_valid)

        shared = train_scaffold_set & test_scaffold_set

        # Overlap percentage: what fraction of test scaffolds are in train
        overlap_pct = 100.0 * len(shared) / len(test_scaffold_set) if test_scaffold_set else 0.0

        return ScaffoldMetrics(
            train_unique_scaffolds=len(train_scaffold_set),
            test_unique_scaffolds=len(test_scaffold_set),
            shared_scaffolds=len(shared),
            scaffold_overlap_percentage=overlap_pct,
            train_scaffold_coverage=100.0 * len(train_valid) / len(train_idx) if train_idx.size > 0 else 0.0,
            test_scaffold_coverage=100.0 * len(test_valid) / len(test_idx) if test_idx.size > 0 else 0.0,
        )

    def _compute_property_distributions(
        self, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> List[PropertyDistribution]:
        """Compute property distribution comparisons."""
        if len(test_idx) == 0 or len(train_idx) == 0:
            return []

        props = self.properties
        distributions = []

        for prop_name in self.compute_properties:
            if prop_name not in props.columns:
                continue

            train_values = props.iloc[train_idx][prop_name].dropna().values
            test_values = props.iloc[test_idx][prop_name].dropna().values

            if len(train_values) < 2 or len(test_values) < 2:
                continue

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(train_values, test_values)

            distributions.append(
                PropertyDistribution(
                    property_name=prop_name,
                    train_mean=float(np.mean(train_values)),
                    train_std=float(np.std(train_values)),
                    train_median=float(np.median(train_values)),
                    test_mean=float(np.mean(test_values)),
                    test_std=float(np.std(test_values)),
                    test_median=float(np.median(test_values)),
                    ks_statistic=float(ks_stat),
                    ks_pvalue=float(ks_pvalue),
                    mean_diff=float(abs(np.mean(train_values) - np.mean(test_values))),
                    median_diff=float(abs(np.median(train_values) - np.median(test_values))),
                )
            )

        return distributions

    def _compute_size_metrics(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        labels: Optional[ArrayLike] = None,
    ) -> SizeMetrics:
        """Compute size and balance metrics."""
        total = len(train_idx) + len(test_idx)
        train_ratio = len(train_idx) / total if total > 0 else 0.0
        test_ratio = len(test_idx) / total if total > 0 else 0.0

        train_pos_ratio = None
        test_pos_ratio = None
        balance_diff = None

        if labels is not None:
            labels = np.array(labels)
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            if len(train_labels) > 0:
                train_pos_ratio = float(np.mean(train_labels))
            if len(test_labels) > 0:
                test_pos_ratio = float(np.mean(test_labels))
            if train_pos_ratio is not None and test_pos_ratio is not None:
                balance_diff = abs(train_pos_ratio - test_pos_ratio)

        return SizeMetrics(
            total_samples=self.n_molecules,
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            train_positive_ratio=train_pos_ratio,
            test_positive_ratio=test_pos_ratio,
            label_balance_diff=balance_diff,
        )

    def analyze_split(
        self,
        train_idx: Union[List[int], np.ndarray],
        test_idx: Union[List[int], np.ndarray],
        splitter_name: str = "unknown",
        split_index: int = 0,
        labels: Optional[ArrayLike] = None,
        compute_similarity: bool = True,
        compute_scaffolds: bool = True,
        compute_properties: bool = True,
    ) -> SplitQualityReport:
        """
        Analyze a single train/test split and compute quality metrics.

        Args:
            train_idx: Indices of training samples.
            test_idx: Indices of test samples.
            splitter_name: Name of the splitting strategy used.
            split_index: Index of this split (for cross-validation).
            labels: Optional labels for computing balance metrics.
            compute_similarity: Whether to compute similarity metrics.
            compute_scaffolds: Whether to compute scaffold metrics.
            compute_properties: Whether to compute property distributions.

        Returns:
            SplitQualityReport containing all computed metrics.

        Example:
            >>> report = analyzer.analyze_split(train_idx, test_idx, "scaffold")
            >>> print(f"Mean similarity: {report.similarity_metrics.mean_sim:.3f}")
        """
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        # Always compute size metrics
        size_metrics = self._compute_size_metrics(train_idx, test_idx, labels)

        # Optionally compute other metrics
        similarity_metrics = None
        if compute_similarity:
            similarity_metrics = self._compute_similarity_metrics(train_idx, test_idx)

        scaffold_metrics = None
        if compute_scaffolds:
            scaffold_metrics = self._compute_scaffold_metrics(train_idx, test_idx)

        property_distributions = []
        if compute_properties:
            property_distributions = self._compute_property_distributions(train_idx, test_idx)

        return SplitQualityReport(
            splitter_name=splitter_name,
            split_index=split_index,
            size_metrics=size_metrics,
            similarity_metrics=similarity_metrics,
            scaffold_metrics=scaffold_metrics,
            property_distributions=property_distributions,
        )

    def analyze_splitter(
        self,
        splitter_name: str,
        n_splits: int = 5,
        test_size: float = 0.2,
        labels: Optional[ArrayLike] = None,
        **splitter_kwargs: Any,
    ) -> List[SplitQualityReport]:
        """
        Analyze multiple splits from a single splitting strategy.

        Args:
            splitter_name: Name of the splitter to use (e.g., "scaffold", "kmeans").
            n_splits: Number of splits to analyze.
            test_size: Fraction of data for test set.
            labels: Optional labels for balance metrics.
            **splitter_kwargs: Additional arguments passed to the splitter.

        Returns:
            List of SplitQualityReport, one for each split.

        Example:
            >>> reports = analyzer.analyze_splitter("scaffold", n_splits=5)
            >>> mean_sim = np.mean([r.similarity_metrics.mean_sim for r in reports])
        """
        from alinemol.splitters import get_splitter

        splitter = get_splitter(splitter_name, n_splits=n_splits, test_size=test_size, **splitter_kwargs)

        reports = []
        for idx, (train_idx, test_idx) in enumerate(splitter.split(self.smiles)):
            report = self.analyze_split(
                train_idx=train_idx,
                test_idx=test_idx,
                splitter_name=splitter_name,
                split_index=idx,
                labels=labels,
            )
            reports.append(report)

        return reports

    def compare_splitters(
        self,
        splitter_names: List[str],
        n_splits: int = 5,
        test_size: float = 0.2,
        labels: Optional[ArrayLike] = None,
        aggregate: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple splitting strategies on the same dataset.

        Args:
            splitter_names: List of splitter names to compare.
            n_splits: Number of splits per splitter.
            test_size: Fraction of data for test set.
            labels: Optional labels for balance metrics.
            aggregate: If True, aggregate metrics across splits (mean ± std).

        Returns:
            DataFrame with comparison metrics. If aggregate=True, shows mean
            values with standard deviations. Otherwise, shows all splits.

        Example:
            >>> comparison = analyzer.compare_splitters(["scaffold", "kmeans", "random"])
            >>> print(comparison[["splitter_name", "sim_mean_sim", "scaffold_overlap_percentage"]])
        """
        all_rows = []

        for splitter_name in splitter_names:
            try:
                reports = self.analyze_splitter(splitter_name, n_splits=n_splits, test_size=test_size, labels=labels)
                for report in reports:
                    all_rows.append(report.to_dataframe_row())
            except Exception as e:
                # Log error but continue with other splitters
                print(f"Warning: Failed to analyze splitter '{splitter_name}': {e}")

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        if aggregate and len(df) > 0:
            # Aggregate by splitter_name
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: ["mean", "std"] for col in numeric_cols if col != "split_index"}

            aggregated = df.groupby("splitter_name").agg(agg_dict)
            # Flatten column names
            aggregated.columns = [f"{col}_{stat}" for col, stat in aggregated.columns]
            aggregated = aggregated.reset_index()

            return aggregated

        return df

    def get_summary_stats(self, reports: List[SplitQualityReport]) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics from a list of reports.

        Args:
            reports: List of SplitQualityReport objects.

        Returns:
            Dictionary with summary statistics for key metrics.
        """
        if not reports:
            return {}

        summary: Dict[str, Dict[str, float]] = {}

        # Similarity metrics
        sim_means = [r.similarity_metrics.mean_sim for r in reports if r.similarity_metrics]
        if sim_means:
            summary["similarity"] = {
                "mean": float(np.mean(sim_means)),
                "std": float(np.std(sim_means)),
                "min": float(np.min(sim_means)),
                "max": float(np.max(sim_means)),
            }

        # Scaffold overlap
        overlaps = [r.scaffold_metrics.scaffold_overlap_percentage for r in reports if r.scaffold_metrics]
        if overlaps:
            summary["scaffold_overlap"] = {
                "mean": float(np.mean(overlaps)),
                "std": float(np.std(overlaps)),
                "min": float(np.min(overlaps)),
                "max": float(np.max(overlaps)),
            }

        return summary
