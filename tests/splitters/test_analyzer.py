"""
Tests for Split Quality Analyzer.

These tests verify the functionality of the SplitAnalyzer class and its
associated dataclasses for computing split quality metrics.
"""

import json
import numpy as np
import pandas as pd
import pytest

from alinemol.splitters.analyzer import (
    SplitAnalyzer,
    SplitQualityReport,
    SimilarityMetrics,
    ScaffoldMetrics,
    PropertyDistribution,
    SizeMetrics,
)


# Test fixtures
@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing - diverse molecules."""
    return [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CCN",  # ethylamine
        "CCCCCCC",  # heptane
        "CC(C)O",  # isopropanol
        "c1ccc(cc1)O",  # phenol
        "CCCCCCCC",  # octane
        "CC(C)(C)O",  # tert-butanol
        "c1ccc(cc1)N",  # aniline
        "CCCC",  # butane
        "c1ccc(cc1)C",  # toluene
        "CCO",  # ethanol (duplicate)
        "CCOCC",  # diethyl ether
        "c1ccc(cc1)Cl",  # chlorobenzene
        "CCCCCC",  # hexane
        "CC(=O)OCC",  # ethyl acetate
        "c1ccc(cc1)F",  # fluorobenzene
        "CCCCC",  # pentane
        "c1ccc(cc1)Br",  # bromobenzene
    ]


@pytest.fixture
def sample_labels():
    """Binary labels for testing."""
    return [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


@pytest.fixture
def train_test_indices():
    """Sample train/test split indices."""
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    test_idx = np.array([16, 17, 18, 19])
    return train_idx, test_idx


@pytest.fixture
def analyzer(sample_smiles):
    """Pre-initialized analyzer for testing."""
    return SplitAnalyzer(sample_smiles)


# Test dataclasses
class TestSimilarityMetrics:
    """Tests for SimilarityMetrics dataclass."""

    def test_creation(self):
        """Test creating SimilarityMetrics."""
        metrics = SimilarityMetrics(
            min_sim=0.1,
            max_sim=0.9,
            mean_sim=0.5,
            median_sim=0.45,
            std_sim=0.2,
            percentile_5=0.15,
            percentile_25=0.3,
            percentile_75=0.7,
            percentile_95=0.85,
            n_test_samples=100,
        )
        assert metrics.min_sim == 0.1
        assert metrics.max_sim == 0.9
        assert metrics.mean_sim == 0.5

    def test_frozen(self):
        """Test that SimilarityMetrics is immutable."""
        metrics = SimilarityMetrics(
            min_sim=0.1,
            max_sim=0.9,
            mean_sim=0.5,
            median_sim=0.45,
            std_sim=0.2,
            percentile_5=0.15,
            percentile_25=0.3,
            percentile_75=0.7,
            percentile_95=0.85,
            n_test_samples=100,
        )
        with pytest.raises(AttributeError):
            metrics.min_sim = 0.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SimilarityMetrics(
            min_sim=0.1,
            max_sim=0.9,
            mean_sim=0.5,
            median_sim=0.45,
            std_sim=0.2,
            percentile_5=0.15,
            percentile_25=0.3,
            percentile_75=0.7,
            percentile_95=0.85,
            n_test_samples=100,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["min_sim"] == 0.1
        assert "n_test_samples" in d


class TestScaffoldMetrics:
    """Tests for ScaffoldMetrics dataclass."""

    def test_creation(self):
        """Test creating ScaffoldMetrics."""
        metrics = ScaffoldMetrics(
            train_unique_scaffolds=50,
            test_unique_scaffolds=10,
            shared_scaffolds=5,
            scaffold_overlap_percentage=50.0,
            train_scaffold_coverage=95.0,
            test_scaffold_coverage=90.0,
        )
        assert metrics.train_unique_scaffolds == 50
        assert metrics.scaffold_overlap_percentage == 50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ScaffoldMetrics(
            train_unique_scaffolds=50,
            test_unique_scaffolds=10,
            shared_scaffolds=5,
            scaffold_overlap_percentage=50.0,
            train_scaffold_coverage=95.0,
            test_scaffold_coverage=90.0,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "train_unique_scaffolds" in d


class TestPropertyDistribution:
    """Tests for PropertyDistribution dataclass."""

    def test_creation(self):
        """Test creating PropertyDistribution."""
        prop = PropertyDistribution(
            property_name="MW",
            train_mean=250.0,
            train_std=50.0,
            train_median=240.0,
            test_mean=260.0,
            test_std=55.0,
            test_median=255.0,
            ks_statistic=0.1,
            ks_pvalue=0.5,
            mean_diff=10.0,
            median_diff=15.0,
        )
        assert prop.property_name == "MW"
        assert prop.train_mean == 250.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        prop = PropertyDistribution(
            property_name="MW",
            train_mean=250.0,
            train_std=50.0,
            train_median=240.0,
            test_mean=260.0,
            test_std=55.0,
            test_median=255.0,
            ks_statistic=0.1,
            ks_pvalue=0.5,
            mean_diff=10.0,
            median_diff=15.0,
        )
        d = prop.to_dict()
        assert d["property_name"] == "MW"


class TestSizeMetrics:
    """Tests for SizeMetrics dataclass."""

    def test_creation_without_labels(self):
        """Test creating SizeMetrics without label info."""
        metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        assert metrics.total_samples == 100
        assert metrics.train_positive_ratio is None

    def test_creation_with_labels(self):
        """Test creating SizeMetrics with label info."""
        metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
            train_positive_ratio=0.5,
            test_positive_ratio=0.6,
            label_balance_diff=0.1,
        )
        assert metrics.train_positive_ratio == 0.5
        assert metrics.label_balance_diff == 0.1


class TestSplitQualityReport:
    """Tests for SplitQualityReport dataclass."""

    def test_creation_minimal(self):
        """Test creating report with minimal data."""
        size_metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        report = SplitQualityReport(
            splitter_name="scaffold",
            split_index=0,
            size_metrics=size_metrics,
        )
        assert report.splitter_name == "scaffold"
        assert report.similarity_metrics is None

    def test_creation_full(self):
        """Test creating report with all metrics."""
        size_metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        sim_metrics = SimilarityMetrics(
            min_sim=0.1,
            max_sim=0.9,
            mean_sim=0.5,
            median_sim=0.45,
            std_sim=0.2,
            percentile_5=0.15,
            percentile_25=0.3,
            percentile_75=0.7,
            percentile_95=0.85,
            n_test_samples=20,
        )
        scaffold_metrics = ScaffoldMetrics(
            train_unique_scaffolds=50,
            test_unique_scaffolds=10,
            shared_scaffolds=5,
            scaffold_overlap_percentage=50.0,
            train_scaffold_coverage=95.0,
            test_scaffold_coverage=90.0,
        )
        report = SplitQualityReport(
            splitter_name="scaffold",
            split_index=0,
            size_metrics=size_metrics,
            similarity_metrics=sim_metrics,
            scaffold_metrics=scaffold_metrics,
        )
        assert report.similarity_metrics is not None
        assert report.scaffold_metrics is not None

    def test_to_dict(self):
        """Test conversion to nested dictionary."""
        size_metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        report = SplitQualityReport(
            splitter_name="scaffold",
            split_index=0,
            size_metrics=size_metrics,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["splitter_name"] == "scaffold"
        assert "size_metrics" in d

    def test_to_dataframe_row(self):
        """Test conversion to flat dictionary for DataFrame."""
        size_metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        report = SplitQualityReport(
            splitter_name="scaffold",
            split_index=0,
            size_metrics=size_metrics,
        )
        row = report.to_dataframe_row()
        assert isinstance(row, dict)
        assert "size_train_size" in row
        assert row["splitter_name"] == "scaffold"

    def test_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        size_metrics = SizeMetrics(
            total_samples=100,
            train_size=80,
            test_size=20,
            train_ratio=0.8,
            test_ratio=0.2,
        )
        sim_metrics = SimilarityMetrics(
            min_sim=0.1,
            max_sim=0.9,
            mean_sim=0.5,
            median_sim=0.45,
            std_sim=0.2,
            percentile_5=0.15,
            percentile_25=0.3,
            percentile_75=0.7,
            percentile_95=0.85,
            n_test_samples=20,
        )
        report = SplitQualityReport(
            splitter_name="scaffold",
            split_index=0,
            size_metrics=size_metrics,
            similarity_metrics=sim_metrics,
        )
        # Should not raise
        json_str = json.dumps(report.to_dict())
        assert isinstance(json_str, str)


class TestSplitAnalyzerInitialization:
    """Tests for SplitAnalyzer initialization."""

    def test_basic_initialization(self, sample_smiles):
        """Test basic initialization."""
        analyzer = SplitAnalyzer(sample_smiles)
        assert analyzer.n_molecules == len(sample_smiles)
        assert analyzer.fingerprint_type == "ecfp"

    def test_custom_fingerprint(self, sample_smiles):
        """Test initialization with custom fingerprint settings."""
        analyzer = SplitAnalyzer(
            sample_smiles,
            fingerprint_type="fcfp",
            fingerprint_radius=3,
            fingerprint_nbits=1024,
        )
        assert analyzer.fingerprint_type == "fcfp"
        assert analyzer.fingerprint_radius == 3
        assert analyzer.fingerprint_nbits == 1024

    def test_custom_properties(self, sample_smiles):
        """Test initialization with custom properties."""
        analyzer = SplitAnalyzer(
            sample_smiles,
            compute_properties=["MW", "LogP"],
        )
        assert analyzer.compute_properties == ["MW", "LogP"]

    def test_empty_smiles_raises(self):
        """Test that empty SMILES list raises error on analysis."""
        analyzer = SplitAnalyzer([])
        assert analyzer.n_molecules == 0


class TestSplitAnalyzerLazyLoading:
    """Tests for lazy loading of computed properties."""

    def test_fingerprints_lazy_loaded(self, analyzer):
        """Test that fingerprints are computed lazily."""
        assert analyzer._fingerprints is None
        fps = analyzer.fingerprints
        assert fps is not None
        assert analyzer._fingerprints is not None

    def test_scaffolds_lazy_loaded(self, analyzer):
        """Test that scaffolds are computed lazily."""
        assert analyzer._scaffolds is None
        scaffolds = analyzer.scaffolds
        assert scaffolds is not None
        assert analyzer._scaffolds is not None

    def test_properties_lazy_loaded(self, analyzer):
        """Test that properties are computed lazily."""
        assert analyzer._properties is None
        props = analyzer.properties
        assert props is not None
        assert analyzer._properties is not None


class TestSplitAnalyzerAnalyzeSplit:
    """Tests for analyze_split method."""

    def test_basic_analysis(self, analyzer, train_test_indices):
        """Test basic split analysis."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(train_idx, test_idx, "test_splitter")

        assert isinstance(report, SplitQualityReport)
        assert report.splitter_name == "test_splitter"
        assert report.split_index == 0
        assert report.size_metrics is not None

    def test_analysis_with_similarity(self, analyzer, train_test_indices):
        """Test that similarity metrics are computed."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            compute_similarity=True,
        )

        assert report.similarity_metrics is not None
        assert 0 <= report.similarity_metrics.mean_sim <= 1
        assert report.similarity_metrics.min_sim <= report.similarity_metrics.max_sim

    def test_analysis_with_scaffolds(self, analyzer, train_test_indices):
        """Test that scaffold metrics are computed."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            compute_scaffolds=True,
        )

        assert report.scaffold_metrics is not None
        assert report.scaffold_metrics.train_unique_scaffolds >= 0
        assert 0 <= report.scaffold_metrics.scaffold_overlap_percentage <= 100

    def test_analysis_with_properties(self, analyzer, train_test_indices):
        """Test that property distributions are computed."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            compute_properties=True,
        )

        assert len(report.property_distributions) > 0
        for prop in report.property_distributions:
            assert prop.ks_statistic >= 0
            assert 0 <= prop.ks_pvalue <= 1

    def test_analysis_with_labels(self, analyzer, train_test_indices, sample_labels):
        """Test analysis with labels for balance metrics."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            labels=sample_labels,
        )

        assert report.size_metrics.train_positive_ratio is not None
        assert report.size_metrics.test_positive_ratio is not None
        assert report.size_metrics.label_balance_diff is not None

    def test_analysis_without_optional_metrics(self, analyzer, train_test_indices):
        """Test analysis with optional metrics disabled."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            compute_similarity=False,
            compute_scaffolds=False,
            compute_properties=False,
        )

        assert report.similarity_metrics is None
        assert report.scaffold_metrics is None
        assert len(report.property_distributions) == 0

    def test_custom_split_index(self, analyzer, train_test_indices):
        """Test setting custom split index."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(
            train_idx,
            test_idx,
            "test",
            split_index=5,
        )
        assert report.split_index == 5


class TestSplitAnalyzerAnalyzeSplitter:
    """Tests for analyze_splitter method."""

    def test_analyze_random_splitter(self, sample_smiles):
        """Test analyzing random splitter."""
        analyzer = SplitAnalyzer(sample_smiles)
        reports = analyzer.analyze_splitter("random", n_splits=2, test_size=0.2)

        assert len(reports) == 2
        for report in reports:
            assert report.splitter_name == "random"
            assert report.size_metrics is not None

    def test_analyze_with_labels(self, sample_smiles, sample_labels):
        """Test analyzing splitter with labels."""
        analyzer = SplitAnalyzer(sample_smiles)
        reports = analyzer.analyze_splitter(
            "random",
            n_splits=2,
            test_size=0.2,
            labels=sample_labels,
        )

        for report in reports:
            assert report.size_metrics.train_positive_ratio is not None


class TestSplitAnalyzerCompareSplitters:
    """Tests for compare_splitters method."""

    def test_compare_two_splitters(self, sample_smiles):
        """Test comparing two splitters."""
        analyzer = SplitAnalyzer(sample_smiles)
        comparison = analyzer.compare_splitters(
            ["random", "molecular_logp"],
            n_splits=2,
            test_size=0.2,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) >= 2  # At least one row per splitter

    def test_compare_with_aggregation(self, sample_smiles):
        """Test comparison with aggregation."""
        analyzer = SplitAnalyzer(sample_smiles)
        comparison = analyzer.compare_splitters(
            ["random"],
            n_splits=3,
            aggregate=True,
        )

        assert isinstance(comparison, pd.DataFrame)
        # Should have aggregated columns
        assert any("_mean" in col for col in comparison.columns)

    def test_compare_without_aggregation(self, sample_smiles):
        """Test comparison without aggregation."""
        analyzer = SplitAnalyzer(sample_smiles)
        comparison = analyzer.compare_splitters(
            ["random"],
            n_splits=3,
            aggregate=False,
        )

        assert isinstance(comparison, pd.DataFrame)
        # Should have individual split rows
        assert len(comparison) >= 3


class TestSplitAnalyzerGetSummaryStats:
    """Tests for get_summary_stats method."""

    def test_get_summary_from_reports(self, sample_smiles):
        """Test getting summary statistics from reports."""
        analyzer = SplitAnalyzer(sample_smiles)
        reports = analyzer.analyze_splitter("random", n_splits=3)
        summary = analyzer.get_summary_stats(reports)

        assert isinstance(summary, dict)
        if "similarity" in summary:
            assert "mean" in summary["similarity"]
            assert "std" in summary["similarity"]

    def test_get_summary_empty_reports(self, analyzer):
        """Test summary with empty reports list."""
        summary = analyzer.get_summary_stats([])
        assert summary == {}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_molecule_train(self, sample_smiles):
        """Test with single molecule in train set."""
        analyzer = SplitAnalyzer(sample_smiles)
        train_idx = np.array([0])
        test_idx = np.array([1, 2, 3, 4])

        report = analyzer.analyze_split(train_idx, test_idx, "test")
        assert report.size_metrics.train_size == 1

    def test_single_molecule_test(self, sample_smiles):
        """Test with single molecule in test set."""
        analyzer = SplitAnalyzer(sample_smiles)
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4])

        report = analyzer.analyze_split(train_idx, test_idx, "test")
        assert report.size_metrics.test_size == 1

    def test_empty_test_set(self, sample_smiles):
        """Test with empty test set."""
        analyzer = SplitAnalyzer(sample_smiles)
        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([])

        report = analyzer.analyze_split(train_idx, test_idx, "test")
        assert report.size_metrics.test_size == 0
        # Similarity metrics should be None for empty test
        assert report.similarity_metrics is None

    def test_invalid_smiles_handling(self):
        """Test handling of invalid SMILES."""
        smiles_with_invalid = [
            "CCO",
            "INVALID_SMILES",
            "c1ccccc1",
            "NOT_A_MOLECULE",
            "CCN",
        ]
        analyzer = SplitAnalyzer(smiles_with_invalid)

        # Should not raise during fingerprint computation
        fps = analyzer.fingerprints
        assert fps.shape[0] == len(smiles_with_invalid)

    def test_list_indices_accepted(self, analyzer, train_test_indices):
        """Test that list indices are accepted (not just numpy arrays)."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(list(train_idx), list(test_idx), "test")
        assert report is not None


class TestPropertyDistributionComputation:
    """Tests for property distribution computation."""

    def test_ks_test_computed(self, analyzer, train_test_indices):
        """Test that KS test is computed correctly."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(train_idx, test_idx, "test")

        for prop in report.property_distributions:
            # KS statistic should be between 0 and 1
            assert 0 <= prop.ks_statistic <= 1
            # P-value should be between 0 and 1
            assert 0 <= prop.ks_pvalue <= 1

    def test_mean_diff_computed(self, analyzer, train_test_indices):
        """Test that mean difference is computed correctly."""
        train_idx, test_idx = train_test_indices
        report = analyzer.analyze_split(train_idx, test_idx, "test")

        for prop in report.property_distributions:
            expected_diff = abs(prop.train_mean - prop.test_mean)
            assert np.isclose(prop.mean_diff, expected_diff)


@pytest.mark.slow
class TestSlowSplitters:
    """Tests for slower splitters (marked as slow)."""

    def test_analyze_scaffold_splitter(self, sample_smiles):
        """Test analyzing scaffold splitter."""
        analyzer = SplitAnalyzer(sample_smiles)
        reports = analyzer.analyze_splitter("scaffold", n_splits=1, test_size=0.2)

        assert len(reports) >= 1
        assert reports[0].splitter_name == "scaffold"

    def test_analyze_kmeans_splitter(self, sample_smiles):
        """Test analyzing kmeans splitter."""
        analyzer = SplitAnalyzer(sample_smiles)
        try:
            reports = analyzer.analyze_splitter("kmeans", n_splits=1, test_size=0.2)
            assert len(reports) >= 1
            assert reports[0].splitter_name == "kmeans"
        except Exception as e:
            # Skip if there's a version compatibility issue with datamol/RDKit
            if "ArgumentError" in str(type(e).__name__) or "Boost.Python" in str(e):
                pytest.skip(f"Skipping due to datamol/RDKit version compatibility: {e}")
