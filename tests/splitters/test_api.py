"""Tests for the one-call interface (split_and_analyze / compare_splits) and registry metadata."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alinemol.splitters import (
    SplitResult,
    compare_splits,
    describe_splitters,
    get_splitter,
    get_splitter_names,
    load_splits,
    select_splits,
    split_and_analyze,
    splitter_params,
)

DATA = Path(__file__).resolve().parents[1] / "conftest" / "dili_standardized.csv"


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(DATA).head(200).reset_index(drop=True)


@pytest.fixture(scope="module")
def smiles(table):
    return table["canonical_smiles"].tolist()


@pytest.fixture(scope="module")
def labels(table):
    return table["label"].to_numpy()


@pytest.fixture(scope="module")
def scaffold_result(smiles, labels):
    return split_and_analyze(smiles, "scaffold", labels, n_splits=3, test_size=0.2)


# ------------------------------------------------------------------ split_and_analyze


def test_returns_disjoint_index_splits_covering_valid_rows(scaffold_result, smiles):
    assert isinstance(scaffold_result, SplitResult)
    assert len(scaffold_result.splits) == 3
    for train, test in scaffold_result.splits:
        assert len(set(train) & set(test)) == 0
        assert max(train.max(), test.max()) < len(smiles)
        assert len(train) > 0 and len(test) > 0


def test_reports_and_shift_per_split(scaffold_result):
    assert len(scaffold_result.reports) == len(scaffold_result.shift) == 3
    shift = scaffold_result.shift[0]
    assert shift["k"] == 5
    assert 0.0 <= shift["nearest"]["median"] <= shift["k_nearest"]["median"] <= 1.0
    report = scaffold_result.reports[0]
    # the analyzer's nearest similarity and the shift's nearest distance are the same quantity
    assert report.similarity_metrics.median_sim == pytest.approx(1 - shift["nearest"]["median"], abs=1e-6)
    assert len(report.nearest_similarities) == len(scaffold_result.splits[0][1])


def test_default_seed_makes_results_reproducible(smiles, scaffold_result, labels):
    again = split_and_analyze(smiles, "scaffold", labels, n_splits=3, test_size=0.2, analyze=False)
    for (train_a, test_a), (train_b, test_b) in zip(scaffold_result.splits, again.splits):
        np.testing.assert_array_equal(np.sort(test_a), np.sort(test_b))


def test_alias_and_params_are_recorded(smiles):
    result = split_and_analyze(smiles, "k-means", n_splits=2, n_clusters=4, analyze=False)
    assert result.splitter == "kmeans"
    assert result.params["n_clusters"] == 4
    assert result.params["random_state"] == 42
    assert result.requested == {"n_splits": 2, "test_size": 0.2, "random_state": 42, "preset": None}


def test_deterministic_splitter_returns_one_split_with_a_note(smiles):
    result = split_and_analyze(smiles, "molecular_weight", n_splits=5, analyze=False)
    assert len(result.splits) == 1
    assert any("deterministic" in w for w in result.warnings)
    assert result.test_fractions[0] == pytest.approx(0.2, abs=0.01)


def test_off_target_test_size_is_reported_not_hidden(smiles):
    result = split_and_analyze(smiles, "kmeans", n_splits=5, test_size=0.2, analyze=False)
    off = [f for f in result.test_fractions if abs(f - 0.2) > 0.05]
    assert bool(off) == any("Realised test fraction" in w for w in result.warnings)


def test_label_balance_filter(smiles, labels):
    result = split_and_analyze(smiles, "random", labels, n_splits=3, max_label_imbalance=0.02, analyze=False)
    for train, test in result.splits:
        assert abs(labels[train].mean() - labels[test].mean()) <= 0.02


def test_select_splits_keeps_first_passing_candidates():
    labels = np.array([1, 1, 0, 0, 1, 0])
    balanced = (np.array([0, 2, 4, 5]), np.array([1, 3]))
    skewed = (np.array([2, 3, 5, 4]), np.array([0, 1]))
    chosen = select_splits([skewed, balanced, balanced], 1, labels, max_label_imbalance=0.1)
    assert len(chosen) == 1 and list(chosen[0][1]) == [1, 3]
    assert len(select_splits([skewed, balanced], 5)) == 2


def test_paper_preset_applies_published_settings(smiles):
    plain = split_and_analyze(smiles, "max_dissimilarity", n_splits=1, analyze=False)
    paper = split_and_analyze(smiles, "max_dissimilarity", n_splits=1, analyze=False, preset="paper")
    assert plain.params["n_clusters"] == 25 and paper.params["n_clusters"] == 10
    with pytest.raises(ValueError, match="Unknown preset"):
        split_and_analyze(smiles, "random", preset="nope")


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"splitter": "not_a_splitter"}, ValueError, "Unknown splitter"),
        ({"splitter": "random", "bogus": 1}, TypeError, "does not accept"),
        ({"splitter": "lo"}, ValueError, "needs activity values"),
    ],
)
def test_clear_errors(smiles, kwargs, error, message):
    with pytest.raises(error, match=message):
        split_and_analyze(smiles, **kwargs)


def test_label_length_mismatch_raises(smiles):
    with pytest.raises(ValueError, match="labels has"):
        split_and_analyze(smiles, "random", labels=[0, 1])


def test_precomputed_distance_matrix_gives_same_shift(smiles, scaffold_result):
    from scipy.spatial.distance import cdist

    from alinemol.splitters.shift import compute_fingerprints

    fps = compute_fingerprints(smiles)
    matrix = cdist(fps, fps, metric="jaccard")
    result = split_and_analyze(smiles, "scaffold", n_splits=3, test_size=0.2, distance_matrix=matrix)
    assert result.shift[0]["k_nearest"]["median"] == pytest.approx(
        scaffold_result.shift[0]["k_nearest"]["median"], abs=1e-6
    )


# ------------------------------------------------------------------ serialisation


def test_to_json_roundtrip_and_schema(scaffold_result):
    payload = json.loads(scaffold_result.to_json())
    assert payload["schema_version"] == 1
    assert payload["splitter"] == "scaffold" and payload["n_splits"] == 3
    assert set(payload["splits"][0]) >= {"train_idx", "test_idx", "test_fraction", "shift", "report"}
    assert "train_idx" not in json.loads(scaffold_result.to_json(include_indices=False))["splits"][0]


def test_summary_has_one_row_per_split(scaffold_result):
    summary = scaffold_result.summary()
    assert len(summary) == 3
    assert {"test_fraction", "nearest_distance_median", "k5_distance_median", "scaffold_overlap_pct"} <= set(summary)


def test_save_and_load_splits(scaffold_result, table, tmp_path):
    out = tmp_path / "scaffold"
    scaffold_result.save(out, data=table)
    assert (out / "result.json").is_file()
    test0 = pd.read_csv(out / "test_0.csv")
    assert list(test0["row_index"]) == list(scaffold_result.splits[0][1])
    assert list(test0["canonical_smiles"]) == list(table["canonical_smiles"].iloc[scaffold_result.splits[0][1]])
    loaded = load_splits(out)
    np.testing.assert_array_equal(loaded[1][0], scaffold_result.splits[1][0])
    with pytest.raises(FileExistsError):
        scaffold_result.save(out)
    scaffold_result.save(out, overwrite=True)


def test_save_refuses_frozen_tree(scaffold_result, tmp_path):
    from alinemol.utils.frozen import FROZEN_MARKER, FrozenDataError

    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / FROZEN_MARKER).write_text("{}")
    with pytest.raises(FrozenDataError):
        scaffold_result.save(tmp_path / "datasets" / "split" / "scaffold")


# ------------------------------------------------------------------ compare_splits


def test_compare_splits_reports_failures_explicitly(smiles):
    comparison = compare_splits(smiles, ["random", "scaffold", "lo", "nonexistent"], n_splits=2)
    assert set(comparison.results) == {"random", "scaffold"}
    assert "needs activity values" in comparison.errors["lo"]
    assert "Unknown splitter" in comparison.errors["nonexistent"]
    frame = comparison.to_frame()
    assert list(frame["splitter"]) == ["random", "scaffold"]
    assert list(frame["n_splits"]) == [2, 2]
    by_name = frame.set_index("splitter")["nearest_distance_median"]
    assert by_name["scaffold"] >= by_name["random"]
    json.loads(comparison.to_json())


# ------------------------------------------------------------------ registry metadata


def test_describe_splitters_covers_registry_and_is_json_friendly():
    described = describe_splitters()
    assert [d["name"] for d in described] == get_splitter_names()
    json.dumps(described)
    by_name = {d["name"]: d for d in described}
    assert by_name["lo"]["needs_labels"] and by_name["datasail"]["requires"] == "datasail"
    assert by_name["random"]["test_size"] == "exact" and by_name["hi"]["test_size"] == "ignored"
    assert "mw" in by_name["molecular_weight"]["aliases"]
    assert by_name["kmeans"]["parameters"]["n_clusters"] == 10
    assert all(d["family"] != "other" and d["summary"] for d in described)


def test_repr_and_params_show_real_configuration():
    splitter = get_splitter("kmeans", n_clusters=7, random_state=3)
    assert "n_clusters=7" in repr(splitter) and "random_state=3" in repr(splitter)
    assert splitter_params(get_splitter("umap", n_clusters=4))["n_clusters"] == 4
    assert splitter_params(get_splitter("butina", cutoff=0.5))["cutoff"] == 0.5


# ------------------------------------------------------------------ datasail wrapper


@pytest.fixture
def fake_datasail(monkeypatch):
    """Stand-in for the optional datasail package with its documented return shape."""
    calls = {}

    def datasail(**kwargs):
        calls.update(kwargs)
        ids = list(kwargs["e_data"])
        assignment = {mol_id: ("test" if i % 5 == 0 else "train") for i, mol_id in enumerate(ids)}
        assignment[ids[1]] = "not selected"
        technique = kwargs["techniques"][0]
        if calls.get("empty"):
            return {}, {}, {}
        return {technique: [assignment] * kwargs["runs"]}, {}, {}

    package = types.ModuleType("datasail")
    sail = types.ModuleType("datasail.sail")
    sail.datasail = datasail
    package.sail = sail
    monkeypatch.setitem(sys.modules, "datasail", package)
    monkeypatch.setitem(sys.modules, "datasail.sail", sail)
    return calls


def test_datasail_result_is_parsed_from_tuple(fake_datasail, smiles):
    splits = list(get_splitter("datasail", n_splits=2, test_size=0.2).split(smiles[:50]))
    assert len(splits) == 2
    train, test = splits[0]
    assert list(test) == list(range(0, 50, 5))
    assert 1 not in train and 1 not in test  # dropped by DataSAIL
    assert fake_datasail["techniques"] == ["C1e"] and fake_datasail["names"] == ["train", "test"]


def test_datasail_never_falls_back_to_random(fake_datasail, smiles):
    fake_datasail["empty"] = True
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        list(get_splitter("datasail").split(smiles[:50]))
