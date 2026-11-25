"""Tests for alinemol.utils.frozen (refuse to write into frozen published data)."""

import pytest

from alinemol.utils.frozen import (
    FROZEN_MARKER,
    FrozenDataError,
    assert_not_frozen,
    default_output_dir,
    find_frozen_root,
    is_frozen,
)


@pytest.fixture
def frozen_tree(tmp_path):
    root = tmp_path / "datasets"
    (root / "TDC" / "HIV").mkdir(parents=True)
    (root / FROZEN_MARKER).write_text("{}")
    (tmp_path / "runs").mkdir()
    return tmp_path


def test_path_inside_frozen_tree_is_detected(frozen_tree):
    target = frozen_tree / "datasets" / "TDC" / "HIV" / "split" / "random" / "train_0.csv"
    assert is_frozen(target)
    assert find_frozen_root(target) == frozen_tree / "datasets"


def test_frozen_root_itself_is_frozen(frozen_tree):
    assert is_frozen(frozen_tree / "datasets")


def test_path_outside_frozen_tree_is_not_frozen(frozen_tree):
    assert not is_frozen(frozen_tree / "runs" / "splits" / "train_0.csv")
    assert find_frozen_root(frozen_tree / "runs") is None


def test_assert_not_frozen_raises_with_helpful_message(frozen_tree):
    target = frozen_tree / "datasets" / "TDC" / "HIV" / "Jaccard_distance.npy"
    with pytest.raises(FrozenDataError) as excinfo:
        assert_not_frozen(target, what="distance matrix")
    message = str(excinfo.value)
    assert "distance matrix" in message
    assert str(frozen_tree / "datasets") in message
    assert "runs/" in message


def test_assert_not_frozen_returns_path_when_allowed(frozen_tree):
    target = frozen_tree / "runs" / "out.csv"
    assert assert_not_frozen(target) == target


def test_relative_paths_are_resolved(frozen_tree, monkeypatch):
    monkeypatch.chdir(frozen_tree)
    assert is_frozen("datasets/TDC/HIV/x.csv")
    assert not is_frozen("runs/x.csv")


def test_default_output_dir_redirects_frozen_inputs(frozen_tree):
    source = frozen_tree / "datasets" / "TDC" / "HIV" / "HIV_standardize.csv"
    out = default_output_dir(source, "splits", fallback=source.parent / "split", runs_dir=frozen_tree / "runs")
    assert out == frozen_tree / "runs" / "splits" / "HIV"


def test_default_output_dir_keeps_fallback_when_not_frozen(tmp_path):
    source = tmp_path / "mydata" / "molecules.csv"
    assert default_output_dir(source, "splits", fallback=source.parent / "split") == source.parent / "split"
