"""Tests for scripts/data_freeze.py (hash manifest + read-only lock of published data)."""

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data_freeze.py"


@pytest.fixture(scope="module")
def df():
    spec = importlib.util.spec_from_file_location("data_freeze", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_freeze"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    """A miniature repo with the same layout as the real frozen trees."""
    files = {
        "datasets/config.yml": "datasets:\n  TDC:\n    - HIV\nsplitting:\n  - random\nmodels:\n  ML:\n    - SVM\n",
        "datasets/TDC/HIV/HIV_standardize.csv": "smiles,label\nCCO,1\n",
        "datasets/TDC/HIV/split/random/train_0.csv": "smiles,label\nCCO,1\n",
        "datasets/TDC/HIV/split/perimeter/train_0.csv": "smiles,label\nCC,0\n",
        "datasets/TDC/DILI/dili.csv": "smiles,label\nC,0\n",
        "classification_results/TDC/HIV/random/SVM/1/metrics.csv": "accuracy\n0.9\n",
        "classification_results/TDC/HIV/random/SVM/1/model.joblib": "weights",
        "classification_results/TDC/HIV/random/LightGBM/1/metrics.csv": "accuracy\n0.8\n",
        "classification_inference_results/1/prediction.csv": "p\n1\n",
        "assets/figures/fig.png": "png",
        "assets/tables/table.tex": "tex",
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    yield tmp_path
    # make sure pytest can clean up even if a test left the tree locked
    for root, dirs, names in os.walk(tmp_path):
        for name in dirs + names:
            p = os.path.join(root, name)
            os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)


def _entries(df, repo, set_name):
    return {e.path: e for e in df.read_manifest(repo, set_name)}


def test_create_assigns_files_to_sets(df, repo):
    df.create(repo)
    datasets = _entries(df, repo, "datasets")
    core = _entries(df, repo, "results-core")
    weights = _entries(df, repo, "results-weights")

    assert "datasets/TDC/HIV/HIV_standardize.csv" in datasets
    assert "classification_results/TDC/HIV/random/SVM/1/metrics.csv" in core
    assert "assets/figures/fig.png" in core
    assert "classification_inference_results/1/prediction.csv" in core
    assert list(weights) == ["classification_results/TDC/HIV/random/SVM/1/model.joblib"]
    assert not set(datasets) & set(core)


def test_create_records_hash_size_and_tag(df, repo):
    df.create(repo)
    datasets = _entries(df, repo, "datasets")
    entry = datasets["datasets/TDC/HIV/split/random/train_0.csv"]
    assert entry.size == len("smiles,label\nCCO,1\n")
    assert len(entry.sha256) == 64
    assert entry.tag == "paper"
    # not in the paper config -> tagged "other" but still frozen
    assert datasets["datasets/TDC/HIV/split/perimeter/train_0.csv"].tag == "other"
    assert datasets["datasets/TDC/DILI/dili.csv"].tag == "other"
    core = _entries(df, repo, "results-core")
    assert core["classification_results/TDC/HIV/random/SVM/1/metrics.csv"].tag == "paper"
    assert core["classification_results/TDC/HIV/random/LightGBM/1/metrics.csv"].tag == "other"


def test_sha256_file_is_sha256sum_compatible(df, repo):
    df.create(repo)
    lines = (repo / "provenance/paper-v1/datasets.sha256").read_text().splitlines()
    digest, path = lines[0].split("  ", 1)
    assert len(digest) == 64 and (repo / path).is_file()


def test_verify_passes_on_untouched_tree(df, repo):
    df.create(repo)
    for full in (False, True):
        report = df.verify(repo, full=full)
        assert report.ok, report


def test_full_verify_detects_same_size_modification(df, repo):
    df.create(repo)
    (repo / "datasets/TDC/HIV/HIV_standardize.csv").write_text("smiles,label\nCCN,1\n")
    assert df.verify(repo, full=False).ok  # same size: quick check cannot see it
    report = df.verify(repo, full=True)
    assert report.changed == ["datasets/TDC/HIV/HIV_standardize.csv"]
    assert not report.ok


def test_quick_verify_detects_size_change_missing_and_unexpected(df, repo):
    df.create(repo)
    (repo / "assets/tables/table.tex").write_text("a longer table")
    (repo / "datasets/TDC/DILI/dili.csv").unlink()
    (repo / "classification_results/TDC/HIV/random/SVM/1/prediction.csv").write_text("new")
    report = df.verify(repo, full=False)
    assert report.changed == ["assets/tables/table.tex"]
    assert report.missing == ["datasets/TDC/DILI/dili.csv"]
    assert report.unexpected == ["classification_results/TDC/HIV/random/SVM/1/prediction.csv"]


def test_verify_can_be_limited_to_a_set(df, repo):
    df.create(repo)
    (repo / "classification_results/TDC/HIV/random/SVM/1/model.joblib").write_text("tampered!")
    assert df.verify(repo, full=True, sets=["datasets", "results-core"]).ok
    assert not df.verify(repo, full=True, sets=["results-weights"]).ok


def test_lock_makes_tree_read_only_and_marks_roots(df, repo):
    df.create(repo)
    df.lock(repo)
    for root in ("datasets", "classification_results", "classification_inference_results", "assets/figures"):
        assert (repo / root / ".frozen").is_file()
    target = repo / "datasets/TDC/HIV/HIV_standardize.csv"
    assert not target.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            target.write_text("overwrite")
        with pytest.raises(PermissionError):
            (repo / "datasets/TDC/HIV/split/random/new.csv").write_text("new")
    # the marker itself is not reported as an unexpected file
    assert df.verify(repo, full=True).ok


def test_lock_refuses_without_manifest(df, repo):
    with pytest.raises(FileNotFoundError):
        df.lock(repo)


def test_unlock_restores_write_access(df, repo):
    df.create(repo)
    df.lock(repo)
    df.unlock(repo)
    assert not (repo / "datasets/.frozen").exists()
    (repo / "datasets/TDC/HIV/HIV_standardize.csv").write_text("writable again")


def test_cli_unlock_requires_confirmation(df, repo):
    df.create(repo)
    df.lock(repo)
    assert df.main(["--repo", str(repo), "unlock"]) != 0
    assert (repo / "datasets/.frozen").exists()
    assert df.main(["--repo", str(repo), "unlock", "--yes-i-am-sure"]) == 0


def test_cli_verify_exit_code(df, repo):
    assert df.main(["--repo", str(repo), "create", "--workers", "2"]) == 0
    assert df.main(["--repo", str(repo), "verify", "--full"]) == 0
    (repo / "assets/figures/fig.png").unlink()
    assert df.main(["--repo", str(repo), "verify"]) == 1
