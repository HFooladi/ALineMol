"""Tests for the `alinemol` command."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from alinemol.cli import main
from alinemol.splitters import get_splitter_names

DATA = Path(__file__).resolve().parents[1] / "conftest" / "dili_standardized.csv"


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "molecules.csv"
    pd.read_csv(DATA).head(150)[["canonical_smiles", "label"]].to_csv(path, index=False)
    return path


def test_list_json_matches_registry(capsys):
    assert main(["list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [d["name"] for d in listed] == get_splitter_names()


def test_list_table(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "scaffold" in out and "needs activity values" in out


def test_split_json_output(csv, capsys):
    assert main(["split", "-f", str(csv), "-s", "random", "-s", "scaffold", "-n", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["results"]) == {"random", "scaffold"}
    assert payload["errors"] == {} and payload["saved"] == {}
    assert payload["input"]["smiles_column"] == "canonical_smiles" and payload["input"]["n_molecules"] == 150
    first = payload["results"]["scaffold"]["splits"][0]
    assert "train_idx" not in first and first["shift"]["k"] == 5


def test_split_saves_and_refuses_to_overwrite(csv, tmp_path, capsys):
    out = tmp_path / "out"
    args = ["split", "-f", str(csv), "-s", "random", "-n", "2", "-o", str(out)]
    assert main(args) == 0
    assert (out / "random" / "result.json").is_file() and (out / "random" / "test_1.csv").is_file()
    capsys.readouterr()
    assert main(args) == 1
    assert "already contains" in capsys.readouterr().err
    assert main(args + ["--overwrite"]) == 0


def test_split_param_and_failure_exit_code(csv, capsys):
    assert main(["split", "-f", str(csv), "-s", "kmeans", "-n", "1", "--param", "n_clusters=4", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"]["kmeans"]["params"]["n_clusters"] == 4
    assert main(["split", "-f", str(csv), "-s", "random", "-s", "nonexistent", "-n", "1", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "nonexistent" in payload["errors"] and "random" in payload["results"]


def test_bad_input_exit_code(tmp_path, capsys):
    assert main(["split", "-f", str(tmp_path / "missing.csv"), "-s", "random"]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["split", "-f", str(bad), "-s", "random"]) == 2
    assert "No SMILES column" in capsys.readouterr().err


def test_split_refuses_frozen_output(csv, tmp_path, capsys):
    frozen = tmp_path / "datasets"
    frozen.mkdir()
    (frozen / ".frozen").write_text("{}")
    assert main(["split", "-f", str(csv), "-s", "random", "-n", "1", "-o", str(frozen / "split")]) == 2
    assert "frozen" in capsys.readouterr().err
    assert not (frozen / "split").exists()


def test_analyze_saved_result_and_plain_csv_splits(csv, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["split", "-f", str(csv), "-s", "scaffold", "-n", "2", "-o", str(out), "--json"]) == 0
    saved = json.loads(capsys.readouterr().out)["results"]["scaffold"]["splits"][0]["shift"]["nearest"]["median"]

    assert main(["analyze", "-f", str(csv), "--splits", str(out / "scaffold"), "--json"]) == 0
    from_json = json.loads(capsys.readouterr().out)
    assert from_json["splits"][0]["shift"]["nearest"]["median"] == pytest.approx(saved)

    # legacy layout: only train_i.csv / test_i.csv without a row_index column
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for name in ("train_0.csv", "test_0.csv"):
        pd.read_csv(out / "scaffold" / name).drop(columns="row_index").to_csv(legacy / name, index=False)
    assert main(["analyze", "-f", str(csv), "--splits", str(legacy), "--json"]) == 0
    from_csv = json.loads(capsys.readouterr().out)
    assert from_csv["splits"][0]["shift"]["nearest"]["median"] == pytest.approx(saved)


def test_guide_prints_packaged_guide(capsys):
    assert main(["guide"]) == 0
    assert "split_and_analyze" in capsys.readouterr().out


def test_module_entry_point_runs():
    result = subprocess.run([sys.executable, "-m", "alinemol.cli", "list", "--json"], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)
