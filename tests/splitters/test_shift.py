"""Tests for alinemol.splitters.shift (nearest-train-neighbour distances)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from alinemol.splitters.shift import compute_fingerprints, nearest_train_distances, summarize_distances

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fps():
    rng = np.random.default_rng(0)
    return rng.random((60, 128)) < 0.2


@pytest.fixture(scope="module")
def split():
    idx = np.random.default_rng(1).permutation(60)
    return idx[:45], idx[45:]


def test_matches_brute_force_jaccard(fps, split):
    train_idx, test_idx = split
    expected = np.sort(cdist(fps[test_idx], fps[train_idx], metric="jaccard"), axis=1)[:, :5]
    got = nearest_train_distances(train_idx, test_idx, fingerprints=fps, k=5)
    assert got.shape == (15, 5)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_rows_are_sorted_nearest_first(fps, split):
    got = nearest_train_distances(*split, fingerprints=fps, k=4)
    assert np.all(np.diff(got, axis=1) >= 0)


def test_chunking_does_not_change_result(fps, split):
    a = nearest_train_distances(*split, fingerprints=fps, k=3, chunk_size=4)
    b = nearest_train_distances(*split, fingerprints=fps, k=3, chunk_size=1000)
    np.testing.assert_allclose(a, b)


def test_distance_matrix_path_matches_fingerprint_path(fps, split, tmp_path):
    matrix = cdist(fps, fps, metric="jaccard")
    from_fps = nearest_train_distances(*split, fingerprints=fps, k=5)
    from_array = nearest_train_distances(*split, distance_matrix=matrix, k=5)
    np.save(tmp_path / "d.npy", matrix)
    from_file = nearest_train_distances(*split, distance_matrix=tmp_path / "d.npy", k=5)
    np.testing.assert_allclose(from_array, from_fps, atol=1e-6)
    np.testing.assert_allclose(from_file, from_array)


def test_k_is_clipped_to_train_size(fps):
    got = nearest_train_distances([0, 1, 2], [3, 4], fingerprints=fps, k=10)
    assert got.shape == (2, 3)


def test_identical_molecule_has_zero_distance(fps):
    doubled = np.vstack([fps, fps[:1]])
    got = nearest_train_distances([0, 1, 2], [60], fingerprints=doubled, k=1)
    assert got[0, 0] == pytest.approx(0.0)


def test_smiles_input_uses_ecfp4():
    smiles = ["CCO", "CCN", "c1ccccc1", "c1ccccc1O", "CCCC"]
    fps = compute_fingerprints(smiles)
    assert fps.shape == (5, 2048) and fps.dtype == bool
    got = nearest_train_distances([0, 1, 2], [3, 4], smiles=smiles, k=1)
    expected = cdist(fps[[3, 4]], fps[[0, 1, 2]], metric="jaccard").min(axis=1)
    np.testing.assert_allclose(got[:, 0], expected, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0, "fingerprints": np.ones((4, 8), bool)}, "k must be"),
        ({}, "Provide one of"),
        ({"distance_matrix": np.zeros((3, 4))}, "square"),
    ],
)
def test_invalid_input_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        nearest_train_distances([0, 1], [2, 3], **kwargs)


def test_distance_matrix_size_must_match_smiles():
    with pytest.raises(ValueError, match="same order"):
        nearest_train_distances([0], [1], smiles=["C", "CC", "CCC"], distance_matrix=np.zeros((2, 2)))


def test_summarize_distances_is_json_friendly():
    summary = summarize_distances(np.array([[0.0, 0.5], [1.0, 0.5]]))
    assert summary["n"] == 4 and summary["median"] == 0.5 and summary["min"] == 0.0 and summary["max"] == 1.0
    assert all(isinstance(v, (int, float)) for v in summary.values())


def test_agrees_with_paper_function_on_synthetic_data(fps, split):
    """Same values as the function used for the paper (which returns them unsorted, per neighbour slot)."""
    from alinemol.utils.split_utils import retrieve_k_nearest_neighbors_Tanimoto

    train_idx, test_idx = np.sort(split[0]), np.sort(split[1])
    matrix = cdist(fps, fps, metric="jaccard")
    original = pd.DataFrame({"smiles": [f"mol{i}" for i in range(60)]})
    paper = retrieve_k_nearest_neighbors_Tanimoto(
        matrix, original, original.iloc[train_idx], original.iloc[test_idx], k=5
    )
    ours = nearest_train_distances(train_idx, test_idx, distance_matrix=matrix, k=5)
    np.testing.assert_allclose(np.sort(np.asarray(paper).reshape(5, -1), axis=0).T, ours, atol=1e-9)


@pytest.mark.slow
def test_reproduces_published_distances_for_one_fold():
    """Recompute HIV / random / fold 0 exactly as the paper did and compare with the frozen CSV."""
    tdc = REPO / "datasets" / "TDC"
    published = tdc / "all_splitters_nearest_distances.csv"
    if not published.is_file():
        pytest.skip("published data not available in this checkout")
    from alinemol.utils.split_utils import retrive_index, sklearn_stratified_random_split

    original = pd.read_csv(tdc / "HIV" / "HIV_standardize.csv")
    id_df = pd.read_csv(tdc / "HIV" / "split" / "random" / "train_0.csv")
    x_train, *_ = sklearn_stratified_random_split(
        np.array(id_df["smiles"]), np.array(id_df["label"]), (0.72, 0.08, 0.2)
    )
    test_df = pd.read_csv(tdc / "HIV" / "split" / "random" / "test_0.csv")
    train_idx = retrive_index(original, pd.DataFrame({"smiles": x_train}))
    test_idx = retrive_index(original, test_df)

    ours = nearest_train_distances(train_idx, test_idx, distance_matrix=tdc / "HIV" / "Jaccard_distance.npy", k=5)

    frozen = pd.read_csv(published, usecols=["tanimoto", "split", "dataset", "index"])
    frozen = frozen[(frozen["dataset"] == "HIV") & (frozen["split"] == "random") & (frozen["index"] == 0)]
    expected = np.sort(frozen["tanimoto"].to_numpy().reshape(5, -1), axis=0).T
    assert ours.shape == expected.shape
    np.testing.assert_allclose(ours, expected, atol=1e-9)
