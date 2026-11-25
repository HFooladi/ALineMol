"""Train/test distribution shift measured as nearest-neighbour distances.

One definition, used everywhere in the package: for every test molecule, the Tanimoto
(Jaccard) *distance* on binary fingerprints to its ``k`` nearest training molecules.

- ``k=1`` is the classic "distance to the nearest training molecule"
  (``1 - max similarity``), which :class:`~alinemol.splitters.SplitAnalyzer` reports.
- ``k=5`` is what the paper reports
  (``alinemol.utils.split_utils.retrieve_k_nearest_neighbors_Tanimoto``).

Larger distances mean the test set is less like the training set, i.e. a harder,
more out-of-distribution split.

Example:
    >>> from alinemol.splitters import get_splitter, nearest_train_distances, summarize_distances
    >>> train_idx, test_idx = next(get_splitter("scaffold", random_state=0).split(smiles))
    >>> distances = nearest_train_distances(train_idx, test_idx, smiles=smiles, k=5)
    >>> distances.shape        # (n_test, 5), each row sorted nearest first
    >>> summarize_distances(distances)["median"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

DEFAULT_FINGERPRINT = {"fp_type": "ecfp", "radius": 2, "fpSize": 2048}  # ECFP4, as in the paper

ArrayOrPath = Union[np.ndarray, str, Path]
IndexArray = Union[Sequence[int], np.ndarray]


def compute_fingerprints(
    smiles: Sequence[str], fp_type: str = "ecfp", radius: int = 2, fpSize: int = 2048
) -> np.ndarray:
    """Binary fingerprints for a list of SMILES as a ``(n, n_bits)`` boolean array.

    Molecules that cannot be parsed get an all-zero fingerprint (distance 1 to everything).
    """
    import datamol as dm

    kwargs = {} if fp_type == "maccs" else {"radius": radius, "fpSize": fpSize}
    rows = []
    width = None
    for smi in smiles:
        mol = dm.to_mol(smi)
        fp = None if mol is None else dm.to_fp(mol, fp_type=fp_type, **kwargs)
        if fp is not None:
            width = len(fp)
        rows.append(fp)
    if width is None:
        raise ValueError("None of the SMILES could be parsed.")
    return np.array([np.zeros(width, dtype=bool) if fp is None else np.asarray(fp, dtype=bool) for fp in rows])


def _load_distance_matrix(distance_matrix: ArrayOrPath, n_molecules: Optional[int]) -> np.ndarray:
    if isinstance(distance_matrix, (str, Path)):
        distance_matrix = np.load(distance_matrix, mmap_mode="r")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"distance_matrix must be square, got shape {distance_matrix.shape}")
    if n_molecules is not None and distance_matrix.shape[0] != n_molecules:
        raise ValueError(
            f"distance_matrix has {distance_matrix.shape[0]} rows but there are {n_molecules} molecules; "
            "its rows must be in the same order as the molecules."
        )
    return distance_matrix


def _k_smallest(block: np.ndarray, k: int) -> np.ndarray:
    """The k smallest values of every row, sorted ascending."""
    if k >= block.shape[1]:
        return np.sort(block, axis=1)[:, :k]
    part = np.partition(block, k - 1, axis=1)[:, :k]
    return np.sort(part, axis=1)


def nearest_train_distances(
    train_idx: IndexArray,
    test_idx: IndexArray,
    *,
    smiles: Optional[Sequence[str]] = None,
    fingerprints: Optional[np.ndarray] = None,
    distance_matrix: Optional[ArrayOrPath] = None,
    k: int = 5,
    chunk_size: int = 512,
) -> np.ndarray:
    """Distance from every test molecule to its ``k`` nearest training molecules.

    Give exactly one source of molecular information, in order of preference:

    Args:
        train_idx: Indices of the training molecules.
        test_idx: Indices of the test molecules.
        smiles: All molecules as SMILES; ECFP4 (2048 bits) fingerprints are computed.
        fingerprints: Precomputed binary fingerprints for all molecules, ``(n, n_bits)``.
        distance_matrix: Precomputed square distance matrix (array or path to ``.npy``,
            memory-mapped) whose rows follow the molecule order. Any metric works here,
            e.g. Tree Mover's Distance.
        k: Number of nearest training molecules per test molecule. Clipped to the
            training-set size.
        chunk_size: Test molecules processed at once (bounds memory).

    Returns:
        Array of shape ``(n_test, k)``; row ``i`` holds the distances for ``test_idx[i]``,
        nearest first.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train_idx and test_idx must both be non-empty.")
    k = min(k, len(train_idx))
    out = np.empty((len(test_idx), k), dtype=np.float64)

    if distance_matrix is not None:
        matrix = _load_distance_matrix(distance_matrix, None if smiles is None else len(smiles))
        order = np.argsort(train_idx)  # sorted column access is much faster on a memory map
        sorted_train = train_idx[order]
        for start in range(0, len(test_idx), chunk_size):
            rows = test_idx[start : start + chunk_size]
            block = np.asarray(matrix[rows][:, sorted_train], dtype=np.float64)
            out[start : start + chunk_size] = _k_smallest(block, k)
        return out

    if fingerprints is None:
        if smiles is None:
            raise ValueError("Provide one of: smiles, fingerprints, distance_matrix.")
        fingerprints = compute_fingerprints(smiles, **DEFAULT_FINGERPRINT)
    fps = np.asarray(fingerprints)
    train = fps[train_idx].astype(np.float32)
    train_bits = train.sum(axis=1)
    for start in range(0, len(test_idx), chunk_size):
        test = fps[test_idx[start : start + chunk_size]].astype(np.float32)
        intersection = test @ train.T
        union = test.sum(axis=1)[:, None] + train_bits[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(union > 0, intersection / union, 0.0)
        out[start : start + chunk_size] = _k_smallest(1.0 - similarity, k)
    return out


def summarize_distances(distances: np.ndarray) -> Dict[str, float]:
    """Summary statistics of a distance array (all values pooled).

    Returns a plain, JSON-serialisable dict: ``n``, ``mean``, ``std``, ``min``, ``p05``,
    ``p25``, ``median``, ``p75``, ``p95``, ``max``.
    """
    values = np.asarray(distances, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("No distances to summarise.")
    p05, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(p05),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(values.max()),
    }
