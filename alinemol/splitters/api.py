"""One-call interface: split a set of molecules and measure the train/test shift.

This is the entry point meant for other projects and for agents::

    from alinemol.splitters import split_and_analyze, compare_splits

    result = split_and_analyze(smiles, "scaffold", n_splits=5, test_size=0.2)
    train_idx, test_idx = result.splits[0]
    result.summary()            # one row per split: sizes, distances, scaffold overlap
    result.warnings             # anything the caller should know (e.g. realised test size)
    result.save("runs/splits/my_data/scaffold", data=df)   # result.json + train_i.csv / test_i.csv

    comparison = compare_splits(smiles, ["random", "scaffold", "kmeans"])
    comparison.to_frame()       # one row per splitter
    comparison.errors           # splitters that failed, with the reason

Everything returned is plain data: numpy index arrays, dataclasses with ``to_dict()``,
and JSON on request. Results are reproducible by default (``random_state=42``) and
record the splitter, its resolved parameters and the package version.
"""

from __future__ import annotations

import json
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from alinemol.splitters.analyzer import SplitAnalyzer, SplitQualityReport
from alinemol.splitters.factory import (
    _SPLITTER_INFO,
    _accepts_var_kwargs,
    _init_parameters,
    describe_splitters,
    get_splitter,
    list_splitters,
    resolve_splitter_name,
    splitter_params,
)
from alinemol.splitters.shift import ArrayOrPath, nearest_train_distances, summarize_distances
from alinemol.utils.frozen import assert_not_frozen

SCHEMA_VERSION = 1
DEFAULT_RANDOM_STATE = 42
TEST_SIZE_WARN_DEVIATION = 0.05
DEFAULT_CANDIDATES = 100

Split = Tuple[np.ndarray, np.ndarray]


def _package_version() -> str:
    try:
        from alinemol import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version lookup must never break a split
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class SplitResult:
    """Splits produced by one splitter, with their shift measurements.

    Attributes:
        splitter: Canonical splitter name.
        params: Constructor parameters the splitter was actually built with.
        requested: What the caller asked for (``n_splits``, ``test_size``, ``random_state``, ``preset``).
        n_molecules: Number of input molecules.
        splits: ``(train_idx, test_idx)`` index arrays into the input, one pair per split.
        reports: One :class:`SplitQualityReport` per split (empty if ``analyze=False``).
        shift: One dict per split with nearest-training-neighbour Tanimoto *distances*:
            ``{"k": k, "nearest": {...}, "k_nearest": {...}}`` where each inner dict is
            :func:`~alinemol.splitters.shift.summarize_distances` output (``nearest`` =
            closest training molecule, ``k_nearest`` = the k closest pooled, as in the paper).
        warnings: Human-readable notes the caller should not miss.
    """

    splitter: str
    params: Dict[str, Any]
    requested: Dict[str, Any]
    n_molecules: int
    splits: List[Split]
    reports: List[SplitQualityReport] = field(default_factory=list)
    shift: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    version: str = field(default_factory=_package_version)
    schema_version: int = SCHEMA_VERSION

    @property
    def test_fractions(self) -> List[float]:
        """Realised test-set fraction of every split."""
        return [len(test) / self.n_molecules for _, test in self.splits]

    def summary(self) -> pd.DataFrame:
        """One row per split: sizes, realised test fraction and the main shift measures."""
        rows = []
        for i, (train, test) in enumerate(self.splits):
            row: Dict[str, Any] = {
                "splitter": self.splitter,
                "split": i,
                "train_size": len(train),
                "test_size": len(test),
                "test_fraction": len(test) / self.n_molecules,
            }
            if i < len(self.shift):
                shift = self.shift[i]
                row["nearest_distance_median"] = shift["nearest"]["median"]
                row["nearest_distance_mean"] = shift["nearest"]["mean"]
                row[f"k{shift['k']}_distance_median"] = shift["k_nearest"]["median"]
                row[f"k{shift['k']}_distance_mean"] = shift["k_nearest"]["mean"]
            if i < len(self.reports):
                report = self.reports[i]
                if report.scaffold_metrics is not None:
                    row["scaffold_overlap_pct"] = report.scaffold_metrics.scaffold_overlap_percentage
                if report.size_metrics.label_balance_diff is not None:
                    row["label_balance_diff"] = report.size_metrics.label_balance_diff
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self, include_indices: bool = True) -> Dict[str, Any]:
        """Plain, JSON-serialisable dict (``include_indices=False`` drops the index lists)."""
        splits = []
        for i, (train, test) in enumerate(self.splits):
            entry: Dict[str, Any] = {
                "split": i,
                "train_size": int(len(train)),
                "test_size": int(len(test)),
                "test_fraction": len(test) / self.n_molecules,
            }
            if include_indices:
                entry["train_idx"] = [int(v) for v in train]
                entry["test_idx"] = [int(v) for v in test]
            if i < len(self.shift):
                entry["shift"] = self.shift[i]
            if i < len(self.reports):
                entry["report"] = self.reports[i].to_dict()
            splits.append(entry)
        return _jsonable(
            {
                "schema_version": self.schema_version,
                "alinemol_version": self.version,
                "splitter": self.splitter,
                "params": self.params,
                "requested": self.requested,
                "n_molecules": self.n_molecules,
                "n_splits": len(self.splits),
                "warnings": self.warnings,
                "splits": splits,
            }
        )

    def to_json(self, include_indices: bool = True, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(include_indices=include_indices), indent=indent)

    def save(self, directory: Union[str, Path], data: Optional[pd.DataFrame] = None, overwrite: bool = False) -> Path:
        """Write ``result.json`` (and, given ``data``, ``train_i.csv`` / ``test_i.csv``) to ``directory``.

        Args:
            directory: Output folder (created if needed). Refused if it lies inside
                frozen published data, or if it already holds a result and
                ``overwrite`` is False.
            data: The table the SMILES came from, in the same row order. When given,
                each split is also written as CSV with an added ``row_index`` column
                pointing back to the row in ``data``.
            overwrite: Allow replacing an existing result.

        Returns:
            Path of the written ``result.json``.
        """
        directory = Path(directory)
        assert_not_frozen(directory, what=f"{self.splitter} splits")
        result_path = directory / "result.json"
        if not overwrite and (result_path.exists() or any(directory.glob("train_*.csv"))):
            raise FileExistsError(f"{directory} already contains a split result. Pass overwrite=True to replace it.")
        if data is not None and len(data) != self.n_molecules:
            raise ValueError(f"data has {len(data)} rows but the result was computed on {self.n_molecules} molecules.")
        directory.mkdir(parents=True, exist_ok=True)
        result_path.write_text(self.to_json() + "\n")
        if data is not None:
            for i, (train, test) in enumerate(self.splits):
                for name, idx in (("train", train), ("test", test)):
                    part = data.iloc[idx].copy()
                    part.insert(0, "row_index", idx)
                    part.to_csv(directory / f"{name}_{i}.csv", index=False)
        return result_path


def load_splits(path: Union[str, Path]) -> List[Split]:
    """Read the ``(train_idx, test_idx)`` pairs back from a saved ``result.json`` (file or its folder)."""
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    payload = json.loads(path.read_text())
    return [(np.asarray(s["train_idx"], dtype=int), np.asarray(s["test_idx"], dtype=int)) for s in payload["splits"]]


@dataclass
class SplitComparison:
    """Results of several splitters on the same molecules.

    Attributes:
        results: Splitter name -> :class:`SplitResult` for every splitter that ran.
        errors: Splitter name -> error message for every splitter that failed.
    """

    results: Dict[str, SplitResult] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_frame(self, aggregate: bool = True) -> pd.DataFrame:
        """One row per splitter (mean over its splits) or, with ``aggregate=False``, per split."""
        frames = [result.summary() for result in self.results.values() if result.splits]
        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, ignore_index=True)
        if not aggregate:
            return table
        numeric = table.drop(columns=["split"]).groupby("splitter", sort=False).mean(numeric_only=True)
        numeric.insert(0, "n_splits", table.groupby("splitter", sort=False).size())
        return numeric.reset_index()

    def to_dict(self, include_indices: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "results": {name: result.to_dict(include_indices) for name, result in self.results.items()},
            "errors": dict(self.errors),
        }

    def to_json(self, include_indices: bool = False, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(include_indices=include_indices), indent=indent)


def _label_imbalance(labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    return abs(float(np.mean(labels[train])) - float(np.mean(labels[test])))


def select_splits(
    candidates: Iterable[Split],
    n_splits: int,
    labels: Optional[Sequence[float]] = None,
    max_label_imbalance: Optional[float] = None,
) -> List[Split]:
    """Take the first ``n_splits`` candidate splits that pass the label-balance filter.

    A candidate passes when the difference between the mean label (the fraction of
    actives for 0/1 labels) in train and in test is at most ``max_label_imbalance``.
    With no labels or no limit, every candidate passes. This is the rule
    ``scripts/splitting.py`` used for the published splits (``--tolerance``).
    """
    label_array = None if labels is None else np.asarray(labels, dtype=float)
    selected: List[Split] = []
    for train, test in candidates:
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if label_array is not None and max_label_imbalance is not None:
            if _label_imbalance(label_array, train, test) > max_label_imbalance:
                continue
        selected.append((train, test))
        if len(selected) >= n_splits:
            break
    return selected


def _build_splitter(
    name: str,
    n_requested: int,
    test_size: Optional[float],
    random_state: Optional[int],
    preset: Optional[str],
    splitter_kwargs: Dict[str, Any],
) -> Any:
    cls = list_splitters()[name]
    accepted = _init_parameters(cls)
    kwargs: Dict[str, Any] = {}
    if preset == "paper":
        from alinemol.splitters.splitting_configures import PAPER_SPLITTER_CONFIGS

        kwargs.update(PAPER_SPLITTER_CONFIGS.get(name, {}))
        if "verbose" in kwargs and "verbose" not in splitter_kwargs:
            kwargs["verbose"] = False
    elif preset is not None:
        raise ValueError(f"Unknown preset '{preset}'. Available presets: ['paper']")
    common = {"n_splits": n_requested, "test_size": test_size, "random_state": random_state}
    for key, value in common.items():
        if key in accepted or _accepts_var_kwargs(cls):
            kwargs[key] = value
    kwargs.update(splitter_kwargs)
    return get_splitter(name, **kwargs)


def _generate(
    smiles: Sequence[str],
    name: str,
    labels: Optional[Sequence[float]],
    n_splits: int,
    test_size: Optional[float],
    random_state: Optional[int],
    max_label_imbalance: Optional[float],
    n_candidates: Optional[int],
    preset: Optional[str],
    splitter_kwargs: Dict[str, Any],
) -> Tuple[Any, List[Split], List[str]]:
    info = _SPLITTER_INFO.get(name, {})
    notes: List[str] = []
    wanted = n_splits
    if info.get("deterministic") and n_splits > 1:
        wanted = 1
        notes.append(f"'{name}' is deterministic: it produces one split, so 1 split is returned instead of {n_splits}.")
    if info.get("needs_labels") and labels is None:
        raise ValueError(f"Splitter '{name}' needs activity values: pass them as labels=.")
    if info.get("test_size") == "ignored" and test_size is not None:
        notes.append(f"'{name}' chooses the split sizes itself; test_size={test_size} is not used.")

    filtering = labels is not None and max_label_imbalance is not None and wanted > 1
    n_requested = max(n_candidates or DEFAULT_CANDIDATES, wanted) if filtering else wanted
    splitter = _build_splitter(name, n_requested, test_size, random_state, preset, splitter_kwargs)

    y = np.asarray(labels) if (labels is not None and info.get("needs_labels")) else None
    candidates = splitter.split(list(smiles), y) if y is not None else splitter.split(list(smiles))
    splits = select_splits(candidates, wanted, labels, max_label_imbalance)

    if not splits:
        raise RuntimeError(
            f"Splitter '{name}' produced no split"
            + (f" within max_label_imbalance={max_label_imbalance}." if max_label_imbalance is not None else ".")
        )
    if len(splits) < wanted:
        notes.append(
            f"Only {len(splits)} of {wanted} requested splits passed max_label_imbalance={max_label_imbalance}."
        )
    n = len(smiles)
    for i, (train, test) in enumerate(splits):
        if len(np.intersect1d(train, test)) > 0:
            raise RuntimeError(f"Splitter '{name}' returned overlapping train and test indices (split {i}).")
    if test_size is not None and info.get("test_size") != "ignored":
        fractions = [len(test) / n for _, test in splits]
        off = [f for f in fractions if abs(f - test_size) > TEST_SIZE_WARN_DEVIATION]
        if off:
            notes.append(
                f"Realised test fraction differs from the requested {test_size:g}: "
                f"min {min(fractions):.2f}, max {max(fractions):.2f} "
                f"('{name}' assigns whole groups, so sizes are approximate)."
            )
    return splitter, splits, notes


def split_and_analyze(
    smiles: Sequence[str],
    splitter: str = "scaffold",
    labels: Optional[Sequence[float]] = None,
    *,
    n_splits: int = 5,
    test_size: Optional[float] = 0.2,
    random_state: Optional[int] = DEFAULT_RANDOM_STATE,
    analyze: bool = True,
    k: int = 5,
    distance_matrix: Optional[ArrayOrPath] = None,
    max_label_imbalance: Optional[float] = None,
    n_candidates: Optional[int] = None,
    preset: Optional[str] = None,
    analyzer: Optional[SplitAnalyzer] = None,
    **splitter_kwargs: Any,
) -> SplitResult:
    """Split molecules with one strategy and measure how far test is from train.

    Args:
        smiles: Molecules as SMILES strings.
        splitter: Splitter name or alias (see :func:`describe_splitters`).
        labels: Optional labels, same order as ``smiles``. Used for class-balance
            reporting and filtering; required by splitters that need activities (``lo``).
        n_splits: Number of splits wanted. Deterministic splitters return one.
        test_size: Requested test fraction. Group-based splitters only approximate it;
            the realised fraction is reported and flagged in ``warnings`` when it is off.
        random_state: Seed (default 42, so results are reproducible unless you pass None).
        analyze: Compute shift measures and quality reports. Set False to only split.
        k: Number of nearest training molecules used for ``shift[...]["k_nearest"]``.
        distance_matrix: Optional precomputed square distance matrix (array or ``.npy``
            path) in ``smiles`` order, reused instead of computing fingerprints.
        max_label_imbalance: If set (and labels given), keep only splits whose train and
            test mean label differ by at most this much; candidates are drawn until
            ``n_splits`` pass or ``n_candidates`` are used up.
        n_candidates: Candidate splits to draw when filtering (default 100).
        preset: ``"paper"`` applies the per-splitter settings used for the published splits.
        analyzer: Reuse a :class:`SplitAnalyzer` built on the same ``smiles``
            (saves recomputing fingerprints across calls).
        **splitter_kwargs: Extra constructor arguments for the splitter (e.g. ``n_clusters=10``).

    Returns:
        :class:`SplitResult`.

    Raises:
        ValueError: Unknown splitter, or labels missing for a splitter that needs them.
        TypeError: A keyword the splitter does not accept.
        RuntimeError: The splitter produced no usable split.
    """
    smiles = list(smiles)
    if len(smiles) < 2:
        raise ValueError("Need at least 2 molecules to split.")
    if labels is not None and len(labels) != len(smiles):
        raise ValueError(f"labels has {len(labels)} entries but there are {len(smiles)} molecules.")
    name = resolve_splitter_name(splitter)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        splitter_obj, splits, notes = _generate(
            smiles,
            name,
            labels,
            n_splits,
            test_size,
            random_state,
            max_label_imbalance,
            n_candidates,
            preset,
            dict(splitter_kwargs),
        )
    notes.extend(sorted({str(w.message) for w in caught if w.filename.startswith(str(Path(__file__).parent))}))

    result = SplitResult(
        splitter=name,
        params=_jsonable({k_: v for k_, v in splitter_params(splitter_obj).items() if k_ != "smiles"}),
        requested={"n_splits": n_splits, "test_size": test_size, "random_state": random_state, "preset": preset},
        n_molecules=len(smiles),
        splits=splits,
        warnings=notes,
    )
    if not analyze:
        return result

    if analyzer is None:
        analyzer = SplitAnalyzer(smiles, precomputed_distance_matrix=distance_matrix)
    elif len(analyzer.smiles) != len(smiles):
        raise ValueError("The given analyzer was built on a different number of molecules.")
    label_array = None if labels is None else np.asarray(labels)
    for i, (train, test) in enumerate(splits):
        result.reports.append(
            analyzer.analyze_split(train, test, splitter_name=name, split_index=i, labels=label_array)
        )
        if distance_matrix is not None:
            distances = nearest_train_distances(train, test, distance_matrix=distance_matrix, k=k)
        else:
            distances = nearest_train_distances(train, test, fingerprints=analyzer.fingerprints, k=k)
        result.shift.append(
            {
                "k": int(distances.shape[1]),
                "metric": "precomputed" if distance_matrix is not None else "tanimoto_distance_ecfp4_2048",
                "nearest": summarize_distances(distances[:, :1]),
                "k_nearest": summarize_distances(distances),
            }
        )
    return result


def compare_splits(
    smiles: Sequence[str],
    splitters: Union[str, Sequence[str]],
    labels: Optional[Sequence[float]] = None,
    *,
    n_splits: int = 5,
    test_size: Optional[float] = 0.2,
    random_state: Optional[int] = DEFAULT_RANDOM_STATE,
    k: int = 5,
    distance_matrix: Optional[ArrayOrPath] = None,
    max_label_imbalance: Optional[float] = None,
    preset: Optional[str] = None,
    splitter_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> SplitComparison:
    """Run several splitters on the same molecules and collect their shift measures.

    Args:
        splitters: Names, or ``"all"`` for every registered splitter that can run without
            labels-as-activities and without optional packages.
        splitter_kwargs: Optional per-splitter constructor arguments, ``{name: {...}}``.
        (other arguments as in :func:`split_and_analyze`)

    Returns:
        :class:`SplitComparison`. A splitter that fails is recorded in ``.errors`` with
        its message; it never disappears silently and never stops the others.
    """
    smiles = list(smiles)
    if isinstance(splitters, str):
        if splitters != "all":
            splitters = [splitters]
        else:
            splitters = [
                d["name"]
                for d in describe_splitters()
                if not d["requires"] and (labels is not None or not d["needs_labels"])
            ]
    analyzer = SplitAnalyzer(smiles, precomputed_distance_matrix=distance_matrix)
    comparison = SplitComparison()
    for requested_name in splitters:
        try:
            name = resolve_splitter_name(requested_name)
            comparison.results[name] = split_and_analyze(
                smiles,
                name,
                labels,
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state,
                k=k,
                distance_matrix=distance_matrix,
                max_label_imbalance=max_label_imbalance,
                preset=preset,
                analyzer=analyzer,
                **(splitter_kwargs or {}).get(requested_name, {}),
            )
        except Exception as error:  # one failing splitter must not hide the others
            comparison.errors[str(requested_name)] = f"{type(error).__name__}: {error}"
    return comparison
