"""``alinemol`` command: split molecules and measure train/test shift from the shell.

    alinemol list [--json]
    alinemol split   -f data.csv -s scaffold [-s kmeans ...] [-o out_dir] [--json]
    alinemol analyze -f data.csv --splits out_dir/scaffold [--json]
    alinemol guide

Every command prints a human-readable table by default and machine-readable JSON with
``--json``. Nothing is written unless ``-o/--output-dir`` is given. Exit code 0 means
success, 1 means at least one splitter failed, 2 means bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SMILES_COLUMNS = ("smiles", "canonical_smiles", "SMILES", "Smiles", "Drug")
LABEL_COLUMNS = ("label", "Label", "Y", "y", "activity")


class InputError(Exception):
    """Bad user input; reported without a traceback."""


def _read_table(path: str, smiles_column: Optional[str], label_column: Optional[str]) -> Tuple[Any, str, Optional[str]]:
    import pandas as pd

    file_path = Path(path)
    if not file_path.is_file():
        raise InputError(f"Input file not found: {path}")
    sep = "\t" if file_path.suffix.lower() in (".tsv", ".tab", ".txt") else ","
    table = pd.read_csv(file_path, sep=sep)
    if smiles_column is None:
        smiles_column = next((c for c in SMILES_COLUMNS if c in table.columns), None)
    if smiles_column is None or smiles_column not in table.columns:
        raise InputError(
            f"No SMILES column found in {path} (columns: {list(table.columns)}). Use --smiles-column to name it."
        )
    if label_column is None:
        label_column = next((c for c in LABEL_COLUMNS if c in table.columns), None)
    elif label_column not in table.columns:
        raise InputError(f"Label column '{label_column}' not in {path} (columns: {list(table.columns)}).")
    if table[smiles_column].isna().any():
        raise InputError(f"{int(table[smiles_column].isna().sum())} rows of {path} have no SMILES; remove them first.")
    return table.reset_index(drop=True), smiles_column, label_column


def _parse_params(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise InputError(f"--param expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            params[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            params[key.strip()] = raw
    return params


def _print_frame(frame: Any) -> None:
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        print(frame.to_string(index=False))


# ---------------------------------------------------------------------------- list


def cmd_list(args: argparse.Namespace) -> int:
    from alinemol.splitters import describe_splitters

    described = describe_splitters()
    if args.json:
        print(json.dumps(described, indent=2))
        return 0
    print(f"{'name':26s} {'family':11s} {'test size':12s} notes")
    for d in described:
        notes = []
        if d["deterministic"]:
            notes.append("one split only")
        if d["needs_labels"]:
            notes.append("needs activity values")
        if d["requires"]:
            notes.append(f"needs package '{d['requires']}'")
        print(f"{d['name']:26s} {d['family']:11s} {d['test_size']:12s} {'; '.join(notes)}")
    print("\nDetails (parameters, aliases): alinemol list --json")
    return 0


# --------------------------------------------------------------------------- split


def cmd_split(args: argparse.Namespace) -> int:
    from alinemol.splitters import compare_splits
    from alinemol.utils.frozen import assert_not_frozen, default_output_dir

    table, smiles_column, label_column = _read_table(args.file, args.smiles_column, args.label_column)
    smiles = table[smiles_column].astype(str).tolist()
    labels = None if (label_column is None or args.no_labels) else table[label_column].to_numpy()
    params = _parse_params(args.param)
    names: Any = "all" if args.splitter == ["all"] else args.splitter
    if len(args.splitter) > 1 and params:
        raise InputError("--param can only be used with a single splitter.")

    output_dir = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.save:
        output_dir = default_output_dir(args.file, "splits", fallback=Path(args.file).resolve().parent / "split")
    if output_dir is not None:
        assert_not_frozen(output_dir, what="splits")

    comparison = compare_splits(
        smiles,
        names,
        labels,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.seed,
        k=args.k,
        distance_matrix=args.distance_matrix,
        max_label_imbalance=args.max_label_imbalance,
        preset=args.preset,
        splitter_kwargs={args.splitter[0]: params} if params else None,
    )

    saved: Dict[str, str] = {}
    if output_dir is not None:
        for name, result in comparison.results.items():
            try:
                saved[name] = str(result.save(output_dir / name, data=table, overwrite=args.overwrite).parent)
            except FileExistsError as error:
                comparison.errors[name] = f"FileExistsError: {error}"
        for name in comparison.errors:
            comparison.results.pop(name, None)

    if args.json:
        payload = comparison.to_dict(include_indices=args.indices)
        payload["input"] = {
            "file": str(args.file),
            "n_molecules": len(smiles),
            "smiles_column": smiles_column,
            "label_column": None if labels is None else label_column,
        }
        payload["saved"] = saved
        print(json.dumps(payload, indent=2))
    else:
        print(f"{len(smiles)} molecules from {args.file} (SMILES column: {smiles_column})")
        if comparison.results:
            frame = comparison.to_frame(aggregate=not args.per_split)
            _print_frame(frame)
            print("\nDistances are Tanimoto distances to the nearest training molecules (higher = harder split).")
        for name, result in comparison.results.items():
            for warning in result.warnings:
                print(f"note [{name}]: {warning}")
        for name, path in saved.items():
            print(f"saved [{name}]: {path}")
        for name, error in comparison.errors.items():
            print(f"FAILED [{name}]: {error}", file=sys.stderr)
    return 1 if comparison.errors else 0


# ------------------------------------------------------------------------- analyze


def _splits_from_directory(directory: Path, table: Any, smiles_column: str) -> Tuple[List[Tuple[Any, Any]], str]:
    """Read splits from a result.json, or from train_i.csv / test_i.csv files."""
    import numpy as np
    import pandas as pd

    from alinemol.splitters import load_splits

    if (directory / "result.json").is_file():
        name = json.loads((directory / "result.json").read_text()).get("splitter", directory.name)
        return load_splits(directory), name

    test_files = sorted(directory.glob("test_*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not test_files:
        raise InputError(f"No result.json and no test_*.csv files in {directory}")
    lookup: Optional[Dict[str, int]] = None
    splits = []
    for test_file in test_files:
        train_file = test_file.with_name(test_file.name.replace("test_", "train_", 1))
        if not train_file.is_file():
            raise InputError(f"Missing {train_file.name} next to {test_file.name}")
        parts = []
        for part_file in (train_file, test_file):
            part = pd.read_csv(part_file)
            if "row_index" in part.columns:
                parts.append(part["row_index"].to_numpy(dtype=int))
                continue
            column = smiles_column if smiles_column in part.columns else "smiles"
            if column not in part.columns:
                raise InputError(f"{part_file} has neither a row_index nor a '{smiles_column}' column.")
            if lookup is None:
                if table[smiles_column].duplicated().any():
                    raise InputError("The input file has duplicate SMILES, so split rows cannot be matched by SMILES.")
                lookup = {smi: i for i, smi in enumerate(table[smiles_column].astype(str))}
            missing = [smi for smi in part[column].astype(str) if smi not in lookup]
            if missing:
                print(
                    f"note: {len(missing)} molecules of {part_file.name} are not in the input file and are skipped.",
                    file=sys.stderr,
                )
            parts.append(np.array([lookup[smi] for smi in part[column].astype(str) if smi in lookup], dtype=int))
        splits.append((parts[0], parts[1]))
    return splits, directory.name


def cmd_analyze(args: argparse.Namespace) -> int:
    import numpy as np

    from alinemol.splitters import SplitAnalyzer, SplitResult, nearest_train_distances, summarize_distances

    table, smiles_column, label_column = _read_table(args.file, args.smiles_column, args.label_column)
    smiles = table[smiles_column].astype(str).tolist()
    labels = None if label_column is None else table[label_column].to_numpy()
    directory = Path(args.splits)
    if not directory.is_dir():
        raise InputError(f"Splits directory not found: {args.splits}")
    splits, name = _splits_from_directory(directory, table, smiles_column)

    analyzer = SplitAnalyzer(smiles, precomputed_distance_matrix=args.distance_matrix)
    result = SplitResult(
        splitter=name,
        params={},
        requested={"source": str(directory)},
        n_molecules=len(smiles),
        splits=[(np.asarray(a), np.asarray(b)) for a, b in splits],
    )
    for i, (train, test) in enumerate(result.splits):
        result.reports.append(analyzer.analyze_split(train, test, splitter_name=name, split_index=i, labels=labels))
        if args.distance_matrix is not None:
            distances = nearest_train_distances(train, test, distance_matrix=args.distance_matrix, k=args.k)
        else:
            distances = nearest_train_distances(train, test, fingerprints=analyzer.fingerprints, k=args.k)
        result.shift.append(
            {
                "k": int(distances.shape[1]),
                "metric": "precomputed" if args.distance_matrix is not None else "tanimoto_distance_ecfp4_2048",
                "nearest": summarize_distances(distances[:, :1]),
                "k_nearest": summarize_distances(distances),
            }
        )
    if args.json:
        print(result.to_json(include_indices=args.indices))
    else:
        print(f"{len(result.splits)} splits from {directory} on {len(smiles)} molecules")
        _print_frame(result.summary())
    return 0


# --------------------------------------------------------------------------- guide


def cmd_guide(_: argparse.Namespace) -> int:
    print((Path(__file__).parent / "splitters" / "AGENT_GUIDE.md").read_text())
    return 0


# ---------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alinemol", description="Split molecular datasets and measure train/test distribution shift."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show the available splitters")
    p_list.add_argument("--json", action="store_true", help="machine-readable output with parameters and aliases")
    p_list.set_defaults(func=cmd_list)

    def add_input(p: argparse.ArgumentParser) -> None:
        p.add_argument("-f", "--file", required=True, help="CSV (or tab-separated .tsv/.tab/.txt) with a SMILES column")
        p.add_argument("--smiles-column", help="name of the SMILES column (default: auto-detect)")
        p.add_argument("--label-column", help="name of the label column (default: auto-detect, optional)")
        p.add_argument("--distance-matrix", help="precomputed .npy distance matrix in the file's row order")
        p.add_argument("-k", type=int, default=5, help="nearest training molecules per test molecule (default: 5)")
        p.add_argument("--json", action="store_true", help="print JSON instead of a table")
        p.add_argument("--indices", action="store_true", help="include train/test indices in the JSON")

    p_split = sub.add_parser("split", help="split a dataset with one or more splitters and measure the shift")
    add_input(p_split)
    p_split.add_argument(
        "-s", "--splitter", action="append", required=True, help="splitter name or alias; repeat for several, or 'all'"
    )
    p_split.add_argument("-n", "--n-splits", type=int, default=5, help="number of splits (default: 5)")
    p_split.add_argument("--test-size", type=float, default=0.2, help="requested test fraction (default: 0.2)")
    p_split.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    p_split.add_argument("--preset", choices=["paper"], help="use the splitter settings of the published splits")
    p_split.add_argument(
        "--max-label-imbalance",
        type=float,
        help="keep only splits whose train/test mean label differs by at most this much",
    )
    p_split.add_argument("--no-labels", action="store_true", help="ignore the label column")
    p_split.add_argument("--param", action="append", metavar="KEY=VALUE", help="extra splitter argument (repeatable)")
    p_split.add_argument("-o", "--output-dir", help="save result.json and train_i/test_i CSVs under DIR/<splitter>/")
    p_split.add_argument("--save", action="store_true", help="save to the default location (next to the input)")
    p_split.add_argument("--overwrite", action="store_true", help="replace existing saved splits")
    p_split.add_argument("--per-split", action="store_true", help="table with one row per split instead of means")
    p_split.set_defaults(func=cmd_split)

    p_analyze = sub.add_parser("analyze", help="measure the shift of splits that already exist on disk")
    add_input(p_analyze)
    p_analyze.add_argument(
        "--splits", required=True, help="folder with result.json, or with train_i.csv / test_i.csv files"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_guide = sub.add_parser("guide", help="print the usage guide for agents and scripts")
    p_guide.set_defaults(func=cmd_guide)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except PermissionError as error:  # includes FrozenDataError
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
