#!/usr/bin/env python
"""Freeze the published data: hash it, verify it, and make it read-only.

The datasets, splits, distance matrices, trained models, predictions, figures and
tables in this checkout are the ones the paper was published with. They are not in
git, so this tool records a SHA-256 manifest of every file (the manifest *is* in
git, under ``provenance/<label>/``) and locks the trees against later runs.

    python scripts/data_freeze.py create            # hash everything (slow, resumable)
    python scripts/data_freeze.py verify            # quick: existence + size
    python scripts/data_freeze.py verify --full     # re-hash and compare
    python scripts/data_freeze.py lock              # read-only + .frozen markers
    python scripts/data_freeze.py unlock --yes-i-am-sure
    python scripts/data_freeze.py status

Files are grouped into three sets so the common check stays fast:

``datasets``         everything under ``datasets/``
``results-core``     metrics, predictions, configs, logs, figures and tables
``results-weights``  model weights and graph caches (hundreds of GB)

Each set has ``<set>.sha256`` (plain ``sha256sum`` format, so
``sha256sum -c provenance/paper-v1/datasets.sha256`` also works from the repo root)
and ``<set>.meta.tsv`` (size, tag, path). The tag is ``paper`` for files that belong
to the dataset x splitter x model grid in ``datasets/config.yml`` and ``other`` for
everything else that happened to be in the trees when they were frozen.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

DEFAULT_LABEL = "paper-v1"
PAPER_DOI = "10.1021/acs.jcim.5c00475"
FROZEN_MARKER = ".frozen"  # keep in sync with alinemol/utils/frozen.py

FROZEN_ROOTS = (
    "datasets",
    "classification_results",
    "classification_inference_results",
    "assets/figures",
    "assets/tables",
)
SETS = ("datasets", "results-core", "results-weights")
WEIGHT_SUFFIXES = (".pth", ".pt", ".joblib", ".pdparams", ".ckpt", ".bin")
AGGREGATE_DATASET_FILES = (
    "all_splitters_nearest_distances.csv",
    "nearest_distances.csv",
    "nearest_distances_id.csv",
)
AGGREGATE_RESULT_FILES = ("results.csv", "hit_rate.csv")
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
CHUNK = 4 * 1024 * 1024


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Entry:
    path: str  # posix path relative to the repo root
    size: int
    sha256: str
    tag: str  # "paper" or "other"


@dataclass
class VerifyReport:
    checked: int = 0
    changed: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.changed or self.missing or self.unexpected)


@dataclass(frozen=True)
class PaperScope:
    """The dataset x splitter x model grid of the paper (from datasets/config.yml)."""

    datasets: frozenset
    splitters: frozenset
    models: frozenset

    @classmethod
    def load(cls, repo: Path) -> "PaperScope":
        config_path = repo / "datasets" / "config.yml"
        if not config_path.is_file():
            return cls(frozenset(), frozenset(), frozenset())
        import yaml  # core dependency of alinemol

        config = yaml.safe_load(config_path.read_text()) or {}
        datasets = (config.get("datasets") or {}).get("TDC") or []
        return cls(
            frozenset(datasets),
            frozenset(config.get("splitting") or []),
            frozenset(_leaves(config.get("models"))),
        )

    def tag(self, rel: str) -> str:
        return "paper" if self._is_paper(rel.split("/")) else "other"

    def _is_paper(self, parts: Sequence[str]) -> bool:
        if parts[0] == "assets":
            return True
        if parts[0] == "datasets":
            if parts == ["datasets", "config.yml"]:
                return True
            if len(parts) < 3 or parts[1] != "TDC":
                return False
            if len(parts) == 3:
                return parts[2] in AGGREGATE_DATASET_FILES
            if parts[2] not in self.datasets:
                return False
            if len(parts) == 4:  # raw / standardized csv, distance matrices
                return True
            # datasets/TDC/<ds>/split/<splitter>/<file>; deeper = archives, not paper
            return len(parts) == 6 and parts[3] == "split" and parts[4] in self.splitters
        if parts[0] == "classification_results":
            if len(parts) < 3 or parts[1] != "TDC":
                return False
            if parts[-1] in AGGREGATE_RESULT_FILES and len(parts) <= 5:
                names = parts[2:-1]
                return (len(names) < 1 or names[0] in self.datasets) and (len(names) < 2 or names[1] in self.splitters)
            if len(parts) < 7:
                return False
            dataset, splitter, model, replicate = parts[2:6]
            return (
                dataset in self.datasets
                and splitter in self.splitters
                and model in self.models
                and replicate.isdigit()
                and 1 <= int(replicate) <= 10
            )
        return False


def _leaves(node) -> Iterator[str]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _leaves(value)
    elif isinstance(node, str):
        yield node


def set_of(rel: str) -> str:
    """Which manifest set a repo-relative path belongs to."""
    if rel.startswith("datasets/"):
        return "datasets"
    if rel.startswith("classification_") and rel.lower().endswith(WEIGHT_SUFFIXES):
        return "results-weights"
    return "results-core"


# --------------------------------------------------------------------------- files


def iter_files(repo: Path, roots: Iterable[str] = FROZEN_ROOTS) -> Iterator[str]:
    """Yield repo-relative posix paths of every regular file under the frozen roots."""
    for root in roots:
        top = repo / root
        if not top.is_dir():
            continue
        stack = [top]
        while stack:
            current = stack.pop()
            with os.scandir(current) as it:
                children = sorted(it, key=lambda e: e.name)
            for child in children:
                if child.is_symlink():
                    print(f"warning: skipping symlink {child.path}", file=sys.stderr)
                elif child.is_dir():
                    stack.append(Path(child.path))
                elif child.is_file() and child.name != FROZEN_MARKER:
                    yield Path(child.path).relative_to(repo).as_posix()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def manifest_dir(repo: Path, label: str = DEFAULT_LABEL) -> Path:
    return repo / "provenance" / label


def read_manifest(repo: Path, set_name: str, label: str = DEFAULT_LABEL) -> List[Entry]:
    directory = manifest_dir(repo, label)
    sha_path = directory / f"{set_name}.sha256"
    meta_path = directory / f"{set_name}.meta.tsv"
    if not sha_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"No manifest for set '{set_name}' in {directory}. Run: data_freeze.py create")
    hashes: Dict[str, str] = {}
    for line in sha_path.read_text().splitlines():
        if line:
            digest, path = line.split("  ", 1)
            hashes[path] = digest
    entries = []
    for line in meta_path.read_text().splitlines()[1:]:
        size, tag, path = line.split("\t", 2)
        entries.append(Entry(path, int(size), hashes[path], tag))
    return entries


def _write_manifest(repo: Path, set_name: str, entries: List[Entry], label: str) -> None:
    directory = manifest_dir(repo, label)
    directory.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: e.path)
    (directory / f"{set_name}.sha256").write_text("".join(f"{e.sha256}  {e.path}\n" for e in entries))
    (directory / f"{set_name}.meta.tsv").write_text(
        "size\ttag\tpath\n" + "".join(f"{e.size}\t{e.tag}\t{e.path}\n" for e in entries)
    )


def _git_commit(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_summary(repo: Path, label: str) -> None:
    directory = manifest_dir(repo, label)
    summary = {
        "label": label,
        "paper_doi": PAPER_DOI,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo),
        "frozen_roots": list(FROZEN_ROOTS),
        "hash": "sha256",
        "sets": {},
    }
    for set_name in SETS:
        try:
            entries = read_manifest(repo, set_name, label)
        except FileNotFoundError:
            continue
        paper = [e for e in entries if e.tag == "paper"]
        summary["sets"][set_name] = {
            "files": len(entries),
            "bytes": sum(e.size for e in entries),
            "paper_files": len(paper),
            "paper_bytes": sum(e.size for e in paper),
            "manifest_sha256": sha256_of(directory / f"{set_name}.sha256"),
        }
    (directory / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


# ------------------------------------------------------------------------ commands


def create(
    repo: Path,
    sets: Optional[Sequence[str]] = None,
    workers: int = 8,
    label: str = DEFAULT_LABEL,
    progress: bool = False,
) -> None:
    """Hash every file under the frozen roots and write the manifests."""
    repo = Path(repo)
    selected = list(sets or SETS)
    scope = PaperScope.load(repo)
    by_set: Dict[str, List[str]] = {name: [] for name in selected}
    for rel in iter_files(repo):
        name = set_of(rel)
        if name in by_set:
            by_set[name].append(rel)

    directory = manifest_dir(repo, label)
    directory.mkdir(parents=True, exist_ok=True)
    for set_name in selected:
        paths = by_set[set_name]
        partial_path = directory / f"{set_name}.partial.tsv"
        done: Dict[str, Entry] = {}
        if partial_path.is_file():  # resume an interrupted run
            for line in partial_path.read_text().splitlines():
                parts = line.split("\t", 2)
                if len(parts) == 3 and len(parts[0]) == 64:
                    rel = parts[2]
                    target = repo / rel
                    if target.is_file() and target.stat().st_size == int(parts[1]):
                        done[rel] = Entry(rel, int(parts[1]), parts[0], scope.tag(rel))
        todo = [rel for rel in paths if rel not in done]
        total_bytes = sum((repo / rel).stat().st_size for rel in todo)
        if progress:
            print(
                f"[{set_name}] {len(paths)} files, {len(done)} already hashed, "
                f"{len(todo)} to hash ({total_bytes / 1e9:.1f} GB)",
                flush=True,
            )

        def work(rel: str) -> Entry:
            target = repo / rel
            return Entry(rel, target.stat().st_size, sha256_of(target), scope.tag(rel))

        hashed_bytes = 0
        next_report = 0
        with open(partial_path, "a") as partial, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(work, rel) for rel in todo]
            for future in as_completed(futures):
                entry = future.result()
                done[entry.path] = entry
                partial.write(f"{entry.sha256}\t{entry.size}\t{entry.path}\n")
                partial.flush()
                hashed_bytes += entry.size
                if progress and hashed_bytes >= next_report:
                    print(f"[{set_name}] {hashed_bytes / 1e9:8.1f} / {total_bytes / 1e9:.1f} GB", flush=True)
                    next_report = hashed_bytes + 10e9
        _write_manifest(repo, set_name, [done[rel] for rel in paths], label)
        partial_path.unlink()
        if progress:
            print(f"[{set_name}] done", flush=True)
    _write_summary(repo, label)


def verify(
    repo: Path,
    full: bool = False,
    sets: Optional[Sequence[str]] = None,
    workers: int = 8,
    label: str = DEFAULT_LABEL,
) -> VerifyReport:
    """Compare the trees with the manifests. Quick = existence + size; full = re-hash."""
    repo = Path(repo)
    selected = list(sets or SETS)
    report = VerifyReport()
    known: Set[str] = set()
    to_hash: List[Entry] = []
    for set_name in selected:
        for entry in read_manifest(repo, set_name, label):
            known.add(entry.path)
            report.checked += 1
            target = repo / entry.path
            if not target.is_file():
                report.missing.append(entry.path)
            elif target.stat().st_size != entry.size:
                report.changed.append(entry.path)
            elif full:
                to_hash.append(entry)
    if to_hash:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            digests = pool.map(lambda e: sha256_of(repo / e.path), to_hash)
            report.changed.extend(e.path for e, digest in zip(to_hash, digests) if digest != e.sha256)
    report.unexpected = [rel for rel in iter_files(repo) if set_of(rel) in selected and rel not in known]
    report.changed.sort()
    report.missing.sort()
    return report


def _existing_roots(repo: Path) -> List[Path]:
    return [repo / root for root in FROZEN_ROOTS if (repo / root).is_dir()]


def lock(repo: Path, label: str = DEFAULT_LABEL) -> None:
    """Drop a .frozen marker in every frozen root and remove all write permissions."""
    repo = Path(repo)
    for set_name in SETS:
        read_manifest(repo, set_name, label)  # refuse to lock what was never hashed
    marker = {
        "label": label,
        "paper_doi": PAPER_DOI,
        "locked": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest": f"provenance/{label}/",
        "note": "Published data. Do not modify. Write new outputs under runs/. See provenance/README.md.",
    }
    for root in _existing_roots(repo):
        marker_path = root / FROZEN_MARKER
        if marker_path.exists():
            os.chmod(marker_path, marker_path.stat().st_mode | stat.S_IWUSR)
        os.chmod(root, root.stat().st_mode | stat.S_IWUSR)
        marker_path.write_text(json.dumps(marker, indent=2) + "\n")
        for current, dirs, files in os.walk(root, topdown=False):
            for name in files + dirs:
                path = os.path.join(current, name)
                if not os.path.islink(path):
                    os.chmod(path, os.stat(path).st_mode & ~WRITE_BITS)
        os.chmod(root, root.stat().st_mode & ~WRITE_BITS)


def unlock(repo: Path) -> None:
    """Give the owner write access back and remove the .frozen markers."""
    repo = Path(repo)
    for root in _existing_roots(repo):
        os.chmod(root, root.stat().st_mode | stat.S_IWUSR)
        for current, dirs, files in os.walk(root, topdown=True):
            for name in dirs + files:
                path = os.path.join(current, name)
                if not os.path.islink(path):
                    os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)
        marker_path = root / FROZEN_MARKER
        if marker_path.exists():
            marker_path.unlink()


def status(repo: Path, label: str = DEFAULT_LABEL) -> str:
    repo = Path(repo)
    lines = []
    summary_path = manifest_dir(repo, label) / "manifest.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        lines.append(f"manifest {label}: created {summary['created']} at commit {summary['git_commit']}")
        for name, info in summary["sets"].items():
            lines.append(
                f"  {name:16s} {info['files']:7d} files {info['bytes'] / 1e9:8.2f} GB"
                f"  (paper: {info['paper_files']} files, {info['paper_bytes'] / 1e9:.2f} GB)"
            )
    else:
        lines.append(f"manifest {label}: not created")
    for root in FROZEN_ROOTS:
        path = repo / root
        state = "missing" if not path.is_dir() else ("locked" if (path / FROZEN_MARKER).is_file() else "NOT locked")
        lines.append(f"  {root:36s} {state}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cli


def _print_report(report: VerifyReport, full: bool, limit: int = 20) -> None:
    mode = "full (sha256)" if full else "quick (existence + size)"
    print(f"verify {mode}: {report.checked} files checked")
    for title, items in (("changed", report.changed), ("missing", report.missing), ("unexpected", report.unexpected)):
        if items:
            print(f"  {title}: {len(items)}")
            for item in items[:limit]:
                print(f"    {item}")
            if len(items) > limit:
                print(f"    ... and {len(items) - limit} more")
    print("OK: data matches the manifest" if report.ok else "FAILED: data differs from the manifest")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="manifest name under provenance/")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="hash every file and write the manifests (resumable)")
    p_create.add_argument("--set", dest="sets", nargs="+", choices=SETS)
    p_create.add_argument("--workers", type=int, default=8)

    p_verify = sub.add_parser("verify", help="compare the data with the manifests")
    mode = p_verify.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="existence + size only (default)")
    mode.add_argument("--full", action="store_true", help="re-hash every file")
    p_verify.add_argument("--set", dest="sets", nargs="+", choices=SETS)
    p_verify.add_argument("--workers", type=int, default=8)

    sub.add_parser("lock", help="make the frozen trees read-only and mark them")
    p_unlock = sub.add_parser("unlock", help="restore write access (think twice)")
    p_unlock.add_argument("--yes-i-am-sure", action="store_true")
    sub.add_parser("status", help="show manifest and lock state")

    args = parser.parse_args(argv)
    repo = args.repo.resolve()

    if args.command == "create":
        create(repo, sets=args.sets, workers=args.workers, label=args.label, progress=True)
        print(status(repo, args.label))
        return 0
    if args.command == "verify":
        try:
            report = verify(repo, full=args.full, sets=args.sets, workers=args.workers, label=args.label)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 2
        _print_report(report, args.full)
        return 0 if report.ok else 1
    if args.command == "lock":
        try:
            lock(repo, args.label)
        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 2
        print(status(repo, args.label))
        return 0
    if args.command == "unlock":
        if not args.yes_i_am_sure:
            print(
                "Refusing to unlock the published data. This removes the protection that keeps later runs\n"
                "from changing it. If you really mean it, re-run with --yes-i-am-sure.",
                file=sys.stderr,
            )
            return 2
        unlock(repo)
        print(status(repo, args.label))
        return 0
    print(status(repo, args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
