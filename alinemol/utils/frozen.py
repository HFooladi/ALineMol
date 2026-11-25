"""Guard against writing into frozen (published) data.

The data the paper was published with is hashed and locked by
``scripts/data_freeze.py``; each locked tree carries a ``.frozen`` marker file at its
root. Anything that writes output should call :func:`assert_not_frozen` on its
target first, so a later run fails with a clear message instead of a bare
``PermissionError`` (or, if the tree was unlocked, instead of silently changing
published files).

This module has no third-party dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

FROZEN_MARKER = ".frozen"  # keep in sync with scripts/data_freeze.py
DEFAULT_RUNS_DIR = "runs"

PathLike = Union[str, "os.PathLike[str]"]


class FrozenDataError(PermissionError):
    """Raised when code tries to write inside a frozen data tree."""


def find_frozen_root(path: PathLike) -> Optional[Path]:
    """Return the frozen tree containing ``path``, or ``None``.

    ``path`` does not need to exist. A directory is frozen when it, or any of its
    parents, contains a ``.frozen`` marker file.
    """
    resolved = Path(os.path.abspath(os.fspath(path)))
    for candidate in (resolved, *resolved.parents):
        if (candidate / FROZEN_MARKER).is_file():
            return candidate
    return None


def is_frozen(path: PathLike) -> bool:
    """Whether ``path`` lies inside a frozen data tree."""
    return find_frozen_root(path) is not None


def default_output_dir(
    input_path: PathLike, kind: str, fallback: PathLike, runs_dir: PathLike = DEFAULT_RUNS_DIR
) -> Path:
    """Pick where outputs derived from ``input_path`` go by default.

    Normally that is ``fallback`` (historically a folder next to the input). When that
    folder would be inside a frozen tree, outputs go to
    ``<runs_dir>/<kind>/<name of the input's folder>`` instead.

    Example: for ``datasets/TDC/HIV/HIV_standardize.csv`` and ``kind="splits"`` the
    result is ``runs/splits/HIV`` once ``datasets/`` is frozen.
    """
    if is_frozen(fallback):
        parent_name = Path(os.path.abspath(os.fspath(input_path))).parent.name
        return Path(runs_dir) / kind / parent_name
    return Path(fallback)


def assert_not_frozen(path: PathLike, what: str = "output") -> PathLike:
    """Raise :class:`FrozenDataError` if ``path`` is inside a frozen tree.

    Args:
        path: Where the caller is about to write.
        what: Short description of what is being written, for the error message.

    Returns:
        ``path`` unchanged, so the call can be used inline.
    """
    root = find_frozen_root(path)
    if root is not None:
        raise FrozenDataError(
            f"Refusing to write {what} to '{path}': it is inside the frozen published data at '{root}'. "
            f"Choose an output location outside it, e.g. under {DEFAULT_RUNS_DIR}/ "
            f"(see provenance/README.md)."
        )
    return path
