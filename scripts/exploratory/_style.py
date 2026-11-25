"""Shared look for exploratory (non-paper) figures.

Exploratory figures are for understanding, not for publication, so they use a plain
sans-serif face without LaTeX and a small fixed set of colours. Every script in
``scripts/exploratory/`` imports its names, order and colours from here so the figures
read as one family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl

REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO / "assets" / "exploratory"

# Order and names follow datasets/config.yml and alinemol.utils.plot_utils.SPLIT_TYPPE_MAPPING.
DATASETS: List[str] = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "HIV", "AMES", "HERG"]
SPLITTER_NAMES: Dict[str, str] = {
    "random": "Random",
    "scaffold": "Scaffold",
    "scaffold_generic": "Scaffold generic",
    "molecular_weight": "Molecular weight",
    "molecular_weight_reverse": "Molecular weight reverse",
    "molecular_logp": "Molecular logP",
    "kmeans": "K-means",
    "max_dissimilarity": "Max dissimilarity",
    "umap": "UMAP",
    "hi": "Lo-Hi",
    "datasail": "DataSAIL",
}

# Ink and surface
INK = "#191919"
INK_SECONDARY = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
SURFACE = "#ffffff"

# Marks: one hue for the thing being studied, gray for the baseline it is compared with.
SERIES = "#257de0"
SERIES_LIGHT = "#b6d5f8"
SERIES_DARK = "#0f4a96"
BASELINE = "#8a8a8a"
BASELINE_LIGHT = "#dcdcdc"
BASELINE_DARK = "#4d4d4d"

# One-hue ramps for magnitude (light -> dark)
SEQUENTIAL_BLUE = ["#f4f9ff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
SEQUENTIAL_ORANGE = ["#fff7f2", "#fde0cf", "#fbc4a3", "#f8a274", "#f47230", "#cf5a1f", "#a44412", "#74300c"]


def apply_style() -> None:
    """Set matplotlib defaults for exploratory figures (call once per script)."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "savefig.dpi": 300,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
        }
    )


def sequential_cmap(colors: List[str], name: str = "sequential"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, colors)


def source_fingerprint(relative_path: str, label: str = "paper-v1") -> Optional[str]:
    """SHA-256 of a frozen source file, looked up in the provenance manifest."""
    for manifest in sorted((REPO / "provenance" / label).glob("*.sha256")):
        for line in manifest.read_text().splitlines():
            digest, _, path = line.partition("  ")
            if path == relative_path:
                return digest
    return None


def save(fig, folder: str, name: str) -> List[Path]:
    """Save ``fig`` as PNG (300 dpi) and PDF under assets/exploratory/<folder>/."""
    out_dir = OUTPUT_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths
