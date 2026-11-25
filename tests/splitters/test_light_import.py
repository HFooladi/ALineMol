"""`import alinemol.splitters` must stay light: no torch, DGL or matplotlib."""

import subprocess
import sys

HEAVY = ["torch", "dgl", "dgllife", "torch_geometric", "matplotlib", "seaborn", "statsmodels"]


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_splitters_does_not_load_heavy_packages():
    out = _run(f"import sys, alinemol.splitters\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))")
    assert out == "", f"heavy modules loaded by import alinemol.splitters: {out}"


def test_legacy_utils_reexports_still_resolve():
    out = _run("from alinemol.utils import assert_not_frozen, visualize_chemspace; print('ok')")
    assert out == "ok"


def test_unknown_utils_attribute_raises_attribute_error():
    out = _run("import alinemol.utils as u\ntry:\n    u.does_not_exist\nexcept AttributeError:\n    print('ok')")
    assert out == "ok"
