"""ALineMol utilities.

Re-exports symbols from submodules so legacy ``from alinemol.utils import X`` style
imports keep working. The re-exports are *lazy*: a submodule is only imported when one
of its names is first accessed. That keeps ``import alinemol.splitters`` (which needs
``alinemol.utils.split_utils`` and ``alinemol.utils.frozen``) from loading matplotlib,
torch or DGL, and lets a splitter-only install work without those packages.

Heavy submodules and the extras they need:

- ``metric_utils``: torch (``[gnn]``) + statsmodels (``[ml]``)
- ``utils``: torch / dgl / dgllife (``[gnn]``)
- ``split_utils.split_molecules_*``: astartes (``[ml]``)
"""

from importlib import import_module
from typing import Any, List

_LAZY = {
    "plot_utils": ["plot_ID_OOD", "plot_ID_OOD_sns", "visualize_chemspace"],
    "metric_utils": [
        "Meter",
        "compute_linear_fit",
        "eval_acc",
        "eval_pr_auc",
        "eval_roc_auc",
        "rescale",
        "compare_rankings",
    ],
    "split_utils": [
        "compute_similarities",
        "featurize",
        "split_molecules_train_test",
        "split_molecules_train_val_test",
    ],
    "utils": [
        "collate_molgraphs",
        "collate_molgraphs_unlabeled",
        "get_configure",
        "init_featurizer",
        "init_inference_trial_path",
        "init_trial_path",
        "load_dataset",
        "load_model",
        "mkdir_p",
        "predict",
        "split_dataset",
    ],
    "frozen": ["FrozenDataError", "assert_not_frozen", "is_frozen"],
}
_NAME_TO_MODULE = {name: module for module, names in _LAZY.items() for name in names}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f"{__name__}.{module_name}")
    except (ImportError, OSError) as error:
        # OSError: DGL raises FileNotFoundError when its graphbolt .so does not match torch.
        raise ImportError(
            f"'{name}' needs optional dependencies that are not installed ({error}). "
            f"Install the matching extra, e.g. pip install 'alinemol[all]'."
        ) from error
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(list(globals()) + __all__)
