"""ALineMol utilities.

Re-exports symbols from submodules so legacy ``from alinemol.utils import X``
style imports continue to work for users with the full install. Heavy
submodules whose dependencies live in optional extras are imported under
try/except so a splitter-only install (no torch, dgl, astartes, statsmodels,
POT, etc.) still imports cleanly. Users who need those symbols on a lean
install should import them from their submodule directly, e.g.::

    from alinemol.utils.utils import load_model
"""

# Light submodules: no heavy dependencies, always importable.
from alinemol.utils.plot_utils import plot_ID_OOD, plot_ID_OOD_sns, visualize_chemspace

# metric_utils requires torch (extra: [gnn]) + statsmodels (extra: [ml]).
try:
    from alinemol.utils.metric_utils import (
        Meter,
        compute_linear_fit,
        eval_acc,
        eval_pr_auc,
        eval_roc_auc,
        rescale,
        compare_rankings,
    )
except ImportError:
    pass

# split_utils requires astartes (extra: [ml]).
try:
    from alinemol.utils.split_utils import (
        compute_similarities,
        featurize,
        split_molecules_train_test,
        split_molecules_train_val_test,
    )
except ImportError:
    pass

# utils.utils requires torch/dgl/dgllife (extra: [gnn]).
try:
    from alinemol.utils.utils import (
        collate_molgraphs,
        collate_molgraphs_unlabeled,
        get_configure,
        init_featurizer,
        init_inference_trial_path,
        init_trial_path,
        load_dataset,
        load_model,
        mkdir_p,
        predict,
        split_dataset,
    )
except ImportError:
    pass
