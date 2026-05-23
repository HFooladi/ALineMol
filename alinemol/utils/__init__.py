"""ALineMol utilities.

Submodules are imported lazily to keep light-weight workflows (e.g. the
splitter-only quick-start notebook) from paying the cost of heavy GNN
dependencies (torch, dgl, dgllife). Users who need those symbols should
import them from their submodule directly, e.g.::

    from alinemol.utils.utils import load_model

The legacy ``from alinemol.utils import load_model`` style still works as
long as the GNN dependencies are installed; if they aren't, those re-exports
are silently skipped instead of raising at package import.
"""

from alinemol.utils.metric_utils import (
    Meter,
    compute_linear_fit,
    eval_acc,
    eval_pr_auc,
    eval_roc_auc,
    rescale,
    compare_rankings,
)
from alinemol.utils.plot_utils import plot_ID_OOD, plot_ID_OOD_sns, visualize_chemspace
from alinemol.utils.split_utils import (
    compute_similarities,
    featurize,
    split_molecules_train_test,
    split_molecules_train_val_test,
)

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
    pass  # GNN dependencies (dgl, torch, dgllife) not available — import directly if needed
