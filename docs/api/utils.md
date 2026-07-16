# `alinemol.utils`

Helper functions for metrics, plotting, featurization, and model training used
throughout the ID/OOD evaluation pipeline.

!!! note "Optional extras"
    `alinemol.utils` re-exports symbols from several submodules. The plotting
    helpers are always available, but the metric, split, and training helpers
    depend on optional extras (`torch`/DGL from `[gnn]`, `statsmodels`/`astartes`
    from `[ml]`). On a lean install, import the always-available symbols from
    `alinemol.utils` and the rest from their submodule directly, e.g.
    `from alinemol.utils.utils import load_model`.

## Plotting

::: alinemol.utils.plot_utils.plot_ID_OOD

::: alinemol.utils.plot_utils.plot_ID_OOD_sns

::: alinemol.utils.plot_utils.visualize_chemspace

## Metrics

::: alinemol.utils.metric_utils.eval_roc_auc

::: alinemol.utils.metric_utils.eval_pr_auc

::: alinemol.utils.metric_utils.eval_acc

::: alinemol.utils.metric_utils.compute_linear_fit

::: alinemol.utils.metric_utils.compare_rankings

::: alinemol.utils.metric_utils.rescale

::: alinemol.utils.metric_utils.Meter

## Splitting helpers

::: alinemol.utils.split_utils.compute_similarities

::: alinemol.utils.split_utils.featurize

::: alinemol.utils.split_utils.split_molecules_train_test

::: alinemol.utils.split_utils.split_molecules_train_val_test

## Training & data loading

::: alinemol.utils.utils.load_dataset

::: alinemol.utils.utils.load_model

::: alinemol.utils.utils.init_featurizer

::: alinemol.utils.utils.get_configure

::: alinemol.utils.utils.collate_molgraphs

::: alinemol.utils.utils.predict

::: alinemol.utils.utils.split_dataset
