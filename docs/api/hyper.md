# `alinemol.hyper`

Hyperparameter search spaces for the GNN model families, used with
[hyperopt](https://github.com/hyperopt/hyperopt)-style tuning in the training
scripts.

```python
from alinemol.hyper import init_hyper_space

space = init_hyper_space("GCN")   # search space for the GCN model
```

## Building a search space

::: alinemol.hyper.hyper.init_hyper_space

## Predefined search spaces

`alinemol.hyper.hyper` also defines per-model hyperparameter dictionaries that
`init_hyper_space` selects from:

| Object | Model |
|---|---|
| `common_hyperparameters` | Shared across models (learning rate, dropout, …) |
| `gcn_hyperparameters` | Graph Convolutional Network |
| `gat_hyperparameters` | Graph Attention Network |
| `weave_hyperparameters` | Weave |
| `mpnn_hyperparameters` | Message Passing Neural Network |
| `attentivefp_hyperparameters` | AttentiveFP |
| `gin_pretrained_hyperparameters` | Pretrained GIN |
| `nf_hyperparameters` | Neural Fingerprint |

See [`scripts/clf_train_gnn.py`](../development.md) for how these feed into the
training loop.
