# `alinemol.models`

ALineMol provides two model families for the ID/OOD benchmarks: a classical
machine-learning baseline and fragment-aware graph neural networks.

## Classical ML baseline

`CML` is a light wrapper around scikit-learn's `RandomForestClassifier` exposing
the standard estimator API. It requires no deep-learning dependencies and serves
as the in-distribution reference model.

```python
from alinemol.models.cml import CML

clf = CML()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

::: alinemol.models.cml.CML

## Graph neural networks

!!! note "Requires the `[gnn]` extra"
    The GNN models depend on `torch`, `torch-geometric`, DGL/DGL-LifeSci, and
    `torch-scatter`. Install with `pip install -e ".[gnn]"` (and a matching
    `torch-scatter` wheel for your torch/CUDA build). They are described here
    rather than auto-documented so this documentation builds without the heavy
    GNN stack.

### `FragGNN`

`alinemol.models.fragGNN.FragGNN` is a fragment-aware graph neural network built
on GIN/GINE convolutions. It encodes atoms, bonds, and molecular fragments with
dedicated encoders and exchanges information between the atom- and fragment-level
representations via message passing. A smaller variant, `FragGNNSmall`, is
available for lighter-weight experiments.

Key building blocks live in `alinemol.models.layers`:

| Layer | Role |
|---|---|
| `AtomEncoder` | Embeds atom features |
| `BondEncoder` | Embeds bond features |
| `FragEncoder` | Embeds fragment-level features |
| `InterMessage` | Passes messages between atom and fragment views |
| `MLP` | Generic multi-layer perceptron head |

See the [training scripts](../development.md) (`scripts/clf_train_gnn.py`) for an
end-to-end GNN training example, and the
[Hyperparameters reference](hyper.md) for the search spaces used to tune them.

## Source

Full model source is on
[GitHub](https://github.com/HFooladi/ALineMol/tree/main/alinemol/models).
