# Splitting Strategies

ALineMol ships 16 splitting strategies behind a single **SMILES-first** API.
Every splitter is created the same way — via the
[`get_splitter()`](../api/splitters.md) factory or by direct class import — and
every splitter's `split()` method accepts a list of SMILES:

```python
from alinemol.splitters import get_splitter

splitter = get_splitter("scaffold", n_splits=5, test_size=0.2)
for train_idx, test_idx in splitter.split(smiles):
    ...
```

This page is a conceptual tour organized by the *kind* of distribution shift each
family produces. For full signatures and parameters, see the
[Splitters API reference](../api/splitters.md).

## Structure-based

Hold out whole structural motifs so the test set contains chemistry the model has
never seen. The strongest, most commonly used notion of molecular OOD.

| Splitter | Idea | Use when |
|---|---|---|
| `scaffold` | Bemis–Murcko scaffold split | Testing generalization to novel scaffolds |
| `scaffold_generic` | Generic (graph-only) scaffold split | Even stricter scaffold generalization |
| `butina` | Taylor–Butina fingerprint clustering | Cluster-disjoint train/test by similarity |

## Property-based

Introduce a systematic shift along a physicochemical axis — a realistic setting
when a screening library skews toward one region of property space.

| Splitter | Idea | Use when |
|---|---|---|
| `molecular_weight` | Train small → test large | Extrapolating to heavier molecules |
| `molecular_weight_reverse` | Train large → test small | The reverse extrapolation |
| `molecular_logp` | Split by lipophilicity (LogP) | Shift in lipophilicity/solubility |

## Clustering-based

Partition chemical space (fingerprint or embedding) into clusters and split along
cluster boundaries, so train and test occupy different regions.

| Splitter | Idea | Use when |
|---|---|---|
| `kmeans` | K-means on ECFP fingerprints | General chemical-space shift |
| `umap` | UMAP embedding + hierarchical clustering | Non-linear chemical-space structure |
| `max_dissimilarity` | Maximum-dissimilarity selection | Worst-case dissimilar test set |
| `perimeter` | Perimeter/boundary sampling | Test on the edges of chemical space |
| `scaffold_kmeans` | K-means on scaffold ECFP | Scaffold-aware clustering |

## Similarity-based

Explicitly bound the train↔test similarity — the setting closest to lead
optimization, where you want to know how models behave near (or far from) known
actives.

| Splitter | Idea | Use when |
|---|---|---|
| `hi` | Ensures **low** train/test similarity | Hard generalization test |
| `lo` | Lead-optimization-style split | Modeling analog series |

## Baseline & advanced

| Splitter | Idea | Use when |
|---|---|---|
| `random` | Random (in-distribution) split | The ID reference every OOD split is compared against |
| `datasail` | DataSAIL-based disjoint splitting | Advanced, information-leakage-aware splits |

## Choosing a strategy

- Start with **`random`** (ID baseline) and **`scaffold`** (the canonical OOD
  test). The gap between them is the headline generalization number.
- Add **`kmeans`** or **`umap`** to probe chemical-space shift, and
  **`molecular_weight`** / **`molecular_logp`** for property extrapolation.
- Use **`hi`** for a deliberately hard, similarity-bounded test.
- Whatever you pick, confirm it actually produced a shift with
  [`SplitAnalyzer`](split-analysis.md) — the splitter name is a *hypothesis*, not
  a guarantee.

<figure markdown>
  ![Model performance by splitter](../assets/figures/model_performance_by_splitter.png){ width="720" }
  <figcaption>Model performance varies substantially with the splitting
  strategy — random splits overstate deployable performance.</figcaption>
</figure>

!!! info "Adding your own splitter"
    New splitters register themselves via the `@register_splitter` decorator and
    become available through `get_splitter()` automatically. See the
    contributor notes in `CLAUDE.md` and the
    [Splitters API](../api/splitters.md) for the base classes.
