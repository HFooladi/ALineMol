# `alinemol.preprocessing`

Utilities for cleaning and standardizing molecular datasets before splitting or
model training. The public entry points wrap RDKit-based normalization so that
duplicate, malformed, or non-canonical SMILES do not leak noise into your splits.

```python
from alinemol.preprocessing import standardization_pipeline

clean_df = standardization_pipeline(raw_df)   # canonicalize + de-duplicate
```

## Pipeline functions

::: alinemol.preprocessing.standardize_smiles

::: alinemol.preprocessing.drop_duplicates

::: alinemol.preprocessing.standardization_pipeline

## Standardizer

The `Standardizer` class implements the underlying normalization steps
(sanitization, salt/solvent stripping, tautomer canonicalization).

::: alinemol.preprocessing.standardizer.Standardizer
