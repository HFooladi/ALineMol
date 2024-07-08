import pytest

from alinemol.preprocessing.standardizer import standardize_smiles, drop_duplicates


def test_drop_duplicates(manual_df_for_drop_duplicate):
    df = manual_df_for_drop_duplicate
    df = standardize_smiles(df)
    df = df[["canonical_smiles", "label"]].rename(columns={"canonical_smiles": "smiles"})
    df = drop_duplicates(df)
    assert df.shape[0] == 2
    assert df.duplicated().sum() == 0
