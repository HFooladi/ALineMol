import pytest

from alinemol.preprocessing import standardize_smiles, drop_duplicates, standardization_pipline

# Test 1: Test standardize_smiles function

def test_standardize_smiles(test_dataset_dili):
    df = test_dataset_dili
    df = standardize_smiles(df)
    assert df.shape[0] == 474
    assert df.shape[1] == 5
    assert "canonical_smiles" in df.columns
    assert "label" in df.columns

# Test 2: Test drop_duplicates function
def test_drop_duplicates(test_dataset_dili_standardize):
    df = test_dataset_dili_standardize
    df = drop_duplicates(df)
    assert df.shape[0] == 474
    assert df.duplicated().sum() == 0

# Test 3: Test drop_duplicates function
def test_drop_duplicates(manual_df_for_drop_duplicate):
    df = manual_df_for_drop_duplicate
    df = drop_duplicates(df)
    # Check if the number of columns is the same (drop duplicate should not remove/add any columns)
    assert manual_df_for_drop_duplicate.shape[1] == df.shape[1]
    # Check if the number of duplicates is 0
    assert df.duplicated().sum() == 0
    # Check if the number of rows is correct
    assert df.shape[0] == 2



# Test 4: Test standardization_pipline function
def test_standardization_pipline(test_dataset_dili):
    assert test_dataset_dili.shape[0] == 474
    assert test_dataset_dili.shape[1] == 2
    assert "smiles" in test_dataset_dili.columns
    assert "label" in test_dataset_dili.columns
    df = test_dataset_dili
    df = standardization_pipline(df)
    assert df.shape[0] == 474
    assert df.shape[1] == 2
    assert "smiles" in df.columns
    assert "label" in df.columns
    assert df.duplicated().sum() == 0




