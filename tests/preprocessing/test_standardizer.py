"""
Test suite for the molecular standardization module.

This module contains tests for the standardization pipeline, including:
- SMILES standardization
- Duplicate removal
- Complete standardization pipeline
"""

import pandas as pd

from alinemol.preprocessing import standardize_smiles, drop_duplicates, standardization_pipeline


def test_standardize_smiles(test_dataset_dili: pd.DataFrame) -> None:
    """
    Test the SMILES standardization function.

    This test verifies that:
    1. The standardization process maintains the expected number of molecules
    2. The output DataFrame has the correct number of columns
    3. The required columns ('canonical_smiles' and 'label') are present

    Args:
        test_dataset_dili (pd.DataFrame): Test dataset containing SMILES strings and labels
    """
    # Apply standardization
    standardized_data = standardize_smiles(test_dataset_dili)

    # Verify output dimensions
    assert standardized_data.shape[0] == 474, "Number of molecules should remain unchanged"
    assert standardized_data.shape[1] == 5, "Output should have 5 columns"

    # Verify required columns
    assert "canonical_smiles" in standardized_data.columns, "Missing 'canonical_smiles' column"
    assert "label" in standardized_data.columns, "Missing 'label' column"


def test_drop_duplicates_standardized(test_dataset_dili_standardize: pd.DataFrame) -> None:
    """
    Test duplicate removal on standardized data.

    This test verifies that:
    1. The duplicate removal process maintains the expected number of molecules
    2. No duplicates remain in the dataset
    3. The standardization process was successful

    Args:
        test_dataset_dili_standardize (pd.DataFrame): Pre-standardized test dataset
    """
    # Remove duplicates
    unique_data = drop_duplicates(test_dataset_dili_standardize)

    # Verify output dimensions
    assert unique_data.shape[0] == 474, "Number of molecules should remain unchanged"
    assert unique_data.duplicated().sum() == 0, "No duplicates should remain"


def test_drop_duplicates_manual(manual_df_for_drop_duplicate: pd.DataFrame) -> None:
    """
    Test duplicate removal on manually created test data.

    This test verifies that:
    1. The duplicate removal process maintains the correct number of columns
    2. No duplicates remain in the dataset
    3. The correct number of unique molecules is retained

    Args:
        manual_df_for_drop_duplicate (pd.DataFrame): Manually created test dataset with known duplicates
    """
    # Remove duplicates
    unique_data = drop_duplicates(manual_df_for_drop_duplicate)

    # Verify output dimensions and content
    assert manual_df_for_drop_duplicate.shape[1] == unique_data.shape[1], "Number of columns should remain unchanged"
    assert unique_data.duplicated().sum() == 0, "No duplicates should remain"
    assert unique_data.shape[0] == 2, "Should retain exactly 2 unique molecules"


def test_standardization_pipeline(test_dataset_dili: pd.DataFrame) -> None:
    """
    Test the complete standardization pipeline.

    This test verifies that:
    1. The input dataset has the correct format
    2. The pipeline maintains the expected number of molecules
    3. The output has the correct structure
    4. No duplicates remain in the final dataset

    Args:
        test_dataset_dili (pd.DataFrame): Test dataset containing SMILES strings and labels
    """
    # Verify input format
    assert test_dataset_dili.shape[0] == 474, "Input should contain 474 molecules"
    assert test_dataset_dili.shape[1] == 2, "Input should have 2 columns"
    assert "smiles" in test_dataset_dili.columns, "Missing 'smiles' column"
    assert "label" in test_dataset_dili.columns, "Missing 'label' column"

    # Apply standardization pipeline
    standardized_data = standardization_pipeline(test_dataset_dili)

    # Verify output format
    assert standardized_data.shape[0] == 474, "Number of molecules should remain unchanged"
    assert standardized_data.shape[1] == 2, "Output should have 2 columns"
    assert "smiles" in standardized_data.columns, "Missing 'smiles' column"
    assert "label" in standardized_data.columns, "Missing 'label' column"
    assert standardized_data.duplicated().sum() == 0, "No duplicates should remain"
