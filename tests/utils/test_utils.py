import pandas as pd
from alinemol.utils.utils import compute_difference

def test_compute_difference():
    # Create a sample results dataframe
    results = pd.DataFrame({
        "model": ["Model A", "Model A", "Model B", "Model B"],
        "ID_test_accuracy": [0.8, 0.7, 0.9, 0.6],
        "OOD_test_accuracy": [0.6, 0.5, 0.7, 0.4],
        "ID_test_roc_auc": [0.85, 0.75, 0.95, 0.65],
        "OOD_test_roc_auc": [0.65, 0.65, 0.75, 0.65],
        "ID_test_pr_auc": [0.75, 0.65, 0.85, 0.55],
        "OOD_test_pr_auc": [0.70, 0.60, 0.65, 0.35]
    })

    # Compute the difference
    diff = compute_difference(results, metrics=["accuracy", "roc_auc", "pr_auc"])

    # Check the expected output
    expected_diff = pd.DataFrame({
        "diff_accuracy": [0.2, 0.2],
        "diff_roc_auc": [0.15, 0.1],
        "diff_pr_auc": [0.05, 0.2]
    }, index=["Model A", "Model B"])

    assert diff.round(3).equals(expected_diff.round(3))
