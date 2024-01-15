import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def eval_roc_auc(df1, df2):
    """
    Evaluate ROC AUC score.

    Args:
        df1 (pd.DataFrame): dataframe containing true labels.
        df2 (pd.DataFrame): dataframe containing predicted labels.

    Returns:
        float: ROC AUC score.
    """
    assert "label" in df1.columns, "Dataframe must have a 'label' column."
    assert "label" in df2.columns, "Dataframe must have a 'label' column."

    y_true = df1["label"].values
    y_pred = df2["label"].values

    return roc_auc_score(y_true, y_pred)


def eval_pr_auc(df1, df2):
    """
    Evaluate PR AUC score.

    Args:
        df1 (pd.DataFrame): dataframe containing true labels.
        df2 (pd.DataFrame): dataframe containing predicted labels.

    Returns:
        float: PR AUC score.
    """
    assert "label" in df1.columns, "Dataframe must have a 'label' column."
    assert "label" in df2.columns, "Dataframe must have a 'label' column."

    y_true = df1["label"].values
    y_pred = df2["label"].values

    return average_precision_score(y_true, y_pred)

def eval_acc(df1, df2, threshold=0.5):
    """
    Evaluate accuracy score.

    Args:
        df1 (pd.DataFrame): dataframe containing true labels.
        df2 (pd.DataFrame): dataframe containing predicted labels.
        threshold (float): threshold for binary classification.

    Returns:
        float: accuracy score.
    """
    assert "label" in df1.columns, "Dataframe must have a 'label' column."
    assert "label" in df2.columns, "Dataframe must have a 'label' column."

    y_true = df1["label"].values
    y_pred = df2["label"].values >= threshold

    return accuracy_score(y_true, y_pred)