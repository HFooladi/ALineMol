import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from typing import Dict, Tuple

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def eval_roc_auc(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
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


def eval_pr_auc(df1: pd.DataFrame, df2:pd.DataFrame) -> float:
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

def eval_acc(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float=0.5) -> float:
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


def rescale(data, scaling=None) -> np.ndarray:
    """Rescale the data.
    
    Args:
        data (list, np.ndarray): data to be rescaled.
        scaling (str): scaling method. Options: 'probit', 'logit', 'linear'.
    
    Returns:
        np.ndarray: rescaled data.
    """
    data = np.asarray(data)
    if scaling == "probit":
        return norm.ppf(data)
    elif scaling == "logit":
        return np.log(data / (1 - data))
    elif scaling == "linear":
        return data
    raise NotImplementedError

##TODO: Needs to check whether better method is available for this function or not.
def compute_linear_fit(x, y) -> Tuple:
    """Returns bias and slope from regression y on x.

    Args:
        x (list, np.ndarray): x values.
        y (list, np.ndarray): y values.
    
    Returns:
        tuple: first element is parameters and second element is rsquared.
    """
    x = np.array(x)
    y = np.array(y)

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared