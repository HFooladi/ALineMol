from typing import Dict, List, Tuple
from typing_extensions import Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn.functional as F
from scipy.stats import norm, pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
)


@dataclass(frozen=True)
class BinaryEvalMetrics:
    size: int
    acc: float
    balanced_acc: float
    f1: float
    prec: float
    recall: float
    roc_auc: float
    avg_precision: float
    kappa: float


BinaryMetricType = Literal["acc", "balanced_acc", "f1", "prec", "recall", "roc_auc", "avg_precision", "kappa"]


def compute_binary_task_metrics(predictions: List[float], labels: List[float]) -> BinaryEvalMetrics:
    """
    Compute binary classification evaluation metrics.

    Args:
        predictions (List[float]): list of predicted probabilities.
        labels (List[float]): list of true labels.
    
    Returns:
        BinaryEvalMetrics: evaluation metrics.
    """
    normalized_predictions = [pred >= 0.5 for pred in predictions]  # Normalise probabilities to bool values

    if np.sum(labels) == len(labels) or np.sum(labels) == 0:
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(labels, predictions)

    return BinaryEvalMetrics(
        size=len(predictions),
        acc=accuracy_score(labels, normalized_predictions),
        balanced_acc=balanced_accuracy_score(labels, normalized_predictions),
        f1=f1_score(labels, normalized_predictions, zero_division=1),
        prec=precision_score(labels, normalized_predictions, zero_division=1),
        recall=recall_score(labels, normalized_predictions, zero_division=1),
        roc_auc=roc_auc,
        avg_precision=average_precision_score(labels, predictions),
        kappa=cohen_kappa_score(labels, normalized_predictions),
    )


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


def eval_pr_auc(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
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


def eval_acc(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.5) -> float:
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


# pylint: disable=E1101
class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    When dealing with multitask learning, quite often we normalize the labels so they are
    roughly at a same scale. During the evaluation, we need to undo the normalization on
    the predicted labels. If mean and std are not None, we will undo the normalization.

    Currently we support evaluation with 4 metrics:

    * ``pearson r2``
    * ``mae``
    * ``rmse``
    * ``roc auc score``

    Args:
        mean : torch.float32 tensor of shape (T) or None.
            Mean of existing training labels across tasks if not ``None``. ``T`` for the
            number of tasks. Default to ``None`` and we assume no label normalization has been
            performed.
        std : torch.float32 tensor of shape (T)
            Std of existing training labels across tasks if not ``None``. Default to ``None``
            and we assume no label normalization has been performed.

    Examples:
        Below gives a demo for a fake evaluation epoch.

        >>> import torch
        >>> from dgllife.utils import Meter

        >>> meter = Meter()
        >>> # Simulate 10 fake mini-batches
        >>> for batch_id in range(10):
        >>>     batch_label = torch.randn(3, 3)
        >>>     batch_pred = torch.randn(3, 3)
        >>>     meter.update(batch_pred, batch_label)

        >>> # Get MAE for all tasks
        >>> print(meter.compute_metric('mae'))
        [1.1325558423995972, 1.0543707609176636, 1.094650149345398]
        >>> # Get MAE averaged over all tasks
        >>> print(meter.compute_metric('mae', reduction='mean'))
        1.0938589175542195
        >>> # Get the sum of MAE over all tasks
        >>> print(meter.compute_metric('mae', reduction='sum'))
        3.2815767526626587
    """

    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration

        Args:
            y_pred : float32 tensor
                Predicted labels with shape ``(B, T)``,
                ``B`` for number of graphs in the batch and ``T`` for the number of tasks
            y_true : float32 tensor
                Ground truth labels with shape ``(B, T)``
            mask : None or float32 tensor
                Binary mask indicating the existence of ground truth labels with
                shape ``(B, T)``. If None, we assume that all labels exist and create
                a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """Prepare for evaluation.

        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.

        Returns:
            mask : float32 tensor
                Binary mask indicating the existence of ground
                truth labels with shape (B, T), B for batch size
                and T for the number of tasks
            y_pred : float32 tensor
                Predicted labels with shape (B, T)
            y_true : float32 tensor
                Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction="none"):
        """Finalize the scores to return.

        Args:
            scores : list of float
                Scores for all tasks.
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == "none":
            return scores
        elif reduction == "mean":
            return np.mean(scores)
        elif reduction == "sum":
            return np.sum(scores)
        else:
            raise ValueError("Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction="none"):
        """Evaluate for multi-label prediction.

        Args:
            score_func : callable
                A score function that takes task-specific ground truth and predicted labels as
                input and return a float as the score. The labels are in the form of 1D tensor.
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson_r2(self, reduction="none"):
        """Compute squared Pearson correlation coefficient.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """

        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2

        return self.multilabel_score(score, reduction)

    def mae(self, reduction="none"):
        """Compute mean absolute error.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """

        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()

        return self.multilabel_score(score, reduction)

    def rmse(self, reduction="none"):
        """Compute root mean square error.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """

        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()

        return self.multilabel_score(score, reduction)

    def accuracy_score(self, reduction="none", threshold=0.5):
        """Compute the accuracy score for binary classification.
        Accuracy scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks.
            threshold (float): threshold for binary classification.

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (
            self.std is None
        ), "Label normalization should not be performed for binary classification."

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print(
                    "Warning: Only one class {} present in y_true for a task. "
                    "ROC AUC score is not defined in that case.".format(y_true[0])
                )
                return None
            else:
                return accuracy_score(y_true.long().numpy(), (torch.sigmoid(y_pred) > threshold).numpy())

        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction="none"):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks.

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (
            self.std is None
        ), "Label normalization should not be performed for binary classification."

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print(
                    "Warning: Only one class {} present in y_true for a task. "
                    "ROC AUC score is not defined in that case.".format(y_true[0])
                )
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())

        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction="none"):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.

        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.

        Args:
            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks.

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (
            self.std is None
        ), "Label normalization should not be performed for binary classification."

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print(
                    "Warning: Only one class {} present in y_true for a task. "
                    "PR AUC score is not defined in that case.".format(y_true[0])
                )
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy()
                )
                return auc(recall, precision)

        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction="none"):
        """Compute metric based on metric name.

        Args:
            metric_name : str

                * ``'r2'``: compute squared Pearson correlation coefficient
                * ``'mae'``: compute mean absolute error
                * ``'rmse'``: compute root mean square error
                * ``'roc_auc_score'``: compute roc-auc score
                * ``'pr_auc_score'``: compute pr-auc score

            reduction : 'none' or 'mean' or 'sum'
                Controls the form of scores for all tasks

        Returns:
            float or list of float
                * If ``reduction == 'none'``, return the list of scores for all tasks.
                * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == "r2":
            return self.pearson_r2(reduction)
        elif metric_name == "mae":
            return self.mae(reduction)
        elif metric_name == "rmse":
            return self.rmse(reduction)
        elif metric_name == "accuracy_score":
            return self.accuracy_score(reduction)
        elif metric_name == "roc_auc_score":
            return self.roc_auc_score(reduction)
        elif metric_name == "pr_auc_score":
            return self.pr_auc_score(reduction)
        else:
            raise ValueError(
                'Expect metric_name to be "r2" or "mae" or "rmse" '
                'or "accuracy_score" or "roc_auc_score" or "pr_auc", got {}'.format(metric_name)
            )
