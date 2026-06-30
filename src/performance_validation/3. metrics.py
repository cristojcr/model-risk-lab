"""
Performance metrics used during model validation.

This module provides functions for evaluating the predictive
performance of statistical and machine learning models.

Author
------
Model Risk Lab
"""

from __future__      import annotations
from scipy.stats     import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

import numpy as np


from common.metric_result import MetricResult


def calculate_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> MetricResult:
    """
    Calculate the Area Under the ROC Curve (AUC).

    Parameters
    ----------
    y_true
        Binary target values (0 or 1).

    y_score
        Predicted probabilities.

    threshold
        Optional minimum acceptable AUC.

    Returns
    -------
    MetricResult
        Object containing the metric result.
    """

    auc = roc_auc_score(y_true, y_score)

    passed = None

    if threshold is not None:
        passed = auc >= threshold

    return MetricResult(
        name="AUC",
        value=float(auc),
        threshold=threshold,
        passed=passed,
    )

def calculate_gini(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> MetricResult:
    """
    Calculate the Gini Coefficient.

    The Gini coefficient is derived from the ROC AUC:

        Gini = 2 × AUC − 1

    Parameters
    ----------
    y_true
        Binary target values (0 or 1).

    y_score
        Predicted probabilities.

    threshold
        Optional minimum acceptable Gini.

    Returns
    -------
    MetricResult
        Object containing the metric result.
    """

    auc_result = calculate_auc(y_true, y_score)

    gini = (2 * auc_result.value) - 1

    passed = None

    if threshold is not None:
        passed = gini >= threshold

    return MetricResult(
        name="Gini",
        value=float(gini),
        threshold=threshold,
        passed=passed,
    )
def calculate_ks(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> MetricResult:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic.

    The KS statistic measures the maximum separation between the
    cumulative distributions of predicted probabilities for the
    positive and negative classes.

    Parameters
    ----------
    y_true
        Binary target values (0 or 1).

    y_score
        Predicted probabilities.

    threshold
        Optional minimum acceptable KS.

    Returns
    -------
    MetricResult
        Object containing the KS statistic.
    """

    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]

    ks_statistic, _ = ks_2samp(positives, negatives)

    passed = None

    if threshold is not None:
        passed = ks_statistic >= threshold

    return MetricResult(
        name="KS",
        value=float(ks_statistic),
        threshold=threshold,
        passed=passed,
    )

def calculate_brier_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> MetricResult:
    """
    Calculate the Brier Score.

    The Brier Score measures the mean squared difference
    between predicted probabilities and observed outcomes.

    Lower values indicate better calibrated predictions.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    threshold
        Optional maximum acceptable Brier Score.

    Returns
    -------
    MetricResult
        Object containing the Brier Score.
    """

    brier = brier_score_loss(y_true, y_score)

    passed = None

    if threshold is not None:
        passed = brier <= threshold

    return MetricResult(
        name="Brier Score",
        value=float(brier),
        threshold=threshold,
        passed=passed,
    )

def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
) -> np.ndarray:
    """
    Compute the Confusion Matrix.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    threshold
        Classification threshold used to convert
        probabilities into predicted classes.

    Returns
    -------
    numpy.ndarray
        Confusion matrix in the following format:

            [[TN FP]
             [FN TP]]
    """

    y_pred = (y_score >= threshold).astype(int)

    return confusion_matrix(y_true, y_pred)

def _get_confusion_values(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.50,
) -> tuple[int, int, int, int]:
    """
    Extract the values of the confusion matrix.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    threshold
        Classification threshold.

    Returns
    -------
    tuple
        (tn, fp, fn, tp)
    """

    cm = calculate_confusion_matrix(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
    )

    tn, fp, fn, tp = cm.ravel()

    return tn, fp, fn, tp

def calculate_accuracy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    threshold: float | None = None,
) -> MetricResult:
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    threshold
        Optional minimum acceptable Accuracy.

    Returns
    -------
    MetricResult
        Accuracy metric.
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true=y_true,
        y_score=y_score,
        threshold=classification_threshold,
    )

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    passed = None

    if threshold is not None:
        passed = accuracy >= threshold

    return MetricResult(
        name="Accuracy",
        value=float(accuracy),
        threshold=threshold,
        passed=passed,
    )

def calculate_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate Precision.

    Precision measures the proportion of predicted positives
    that are actually positive.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional minimum acceptable Precision.

    Returns
    -------
    MetricResult
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true,
        y_score,
        classification_threshold,
    )

    precision = 0.0

    if (tp + fp) > 0:
        precision = tp / (tp + fp)

    passed = None

    if validation_threshold is not None:
        passed = precision >= validation_threshold

    return MetricResult(
        name="Precision",
        value=float(precision),
        threshold=validation_threshold,
        passed=passed,
    )

def calculate_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate Recall (Sensitivity / True Positive Rate).

    Recall measures the proportion of actual positives
    correctly identified by the model.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional minimum acceptable Recall.

    Returns
    -------
    MetricResult
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true,
        y_score,
        classification_threshold,
    )

    recall = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn)

    passed = None

    if validation_threshold is not None:
        passed = recall >= validation_threshold

    return MetricResult(
        name="Recall",
        value=float(recall),
        threshold=validation_threshold,
        passed=passed,
    )

def calculate_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate the F1 Score.

    The F1 Score is the harmonic mean of Precision and Recall.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional minimum acceptable F1 Score.

    Returns
    -------
    MetricResult
        F1 Score metric.
    """

    precision = calculate_precision(
        y_true=y_true,
        y_score=y_score,
        classification_threshold=classification_threshold,
    ).value

    recall = calculate_recall(
        y_true=y_true,
        y_score=y_score,
        classification_threshold=classification_threshold,
    ).value

    f1 = 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    passed = None

    if validation_threshold is not None:
        passed = f1 >= validation_threshold

    return MetricResult(
        name="F1 Score",
        value=float(f1),
        threshold=validation_threshold,
        passed=passed,
    )

def calculate_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate Specificity (True Negative Rate).

    Specificity measures the proportion of actual negatives
    correctly identified by the model.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional minimum acceptable Specificity.

    Returns
    -------
    MetricResult
        Specificity metric.
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true=y_true,
        y_score=y_score,
        threshold=classification_threshold,
    )

    specificity = 0.0

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)

    passed = None

    if validation_threshold is not None:
        passed = specificity >= validation_threshold

    return MetricResult(
        name="Specificity",
        value=float(specificity),
        threshold=validation_threshold,
        passed=passed,
    )

def calculate_negative_predictive_value(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate Negative Predictive Value (NPV).

    NPV measures the proportion of predicted negatives
    that are actually negative.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional minimum acceptable NPV.

    Returns
    -------
    MetricResult
        Negative Predictive Value metric.
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true=y_true,
        y_score=y_score,
        threshold=classification_threshold,
    )

    npv = 0.0

    if (tn + fn) > 0:
        npv = tn / (tn + fn)

    passed = None

    if validation_threshold is not None:
        passed = npv >= validation_threshold

    return MetricResult(
        name="Negative Predictive Value",
        value=float(npv),
        threshold=validation_threshold,
        passed=passed,
    )

def calculate_false_positive_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classification_threshold: float = 0.50,
    validation_threshold: float | None = None,
) -> MetricResult:
    """
    Calculate False Positive Rate (FPR).

    FPR measures the proportion of actual negatives
    incorrectly classified as positives.

    Parameters
    ----------
    y_true
        Binary target values.

    y_score
        Predicted probabilities.

    classification_threshold
        Probability threshold used to classify observations.

    validation_threshold
        Optional maximum acceptable FPR.

    Returns
    -------
    MetricResult
        False Positive Rate metric.
    """

    tn, fp, fn, tp = _get_confusion_values(
        y_true=y_true,
        y_score=y_score,
        threshold=classification_threshold,
    )

    fpr = 0.0

    if (fp + tn) > 0:
        fpr = fp / (fp + tn)

    passed = None

    if validation_threshold is not None:
        passed = fpr <= validation_threshold

    return MetricResult(
        name="False Positive Rate",
        value=float(fpr),
        threshold=validation_threshold,
        passed=passed,
    )

