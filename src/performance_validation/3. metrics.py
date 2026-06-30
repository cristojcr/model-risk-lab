"""
Performance metrics used during model validation.

This module provides functions for evaluating the predictive
performance of statistical and machine learning models.

Author
------
Model Risk Lab
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

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
