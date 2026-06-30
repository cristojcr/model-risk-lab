"""
Performance Validation Module

This module provides performance validation metrics commonly used in
Model Risk Management (MRM) and Credit Risk Model Validation.

Available metrics:
- ROC AUC
- Gini Coefficient
- Kolmogorov-Smirnov (KS)
- Brier Score

Author:
Model Risk Lab
"""

from .validator import PerformanceValidator
from .report import PerformanceValidationReport

from .metrics import (
    calculate_auc,
    calculate_gini,
    calculate_ks,
    calculate_brier_score,
)

__all__ = [
    "calculate_auc",
    "calculate_gini",
    "calculate_ks",
    "calculate_brier_score",
]
