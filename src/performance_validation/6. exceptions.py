"""
Custom exceptions for Performance Validation.
"""


class PerformanceValidationError(Exception):
    """Base exception for Performance Validation."""


class MissingColumnError(PerformanceValidationError):
    """Required column not found."""


class InvalidProbabilityError(PerformanceValidationError):
    """Predicted probabilities must be between 0 and 1."""


class DifferentLengthError(PerformanceValidationError):
    """Arrays have different lengths."""


class MissingValuesError(PerformanceValidationError):
    """Input data contains missing values."""
