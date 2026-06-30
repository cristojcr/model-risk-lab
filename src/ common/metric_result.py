"""
Shared objects used across the Model Risk Lab framework.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class MetricResult:
    """
    Represents the result of a validation metric.

    Attributes
    ----------
    name : str
        Metric name.

    value : float
        Metric value.

    threshold : float | None
        Threshold used during validation.

    passed : bool | None
        Indicates whether the metric passed the validation threshold.
    """

    name: str
    value: float
    threshold: float | None = None
    passed: bool | None = None
