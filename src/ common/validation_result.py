"""
Represents the overall result of a validation module.
"""

from dataclasses import dataclass, field

from .metric_result import MetricResult
from .finding import Finding


@dataclass(slots=True)
class ValidationResult:
    """
    Represents the output of a validation module.

    Attributes
    ----------
    module : str
        Name of the validation module.

    status : str
        Overall validation status (PASS, WARNING or FAIL).

    metrics : list[MetricResult]
        List of calculated metrics.

    findings : list[Finding]
        Validation findings identified during the assessment.
    """

    module: str

    status: str = "PASS"

    metrics: list[MetricResult] = field(default_factory=list)

    findings: list[Finding] = field(default_factory=list)
