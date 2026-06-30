"""
Represents a validation finding identified during model validation.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Finding:
    """
    Represents an issue identified during model validation.

    Attributes
    ----------
    title : str
        Short description of the finding.

    severity : str
        Severity level (Low, Medium, High or Critical).

    description : str
        Detailed explanation.

    recommendation : str
        Suggested action to address the finding.
    """

    title: str
    severity: str
    description: str
    recommendation: str
