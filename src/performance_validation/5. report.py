from dataclasses import dataclass


@dataclass
class PerformanceValidationReport:

    auc: float

    gini: float

    ks: float

    brier: float

    status: str

    findings: list

    recommendations: list

    def summary(self):

        return f"""
Performance Validation Report

AUC ............ {self.auc:.4f}

Gini ........... {self.gini:.4f}

KS ............. {self.ks:.4f}

Brier Score .... {self.brier:.4f}

Overall Status . {self.status}
"""
