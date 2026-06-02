from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def compute_expected_risk(
    occurrence_probs: pd.Series,
    severity_preds: pd.Series,
) -> pd.Series:
    return occurrence_probs * severity_preds


def evaluate_risk_ranking(
    risk_scores: pd.Series,
    y_true_severity: pd.Series,
    y_true_occurrence: pd.Series,
) -> dict[str, float | int | str]:
    mask = (y_true_occurrence == 1) & y_true_severity.notna()
    if int(mask.sum()) < 10:
        return {
            "warning": "Insufficient severity samples for risk evaluation",
            "n_evaluable": int(mask.sum()),
        }
    rho, p_value = spearmanr(risk_scores[mask], y_true_severity[mask])
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "n_evaluable": int(mask.sum()),
    }
