from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RANKING_FRACTIONS = (0.10, 0.20)


def compute_expected_risk(
    occurrence_probs: pd.Series,
    severity_preds: pd.Series,
) -> pd.Series:
    return occurrence_probs * severity_preds


def _population_ranking_metrics(
    risk: np.ndarray,
    occurrence: np.ndarray,
    severity: np.ndarray,
) -> dict[str, float | int]:
    """Operational top-k ranking quality over the whole test population.

    Unlike the conditional Spearman correlation (which only looks at realized
    fires), these metrics evaluate how well the risk score concentrates actual
    fires and realized burn severity into the highest-ranked patches, which is
    what matters for a finite alerting/inspection budget.
    """
    n = int(risk.shape[0])
    metrics: dict[str, float | int] = {"n_test": n}
    if n == 0:
        return metrics

    order = np.argsort(-risk, kind="stable")
    occurrence_sorted = occurrence[order]
    severity_sorted = np.nan_to_num(severity[order], nan=0.0)
    total_severity = float(severity_sorted.sum())

    for fraction in RANKING_FRACTIONS:
        k = max(1, int(round(fraction * n)))
        pct = int(round(fraction * 100))
        metrics[f"precision_at_{pct}pct"] = float(np.mean(occurrence_sorted[:k] == 1))
        if total_severity > 0.0:
            metrics[f"severity_capture_at_{pct}pct"] = float(
                severity_sorted[:k].sum() / total_severity
            )
        else:
            metrics[f"severity_capture_at_{pct}pct"] = float("nan")
    return metrics


def evaluate_risk_ranking(
    risk_scores: pd.Series,
    y_true_severity: pd.Series,
    y_true_occurrence: pd.Series,
) -> dict[str, float | int | str]:
    risk = np.asarray(risk_scores, dtype=float)
    occurrence = np.asarray(y_true_occurrence, dtype=float)
    severity = np.asarray(y_true_severity, dtype=float)

    metrics: dict[str, float | int | str] = {}
    metrics.update(_population_ranking_metrics(risk, occurrence, severity))

    evaluable = (occurrence == 1) & ~np.isnan(severity)
    n_evaluable = int(evaluable.sum())
    metrics["n_evaluable"] = n_evaluable
    if n_evaluable < 10:
        metrics["warning"] = "Insufficient severity samples for conditional risk evaluation"
        return metrics

    rho, p_value = spearmanr(risk[evaluable], severity[evaluable])
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_p"] = float(p_value)
    return metrics
