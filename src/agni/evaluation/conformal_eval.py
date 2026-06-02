from __future__ import annotations

import pandas as pd

from agni.models.conformal import ConformalRiskSet


def conformal_sets_to_frame(sets: list[ConformalRiskSet]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "risk_point": item.risk_point,
                "risk_lower": item.risk_lower,
                "risk_upper": item.risk_upper,
                "coverage_level": item.coverage_level,
            }
            for item in sets
        ]
    )


def summarize_high_confidence_alerts(sets: list[ConformalRiskSet], threshold: float) -> dict[str, float]:
    frame = conformal_sets_to_frame(sets)
    confident = frame["risk_lower"] > threshold
    return {
        "threshold": float(threshold),
        "n_confident_alerts": int(confident.sum()),
        "fraction_confident_alerts": float(confident.mean()) if len(frame) else 0.0,
        "mean_confident_lower": float(frame.loc[confident, "risk_lower"].mean()) if confident.any() else 0.0,
    }
