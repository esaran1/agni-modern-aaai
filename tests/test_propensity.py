from __future__ import annotations

import numpy as np
import pandas as pd

from agni.models.propensity_severity import NaiveSeverityModel, PropensityWeightedSeverityModel


def _severity_frame() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    rng = np.random.default_rng(42)
    for idx in range(120):
        propensity = rng.uniform(0.05, 0.95)
        dryness = rng.normal()
        rare_fire_boost = (1.0 - propensity) * 2.0
        severity = 0.8 * dryness + rare_fire_boost + rng.normal(scale=0.1)
        rows.append(
            {
                "weather_vpd_mean_l7d": dryness + propensity,
                "terrain_twi_mean": 1.0 - dryness * 0.1,
                "propensity_score": propensity,
                "y_sev_dnbr": severity,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, ["weather_vpd_mean_l7d", "terrain_twi_mean"]


def test_ipw_weights_clipped() -> None:
    model = PropensityWeightedSeverityModel({}, clip_min=0.05, clip_max=0.95, normalize_weights=False)
    scores = np.array([0.01, 0.5, 0.99])
    weights = model.compute_ipw_weights(scores)
    assert weights[0] == 1.0 / 0.05
    assert weights[2] == 1.0 / 0.95


def test_ipw_weights_normalized() -> None:
    model = PropensityWeightedSeverityModel({}, normalize_weights=True)
    scores = np.array([0.1, 0.3, 0.5, 0.8])
    weights = model.compute_ipw_weights(scores)
    assert abs(weights.sum() - len(scores)) < 1e-6


def test_naive_vs_ipw_different_predictions() -> None:
    frame, feature_columns = _severity_frame()
    train = frame.iloc[:80].copy()
    val = frame.iloc[80:100].copy()
    test = frame.iloc[100:].copy()

    naive = NaiveSeverityModel({"params": {"n_estimators": 40, "max_depth": 3}})
    naive.fit(train, val, feature_columns, "y_sev_dnbr")
    ipw = PropensityWeightedSeverityModel({"params": {"n_estimators": 40, "max_depth": 3}})
    ipw.fit(train, val, feature_columns, "y_sev_dnbr", propensity_column="propensity_score")

    naive_pred = naive.predict(test, feature_columns)
    ipw_pred = ipw.predict(test, feature_columns)
    assert not np.allclose(naive_pred, ipw_pred)
