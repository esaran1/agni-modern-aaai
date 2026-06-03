from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from agni.experiment_utils import attach_occurrence_propensity, train_model_on_existing_split
from agni.models.propensity_severity import NaiveSeverityModel, PropensityWeightedSeverityModel


def _severity_frame() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    rng = np.random.default_rng(42)
    for _idx in range(120):
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
    model = PropensityWeightedSeverityModel(
        {},
        clip_min=0.05,
        clip_max=0.95,
        normalize_weights=False,
    )
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


def test_attach_occurrence_propensity_populates_all_splits() -> None:
    rows = []
    for patch in range(6):
        for step in range(6):
            split = "train" if step < 3 else "val" if step == 3 else "test"
            rows.append(
                {
                    "patch_id": f"{patch}",
                    "reference_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * step),
                    "split": split,
                    "weather_vpd_mean_l7d": patch + step,
                    "terrain_twi_mean": patch - step / 10.0,
                    "y_occ_30d": int((patch + step) % 2 == 0),
                }
            )
    df = pd.DataFrame(rows)
    result = attach_occurrence_propensity(
        df,
        horizon_days=30,
        feature_columns=["weather_vpd_mean_l7d", "terrain_twi_mean"],
        model_name="logreg",
        model_params={"max_iter": 200},
    )
    assert result.predictions["propensity_score"].notna().all()


def test_attach_occurrence_propensity_uses_cross_fitted_train_scores() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for patch in range(12):
        for step in range(7):
            split = "train" if step < 4 else "val" if step == 4 else "test"
            weather = rng.normal(loc=patch * 0.2 + step * 0.1, scale=0.5)
            terrain = rng.normal(loc=-patch * 0.1, scale=0.3)
            logit = 0.4 * weather - 0.2 * terrain + rng.normal(scale=0.4)
            rows.append(
                {
                    "patch_id": f"{patch}",
                    "reference_date": pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=7 * step),
                    "split": split,
                    "weather_vpd_mean_l7d": weather,
                    "terrain_twi_mean": terrain,
                    "y_occ_30d": int(logit > 0.0),
                }
            )
    df = pd.DataFrame(rows)
    feature_columns = ["weather_vpd_mean_l7d", "terrain_twi_mean"]
    cross_fit = attach_occurrence_propensity(
        df,
        horizon_days=30,
        feature_columns=feature_columns,
        model_name="logreg",
        model_params={"max_iter": 200},
    )
    in_sample = train_model_on_existing_split(
        df,
        model_name="logreg",
        model_task="classification",
        model_params={"max_iter": 200},
        target_column="y_occ_30d",
        feature_columns=feature_columns,
    )

    train_mask = df["split"] == "train"
    assert not np.allclose(
        cross_fit.predictions.loc[train_mask, "propensity_score"].to_numpy(),
        in_sample.predictions.loc[train_mask, "prediction"].to_numpy(),
    )


def test_severity_models_honor_configured_estimator_family() -> None:
    frame, feature_columns = _severity_frame()
    train = frame.iloc[:80].copy()
    val = frame.iloc[80:100].copy()

    naive = NaiveSeverityModel(
        {
            "estimator_name": "logreg",
            "params": {"n_estimators": 40, "max_depth": 3},
        }
    )
    naive.fit(train, val, feature_columns, "y_sev_dnbr")

    ipw = PropensityWeightedSeverityModel(
        {
            "estimator_name": "logreg",
            "params": {"n_estimators": 40, "max_depth": 3},
        }
    )
    ipw.fit(train, val, feature_columns, "y_sev_dnbr", propensity_column="propensity_score")

    assert isinstance(naive.model.estimator, LinearRegression)
    assert isinstance(ipw.model.estimator, LinearRegression)
