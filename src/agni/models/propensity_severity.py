from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from agni.features.guard import assert_no_leakage
from agni.models.base import BaseModel


@dataclass
class _SklearnSeverityArtifacts:
    pipeline: Pipeline


def _get_model_params(config: dict) -> dict:
    params = config.get("params", config)
    return dict(params)


class PropensityWeightedSeverityModel(BaseModel):
    """Severity regression with inverse-propensity weighting."""

    def __init__(
        self,
        config: dict,
        clip_min: float = 0.05,
        clip_max: float = 0.95,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__(config)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize_weights = normalize_weights

    def compute_ipw_weights(self, propensity_scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(propensity_scores, dtype=float), self.clip_min, self.clip_max)
        weights = 1.0 / clipped
        if self.normalize_weights and len(weights) > 0:
            weights = weights * len(weights) / weights.sum()
        return weights

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        propensity_column: str = "propensity_score",
    ) -> None:
        assert_no_leakage(feature_columns)
        self._feature_columns = feature_columns
        self._fit(
            df_train=df_train,
            df_val=df_val,
            feature_columns=feature_columns,
            target_column=target_column,
            propensity_column=propensity_column,
        )

    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        propensity_column: str = "propensity_score",
    ) -> None:
        if propensity_column not in df_train.columns:
            raise ValueError(
                f"Propensity column '{propensity_column}' not found. "
                "Run the occurrence model first and attach its predictions."
            )

        train_weights = self.compute_ipw_weights(df_train[propensity_column].to_numpy())
        params = _get_model_params(self.config)
        estimator = None
        try:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.03),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                n_estimators=params.get("n_estimators", 500),
                random_state=params.get("random_state", 42),
            )
        except ImportError:
            estimator = HistGradientBoostingRegressor(
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 6),
                max_iter=params.get("n_estimators", 500),
                random_state=params.get("random_state", 42),
            )

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(
            df_train[feature_columns],
            df_train[target_column],
            estimator__sample_weight=train_weights,
        )
        self.model = _SklearnSeverityArtifacts(pipeline=pipeline)

    def predict(self, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self.model.pipeline.predict(df[feature_columns])

    def predict_proba(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        return pd.Series(self.predict(df, feature_columns), index=df.index)


class NaiveSeverityModel(PropensityWeightedSeverityModel):
    """Baseline severity model with uniform weights."""

    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        propensity_column: str = "propensity_score",
    ) -> None:
        params = _get_model_params(self.config)
        try:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.03),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                n_estimators=params.get("n_estimators", 500),
                random_state=params.get("random_state", 42),
            )
        except ImportError:
            estimator = HistGradientBoostingRegressor(
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 6),
                max_iter=params.get("n_estimators", 500),
                random_state=params.get("random_state", 42),
            )

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(df_train[feature_columns], df_train[target_column])
        self.model = _SklearnSeverityArtifacts(pipeline=pipeline)
