from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from agni.features.guard import assert_no_leakage
from agni.models.base import BaseModel

SUPPORTED_SEVERITY_ESTIMATORS = {"xgboost", "random_forest", "logreg"}


@dataclass
class _SklearnSeverityArtifacts:
    imputer: SimpleImputer
    estimator: object


def _get_model_params(config: dict) -> dict:
    params = config.get("params", config)
    return dict(params)


def _filter_supported_params(factory, params: dict) -> dict:
    supported = set(inspect.signature(factory).parameters)
    return {key: value for key, value in params.items() if key in supported}


def _fit_regressor(
    estimator_name: str,
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    train_weights: np.ndarray | None,
    val_weights: np.ndarray | None,
) -> object:
    estimator_name = estimator_name.lower()
    if estimator_name not in SUPPORTED_SEVERITY_ESTIMATORS:
        supported = ", ".join(sorted(SUPPORTED_SEVERITY_ESTIMATORS))
        raise ValueError(
            f"Unsupported severity estimator family '{estimator_name}'. "
            f"Supported options are: {supported}."
        )

    if estimator_name == "xgboost":
        try:
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **_filter_supported_params(XGBRegressor, params),
            )
            fit_kwargs = {"verbose": False}
            if train_weights is not None:
                fit_kwargs["sample_weight"] = train_weights
            if x_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(x_val, y_val)]
                if val_weights is not None and train_weights is not None:
                    fit_kwargs["sample_weight_eval_set"] = [val_weights]
            estimator.fit(x_train, y_train, **fit_kwargs)
            return estimator
        except ImportError:
            estimator = HistGradientBoostingRegressor(
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 6),
                max_iter=params.get("n_estimators", 500),
                random_state=params.get("random_state", 42),
            )
            fit_kwargs = {}
            if train_weights is not None:
                fit_kwargs["sample_weight"] = train_weights
            estimator.fit(x_train, y_train, **fit_kwargs)
            return estimator

    if estimator_name == "random_forest":
        estimator = RandomForestRegressor(
            **_filter_supported_params(RandomForestRegressor, params),
        )
        fit_kwargs = {}
        if train_weights is not None:
            fit_kwargs["sample_weight"] = train_weights
        estimator.fit(x_train, y_train, **fit_kwargs)
        return estimator

    if estimator_name == "logreg":
        estimator = LinearRegression(**_filter_supported_params(LinearRegression, params))
        fit_kwargs = {}
        if train_weights is not None:
            fit_kwargs["sample_weight"] = train_weights
        estimator.fit(x_train, y_train, **fit_kwargs)
        return estimator

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
        val_weights = (
            self.compute_ipw_weights(df_val[propensity_column].to_numpy())
            if propensity_column in df_val.columns and not df_val.empty
            else None
        )
        params = _get_model_params(self.config)
        estimator_name = self.config.get("estimator_name", "xgboost")
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(df_train[feature_columns])
        y_train = df_train[target_column].to_numpy()
        x_val = imputer.transform(df_val[feature_columns]) if not df_val.empty else None
        y_val = df_val[target_column].to_numpy() if not df_val.empty else None
        estimator = _fit_regressor(
            estimator_name=estimator_name,
            params=params,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            train_weights=train_weights,
            val_weights=val_weights,
        )
        self.model = _SklearnSeverityArtifacts(imputer=imputer, estimator=estimator)

    def predict(self, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        x = self.model.imputer.transform(df[feature_columns])
        return self.model.estimator.predict(x)

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
        estimator_name = self.config.get("estimator_name", "xgboost")
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(df_train[feature_columns])
        y_train = df_train[target_column].to_numpy()
        x_val = imputer.transform(df_val[feature_columns]) if not df_val.empty else None
        y_val = df_val[target_column].to_numpy() if not df_val.empty else None
        estimator = _fit_regressor(
            estimator_name=estimator_name,
            params=params,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            train_weights=None,
            val_weights=None,
        )
        self.model = _SklearnSeverityArtifacts(imputer=imputer, estimator=estimator)
