from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from agni.models.base import BaseModel


class RandomForestModel(BaseModel):
    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        params = self.config.get("params", {})
        task = self.config.get("task", "classification")
        estimator = RandomForestClassifier(**params) if task == "classification" else RandomForestRegressor(**params)
        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("estimator", estimator),
            ]
        )
        self.model.fit(df_train[feature_columns], df_train[target_column])

    def predict_proba(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        task = self.config.get("task", "classification")
        if task == "classification":
            values = self.model.predict_proba(df[feature_columns])[:, 1]
        else:
            values = self.model.predict(df[feature_columns])
        return pd.Series(values, index=df.index)
