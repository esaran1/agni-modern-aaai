from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from agni.models.base import BaseModel


class XGBoostModel(BaseModel):
    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        params = self.config.get("params", {}).copy()
        task = self.config.get("task", "classification")
        try:
            from xgboost import XGBClassifier, XGBRegressor

            if task == "classification":
                params.setdefault("eval_metric", "logloss")
                self.model = XGBClassifier(**params)
            else:
                params.setdefault("objective", "reg:squarederror")
                self.model = XGBRegressor(**params)
        except ImportError:
            self.model = (
                HistGradientBoostingClassifier(random_state=params.get("random_state", 42))
                if task == "classification"
                else HistGradientBoostingRegressor(random_state=params.get("random_state", 42))
            )

        self.model.fit(df_train[feature_columns].fillna(0.0), df_train[target_column])

    def predict_proba(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        task = self.config.get("task", "classification")
        if task == "classification":
            values = self.model.predict_proba(df[feature_columns].fillna(0.0))[:, 1]
        else:
            values = self.model.predict(df[feature_columns].fillna(0.0))
        return pd.Series(values, index=df.index)
