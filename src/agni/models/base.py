from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from agni.features.guard import assert_no_leakage


class BaseModel(ABC):
    def __init__(self, config: dict):
        self.config = config
        self._feature_columns: list[str] | None = None
        self.model = None

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        assert_no_leakage(feature_columns)
        self._feature_columns = feature_columns
        self._fit(df_train, df_val, feature_columns, target_column)

    @abstractmethod
    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)
