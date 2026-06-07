from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor

from agni.models.base import BaseModel


@dataclass
class _TorchArtifacts:
    module: object
    imputer: SimpleImputer
    feature_count: int


class TransformerModel(BaseModel):
    def _fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            task = self.config.get("task", "classification")
            estimator = (
                MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42)
                if task == "classification"
                else MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200, random_state=42)
            )
            estimator.fit(df_train[feature_columns].fillna(0.0), df_train[target_column])
            self.model = estimator
            return

        seed = int(self.config.get("params", {}).get("random_state", 42))
        torch.manual_seed(seed)
        hidden_dim = int(self.config.get("params", {}).get("hidden_dim", 64))
        n_heads = int(self.config.get("params", {}).get("n_heads", 4))
        n_layers = int(self.config.get("params", {}).get("n_layers", 2))
        dropout = float(self.config.get("params", {}).get("dropout", 0.1))
        epochs = int(self.config.get("params", {}).get("epochs", 15))
        lr = float(self.config.get("params", {}).get("lr", 1e-3))
        batch_size = int(self.config.get("params", {}).get("batch_size", 128))
        task = self.config.get("task", "classification")

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(df_train[feature_columns]).astype(np.float32)
        y_train = df_train[target_column].to_numpy(dtype=np.float32)

        class TabularTransformer(nn.Module):
            def __init__(self, feature_count: int) -> None:
                super().__init__()
                self.input_projection = nn.Linear(1, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Linear(hidden_dim, 1)
                self.feature_count = feature_count

            def forward(self, x):
                x = x.unsqueeze(-1)
                x = self.input_projection(x)
                x = self.encoder(x)
                x = x.mean(dim=1)
                return self.head(x).squeeze(-1)

        module = TabularTransformer(x_train.shape[1])
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        module.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = module(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.model = _TorchArtifacts(
            module=module.eval(),
            imputer=imputer,
            feature_count=x_train.shape[1],
        )

    def predict_proba(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        if isinstance(self.model, _TorchArtifacts):
            import torch

            x = self.model.imputer.transform(df[feature_columns]).astype(np.float32)
            with torch.no_grad():
                logits = self.model.module(torch.from_numpy(x))
                probs = torch.sigmoid(logits).numpy()
            return pd.Series(probs, index=df.index)

        task = self.config.get("task", "classification")
        if task == "classification":
            values = self.model.predict_proba(df[feature_columns].fillna(0.0))[:, 1]
        else:
            values = self.model.predict(df[feature_columns].fillna(0.0))
        return pd.Series(values, index=df.index)
