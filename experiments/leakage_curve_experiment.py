from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.evaluation.leakage_taxonomy import compute_leakage_curve, detect_type3_leakage
from agni.experiment_utils import occurrence_target_column, safe_classification_metrics
from agni.features.guard import infer_feature_columns
from agni.models import build_model
from agni.pipeline import load_dataset, split_dataset

app = typer.Typer()


@app.command()
def main(config: str, horizon_days: int | None = None) -> None:
    experiment = load_experiment_config(config)
    feature_df = load_dataset(experiment)
    split_df = split_dataset(feature_df, experiment)
    feature_columns = infer_feature_columns(split_df)
    resolved_horizon = horizon_days or experiment.data.temporal.horizon_days
    target_column = occurrence_target_column(resolved_horizon)
    occurrence_model_name = experiment.model.resolve_occurrence_model_name()
    occurrence_model_params = experiment.model.resolve_occurrence_model_params()

    def train_model_fn(df_train: pd.DataFrame, df_val: pd.DataFrame):
        model = build_model(
            occurrence_model_name,
            {"task": "classification", "params": occurrence_model_params},
        )
        model.fit(df_train, df_val, feature_columns, target_column)
        return model

    def evaluate_fn(model, df_test: pd.DataFrame) -> float:
        metrics = safe_classification_metrics(
            df_test[target_column],
            model.predict_proba(df_test, feature_columns),
        )
        return metrics["roc_auc"] if not np.isnan(metrics["roc_auc"]) else float("nan")

    temporal = compute_leakage_curve(
        feature_df,
        train_model_fn,
        evaluate_fn,
        horizon_days=resolved_horizon,
        split_boundaries=(
            pd.Timestamp(experiment.data.split.train_end),
            pd.Timestamp(experiment.data.split.val_end),
        ),
    )
    spatial = detect_type3_leakage(
        split_df,
        train_model_fn,
        evaluate_fn,
        grid_km=experiment.data.grid.grid_km,
    )
    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(output_dir / "leakage_curve_temporal.csv", index=False)
    spatial.to_csv(output_dir / "leakage_curve_spatial.csv", index=False)


if __name__ == "__main__":
    app()
