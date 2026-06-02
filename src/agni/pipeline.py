from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agni.config import ExperimentConfig
from agni.evaluation.metrics import classification_metrics, regression_metrics
from agni.features.guard import infer_feature_columns
from agni.features.physical import compute_terrain_range, compute_vpd_features, compute_wind_speed
from agni.features.spectral import compute_spectral_indices
from agni.features.temporal import (
    compute_desiccation_index,
    compute_feature_ratios,
    compute_temporal_diffs,
)
from agni.models import build_model
from agni.splits.spatiotemporal import spatiotemporal_purged_split
from agni.splits.temporal import temporal_purged_split


def dataset_path_for_stage(config: ExperimentConfig, stage: str) -> Path:
    return Path(config.data.processed_dir) / f"{stage}.parquet"


def load_dataset(config: ExperimentConfig, preferred_stage: str = "features") -> pd.DataFrame:
    preferred = dataset_path_for_stage(config, preferred_stage)
    if preferred.exists():
        return pd.read_parquet(preferred)

    fallback = dataset_path_for_stage(config, "dataset")
    if fallback.exists():
        return pd.read_parquet(fallback)

    raise FileNotFoundError(f"No dataset found at {preferred} or {fallback}")


def enrich_feature_table(df: pd.DataFrame, stride_days: int = 7) -> pd.DataFrame:
    frame = df.copy()
    frame = compute_vpd_features(frame)
    frame = compute_wind_speed(frame)
    frame = compute_terrain_range(frame)
    frame = compute_spectral_indices(frame)
    frame = compute_temporal_diffs(
        frame,
        columns=[
            "weather_temperature_2m_mean_l7d",
            "weather_total_precipitation_sum_mean_l7d",
            "weather_vpd_mean_l7d",
        ],
        stride_days=stride_days,
    )
    frame = compute_desiccation_index(frame)
    frame = compute_feature_ratios(frame)
    return frame


def split_dataset(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if config.data.spatial_blocks is not None:
        return spatiotemporal_purged_split(df, config.data)
    return temporal_purged_split(df, config.data.split, config.data.temporal.horizon_days)


def target_column_for_task(config: ExperimentConfig) -> str:
    if config.task == "occurrence":
        return f"y_occ_{config.data.temporal.horizon_days}d"
    if config.task == "severity":
        return "y_sev_dnbr"
    raise ValueError("Risk is derived from occurrence and severity predictions, not trained directly")


def fit_and_predict(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    split_df = split_dataset(df, config)
    target_column = target_column_for_task(config)

    if config.task == "severity":
        split_df = split_df[split_df["y_sev_available"] == 1].copy()

    feature_columns = infer_feature_columns(split_df)
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split assignment produced an empty partition: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    model = build_model(
        config.model.name,
        {"task": config.model.task, "params": config.model.params},
    )
    model.fit(train_df, val_df, feature_columns, target_column)

    predictions = split_df.copy()
    predictions["prediction"] = model.predict_proba(predictions, feature_columns)
    test_predictions = predictions[predictions["split"] == "test"].copy()
    metrics = (
        classification_metrics(test_predictions[target_column], test_predictions["prediction"])
        if config.model.task == "classification"
        else regression_metrics(test_predictions[target_column], test_predictions["prediction"])
    )
    return model, predictions, metrics


def save_training_outputs(
    config: ExperimentConfig,
    model,
    predictions: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.pkl")
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
