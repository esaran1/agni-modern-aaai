from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.data.builder import build_dataset
from agni.data.grid import build_patch_grid
from agni.data.manifest import build_dataset_manifest, hash_file
from agni.data.sources import build_adapters
from agni.data.sources.ee_session import initialize_earth_engine
from agni.experiment_utils import fit_risk_pipeline
from agni.features.guard import infer_feature_columns
from agni.labels.materialize import extract_sentinel2_severity, materialize_labels
from agni.pipeline import (
    enrich_feature_table,
    fit_and_predict,
    labeled_dataset_path,
    save_training_outputs,
    split_dataset,
)

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(
    config: str,
    max_workers: int = 8,
    max_retries: int = 3,
    ee_project: str = "",
    ee_key: str = "",
    high_volume: bool = True,
) -> None:
    experiment = load_experiment_config(config)
    initialize_earth_engine(
        project=ee_project or None,
        high_volume=high_volume,
        key_file=ee_key or None,
    )
    grid = build_patch_grid(experiment.data.grid.bbox, experiment.data.grid.grid_km)
    Path(experiment.data.processed_dir).mkdir(parents=True, exist_ok=True)
    grid.to_parquet(Path(experiment.data.processed_dir) / "grid.parquet", index=False)

    dataset_result = build_dataset(
        experiment.data,
        grid,
        build_adapters(experiment.data.sources),
        output_name="dataset.parquet",
        max_workers=max_workers,
        max_retries=max_retries,
    )
    features = enrich_feature_table(
        pd.read_parquet(dataset_result.dataset_path),
        stride_days=experiment.data.temporal.reference_stride_days,
    )
    features_path = Path(experiment.data.processed_dir) / "features.parquet"
    features.to_parquet(features_path, index=False)
    labeled_features = materialize_labels(
        features_df=features,
        grid_df=grid,
        horizon_days=experiment.data.temporal.horizon_days,
        lookback_days=experiment.data.temporal.lookback_days,
        reference_stride_days=experiment.data.temporal.reference_stride_days,
        label_reference_end=experiment.data.temporal.reference_end,
        severity_extractor=(
            extract_sentinel2_severity if experiment.task in {"severity", "risk"} else None
        ),
        materialize_severity=experiment.task in {"severity", "risk"},
    )
    labeled_features_path = labeled_dataset_path(experiment)
    labeled_features.to_parquet(labeled_features_path, index=False)
    label_manifest = build_dataset_manifest(
        dataset_path=labeled_features_path,
        config_dict=experiment.model_dump(mode="json"),
        row_count=len(labeled_features),
        extra={
            "output_name": labeled_features_path.name,
            "label_task": experiment.task,
            "source_features_sha256": hash_file(features_path),
            "grid_sha256": hash_file(Path(experiment.data.processed_dir) / "grid.parquet"),
            "label_reference_end": experiment.data.temporal.reference_end.isoformat(),
            "severity_window_days": 30,
        },
    )
    labeled_features_path.with_suffix(".manifest.json").write_text(
        json.dumps(label_manifest, indent=2),
        encoding="utf-8",
    )

    if experiment.task == "risk":
        split_df = split_dataset(labeled_features, experiment)
        result = fit_risk_pipeline(
            split_df,
            horizon_days=experiment.data.temporal.horizon_days,
            feature_columns=infer_feature_columns(split_df),
            occurrence_model_name=experiment.model.resolve_occurrence_model_name(),
            occurrence_model_params=experiment.model.resolve_occurrence_model_params(),
            severity_estimator_name=experiment.model.resolve_severity_model_name(),
            severity_model_params=experiment.model.resolve_severity_model_params(),
        )
        model, predictions, metrics = result.model, result.predictions, result.metrics
    else:
        model, predictions, metrics = fit_and_predict(labeled_features, experiment)
    save_training_outputs(experiment, model, predictions, metrics)
    LOGGER.info("Experiment complete with metrics: %s", metrics)


if __name__ == "__main__":
    app()
