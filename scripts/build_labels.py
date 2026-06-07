from __future__ import annotations

import json
import logging

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.data.manifest import build_dataset_manifest, hash_file
from agni.data.sources.ee_session import initialize_earth_engine
from agni.labels.materialize import extract_sentinel2_severity, materialize_labels
from agni.pipeline import dataset_path_for_stage, labeled_dataset_path

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(
    config: str,
    severity_window_days: int = 30,
    ee_project: str = "",
    ee_key: str = "",
    high_volume: bool = True,
) -> None:
    experiment = load_experiment_config(config)
    # Severity/risk label materialization calls Earth Engine; occurrence-only is offline.
    if experiment.task in {"severity", "risk"}:
        initialize_earth_engine(
            project=ee_project or None,
            high_volume=high_volume,
            key_file=ee_key or None,
        )
    grid_path = dataset_path_for_stage(experiment, "grid")
    features_path = dataset_path_for_stage(experiment, "features")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found at {grid_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at {features_path}")

    output_path = labeled_dataset_path(experiment)
    features_df = pd.read_parquet(features_path)
    grid_df = pd.read_parquet(grid_path)
    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=experiment.data.temporal.horizon_days,
        lookback_days=experiment.data.temporal.lookback_days,
        severity_window_days=severity_window_days,
        reference_stride_days=experiment.data.temporal.reference_stride_days,
        label_reference_end=experiment.data.temporal.reference_end,
        severity_extractor=(
            extract_sentinel2_severity if experiment.task in {"severity", "risk"} else None
        ),
        materialize_severity=experiment.task in {"severity", "risk"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output_path, index=False)
    labeled_path = output_path

    manifest = build_dataset_manifest(
        dataset_path=labeled_path,
        config_dict=experiment.model_dump(mode="json"),
        row_count=len(labeled),
        extra={
            "output_name": labeled_path.name,
            "label_task": experiment.task,
            "source_features_sha256": hash_file(features_path),
            "grid_sha256": hash_file(grid_path),
            "label_reference_end": experiment.data.temporal.reference_end.isoformat(),
            "severity_window_days": severity_window_days,
        },
    )
    manifest_path = labeled_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote labeled features to %s", labeled_path)
    LOGGER.info("Wrote label manifest to %s", manifest_path)


if __name__ == "__main__":
    app()
