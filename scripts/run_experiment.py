from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.data.builder import build_dataset
from agni.data.grid import build_patch_grid
from agni.data.sources import build_adapters
from agni.pipeline import enrich_feature_table, fit_and_predict, save_training_outputs

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    grid = build_patch_grid(experiment.data.grid.bbox, experiment.data.grid.grid_km)
    Path(experiment.data.processed_dir).mkdir(parents=True, exist_ok=True)
    grid.to_parquet(Path(experiment.data.processed_dir) / "grid.parquet", index=False)

    dataset_result = build_dataset(
        experiment.data,
        grid,
        build_adapters(experiment.data.sources),
        output_name="dataset.parquet",
    )
    features = enrich_feature_table(
        pd.read_parquet(dataset_result.dataset_path),
        stride_days=experiment.data.temporal.reference_stride_days,
    )
    features_path = Path(experiment.data.processed_dir) / "features.parquet"
    features.to_parquet(features_path, index=False)

    model, predictions, metrics = fit_and_predict(features, experiment)
    save_training_outputs(experiment, model, predictions, metrics)
    LOGGER.info("Experiment complete with metrics: %s", metrics)


if __name__ == "__main__":
    app()
