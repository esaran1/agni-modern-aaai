from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.pipeline import dataset_path_for_stage, enrich_feature_table

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    dataset_path = dataset_path_for_stage(experiment, "dataset")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_parquet(dataset_path)
    enriched = enrich_feature_table(df, stride_days=experiment.data.temporal.reference_stride_days)
    output_path = dataset_path_for_stage(experiment, "features")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    LOGGER.info("Wrote enriched features to %s", output_path)


if __name__ == "__main__":
    app()
