from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.data.builder import build_dataset
from agni.data.grid import build_patch_grid
from agni.data.sources import build_adapters

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    grid_path = Path(experiment.data.processed_dir) / "grid.parquet"
    if grid_path.exists():
        grid = pd.read_parquet(grid_path)
    else:
        grid = build_patch_grid(experiment.data.grid.bbox, experiment.data.grid.grid_km)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(grid_path, index=False)

    adapters = build_adapters(experiment.data.sources)
    result = build_dataset(experiment.data, grid, adapters, output_name="dataset.parquet")
    LOGGER.info("Wrote dataset to %s", result.dataset_path)
    LOGGER.info("Wrote manifest to %s", result.manifest_path)


if __name__ == "__main__":
    app()
