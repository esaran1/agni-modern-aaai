from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.data.builder import build_dataset, build_patch_shards
from agni.data.grid import build_patch_grid
from agni.data.sources import build_adapters
from agni.data.sources.ee_session import initialize_earth_engine

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(
    config: str,
    max_workers: int = 8,
    max_retries: int = 3,
    resume: bool = True,
    ee_project: str = "",
    ee_key: str = "",
    high_volume: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    merge: bool = False,
) -> None:
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    project = ee_project or None

    experiment = load_experiment_config(config)
    grid_path = Path(experiment.data.processed_dir) / "grid.parquet"
    if grid_path.exists():
        grid = pd.read_parquet(grid_path)
    else:
        grid = build_patch_grid(experiment.data.grid.bbox, experiment.data.grid.grid_km)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(grid_path, index=False)

    adapters = build_adapters(experiment.data.sources)

    # Merge step needs no Earth Engine access: it only concatenates existing shards.
    if merge:
        result = build_dataset(
            experiment.data,
            grid,
            adapters,
            output_name="dataset.parquet",
            max_workers=max_workers,
            max_retries=max_retries,
            resume=True,
        )
        LOGGER.info("Merged %d shards into %s", num_shards, result.dataset_path)
        LOGGER.info("Wrote manifest to %s", result.manifest_path)
        return

    initialize_earth_engine(project=project, high_volume=high_volume, key_file=ee_key or None)

    if num_shards > 1:
        grid_shard = grid.iloc[shard_index::num_shards].reset_index(drop=True)
        built = build_patch_shards(
            experiment.data,
            grid_shard,
            adapters,
            output_name="dataset.parquet",
            max_workers=max_workers,
            max_retries=max_retries,
            resume=resume,
        )
        LOGGER.info(
            "Shard %d/%d built %d patches; run with --merge after all shards finish",
            shard_index,
            num_shards,
            built,
        )
        return

    result = build_dataset(
        experiment.data,
        grid,
        adapters,
        output_name="dataset.parquet",
        max_workers=max_workers,
        max_retries=max_retries,
        resume=resume,
    )
    LOGGER.info("Wrote dataset to %s", result.dataset_path)
    LOGGER.info("Wrote manifest to %s", result.manifest_path)


if __name__ == "__main__":
    app()
