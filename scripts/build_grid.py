from __future__ import annotations

import logging
from pathlib import Path

import typer

from agni.config import load_experiment_config
from agni.data.grid import build_patch_grid

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    grid = build_patch_grid(experiment.data.grid.bbox, experiment.data.grid.grid_km)
    output_path = Path(experiment.data.processed_dir) / "grid.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s patches to %s", len(grid), output_path)


if __name__ == "__main__":
    app()
