from __future__ import annotations

from pathlib import Path

import typer

from agni.config import load_experiment_config
from agni.pipeline import load_dataset, split_dataset

app = typer.Typer()


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    split_df = split_dataset(load_dataset(experiment), experiment)
    summary = split_df.groupby("split")["patch_id"].agg(["count", "nunique"]).reset_index()
    output_path = Path(experiment.output_dir) / "spatial_block_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
