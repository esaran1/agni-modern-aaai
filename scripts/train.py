from __future__ import annotations

import logging

import typer

from agni.config import load_experiment_config
from agni.pipeline import fit_and_predict, load_dataset, save_training_outputs

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    df = load_dataset(experiment)
    model, predictions, metrics = fit_and_predict(df, experiment)
    save_training_outputs(experiment, model, predictions, metrics)
    LOGGER.info("Training complete with metrics: %s", metrics)


if __name__ == "__main__":
    app()
