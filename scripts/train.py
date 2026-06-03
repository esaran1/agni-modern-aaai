from __future__ import annotations

import logging

import typer

from agni.config import load_experiment_config
from agni.experiment_utils import fit_risk_pipeline
from agni.features.guard import infer_feature_columns
from agni.pipeline import fit_and_predict, load_dataset, save_training_outputs, split_dataset

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    df = load_dataset(experiment)
    if experiment.task == "risk":
        split_df = split_dataset(df, experiment)
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
        model, predictions, metrics = fit_and_predict(df, experiment)
    save_training_outputs(experiment, model, predictions, metrics)
    LOGGER.info("Training complete with metrics: %s", metrics)


if __name__ == "__main__":
    app()
