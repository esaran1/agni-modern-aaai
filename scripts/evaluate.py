from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.evaluation.metrics import classification_metrics, regression_metrics
from agni.pipeline import fit_and_predict, load_dataset, target_column_for_task
from agni.risk.expected_risk import compute_expected_risk, evaluate_risk_ranking

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    predictions_path = Path(experiment.output_dir) / "predictions.parquet"
    metrics_path = Path(experiment.output_dir) / "metrics.json"

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
    else:
        _, predictions, _ = fit_and_predict(load_dataset(experiment), experiment)

    if experiment.task == "risk":
        required_cols = {"occurrence_prediction", "severity_prediction", "y_occ_30d", "y_sev_dnbr"}
        missing = required_cols - set(predictions.columns)
        if missing:
            raise ValueError(f"Risk evaluation requires columns: {sorted(missing)}")
        predictions["risk_score"] = compute_expected_risk(
            predictions["occurrence_prediction"],
            predictions["severity_prediction"],
        )
        metrics = evaluate_risk_ranking(
            predictions["risk_score"],
            predictions["y_sev_dnbr"],
            predictions["y_occ_30d"],
        )
    else:
        target_column = target_column_for_task(experiment)
        test_df = predictions[predictions["split"] == "test"].copy()
        metrics = (
            classification_metrics(test_df[target_column], test_df["prediction"])
            if experiment.model.task == "classification"
            else regression_metrics(test_df[target_column], test_df["prediction"])
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", metrics_path)
    LOGGER.info("Metrics: %s", metrics)


if __name__ == "__main__":
    app()
