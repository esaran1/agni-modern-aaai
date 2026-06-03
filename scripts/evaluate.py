from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.evaluation.metrics import classification_metrics, regression_metrics
from agni.experiment_utils import fit_risk_pipeline, occurrence_target_column
from agni.features.guard import infer_feature_columns
from agni.pipeline import (
    fit_and_predict,
    load_dataset,
    split_dataset,
    target_column_for_task,
)
from agni.risk.expected_risk import compute_expected_risk, evaluate_risk_ranking

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    predictions_path = Path(experiment.output_dir) / "predictions.parquet"
    metrics_path = Path(experiment.output_dir) / "metrics.json"
    generated_predictions = False

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
    elif experiment.task == "risk":
        split_df = split_dataset(load_dataset(experiment), experiment)
        risk_result = fit_risk_pipeline(
            split_df,
            horizon_days=experiment.data.temporal.horizon_days,
            feature_columns=infer_feature_columns(split_df),
            occurrence_model_name=experiment.model.resolve_occurrence_model_name(),
            occurrence_model_params=experiment.model.resolve_occurrence_model_params(),
            severity_estimator_name=experiment.model.resolve_severity_model_name(),
            severity_model_params=experiment.model.resolve_severity_model_params(),
        )
        predictions = risk_result.predictions
        generated_predictions = True
    else:
        _, predictions, _ = fit_and_predict(load_dataset(experiment), experiment)
        generated_predictions = True

    if experiment.task == "risk":
        test_df = predictions[predictions["split"] == "test"].copy()
        occ_target = occurrence_target_column(experiment.data.temporal.horizon_days)
        required_cols = {occ_target, "y_sev_dnbr"}
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"Risk evaluation requires columns: {sorted(missing)}")
        if "risk_score" in test_df.columns:
            risk_col = "risk_score"
        elif "risk_prediction" in test_df.columns:
            risk_col = "risk_prediction"
        else:
            if "severity_prediction" not in test_df.columns:
                raise ValueError(
                    "Risk evaluation requires severity_prediction when risk_score is absent"
                )
            if "occurrence_prediction" in test_df.columns:
                occurrence_col = "occurrence_prediction"
            elif "propensity_score" in test_df.columns:
                occurrence_col = "propensity_score"
            else:
                raise ValueError(
                    "Risk evaluation requires risk_score, risk_prediction, "
                    "or occurrence_prediction/propensity_score with severity_prediction"
                )
            test_df["risk_score"] = compute_expected_risk(
                test_df[occurrence_col],
                test_df["severity_prediction"],
            )
            risk_col = "risk_score"
        metrics = evaluate_risk_ranking(
            test_df[risk_col],
            test_df["y_sev_dnbr"],
            test_df[occ_target],
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
    if generated_predictions:
        predictions.to_parquet(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", metrics_path)
    LOGGER.info("Metrics: %s", metrics)


if __name__ == "__main__":
    app()
