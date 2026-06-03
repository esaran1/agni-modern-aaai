from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config, require_experiment_task
from agni.evaluation.conformal_eval import conformal_sets_to_frame, summarize_high_confidence_alerts
from agni.experiment_utils import (
    attach_occurrence_propensity,
    carve_conformal_calibration_split,
    train_severity_variant,
)
from agni.features.guard import infer_feature_columns
from agni.models.conformal import SplitConformalRiskPredictor
from agni.pipeline import load_dataset, split_dataset
from agni.risk.expected_risk import compute_expected_risk

app = typer.Typer()


@app.command()
def main(config: str, alpha: float = 0.10, threshold: float = 0.5) -> None:
    experiment = load_experiment_config(config)
    require_experiment_task(experiment, "risk", "conformal_risk_experiment")
    split_df = split_dataset(load_dataset(experiment), experiment)
    split_df = carve_conformal_calibration_split(
        split_df,
        min_required_rows=20,
        required_columns=("y_sev_available", "y_sev_dnbr"),
    )
    feature_columns = infer_feature_columns(split_df)
    occurrence_model_name = experiment.model.resolve_occurrence_model_name()
    occurrence_model_params = experiment.model.resolve_occurrence_model_params()
    severity_model_name = experiment.model.resolve_severity_model_name()
    severity_model_params = experiment.model.resolve_severity_model_params()
    occurrence = attach_occurrence_propensity(
        split_df,
        horizon_days=experiment.data.temporal.horizon_days,
        feature_columns=feature_columns,
        model_name=occurrence_model_name,
        model_params=occurrence_model_params,
    )
    severity = train_severity_variant(
        occurrence.predictions,
        model_type="ipw",
        model_params=severity_model_params,
        feature_columns=feature_columns,
        estimator_name=severity_model_name,
    )
    evaluable = severity.predictions.copy()
    evaluable["risk_score"] = compute_expected_risk(
        evaluable["propensity_score"],
        evaluable["severity_prediction"],
    )
    calibration = evaluable[evaluable["split"] == "calibration"]
    test = evaluable[evaluable["split"] == "test"]

    predictor = SplitConformalRiskPredictor(alpha=alpha).calibrate(
        calibration["risk_score"],
        calibration["y_sev_dnbr"],
    )
    sets = predictor.predict(test["risk_score"])
    coverage = predictor.evaluate_coverage(test["risk_score"], test["y_sev_dnbr"])
    alerts = summarize_high_confidence_alerts(sets, threshold=threshold)

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluable.to_parquet(output_dir / "predictions.parquet", index=False)
    conformal_sets_to_frame(sets).to_csv(output_dir / "conformal_intervals.csv", index=False)
    summary = {**coverage, **alerts}
    pd.DataFrame([summary]).to_csv(output_dir / "conformal_coverage.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
