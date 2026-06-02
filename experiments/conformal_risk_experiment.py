from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.experiment_utils import attach_occurrence_propensity, train_severity_variant
from agni.evaluation.conformal_eval import conformal_sets_to_frame, summarize_high_confidence_alerts
from agni.models.conformal import SplitConformalRiskPredictor
from agni.features.guard import infer_feature_columns
from agni.pipeline import load_dataset, split_dataset
from agni.risk.expected_risk import compute_expected_risk

app = typer.Typer()


@app.command()
def main(config: str, alpha: float = 0.10, threshold: float = 0.5) -> None:
    experiment = load_experiment_config(config)
    split_df = split_dataset(load_dataset(experiment), experiment)
    feature_columns = infer_feature_columns(split_df)
    occurrence = attach_occurrence_propensity(
        split_df,
        horizon_days=experiment.data.temporal.horizon_days,
        feature_columns=feature_columns,
        model_name="xgboost",
        model_params=experiment.model.params,
    )
    severity = train_severity_variant(
        occurrence.predictions,
        model_type="ipw",
        model_params=experiment.model.params,
        feature_columns=feature_columns,
    )
    evaluable = severity.predictions.copy()
    evaluable["risk_pred"] = compute_expected_risk(
        evaluable["propensity_score"],
        evaluable["severity_prediction"],
    )
    calibration = evaluable[evaluable["split"] == "val"]
    test = evaluable[evaluable["split"] == "test"]

    predictor = SplitConformalRiskPredictor(alpha=alpha).calibrate(
        calibration["risk_pred"],
        calibration["y_sev_dnbr"],
    )
    sets = predictor.predict(test["risk_pred"])
    coverage = predictor.evaluate_coverage(test["risk_pred"], test["y_sev_dnbr"])
    alerts = summarize_high_confidence_alerts(sets, threshold=threshold)

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conformal_sets_to_frame(sets).to_csv(output_dir / "conformal_intervals.csv", index=False)
    pd.DataFrame([{**coverage, **alerts}]).to_csv(output_dir / "conformal_coverage.csv", index=False)


if __name__ == "__main__":
    app()
