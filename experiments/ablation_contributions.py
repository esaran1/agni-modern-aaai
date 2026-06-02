from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.experiment_utils import (
    attach_occurrence_propensity,
    train_joint_risk_variant,
    train_severity_variant,
)
from agni.evaluation.leakage_taxonomy import compute_leakage_curve, detect_type3_leakage
from agni.features.guard import infer_feature_columns
from agni.models.conformal import SplitConformalRiskPredictor
from agni.pipeline import load_dataset, split_dataset
from agni.risk.expected_risk import compute_expected_risk

app = typer.Typer()


@app.command()
def main(config: str) -> None:
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
    df = occurrence.predictions
    baseline = train_severity_variant(df, "naive", experiment.model.params, feature_columns)
    ipw = train_severity_variant(df, "ipw", experiment.model.params, feature_columns)
    joint = train_joint_risk_variant(df, feature_columns, experiment.model.params, lambda_rank=experiment.model.params.get("lambda_rank", 0.1))

    joint_eval = joint.predictions[joint.predictions["y_sev_available"] == 1].copy()
    joint_eval["risk_pred"] = compute_expected_risk(
        joint_eval["occurrence_prediction"],
        joint_eval["severity_prediction"],
    )
    calibration = joint_eval[joint_eval["split"] == "val"]
    test = joint_eval[joint_eval["split"] == "test"]
    conformal = SplitConformalRiskPredictor(alpha=0.10).calibrate(calibration["risk_pred"], calibration["y_sev_dnbr"])
    conformal_metrics = conformal.evaluate_coverage(test["risk_pred"], test["y_sev_dnbr"])

    def train_model_fn(df_train: pd.DataFrame, df_val: pd.DataFrame):
        from agni.models import build_model

        model = build_model("xgboost", {"task": "classification", "params": experiment.model.params})
        model.fit(df_train, df_val, feature_columns, f"y_occ_{experiment.data.temporal.horizon_days}d")
        return model

    def evaluate_fn(model, df_test: pd.DataFrame) -> float:
        from agni.experiment_utils import safe_classification_metrics

        return safe_classification_metrics(
            df_test[f"y_occ_{experiment.data.temporal.horizon_days}d"],
            model.predict_proba(df_test, feature_columns),
        )["roc_auc"]

    temporal_curve = compute_leakage_curve(
        load_dataset(experiment),
        train_model_fn,
        evaluate_fn,
        horizon_days=experiment.data.temporal.horizon_days,
        split_boundaries=(
            pd.Timestamp(experiment.data.split.train_end),
            pd.Timestamp(experiment.data.split.val_end),
        ),
    )
    spatial_curve = detect_type3_leakage(
        split_df,
        train_model_fn,
        evaluate_fn,
        grid_km=experiment.data.grid.grid_km,
    )
    temporal_drop = (
        float(temporal_curve.iloc[0]["roc_auc"] - temporal_curve.iloc[-1]["roc_auc"])
        if len(temporal_curve) >= 2
        else float("nan")
    )
    spatial_drop = (
        float(spatial_curve.iloc[0]["mean_auc"] - spatial_curve.iloc[-1]["mean_auc"])
        if len(spatial_curve) >= 2
        else float("nan")
    )

    rows = [
        {
            "variant": "baseline",
            "occurrence_roc_auc": occurrence.metrics["roc_auc"],
            "severity_rmse": baseline.metrics["rmse"],
            "severity_mae": baseline.metrics["mae"],
            "risk_spearman": baseline.metrics["risk_spearman"],
        },
        {
            "variant": "+ipw",
            "occurrence_roc_auc": occurrence.metrics["roc_auc"],
            "severity_rmse": ipw.metrics["rmse"],
            "severity_mae": ipw.metrics["mae"],
            "risk_spearman": ipw.metrics["risk_spearman"],
        },
        {
            "variant": "+joint_risk",
            **joint.metrics,
        },
        {
            "variant": "+conformal",
            **joint.metrics,
            "conformal_coverage": conformal_metrics["empirical_coverage"],
            "conformal_width": conformal_metrics["mean_interval_width"],
        },
        {
            "variant": "+leakage_diagnosis",
            **joint.metrics,
            "conformal_coverage": conformal_metrics["empirical_coverage"],
            "temporal_leakage_auc_drop": temporal_drop,
            "spatial_leakage_auc_drop": spatial_drop,
        },
    ]
    out = Path(experiment.output_dir) / "contribution_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    app()
