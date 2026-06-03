from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config, require_experiment_task
from agni.experiment_utils import (
    attach_occurrence_propensity,
    occurrence_target_column,
    train_joint_risk_variant,
    train_severity_variant,
)
from agni.features.guard import infer_feature_columns
from agni.pipeline import load_dataset, split_dataset

app = typer.Typer()


@app.command()
def main(config: str, epochs: int = 5) -> None:
    experiment = load_experiment_config(config)
    require_experiment_task(experiment, "risk", "joint_risk_experiment")
    split_df = split_dataset(load_dataset(experiment), experiment)
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
    df = occurrence.predictions

    naive = train_severity_variant(
        df,
        model_type="naive",
        model_params=severity_model_params,
        feature_columns=feature_columns,
        estimator_name=severity_model_name,
    )
    ipw = train_severity_variant(
        df,
        model_type="ipw",
        model_params=severity_model_params,
        feature_columns=feature_columns,
        estimator_name=severity_model_name,
    )
    occ_target = occurrence_target_column(experiment.data.temporal.horizon_days)

    joint_params = dict(experiment.model.params)
    joint_params["epochs"] = epochs
    joint_no_rank = train_joint_risk_variant(
        df,
        feature_columns,
        joint_params,
        lambda_rank=0.0,
        occurrence_target_column_name=occ_target,
    )
    joint_full = train_joint_risk_variant(
        df,
        feature_columns,
        joint_params,
        lambda_rank=joint_params.get("lambda_rank", 0.1),
        occurrence_target_column_name=occ_target,
    )

    out = Path(experiment.output_dir) / "joint_risk_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        [
            {
                "approach": "independent",
                "occurrence_roc_auc": occurrence.metrics["roc_auc"],
                "severity_rmse": naive.metrics["rmse"],
                "severity_mae": naive.metrics["mae"],
                "risk_spearman": naive.metrics["risk_spearman"],
            },
            {
                "approach": "ipw",
                "occurrence_roc_auc": occurrence.metrics["roc_auc"],
                "severity_rmse": ipw.metrics["rmse"],
                "severity_mae": ipw.metrics["mae"],
                "risk_spearman": ipw.metrics["risk_spearman"],
            },
            {"approach": "joint_no_rank", **joint_no_rank.metrics},
            {"approach": "joint_full", **joint_full.metrics},
        ]
    )
    result.to_csv(out, index=False)
    predictions = joint_full.predictions.copy()
    predictions["risk_score"] = predictions["risk_prediction"]
    predictions.to_parquet(
        Path(experiment.output_dir) / "predictions.parquet",
        index=False,
    )
    (Path(experiment.output_dir) / "metrics.json").write_text(
        json.dumps(joint_full.metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    app()
