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
from agni.features.guard import infer_feature_columns
from agni.pipeline import load_dataset, split_dataset

app = typer.Typer()


@app.command()
def main(config: str, epochs: int = 5) -> None:
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

    naive = train_severity_variant(df, "naive", experiment.model.params, feature_columns)
    ipw = train_severity_variant(df, "ipw", experiment.model.params, feature_columns)

    joint_params = dict(experiment.model.params)
    joint_params["epochs"] = epochs
    joint_no_rank = train_joint_risk_variant(df, feature_columns, joint_params, lambda_rank=0.0)
    joint_full = train_joint_risk_variant(
        df,
        feature_columns,
        joint_params,
        lambda_rank=joint_params.get("lambda_rank", 0.1),
    )

    out = Path(experiment.output_dir) / "joint_risk_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
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
    ).to_csv(out, index=False)


if __name__ == "__main__":
    app()
