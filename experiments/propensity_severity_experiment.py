from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config, require_experiment_task
from agni.experiment_utils import attach_occurrence_propensity, train_severity_variant
from agni.features.guard import infer_feature_columns
from agni.pipeline import load_dataset, split_dataset
from agni.risk.expected_risk import compute_expected_risk

app = typer.Typer()


@app.command()
def main(config: str, propensity_column: str = "propensity_score") -> None:
    experiment = load_experiment_config(config)
    require_experiment_task(experiment, "risk", "propensity_severity_experiment")
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
        propensity_column=propensity_column,
    )
    ipw = train_severity_variant(
        df,
        model_type="ipw",
        model_params=severity_model_params,
        feature_columns=feature_columns,
        estimator_name=severity_model_name,
        propensity_column=propensity_column,
    )
    results = [
        {"model": "naive", "occurrence_roc_auc": occurrence.metrics["roc_auc"], **naive.metrics},
        {"model": "ipw", "occurrence_roc_auc": occurrence.metrics["roc_auc"], **ipw.metrics},
    ]

    merged = occurrence.predictions.merge(
        naive.predictions[["patch_id", "reference_date", "severity_prediction"]].rename(
            columns={"severity_prediction": "naive_severity_prediction"}
        ),
        on=["patch_id", "reference_date"],
        how="left",
    ).merge(
        ipw.predictions[["patch_id", "reference_date", "severity_prediction"]].rename(
            columns={"severity_prediction": "ipw_severity_prediction"}
        ),
        on=["patch_id", "reference_date"],
        how="left",
    )
    merged["severity_prediction"] = merged["ipw_severity_prediction"]
    merged["risk_score"] = compute_expected_risk(
        merged["occurrence_prediction"],
        merged["severity_prediction"],
    )

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "propensity_severity_comparison.csv", index=False)
    merged.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps({"rows": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    app()
