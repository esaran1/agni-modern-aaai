from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.experiment_utils import attach_occurrence_propensity, train_severity_variant
from agni.features.guard import infer_feature_columns
from agni.pipeline import load_dataset, split_dataset

app = typer.Typer()


@app.command()
def main(config: str, propensity_column: str = "propensity_score") -> None:
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
    naive = train_severity_variant(
        df,
        model_type="naive",
        model_params=experiment.model.params,
        feature_columns=feature_columns,
        propensity_column=propensity_column,
    )
    ipw = train_severity_variant(
        df,
        model_type="ipw",
        model_params=experiment.model.params,
        feature_columns=feature_columns,
        propensity_column=propensity_column,
    )
    results = [
        {"model": "naive", "occurrence_roc_auc": occurrence.metrics["roc_auc"], **naive.metrics},
        {"model": "ipw", "occurrence_roc_auc": occurrence.metrics["roc_auc"], **ipw.metrics},
    ]

    out = Path(experiment.output_dir) / "propensity_severity_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)


if __name__ == "__main__":
    app()
