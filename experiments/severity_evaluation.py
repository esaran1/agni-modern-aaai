from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.evaluation.metrics import regression_metrics

app = typer.Typer()


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    predictions_path = Path(experiment.output_dir) / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    evaluable = predictions[
        (predictions["split"] == "test") & (predictions["y_sev_available"] == 1)
    ].copy()
    prediction_col = (
        "severity_prediction"
        if "severity_prediction" in evaluable.columns
        else "prediction"
    )
    metrics = regression_metrics(evaluable["y_sev_dnbr"], evaluable[prediction_col])
    output_path = Path(experiment.output_dir) / "severity_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
