from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from agni.config import load_experiment_config
from agni.pipeline import target_column_for_task

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    predictions_path = Path(experiment.output_dir) / "predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")

    predictions = pd.read_parquet(predictions_path)
    test_df = predictions[predictions["split"] == "test"].copy()
    target_col = target_column_for_task(experiment)
    plot_dir = Path(experiment.output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if experiment.model.task == "classification":
        RocCurveDisplay.from_predictions(test_df[target_col], test_df["prediction"])
        plt.savefig(plot_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(test_df[target_col], test_df["prediction"])
        plt.savefig(plot_dir / "pr_curve.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.scatter(test_df[target_col], test_df["prediction"], alpha=0.5)
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.savefig(plot_dir / "severity_scatter.png", dpi=200, bbox_inches="tight")
        plt.close()

    LOGGER.info("Wrote plots to %s", plot_dir)


if __name__ == "__main__":
    app()
