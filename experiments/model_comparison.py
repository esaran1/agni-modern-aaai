from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.pipeline import fit_and_predict, load_dataset

app = typer.Typer()


@app.command()
def main(config: str, models: list[str]) -> None:
    experiment = load_experiment_config(config)
    base_model_config = experiment.model.model_copy(deep=True)
    df = load_dataset(experiment)
    rows = []
    for model_name in models:
        resolved_params = base_model_config.resolve_comparison_model_params(model_name)
        experiment.model.name = model_name
        experiment.model.params = resolved_params
        model, _, metrics = fit_and_predict(df, experiment)
        rows.append({"model": model_name, **metrics, "artifact_type": type(model).__name__})
    result = pd.DataFrame(rows)
    output_path = Path(experiment.output_dir) / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
