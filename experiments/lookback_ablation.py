from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni.config import load_experiment_config
from agni.features.guard import infer_feature_columns
from agni.pipeline import fit_and_predict, load_dataset

app = typer.Typer()


def filter_features_by_max_window(feature_columns: list[str], max_window: int) -> list[str]:
    kept = []
    for column in feature_columns:
        if "_l" in column and column.endswith("d"):
            try:
                window = int(column.rsplit("_l", 1)[1].rstrip("d"))
            except ValueError:
                kept.append(column)
                continue
            if window <= max_window:
                kept.append(column)
        else:
            kept.append(column)
    return kept


@app.command()
def main(config: str, windows: list[int] | None = None) -> None:
    experiment = load_experiment_config(config)
    df = load_dataset(experiment)
    all_features = infer_feature_columns(df)
    rows = []
    for window in (windows or [7, 14, 30, 60]):
        filtered = filter_features_by_max_window(all_features, window)
        working = df.copy()
        drop_columns = [column for column in all_features if column not in filtered]
        working = working.drop(columns=drop_columns)
        _, _, metrics = fit_and_predict(working, experiment)
        rows.append({"max_window_days": window, **metrics})
    output_path = Path(experiment.output_dir) / "lookback_ablation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
