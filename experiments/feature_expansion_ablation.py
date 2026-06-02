from __future__ import annotations

from pathlib import Path

import typer

from agni.config import load_experiment_config
from agni.evaluation.ablation import leave_one_source_out_ablation
from agni.features.guard import infer_feature_columns
from agni.models import build_model
from agni.pipeline import load_dataset, split_dataset, target_column_for_task

app = typer.Typer()


@app.command()
def main(config: str) -> None:
    experiment = load_experiment_config(config)
    df = split_dataset(load_dataset(experiment), experiment)
    target = target_column_for_task(experiment)
    feature_columns = infer_feature_columns(df)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    result = leave_one_source_out_ablation(
        model_builder=lambda: build_model(
            experiment.model.name,
            {"task": experiment.model.task, "params": experiment.model.params},
        ),
        df_train=train_df,
        df_val=val_df,
        df_test=test_df,
        feature_columns=feature_columns,
        target_column=target,
        task=experiment.model.task,
    )
    output_path = Path(experiment.output_dir) / "feature_ablation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
