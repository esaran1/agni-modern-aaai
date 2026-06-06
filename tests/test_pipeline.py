from __future__ import annotations

from pathlib import Path

import pandas as pd

from agni.config import ExperimentConfig
from agni.pipeline import labeled_dataset_path, load_dataset


def _experiment_config(tmp_path: Path, task: str) -> ExperimentConfig:
    model_task = "classification" if task != "severity" else "regression"
    return ExperimentConfig.model_validate(
        {
            "name": f"{task}-pipeline-test",
            "task": task,
            "data": {
                "grid": {
                    "grid_km": 10,
                    "bbox": {
                        "lon_min": 0.0,
                        "lon_max": 1.0,
                        "lat_min": 0.0,
                        "lat_max": 1.0,
                    },
                },
                "temporal": {
                    "reference_start": "2020-01-01",
                    "reference_end": "2020-01-31",
                    "reference_stride_days": 7,
                    "lookback_days": 30,
                    "temporal_windows": [7, 14, 30],
                    "horizon_days": 14,
                },
                "split": {
                    "buffer_days": 14,
                    "train_end": "2020-01-10",
                    "val_end": "2020-01-20",
                    "test_end": "2020-01-31",
                },
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "model": {
                "name": "logreg",
                "task": model_task,
                "params": {"max_iter": 100},
            },
            "evaluation": {"bootstrap_samples": 0, "confidence_level": 0.95},
            "output_dir": str(tmp_path / "outputs"),
        }
    )


def test_load_dataset_prefers_task_specific_labeled_stage(tmp_path: Path) -> None:
    experiment = _experiment_config(tmp_path, "occurrence")
    processed_dir = Path(experiment.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.DataFrame({"stage": ["features"]})
    occurrence_df = pd.DataFrame({"stage": ["occurrence_labels"]})
    severity_df = pd.DataFrame({"stage": ["severity_labels"]})

    features_df.to_parquet(processed_dir / "features.parquet", index=False)
    occurrence_df.to_parquet(labeled_dataset_path(experiment), index=False)
    severity_df.to_parquet(processed_dir / "labeled_features_severity.parquet", index=False)

    loaded = load_dataset(experiment)

    assert loaded["stage"].tolist() == ["occurrence_labels"]


def test_load_dataset_ignores_other_task_labeled_stage(tmp_path: Path) -> None:
    experiment = _experiment_config(tmp_path, "risk")
    processed_dir = Path(experiment.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.DataFrame({"stage": ["features"]})
    occurrence_df = pd.DataFrame({"stage": ["occurrence_labels"]})

    features_df.to_parquet(processed_dir / "features.parquet", index=False)
    occurrence_df.to_parquet(processed_dir / "labeled_features_occurrence.parquet", index=False)

    loaded = load_dataset(experiment)

    assert loaded["stage"].tolist() == ["features"]
