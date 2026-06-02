from __future__ import annotations

from pathlib import Path

import pytest

from agni.config import ExperimentConfig, load_experiment_config


def test_load_experiment_config() -> None:
    config = load_experiment_config(Path("configs/experiments/kalimantan_pilot.yaml"))
    assert config.name == "kalimantan_pilot"
    assert config.data.grid.grid_km == 10


def test_split_buffer_validation() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "task": "occurrence",
                "data": {
                    "grid": {
                        "grid_km": 10,
                        "bbox": {"lon_min": 0, "lon_max": 1, "lat_min": 0, "lat_max": 1},
                    },
                    "temporal": {
                        "reference_start": "2020-01-01",
                        "reference_end": "2020-12-31",
                        "reference_stride_days": 7,
                        "lookback_days": 60,
                        "temporal_windows": [7, 30, 60],
                        "horizon_days": 30,
                    },
                    "split": {
                        "buffer_days": 14,
                        "train_end": "2020-06-30",
                        "val_end": "2020-09-30",
                        "test_end": "2020-12-31",
                    },
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                },
                "model": {"name": "logreg", "task": "classification", "params": {}},
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": "outputs/test",
            }
        )
