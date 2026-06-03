from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agni.config import load_experiment_config


def _base_config(tmp_path: Path) -> dict:
    return {
        "name": "config-test",
        "task": "risk",
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
                "reference_end": "2020-03-31",
                "reference_stride_days": 7,
                "lookback_days": 30,
                "temporal_windows": [7, 14, 30],
                "horizon_days": 14,
            },
            "split": {
                "buffer_days": 14,
                "train_end": "2020-01-31",
                "val_end": "2020-02-29",
                "test_end": "2020-03-31",
            },
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "model": {
            "name": "transformer",
            "task": "classification",
            "params": {"epochs": 1},
        },
        "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
        "output_dir": str(tmp_path / "outputs"),
    }


def test_risk_config_requires_supported_severity_family(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_risk.yaml"
    config_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")

    with pytest.raises(ValueError, match="severity estimator family"):
        load_experiment_config(config_path)


def test_risk_config_allows_separate_occurrence_and_severity_models(tmp_path: Path) -> None:
    config_dict = _base_config(tmp_path)
    config_dict["model"]["occurrence"] = {"name": "transformer", "params": {"epochs": 2}}
    config_dict["model"]["severity"] = {"name": "logreg", "params": {}}
    config_dict["model"]["comparison_params"] = {"random_forest": {"n_estimators": 25}}
    config_path = tmp_path / "valid_risk.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    config = load_experiment_config(config_path)

    assert config.model.resolve_occurrence_model_name() == "transformer"
    assert config.model.resolve_occurrence_model_params() == {"epochs": 2}
    assert config.model.resolve_severity_model_name() == "logreg"
    assert config.model.resolve_severity_model_params() == {}
    assert config.model.resolve_comparison_model_params("random_forest") == {
        "n_estimators": 25
    }


def test_stage_specific_param_overrides_merge_with_base_or_defaults(tmp_path: Path) -> None:
    config_dict = _base_config(tmp_path)
    config_dict["model"] = {
        "name": "logreg",
        "task": "classification",
        "params": {"max_iter": 500, "random_state": 42},
        "occurrence": {"params": {"max_iter": 900}},
        "severity": {"name": "random_forest", "params": {"n_estimators": 50}},
    }
    config_path = tmp_path / "merged_overrides.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    config = load_experiment_config(config_path)

    assert config.model.resolve_occurrence_model_name() == "logreg"
    assert config.model.resolve_occurrence_model_params() == {
        "max_iter": 900,
        "random_state": 42,
    }
    assert config.model.resolve_severity_model_name() == "random_forest"
    assert config.model.resolve_severity_model_params() == {
        "n_estimators": 50,
        "random_state": 42,
        "n_jobs": -1,
    }


@pytest.mark.parametrize(
    ("experiment_task", "model_task", "message"),
    [
        ("occurrence", "regression", "Occurrence experiments require model.task"),
        ("severity", "classification", "Severity experiments require model.task"),
        ("risk", "regression", "Risk experiments require model.task"),
    ],
)
def test_experiment_and_model_tasks_must_be_semantically_compatible(
    tmp_path: Path,
    experiment_task: str,
    model_task: str,
    message: str,
) -> None:
    config_dict = _base_config(tmp_path)
    config_dict["task"] = experiment_task
    config_dict["model"]["task"] = model_task
    if experiment_task != "risk":
        config_dict["model"]["name"] = "logreg"
        config_dict["model"]["params"] = {"max_iter": 200}
        config_dict["model"].pop("severity", None)
        config_dict["model"].pop("occurrence", None)
    config_path = tmp_path / f"{experiment_task}_{model_task}.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(config_path)
