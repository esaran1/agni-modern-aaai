from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("propensity_severity_experiment", "experiments/propensity_severity_experiment.py"),
        ("joint_risk_experiment", "experiments/joint_risk_experiment.py"),
        ("conformal_risk_experiment", "experiments/conformal_risk_experiment.py"),
        ("ablation_contributions", "experiments/ablation_contributions.py"),
    ],
)
def test_risk_experiments_use_base_occurrence_model_without_stage_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
) -> None:
    module = _load_module(module_name, relative_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "entrypoint-test",
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
                    "name": "logreg",
                    "task": "classification",
                    "params": {"max_iter": 200},
                    "severity": {"name": "logreg", "params": {}},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    split_df = pd.DataFrame(
        {
            "patch_id": ["0"],
            "reference_date": [pd.Timestamp("2020-01-01")],
            "split": ["train"],
            "weather_vpd_mean_l7d": [1.0],
            "terrain_twi_mean": [0.2],
            "y_occ_14d": [0],
            "y_sev_available": [0],
            "y_sev_dnbr": [0.0],
        }
    )

    if hasattr(module, "load_dataset"):
        monkeypatch.setattr(module, "load_dataset", lambda experiment: split_df.copy())
    if hasattr(module, "split_dataset"):
        monkeypatch.setattr(module, "split_dataset", lambda df, experiment: df.copy())
    if hasattr(module, "carve_conformal_calibration_split"):
        monkeypatch.setattr(module, "carve_conformal_calibration_split", lambda df, **kwargs: df)
    monkeypatch.setattr(module, "infer_feature_columns", lambda df: ["weather_vpd_mean_l7d"])

    class _StopHereError(Exception):
        pass

    def _fake_attach(*args, **kwargs):
        assert kwargs["model_name"] == "logreg"
        assert kwargs["model_params"] == {"max_iter": 200}
        raise _StopHereError

    monkeypatch.setattr(module, "attach_occurrence_propensity", _fake_attach)

    with pytest.raises(_StopHereError):
        module.main(str(config_path))


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("propensity_severity_experiment", "experiments/propensity_severity_experiment.py"),
        ("joint_risk_experiment", "experiments/joint_risk_experiment.py"),
        ("conformal_risk_experiment", "experiments/conformal_risk_experiment.py"),
        ("ablation_contributions", "experiments/ablation_contributions.py"),
    ],
)
def test_experiments_use_separate_risk_stage_model_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
) -> None:
    module = _load_module(module_name, relative_path)
    config_path = tmp_path / "risk_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "entrypoint-risk-test",
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
                    "occurrence": {
                        "name": "random_forest",
                        "params": {"n_estimators": 17, "random_state": 3},
                    },
                    "severity": {"name": "logreg", "params": {}},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    split_df = pd.DataFrame(
        {
            "patch_id": ["0"],
            "reference_date": [pd.Timestamp("2020-01-01")],
            "split": ["train"],
            "weather_vpd_mean_l7d": [1.0],
            "terrain_twi_mean": [0.2],
            "y_occ_14d": [0],
            "y_sev_available": [1],
            "y_sev_dnbr": [0.1],
        }
    )

    if hasattr(module, "load_dataset"):
        monkeypatch.setattr(module, "load_dataset", lambda experiment: split_df.copy())
    if hasattr(module, "split_dataset"):
        monkeypatch.setattr(module, "split_dataset", lambda df, experiment: df.copy())
    if hasattr(module, "carve_conformal_calibration_split"):
        monkeypatch.setattr(module, "carve_conformal_calibration_split", lambda df, **kwargs: df)
    monkeypatch.setattr(module, "infer_feature_columns", lambda df: ["weather_vpd_mean_l7d"])

    class _StopHereError(Exception):
        pass

    def _fake_attach(*args, **kwargs):
        assert kwargs["model_name"] == "random_forest"
        assert kwargs["model_params"] == {
            "n_estimators": 17,
            "random_state": 3,
            "n_jobs": -1,
        }
        return type(
            "OccurrenceResult",
            (),
            {
                "predictions": split_df.assign(
                    propensity_score=0.4,
                    occurrence_prediction=0.4,
                ),
                "metrics": {"roc_auc": 0.5},
            },
        )()

    def _fake_train(*args, **kwargs):
        assert kwargs["estimator_name"] == "logreg"
        assert kwargs["model_params"] == {}
        raise _StopHereError

    monkeypatch.setattr(module, "attach_occurrence_propensity", _fake_attach)
    monkeypatch.setattr(module, "train_severity_variant", _fake_train)

    with pytest.raises(_StopHereError):
        module.main(str(config_path))


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("propensity_severity_experiment", "experiments/propensity_severity_experiment.py"),
        ("joint_risk_experiment", "experiments/joint_risk_experiment.py"),
        ("conformal_risk_experiment", "experiments/conformal_risk_experiment.py"),
        ("ablation_contributions", "experiments/ablation_contributions.py"),
    ],
)
def test_risk_experiment_entrypoints_reject_non_risk_configs(
    tmp_path: Path,
    module_name: str,
    relative_path: str,
) -> None:
    module = _load_module(module_name, relative_path)
    config_path = tmp_path / "occurrence_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "wrong-task",
                "task": "occurrence",
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
                    "name": "logreg",
                    "task": "classification",
                    "params": {"max_iter": 200},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires config.task == 'risk'"):
        module.main(str(config_path))
