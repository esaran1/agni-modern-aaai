from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _risk_config_dict(tmp_path: Path) -> dict:
    return {
        "name": "runner-risk-test",
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
                "reference_end": "2020-04-30",
                "reference_stride_days": 7,
                "lookback_days": 30,
                "temporal_windows": [7, 14, 30],
                "horizon_days": 14,
            },
            "split": {
                "buffer_days": 14,
                "train_end": "2020-02-15",
                "val_end": "2020-03-15",
                "test_end": "2020-04-30",
            },
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "model": {
            "name": "transformer",
            "task": "classification",
            "params": {"epochs": 1},
            "occurrence": {"name": "random_forest", "params": {"n_estimators": 11}},
            "severity": {"name": "logreg", "params": {}},
        },
        "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
        "output_dir": str(tmp_path / "outputs"),
    }


def test_train_script_routes_risk_configs_to_risk_pipeline(tmp_path, monkeypatch) -> None:
    module = _load_module("train_script", "scripts/train.py")
    config_path = tmp_path / "risk_train.yaml"
    config_path.write_text(yaml.safe_dump(_risk_config_dict(tmp_path)), encoding="utf-8")

    df = pd.DataFrame(
        {
            "patch_id": ["0"],
            "reference_date": [pd.Timestamp("2020-01-01")],
            "split": ["test"],
            "weather_vpd_mean_l7d": [1.0],
            "terrain_twi_mean": [0.2],
            "y_occ_14d": [1],
            "y_sev_available": [1],
            "y_sev_dnbr": [0.3],
        }
    )
    monkeypatch.setattr(module, "load_dataset", lambda experiment: df.copy())
    monkeypatch.setattr(module, "split_dataset", lambda frame, experiment: frame.copy())
    monkeypatch.setattr(module, "infer_feature_columns", lambda frame: ["weather_vpd_mean_l7d"])

    captured = {}

    def _fake_fit_risk_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return type(
            "RiskResult",
            (),
            {
                "model": {"occurrence_model": object(), "severity_model": object()},
                "predictions": df.assign(
                    occurrence_prediction=0.7,
                    severity_prediction=0.4,
                    risk_score=0.28,
                ),
                "metrics": {"spearman_rho": 0.5},
            },
        )()

    def _fake_fit_and_predict(*args, **kwargs):
        raise AssertionError("fit_and_predict should not be called for risk configs")

    def _fake_save_outputs(experiment, model, predictions, metrics):
        captured["saved_model"] = model
        captured["saved_metrics"] = metrics
        output_dir = Path(experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    monkeypatch.setattr(module, "fit_risk_pipeline", _fake_fit_risk_pipeline)
    monkeypatch.setattr(module, "fit_and_predict", _fake_fit_and_predict)
    monkeypatch.setattr(module, "save_training_outputs", _fake_save_outputs)

    module.main(str(config_path))

    assert captured["occurrence_model_name"] == "random_forest"
    assert captured["severity_estimator_name"] == "logreg"
    assert captured["saved_metrics"] == {"spearman_rho": 0.5}
    assert (tmp_path / "outputs" / "predictions.parquet").exists()


def test_run_experiment_script_routes_risk_configs_to_risk_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_module("run_experiment_script", "scripts/run_experiment.py")
    config_path = tmp_path / "risk_run.yaml"
    config_path.write_text(yaml.safe_dump(_risk_config_dict(tmp_path)), encoding="utf-8")

    grid = pd.DataFrame({"patch_id": ["0"]})
    dataset = pd.DataFrame(
        {
            "patch_id": ["0"],
            "reference_date": [pd.Timestamp("2020-01-01")],
            "weather_vpd_mean_l7d": [1.0],
            "terrain_twi_mean": [0.2],
            "y_occ_14d": [1],
            "y_sev_available": [1],
            "y_sev_dnbr": [0.3],
        }
    )
    split_df = dataset.assign(split="test")
    monkeypatch.setattr(module, "build_patch_grid", lambda bbox, grid_km: grid.copy())
    monkeypatch.setattr(module, "build_adapters", lambda sources: [])
    monkeypatch.setattr(
        module,
        "build_dataset",
        lambda data_config, built_grid, adapters, output_name: type(
            "DatasetResult",
            (),
            {"dataset_path": tmp_path / "processed" / "dataset.parquet"},
        )(),
    )
    monkeypatch.setattr(module.pd, "read_parquet", lambda path: dataset.copy())
    monkeypatch.setattr(module, "enrich_feature_table", lambda frame, stride_days: frame.copy())
    monkeypatch.setattr(module, "split_dataset", lambda frame, experiment: split_df.copy())
    monkeypatch.setattr(module, "infer_feature_columns", lambda frame: ["weather_vpd_mean_l7d"])

    captured = {}

    def _fake_fit_risk_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return type(
            "RiskResult",
            (),
            {
                "model": {"occurrence_model": object(), "severity_model": object()},
                "predictions": split_df.assign(
                    occurrence_prediction=0.7,
                    severity_prediction=0.4,
                    risk_score=0.28,
                ),
                "metrics": {"spearman_rho": 0.6},
            },
        )()

    def _fake_fit_and_predict(*args, **kwargs):
        raise AssertionError("fit_and_predict should not be called for risk configs")

    def _fake_save_outputs(experiment, model, predictions, metrics):
        captured["saved_model"] = model
        captured["saved_metrics"] = metrics

    monkeypatch.setattr(module, "fit_risk_pipeline", _fake_fit_risk_pipeline)
    monkeypatch.setattr(module, "fit_and_predict", _fake_fit_and_predict)
    monkeypatch.setattr(module, "save_training_outputs", _fake_save_outputs)

    module.main(str(config_path))

    assert captured["occurrence_model_name"] == "random_forest"
    assert captured["severity_estimator_name"] == "logreg"
    assert captured["saved_metrics"] == {"spearman_rho": 0.6}
