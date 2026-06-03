from __future__ import annotations

import importlib.util
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


def test_model_comparison_uses_family_specific_params(tmp_path, monkeypatch) -> None:
    module = _load_module("model_comparison_experiment", "experiments/model_comparison.py")
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "comparison-test",
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
                    "name": "xgboost",
                    "task": "classification",
                    "params": {"n_estimators": 55, "max_depth": 3},
                    "comparison_params": {
                        "random_forest": {"n_estimators": 21, "random_state": 9}
                    },
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(tmp_path / "outputs"),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "load_dataset", lambda experiment: pd.DataFrame())

    seen = []

    def _fake_fit_and_predict(df, experiment):
        seen.append((experiment.model.name, dict(experiment.model.params)))
        return object(), pd.DataFrame(), {"metric": 1.0}

    monkeypatch.setattr(module, "fit_and_predict", _fake_fit_and_predict)

    module.main(str(config_path), ["xgboost", "random_forest", "logreg"])

    assert seen[0] == ("xgboost", {"n_estimators": 55, "max_depth": 3})
    assert seen[1] == ("random_forest", {"n_estimators": 21, "random_state": 9})
    assert seen[2] == ("logreg", {"max_iter": 500, "random_state": 42})
