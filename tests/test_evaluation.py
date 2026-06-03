from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agni.evaluation.bootstrap import bootstrap_metric
from agni.evaluation.delong import delong_roc_test
from agni.evaluation.metrics import classification_metrics

_EVALUATE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_EVALUATE_SPEC = importlib.util.spec_from_file_location("evaluate_script", _EVALUATE_SCRIPT)
assert _EVALUATE_SPEC is not None and _EVALUATE_SPEC.loader is not None
_EVALUATE_MODULE = importlib.util.module_from_spec(_EVALUATE_SPEC)
_EVALUATE_SPEC.loader.exec_module(_EVALUATE_MODULE)
evaluate_main = _EVALUATE_MODULE.main


def test_classification_metrics_and_bootstrap() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    metrics = classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] > 0.9
    boot = bootstrap_metric(
        y_true,
        y_prob,
        lambda a, b: classification_metrics(a, b)["roc_auc"],
        50,
    )
    assert boot["n_bootstrap"] > 0


def test_classification_metrics_handles_single_class_targets() -> None:
    y_true = np.zeros(6, dtype=int)
    y_prob = np.linspace(0.1, 0.9, 6)

    metrics = classification_metrics(y_true, y_prob)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert "f1" in metrics


def test_delong_test_runs() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.2, 0.9, 0.85, 0.3, 0.8, 0.35, 0.75])
    b = np.array([0.2, 0.25, 0.8, 0.7, 0.4, 0.65, 0.45, 0.6])
    result = delong_roc_test(y_true, a, b)
    assert "p_value" in result


def _risk_dataset(horizon_days: int) -> pd.DataFrame:
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    occurrence_col = f"y_occ_{horizon_days}d"
    for patch in range(6):
        for step in range(15):
            reference_date = base_date + pd.Timedelta(days=7 * step)
            dryness = 0.4 * step + 0.2 * patch
            terrain = 1.0 - 0.05 * patch
            occurrence = int(step >= 4)
            severity = dryness * 0.5 if occurrence else np.nan
            rows.append(
                {
                    "patch_id": f"{patch}",
                    "reference_date": reference_date,
                    "weather_vpd_mean_l7d": dryness,
                    "terrain_twi_mean": terrain,
                    occurrence_col: occurrence,
                    "y_sev_available": occurrence,
                    "y_sev_dnbr": severity,
                }
            )
    return pd.DataFrame(rows)


def test_risk_evaluate_generates_predictions_for_non_default_horizon(tmp_path: Path) -> None:
    horizon_days = 14
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_dir.mkdir()
    output_dir.mkdir()
    _risk_dataset(horizon_days).to_parquet(processed_dir / "features.parquet", index=False)

    config_path = tmp_path / "risk_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "risk-eval",
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
                        "horizon_days": horizon_days,
                    },
                    "split": {
                        "buffer_days": horizon_days,
                        "train_end": "2020-02-15",
                        "val_end": "2020-03-15",
                        "test_end": "2020-04-30",
                    },
                    "raw_dir": str(tmp_path / "raw"),
                    "processed_dir": str(processed_dir),
                },
                "model": {
                    "name": "logreg",
                    "task": "classification",
                    "params": {"max_iter": 200},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    evaluate_main(str(config_path))

    assert (output_dir / "predictions.parquet").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_evaluable"] >= 10
    assert "spearman_rho" in metrics


def test_occurrence_evaluate_persists_generated_predictions(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_dir.mkdir()
    output_dir.mkdir()
    occurrence_df = _risk_dataset(14).copy()
    occurrence_df.to_parquet(processed_dir / "features.parquet", index=False)

    config_path = tmp_path / "occ_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "occ-eval",
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
                    "processed_dir": str(processed_dir),
                },
                "model": {
                    "name": "logreg",
                    "task": "classification",
                    "params": {"max_iter": 200},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    evaluate_main(str(config_path))

    assert (output_dir / "predictions.parquet").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "roc_auc" in metrics


def test_risk_evaluate_uses_separate_stage_model_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_dir.mkdir()
    output_dir.mkdir()
    _risk_dataset(14).to_parquet(processed_dir / "features.parquet", index=False)

    config_path = tmp_path / "risk_stage_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "risk-stage-eval",
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
                    "processed_dir": str(processed_dir),
                },
                "model": {
                    "name": "transformer",
                    "task": "classification",
                    "params": {"epochs": 1},
                    "occurrence": {"name": "random_forest", "params": {"n_estimators": 11}},
                    "severity": {"name": "logreg", "params": {}},
                },
                "evaluation": {"bootstrap_samples": 10, "confidence_level": 0.95},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_fit_risk_pipeline(*args, **kwargs):
        captured.update(kwargs)
        frame = pd.DataFrame(
            {
                "split": ["test"],
                "y_occ_14d": [1],
                "y_sev_dnbr": [0.3],
                "occurrence_prediction": [0.7],
                "severity_prediction": [0.4],
                "risk_score": [0.28],
            }
        )
        return type("RiskResult", (), {"predictions": frame})()

    monkeypatch.setattr(_EVALUATE_MODULE, "fit_risk_pipeline", _fake_fit_risk_pipeline)

    evaluate_main(str(config_path))

    assert captured["occurrence_model_name"] == "random_forest"
    assert captured["occurrence_model_params"] == {
        "n_estimators": 11,
        "random_state": 42,
        "n_jobs": -1,
    }
    assert captured["severity_estimator_name"] == "logreg"
    assert captured["severity_model_params"] == {}
