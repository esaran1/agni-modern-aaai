from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agni.config import ExperimentConfig
from agni.pipeline import enrich_feature_table, fit_and_predict, save_training_outputs


def build_synthetic_dataset() -> pd.DataFrame:
    rows = []
    start = date(2020, 1, 1)
    for patch_row in range(6):
        for patch_col in range(6):
            for step in range(18):
                ref_date = start + timedelta(days=step * 21)
                temp = 295 + 0.7 * step + 0.2 * patch_row
                dew = temp - 4.0 + 0.1 * patch_col
                precip = max(0.1, 15 - 0.6 * step + 0.1 * patch_col)
                vpd_proxy = temp - dew
                score = 0.5 * vpd_proxy + 0.15 * step + 0.2 * (patch_row % 2) - 0.1 * precip
                y_occ = int(score > 1.8 and step % 4 != 0)
                y_sev = max(0.0, score / 4.0) if y_occ else np.nan
                rows.append(
                    {
                        "patch_id": f"{patch_row}_{patch_col}",
                        "patch_row": patch_row,
                        "patch_col": patch_col,
                        "reference_date": ref_date,
                        "weather_temperature_2m_mean_l7d": temp,
                        "weather_temperature_2m_mean_l30d": temp - 0.5,
                        "weather_temperature_2m_mean_l60d": temp - 1.0,
                        "weather_temperature_2m_max_l7d": temp + 3.0,
                        "weather_dewpoint_temperature_2m_mean_l7d": dew,
                        "weather_dewpoint_temperature_2m_mean_l30d": dew - 0.2,
                        "weather_dewpoint_temperature_2m_mean_l60d": dew - 0.4,
                        "weather_total_precipitation_sum_mean_l7d": precip,
                        "weather_total_precipitation_sum_mean_l30d": precip + 2.0,
                        "weather_total_precipitation_sum_mean_l60d": precip + 4.0,
                        "weather_u_component_of_wind_10m_mean_l7d": 1.5 + 0.1 * patch_col,
                        "weather_v_component_of_wind_10m_mean_l7d": 0.8 + 0.1 * patch_row,
                        "optical_b2_mean_l7d": 0.15,
                        "optical_b3_mean_l7d": 0.22,
                        "optical_b4_mean_l7d": 0.27,
                        "optical_b8_mean_l7d": 0.58,
                        "optical_b11_mean_l7d": 0.21,
                        "optical_b12_mean_l7d": 0.18,
                        "terrain_twi_mean": 3.0 + 0.05 * patch_row,
                        "terrain_elevation_min": 5.0 + patch_row,
                        "terrain_elevation_max": 12.0 + patch_row,
                        "y_occ_30d": y_occ,
                        "y_sev_available": int(y_occ == 1),
                        "y_sev_dnbr": y_sev,
                    }
                )
    return pd.DataFrame(rows)


def test_end_to_end_synthetic(tmp_path: Path) -> None:
    df = enrich_feature_table(build_synthetic_dataset())
    config = ExperimentConfig.model_validate(
        {
            "name": "synthetic",
            "task": "occurrence",
            "data": {
                "grid": {
                    "grid_km": 10,
                    "bbox": {"lon_min": 0, "lon_max": 1, "lat_min": 0, "lat_max": 1},
                },
                "temporal": {
                    "reference_start": "2020-01-01",
                    "reference_end": "2020-12-31",
                    "reference_stride_days": 21,
                    "lookback_days": 60,
                    "temporal_windows": [7, 30, 60],
                    "horizon_days": 30,
                },
                "split": {
                    "buffer_days": 30,
                    "train_end": "2020-06-30",
                    "val_end": "2020-09-30",
                    "test_end": "2020-12-31",
                },
                "spatial_blocks": {"block_size_km": 20, "n_folds": 3, "seed": 42},
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "model": {"name": "xgboost", "task": "classification", "params": {"random_state": 42}},
            "evaluation": {"bootstrap_samples": 50, "confidence_level": 0.95},
            "output_dir": str(tmp_path / "outputs"),
        }
    )
    model, predictions, metrics = fit_and_predict(df, config)
    save_training_outputs(config, model, predictions, metrics)
    assert (tmp_path / "outputs" / "model.pkl").exists()
    assert (tmp_path / "outputs" / "predictions.parquet").exists()
    assert metrics["roc_auc"] >= 0.5
