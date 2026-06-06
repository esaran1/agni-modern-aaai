from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agni.data.builder import build_dataset
from agni.data.sources import build_adapters
from agni.data.sources.base import month_after_iso, month_start_iso


def _data_config(tmp_path: Path) -> dict:
    return {
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
            "reference_end": "2020-01-01",
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
    }


def test_build_adapters_rejects_unknown_enabled_sources() -> None:
    with pytest.raises(ValueError, match="Unknown enabled data source"):
        build_adapters([SimpleNamespace(name="not_a_real_source", enabled=True, params={})])


def test_build_dataset_raises_on_adapter_failure_by_default(tmp_path: Path) -> None:
    from agni.config import DataConfig

    class FailingAdapter:
        def extract_patch(self, **kwargs):
            raise RuntimeError("boom")

    patch_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "patch_row": [0],
            "patch_col": [0],
            "centroid_lon": [0.5],
            "centroid_lat": [0.5],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )
    config = DataConfig.model_validate(_data_config(tmp_path))

    with pytest.raises(RuntimeError, match="Adapter FailingAdapter failed"):
        build_dataset(config, patch_df, [FailingAdapter()])


def test_build_dataset_extends_reference_dates_by_horizon(tmp_path: Path) -> None:
    from agni.config import DataConfig

    class EchoAdapter:
        def extract_patch(self, **kwargs):
            return {"weather_vpd_mean_l7d": 1.0}

    patch_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "patch_row": [0],
            "patch_col": [0],
            "centroid_lon": [0.5],
            "centroid_lat": [0.5],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )
    config = DataConfig.model_validate(_data_config(tmp_path))

    result = build_dataset(config, patch_df, [EchoAdapter()])

    assert result.dataset["reference_date"].tolist() == [
        date(2020, 1, 1),
        date(2020, 1, 8),
        date(2020, 1, 15),
        date(2020, 1, 22),
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["future_padding_days"] == 21


def test_build_dataset_uses_event_observation_window_for_padding(tmp_path: Path) -> None:
    from agni.config import DataConfig

    class EchoAdapter:
        def extract_patch(self, **kwargs):
            return {"weather_vpd_mean_l14d": 1.0}

    patch_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "patch_row": [0],
            "patch_col": [0],
            "centroid_lon": [0.5],
            "centroid_lat": [0.5],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )
    config_dict = _data_config(tmp_path)
    config_dict["temporal"]["temporal_windows"] = [14, 30]
    config = DataConfig.model_validate(config_dict)

    result = build_dataset(config, patch_df, [EchoAdapter()])

    assert result.dataset["reference_date"].tolist() == [
        date(2020, 1, 1),
        date(2020, 1, 8),
        date(2020, 1, 15),
        date(2020, 1, 22),
        date(2020, 1, 29),
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["future_padding_days"] == 28


def test_month_start_iso_floors_mid_month_dates() -> None:
    assert month_start_iso("2020-01-15") == "2020-01-01"
    assert month_start_iso(date(2020, 2, 29)) == "2020-02-01"


def test_month_after_iso_advances_to_next_month_start() -> None:
    assert month_after_iso("2020-01-15") == "2020-02-01"
    assert month_after_iso(date(2020, 12, 31)) == "2021-01-01"
