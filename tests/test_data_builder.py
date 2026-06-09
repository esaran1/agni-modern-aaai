from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agni.data.builder import _is_transient_error, build_dataset
from agni.data.sources import build_adapters
from agni.data.sources.base import month_after_iso, month_start_iso


def test_transient_classifier_retries_only_genuine_transients() -> None:
    # Network/quota/5xx style errors are retryable.
    assert _is_transient_error(TimeoutError("read timed out"))
    assert _is_transient_error(ConnectionError("connection reset by peer"))
    assert _is_transient_error(RuntimeError("HTTP 503 service unavailable"))
    assert _is_transient_error(RuntimeError("Too Many Requests (429)"))
    assert _is_transient_error(RuntimeError("Earth Engine capacity exceeded; try again later"))

    # Deterministic Earth Engine bugs must fail fast, not burn minutes of backoff.
    class EEException(Exception):  # noqa: N818
        pass

    band_error = EEException(
        "Image.select: Band pattern 'B12' was applied to an Image with no bands."
    )
    assert not _is_transient_error(band_error)
    assert not _is_transient_error(EEException("User memory limit exceeded."))
    assert not _is_transient_error(ValueError("unknown column"))


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


def _multi_patch_df(n_patches: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch_id": [f"0_{i}" for i in range(n_patches)],
            "patch_row": [0] * n_patches,
            "patch_col": list(range(n_patches)),
            "centroid_lon": [0.5] * n_patches,
            "centroid_lat": [0.5] * n_patches,
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"] * n_patches,
        }
    )


class _EchoAdapter:
    def extract_patch(self, **kwargs):
        return {"weather_vpd_mean_l7d": 1.0}


def test_build_dataset_checkpoints_and_resumes(tmp_path: Path) -> None:
    from agni.config import DataConfig

    class CountingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def extract_patch(self, **kwargs):
            self.calls += 1
            return {"weather_vpd_mean_l7d": 1.0}

    config = DataConfig.model_validate(_data_config(tmp_path))
    patch_df = _multi_patch_df(3)

    adapter = CountingAdapter()
    first = build_dataset(config, patch_df, [adapter])
    calls_after_first = adapter.calls
    assert calls_after_first > 0

    # Second run should reuse the checkpoint shards and not re-extract anything.
    second = build_dataset(config, patch_df, [adapter])
    assert adapter.calls == calls_after_first
    assert len(second.dataset) == len(first.dataset)


def test_build_dataset_parallel_matches_sequential(tmp_path: Path) -> None:
    from agni.config import DataConfig

    base = _data_config(tmp_path)
    seq_cfg = DataConfig.model_validate({**base, "processed_dir": str(tmp_path / "seq")})
    par_cfg = DataConfig.model_validate({**base, "processed_dir": str(tmp_path / "par")})
    patch_df = _multi_patch_df(4)

    seq = build_dataset(seq_cfg, patch_df, [_EchoAdapter()], max_workers=1)
    par = build_dataset(par_cfg, patch_df, [_EchoAdapter()], max_workers=4)

    left = seq.dataset.sort_values(["patch_id", "reference_date"]).reset_index(drop=True)
    right = par.dataset.sort_values(["patch_id", "reference_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_build_dataset_retries_transient_errors(tmp_path: Path) -> None:
    from agni.config import DataConfig

    class FlakyAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def extract_patch(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporarily unavailable: rate limit exceeded")
            return {"weather_vpd_mean_l7d": 1.0}

    config = DataConfig.model_validate(_data_config(tmp_path))
    adapter = FlakyAdapter()
    result = build_dataset(
        config,
        _multi_patch_df(1),
        [adapter],
        max_retries=5,
        backoff_seconds=0.0,
    )
    assert adapter.calls >= 3
    assert not result.dataset.empty


def test_build_patch_shards_then_merge(tmp_path: Path) -> None:
    from agni.config import DataConfig
    from agni.data.builder import build_patch_shards

    config = DataConfig.model_validate(_data_config(tmp_path))
    patch_df = _multi_patch_df(4)

    # Simulate two cluster array tasks building disjoint patch slices.
    build_patch_shards(config, patch_df.iloc[0::2].reset_index(drop=True), [_EchoAdapter()])
    build_patch_shards(config, patch_df.iloc[1::2].reset_index(drop=True), [_EchoAdapter()])

    # Merge step: every shard already exists, so no extraction is needed.
    merge_adapter = _EchoAdapter()
    result = build_dataset(config, patch_df, [merge_adapter], resume=True)
    assert set(result.dataset["patch_id"]) == set(patch_df["patch_id"])


def test_initialize_earth_engine_is_tolerant_without_credentials() -> None:
    from agni.data.sources.ee_session import initialize_earth_engine

    # Offline/sandbox has no EE credentials; the helper must warn and return a bool,
    # never raise, so mocked/offline pipelines keep working.
    assert isinstance(initialize_earth_engine(high_volume=True), bool)


def test_initialize_earth_engine_tolerates_missing_key_file(tmp_path: Path) -> None:
    from agni.data.sources.ee_session import initialize_earth_engine

    # A bad/missing service-account key must not raise; it should fail soft.
    missing_key = tmp_path / "nope.json"
    assert initialize_earth_engine(key_file=str(missing_key)) is False


def test_month_start_iso_floors_mid_month_dates() -> None:
    assert month_start_iso("2020-01-15") == "2020-01-01"
    assert month_start_iso(date(2020, 2, 29)) == "2020-02-01"


def test_month_after_iso_advances_to_next_month_start() -> None:
    assert month_after_iso("2020-01-15") == "2020-02-01"
    assert month_after_iso(date(2020, 12, 31)) == "2021-01-01"
