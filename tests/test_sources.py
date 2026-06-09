from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agni.data.sources.base import ms_timestamp_to_iso_date


def test_ms_timestamp_to_iso_date_handles_none_and_value() -> None:
    assert ms_timestamp_to_iso_date(None) is None
    # 2020-01-12 00:00:00 UTC in epoch milliseconds.
    ms = 1578787200000
    assert ms_timestamp_to_iso_date(ms) == "2020-01-12"


def _install_fake_ee(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal fake ``ee`` that mimics lazy server-side objects.

    ``reduceRegion(...).get(key)`` returns an unevaluated object; only
    ``ee.Dictionary(...).getInfo()`` resolves it to a plain Python scalar. This is
    exactly the shape that previously caused parquet serialization to fail when an
    adapter forgot to materialize.
    """

    class FakeValue:
        def __init__(self, value: float) -> None:
            self.value = value

    class FakeRegionResult:
        def get(self, key: str) -> FakeValue:
            return FakeValue(1.5)

    class FakeReducer:
        def combine(self, *args, **kwargs) -> FakeReducer:
            return self

    class FakeImage:
        def select(self, band: str) -> FakeImage:
            return self

        def mean(self) -> FakeImage:
            return self

        def reduceRegion(self, **kwargs) -> FakeRegionResult:  # noqa: N802
            return FakeRegionResult()

    class FakeCollection(FakeImage):
        def filterDate(self, start, end) -> FakeCollection:  # noqa: N802
            return self

        def filterBounds(self, geometry) -> FakeCollection:  # noqa: N802
            return self

    class FakeDictionary:
        def __init__(self, mapping: dict) -> None:
            self.mapping = mapping

        def getInfo(self):  # noqa: N802
            return {
                key: (value.value if isinstance(value, FakeValue) else value)
                for key, value in self.mapping.items()
            }

    fake_ee = SimpleNamespace(
        Geometry=lambda geo: geo,
        ImageCollection=lambda collection_id: FakeCollection(),
        Reducer=SimpleNamespace(
            mean=lambda: FakeReducer(),
            stdDev=lambda: FakeReducer(),
            minMax=lambda: FakeReducer(),
        ),
        Dictionary=FakeDictionary,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)


def test_era5_adapter_returns_materialized_python_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ee(monkeypatch)
    from agni.data.sources.era5 import ERA5LandAdapter

    geometry = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0.0, 0.0]})
    features = ERA5LandAdapter().extract_patch(
        geometry=geometry,
        reference_date="2019-08-15",
        lookback_days=30,
        temporal_windows=[7],
    )

    assert features, "expected ERA5 features to be produced"
    # The whole point of the fix: values must be plain Python numbers, never
    # unevaluated Earth Engine objects (which break parquet serialization).
    assert all(isinstance(value, float | int) for value in features.values())
    assert "weather_temperature_2m_mean_l7d" in features


def test_sentinel2_adapter_handles_empty_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSize:
        def getInfo(self):  # noqa: N802
            return 0

    class FakeCollection:
        def filterDate(self, start, end):  # noqa: N802
            return self

        def filterBounds(self, geometry):  # noqa: N802
            return self

        def filter(self, predicate):
            return self

        def size(self):
            return FakeSize()

    class FakeReducer:
        def combine(self, *args, **kwargs) -> FakeReducer:
            return self

    fake_ee = SimpleNamespace(
        Geometry=lambda geo: geo,
        ImageCollection=lambda collection_id: FakeCollection(),
        Filter=SimpleNamespace(lt=lambda field, value: (field, value)),
        Reducer=SimpleNamespace(
            mean=lambda: FakeReducer(),
            stdDev=lambda: FakeReducer(),
            minMax=lambda: FakeReducer(),
        ),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    from agni.data.sources.sentinel2 import Sentinel2Adapter

    geometry = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0.0, 0.0]})
    features = Sentinel2Adapter().extract_patch(
        geometry=geometry,
        reference_date="2019-08-15",
        lookback_days=30,
        temporal_windows=[7],
    )

    # An empty window must yield explicit nulls for every band/stat, never raise.
    assert features["optical_b12_mean_l7d"] is None
    assert features["optical_b2_max_l7d"] is None
    assert len(features) == len(Sentinel2Adapter.BANDS) * 4
