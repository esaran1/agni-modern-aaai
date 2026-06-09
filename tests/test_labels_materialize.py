from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from agni.data.sources.modis_burn import MODISBurnAdapter
from agni.labels.materialize import (
    OccurrenceLabel,
    SeverityLabel,
    _event_dates_from_burn_timestamps,
    extract_future_modis_occurrence,
    materialize_labels,
)
from agni.labels.occurrence import build_occurrence_labels
from agni.labels.severity import build_severity_labels

JAN_12_2020_MS = pd.Timestamp("2020-01-12").value // 1_000_000
JAN_20_2020_MS = pd.Timestamp("2020-01-20").value // 1_000_000


def test_materialize_labels_uses_extractors_and_writes_expected_columns() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0"],
            "reference_date": [date(2020, 1, 1), date(2020, 1, 8)],
            "weather_vpd_mean_l7d": [1.0, 1.2],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    def fake_occurrence(geometry, reference_date, horizon_days):
        if reference_date == date(2020, 1, 1):
            return OccurrenceLabel(event_date=date(2020, 1, 10), label=1)
        return OccurrenceLabel(event_date=None, label=0)

    def fake_severity(geometry, burn_date, severity_window_days):
        return SeverityLabel(
            prefire_nbr=0.6,
            postfire_nbr=0.2,
            dnbr=0.4,
            severity_available=1,
            severity_class="high",
        )

    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=14,
        occurrence_extractor=fake_occurrence,
        severity_extractor=fake_severity,
    )

    first = labeled.iloc[0]
    second = labeled.iloc[1]
    assert first["y_occ_14d"] == 1
    assert str(first["event_date"]).startswith("2020-01-10")
    assert first["label_nbr_prefire"] == 0.6
    assert first["label_nbr_postfire"] == 0.2
    assert first["y_sev_available"] == 1
    assert first["y_sev_dnbr"] == pytest.approx(0.4)
    assert first["y_sev_class"] == "high"
    assert second["y_occ_14d"] == 0
    assert pd.isna(second["y_sev_dnbr"])


def test_materialize_labels_derives_occurrence_and_severity_locally() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0", "0_0", "0_0"],
            "reference_date": [
                date(2020, 1, 1),
                date(2020, 1, 8),
                date(2020, 1, 15),
                date(2020, 1, 22),
                date(2020, 1, 29),
            ],
            "temporal_burn_count_l7d": [0.0, 0.0, 0.0, 1.0, 0.0],
            "temporal_burn_date_l7d": [pd.NA, pd.NA, pd.NA, 20.0, pd.NA],
            "temporal_burn_timestamp_l7d": [pd.NA, pd.NA, pd.NA, JAN_20_2020_MS, pd.NA],
            "optical_nbr_mean_l7d": [0.7, 0.65, 0.6, 0.4, 0.2],
            "weather_vpd_mean_l7d": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=14,
        lookback_days=30,
        severity_window_days=14,
        reference_stride_days=7,
        label_reference_end=date(2020, 1, 15),
    )

    assert labeled["reference_date"].tolist() == [
        date(2020, 1, 1),
        date(2020, 1, 8),
        date(2020, 1, 15),
    ]

    first = labeled.iloc[0]
    second = labeled.iloc[1]
    third = labeled.iloc[2]
    assert first["y_occ_14d"] == 0
    assert pd.isna(first["y_sev_dnbr"])
    assert second["y_occ_14d"] == 1
    assert str(second["event_date"]).startswith("2020-01-20")
    assert second["label_nbr_prefire"] == 0.6
    assert second["label_nbr_postfire"] == 0.2
    assert second["y_sev_available"] == 1
    assert second["y_sev_dnbr"] == pytest.approx(0.4)
    assert second["y_sev_class"] == "high"
    assert third["y_occ_14d"] == 1
    assert str(third["event_date"]).startswith("2020-01-20")


def test_materialize_labels_uses_sparse_severity_queries_for_shared_event() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0", "0_0"],
            "reference_date": [
                date(2020, 1, 1),
                date(2020, 1, 8),
                date(2020, 1, 15),
                date(2020, 1, 22),
            ],
            "temporal_burn_count_l7d": [0.0, 0.0, 1.0, 0.0],
            "temporal_burn_date_l7d": [pd.NA, pd.NA, 12.0, pd.NA],
            "temporal_burn_timestamp_l7d": [pd.NA, pd.NA, JAN_12_2020_MS, pd.NA],
            "optical_nbr_mean_l7d": [0.7, 0.6, 0.4, 0.2],
            "weather_vpd_mean_l7d": [1.0, 1.1, 1.2, 1.3],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )
    calls: list[date] = []

    def fake_severity(geometry, burn_date, severity_window_days):
        calls.append(burn_date)
        return SeverityLabel(
            prefire_nbr=0.55,
            postfire_nbr=0.25,
            dnbr=0.3,
            severity_available=1,
            severity_class="high",
        )

    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=14,
        severity_window_days=14,
        reference_stride_days=7,
        label_reference_end=date(2020, 1, 8),
        severity_extractor=fake_severity,
    )

    assert calls == [date(2020, 1, 12)]
    assert labeled["y_occ_14d"].tolist() == [1, 1]
    assert labeled["y_sev_dnbr"].tolist() == pytest.approx([0.3, 0.3])


def test_materialize_labels_raises_on_observed_event_without_burn_date() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0"],
            "reference_date": [date(2020, 1, 1), date(2020, 1, 8)],
            "temporal_fire_count_l7d": [0.0, 1.0],
            "optical_nbr_mean_l7d": [0.7, 0.4],
            "weather_vpd_mean_l7d": [1.0, 1.1],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    with pytest.raises(ValueError, match="require an inferable temporal_burn_date"):
        materialize_labels(
            features_df=features_df,
            grid_df=grid_df,
            horizon_days=14,
            reference_stride_days=7,
            label_reference_end=date(2020, 1, 1),
        )


def test_materialize_labels_can_skip_severity_for_occurrence_workflows() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0", "0_0"],
            "reference_date": [
                date(2020, 1, 1),
                date(2020, 1, 8),
                date(2020, 1, 15),
                date(2020, 1, 22),
            ],
            "temporal_burn_count_l7d": [0.0, 0.0, 1.0, 0.0],
            "temporal_burn_date_l7d": [pd.NA, pd.NA, 12.0, pd.NA],
            "temporal_burn_timestamp_l7d": [pd.NA, pd.NA, JAN_12_2020_MS, pd.NA],
            "optical_nbr_mean_l7d": [0.7, 0.6, 0.4, 0.2],
            "weather_vpd_mean_l7d": [1.0, 1.1, 1.2, 1.3],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("severity extractor should not run for occurrence-only labels")

    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=14,
        reference_stride_days=7,
        label_reference_end=date(2020, 1, 8),
        severity_extractor=fail_if_called,
        materialize_severity=False,
    )

    assert labeled["y_occ_14d"].tolist() == [1, 1]
    assert labeled["y_sev_available"].tolist() == [0, 0]
    assert labeled["y_sev_dnbr"].isna().all()


def test_materialize_labels_prefers_absolute_burn_timestamp_across_year_boundary() -> None:
    burn_timestamp = pd.Timestamp("2020-12-31").value // 1_000_000
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0", "0_0"],
            "reference_date": [
                date(2020, 12, 24),
                date(2020, 12, 31),
                date(2021, 1, 7),
                date(2021, 1, 14),
            ],
            "temporal_burn_count_l7d": [0.0, 0.0, 1.0, 0.0],
            "temporal_burn_date_l7d": [pd.NA, pd.NA, 2.0, pd.NA],
            "temporal_burn_timestamp_l7d": [pd.NA, pd.NA, burn_timestamp, pd.NA],
            "optical_nbr_mean_l7d": [0.7, 0.6, 0.4, 0.2],
            "weather_vpd_mean_l7d": [1.0, 1.1, 1.2, 1.3],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=14,
        reference_stride_days=7,
        label_reference_end=date(2020, 12, 24),
        materialize_severity=False,
    )

    assert labeled["y_occ_14d"].tolist() == [1]
    assert str(labeled.iloc[0]["event_date"]).startswith("2020-12-31")


def test_materialize_labels_rejects_burn_features_without_timestamp_column() -> None:
    features_df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0"],
            "reference_date": [date(2020, 1, 1), date(2020, 1, 8)],
            "temporal_burn_count_l7d": [0.0, 1.0],
            "temporal_burn_date_l7d": [pd.NA, 8.0],
            "optical_nbr_mean_l7d": [0.7, 0.4],
            "weather_vpd_mean_l7d": [1.0, 1.1],
        }
    )
    grid_df = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "geometry_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    )

    with pytest.raises(ValueError, match="temporal_burn_timestamp features"):
        materialize_labels(
            features_df=features_df,
            grid_df=grid_df,
            horizon_days=14,
            reference_stride_days=7,
            label_reference_end=date(2020, 1, 1),
            materialize_severity=False,
        )


def test_event_dates_from_burn_timestamps_use_absolute_dates() -> None:
    timestamps = [
        pd.Timestamp("2020-12-31").value // 1_000_000,
        pd.Timestamp("2021-01-02").value // 1_000_000,
        None,
        0,
    ]

    event_dates = _event_dates_from_burn_timestamps(
        timestamps=timestamps,
        reference_date=date(2020, 12, 24),
        horizon_days=14,
    )

    assert event_dates == [date(2020, 12, 31), date(2021, 1, 2)]


def test_extract_future_modis_occurrence_includes_overlapping_start_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_calls: list[tuple[str, str]] = []

    class FakeAggregate:
        def __init__(self, payload):
            self.payload = payload

        def getInfo(self):  # noqa: N802
            return self.payload

    class FakeCollection:
        def filterDate(self, start, end):  # noqa: N802
            filter_calls.append((start, end))
            return self

        def filterBounds(self, geometry):  # noqa: N802
            return self

        def map(self, fn):
            return self

        def aggregate_array(self, name):
            return FakeAggregate([])

    fake_ee = SimpleNamespace(
        Geometry=lambda geo: geo,
        ImageCollection=lambda collection_id: FakeCollection(),
        Reducer=SimpleNamespace(mean=lambda: None, min=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    geometry = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0.0, 0.0]})
    label = extract_future_modis_occurrence(geometry, date(2020, 1, 15), 14)

    assert label == OccurrenceLabel(event_date=None, label=0)
    assert filter_calls == [("2020-01-01", "2020-02-01")]


def test_modis_burn_adapter_includes_overlapping_start_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_calls: list[tuple[str, str]] = []

    class FakeRegionResult:
        def get(self, key):
            return None

    class FakeDictionary:
        def __init__(self, mapping):
            self.mapping = mapping

        def getInfo(self):  # noqa: N802
            return dict(self.mapping)

    class FakeImage:
        def reduceRegion(self, **kwargs):  # noqa: N802
            return FakeRegionResult()

    class FakeCollection:
        def filterDate(self, start, end):  # noqa: N802
            filter_calls.append((start, end))
            return self

        def filterBounds(self, geometry):  # noqa: N802
            return self

        def map(self, fn):
            return self

        def max(self):
            return FakeImage()

        def min(self):
            return FakeImage()

        def aggregate_min(self, name):
            return None

    fake_ee = SimpleNamespace(
        Geometry=lambda geo: geo,
        ImageCollection=lambda collection_id: FakeCollection(),
        Reducer=SimpleNamespace(mean=lambda: None, min=lambda: None),
        Dictionary=FakeDictionary,
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)

    geometry = SimpleNamespace(__geo_interface__={"type": "Point", "coordinates": [0.0, 0.0]})
    adapter = MODISBurnAdapter()
    features = adapter.extract_patch(
        geometry=geometry,
        reference_date="2020-01-15",
        lookback_days=30,
        temporal_windows=[7],
    )

    assert ("2020-01-01", "2020-02-01") in filter_calls
    assert ("2019-12-01", "2020-02-01") in filter_calls
    assert features["temporal_burn_count_l7d"] is None


def test_build_occurrence_labels_defaults_to_observed_event_contract() -> None:
    frame = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_0", "0_0"],
            "reference_date": [date(2020, 1, 1), date(2020, 1, 8), date(2020, 1, 15)],
            "observed_event": [0, 0, 1],
            "observed_event_date": [pd.NaT, pd.NaT, date(2020, 1, 20)],
        }
    )

    labeled = build_occurrence_labels(frame, horizon_days=14)

    assert labeled["y_occ_14d"].tolist() == [0, 1, 1]


def test_build_occurrence_labels_rejects_legacy_missing_columns() -> None:
    frame = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "reference_date": [date(2020, 1, 1)],
        }
    )

    with pytest.raises(ValueError, match="Missing: observed_event, observed_event_date"):
        build_occurrence_labels(frame)


def test_build_occurrence_labels_rejects_positive_events_without_dates() -> None:
    frame = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "reference_date": [date(2020, 1, 1)],
            "observed_event": [1],
            "observed_event_date": [pd.NaT],
        }
    )

    with pytest.raises(ValueError, match="requires dated positive events"):
        build_occurrence_labels(frame)


def test_build_severity_labels_defaults_to_event_date_contract() -> None:
    frame = pd.DataFrame(
        {
            "event_date": [date(2020, 1, 10), pd.NaT],
            "label_nbr_prefire": [0.7, 0.5],
            "label_nbr_postfire": [0.2, 0.4],
            "y_occ_14d": [1, 0],
        }
    )

    labeled = build_severity_labels(frame)

    assert labeled["y_sev_available"].tolist() == [1, 0]
    assert labeled["y_sev_dnbr"].tolist()[0] == pytest.approx(0.5)
    assert labeled["y_sev_class"].tolist()[0] == "high"


def test_build_severity_labels_rejects_missing_event_columns() -> None:
    frame = pd.DataFrame(
        {
            "modis_burn_date": [12.0],
            "label_nbr_prefire": [0.6],
            "label_nbr_postfire": [0.3],
            "y_occ_14d": [1],
        }
    )

    with pytest.raises(ValueError, match="Missing: event_date"):
        build_severity_labels(frame)


def test_build_severity_labels_requires_explicit_occurrence_column_when_ambiguous() -> None:
    frame = pd.DataFrame(
        {
            "event_date": [date(2020, 1, 10)],
            "label_nbr_prefire": [0.7],
            "label_nbr_postfire": [0.2],
            "y_occ_14d": [1],
            "y_occ_30d": [1],
        }
    )

    with pytest.raises(ValueError, match="requires an explicit occurrence column"):
        build_severity_labels(frame)
