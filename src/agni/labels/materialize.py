from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt

from agni.data.sources.base import month_after_iso, month_start_iso


@dataclass(frozen=True)
class OccurrenceLabel:
    event_date: date | None
    label: int


@dataclass(frozen=True)
class SeverityLabel:
    prefire_nbr: float | None
    postfire_nbr: float | None
    dnbr: float | None
    severity_available: int
    severity_class: str | None


def classify_severity(dnbr: float) -> str:
    if dnbr <= 0.1:
        return "low"
    if dnbr <= 0.27:
        return "moderate"
    if dnbr <= 0.66:
        return "high"
    return "extreme"


NBR_WINDOW_CANDIDATES = (7, 14, 30, 60)


def _positive_burn_dates(image):
    image = image.select("BurnDate")
    return image.updateMask(image.gt(0))


def _event_dates_from_burn_timestamps(
    timestamps: list[object],
    reference_date: date,
    horizon_days: int,
) -> list[date]:
    horizon_end = reference_date + timedelta(days=horizon_days)
    event_dates: list[date] = []
    for timestamp in timestamps:
        event_date = infer_observed_burn_timestamp(timestamp)
        if event_date is None:
            continue
        if reference_date < event_date <= horizon_end:
            event_dates.append(event_date)
    return event_dates


def extract_future_modis_occurrence(
    geometry,
    reference_date: date,
    horizon_days: int,
) -> OccurrenceLabel:
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("earthengine-api is required for occurrence labels") from exc

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    start_date = reference_date.isoformat()
    end_date = (reference_date + timedelta(days=horizon_days)).isoformat()
    collection_start = month_start_iso(reference_date)
    collection_end = month_after_iso(end_date)
    collection = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(collection_start, collection_end)
        .filterBounds(ee_geometry)
    )
    milliseconds_per_day = 86_400_000

    def future_burn_timestamps(image):
        image = ee.Image(image)
        burn_date = image.select("BurnDate")
        image_year = ee.Date(image.get("system:time_start")).get("year")
        year_start_ms = ee.Date.fromYMD(image_year, 1, 1).millis()
        burn_timestamps = (
            burn_date.toDouble()
            .multiply(milliseconds_per_day)
            .add(ee.Number(year_start_ms).subtract(milliseconds_per_day))
            .updateMask(burn_date.gt(0))
            .rename("burn_timestamp")
        )
        valid = burn_timestamps.gt(ee.Date(start_date).millis()).And(
            burn_timestamps.lte(ee.Date(end_date).millis())
        )
        return burn_timestamps.updateMask(valid)

    def attach_patch_burn_timestamp(image):
        min_burn_timestamp = future_burn_timestamps(image).reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=ee_geometry,
            scale=500,
            maxPixels=1_000_000,
        ).get("burn_timestamp")
        return image.set("patch_burn_timestamp", min_burn_timestamp)

    burn_images = collection.map(attach_patch_burn_timestamp)
    burn_timestamps = burn_images.aggregate_array("patch_burn_timestamp").getInfo() or []
    event_dates = _event_dates_from_burn_timestamps(
        burn_timestamps,
        reference_date=reference_date,
        horizon_days=horizon_days,
    )

    if not event_dates:
        return OccurrenceLabel(event_date=None, label=0)
    return OccurrenceLabel(event_date=min(event_dates), label=1)


def _sentinel_nbr_mean(geometry, start_date: date, end_date: date) -> float | None:
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("earthengine-api is required for severity labels") from exc

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filterBounds(ee_geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    image_count = collection.size().getInfo()
    if not image_count:
        return None

    nbr = collection.median().normalizedDifference(["B8", "B12"]).rename("nbr")
    stat = nbr.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee_geometry,
        scale=30,
        maxPixels=1_000_000,
    ).get("nbr")
    return None if stat is None else float(stat.getInfo())


def extract_sentinel2_severity(
    geometry,
    burn_date: date,
    severity_window_days: int = 30,
) -> SeverityLabel:
    pre_start = burn_date - timedelta(days=severity_window_days)
    pre_end = burn_date
    post_start = burn_date
    post_end = burn_date + timedelta(days=severity_window_days)

    prefire_nbr = _sentinel_nbr_mean(geometry, pre_start, pre_end)
    postfire_nbr = _sentinel_nbr_mean(geometry, post_start, post_end)
    if prefire_nbr is None or postfire_nbr is None:
        return SeverityLabel(
            prefire_nbr=prefire_nbr,
            postfire_nbr=postfire_nbr,
            dnbr=None,
            severity_available=0,
            severity_class=None,
        )

    dnbr = max(prefire_nbr - postfire_nbr, 0.0)
    return SeverityLabel(
        prefire_nbr=prefire_nbr,
        postfire_nbr=postfire_nbr,
        dnbr=dnbr,
        severity_available=1,
        severity_class=classify_severity(dnbr),
    )


def infer_observed_burn_date(
    reference_date: date,
    burn_day_of_year: float | int | None,
    lookback_days: int,
) -> date | None:
    if burn_day_of_year is None or pd.isna(burn_day_of_year):
        return None
    burn_day = int(round(float(burn_day_of_year)))
    if burn_day <= 0:
        return None

    current_year_candidate = date(reference_date.year, 1, 1) + timedelta(days=burn_day - 1)
    if current_year_candidate <= reference_date:
        candidate = current_year_candidate
    else:
        candidate = date(reference_date.year - 1, 1, 1) + timedelta(days=burn_day - 1)

    if candidate > reference_date:
        return None
    if (reference_date - candidate).days > lookback_days:
        return None
    return candidate


def infer_observed_burn_timestamp(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return pd.Timestamp(value).date()
    if isinstance(value, int | float):
        numeric_value = float(value)
        if numeric_value <= 0:
            return None
        if numeric_value >= 10_000_000_000:
            return pd.Timestamp(numeric_value, unit="ms").date()
        return None
    return None


def select_nbr_value(row: pd.Series) -> float | None:
    for window in NBR_WINDOW_CANDIDATES:
        column = f"optical_nbr_mean_l{window}d"
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return None


def select_event_window(frame: pd.DataFrame, stride_days: int) -> int | None:
    available_windows = sorted(
        {
            int(column.removeprefix("temporal_burn_count_l").removesuffix("d"))
            for column in frame.columns
            if column.startswith("temporal_burn_count_l") and column.endswith("d")
        }
        | {
            int(column.removeprefix("temporal_fire_count_l").removesuffix("d"))
            for column in frame.columns
            if column.startswith("temporal_fire_count_l") and column.endswith("d")
        }
    )
    for window in available_windows:
        if window >= stride_days:
            return window
    return None


def derive_observed_events(
    frame: pd.DataFrame,
    stride_days: int,
) -> pd.DataFrame:
    event_window = select_event_window(frame, stride_days)
    if event_window is None:
        raise ValueError(
            "Local label materialization requires temporal burn/fire count features. "
            "Expected a temporal burn/fire window at least as long as the reference stride."
        )

    burn_count_col = f"temporal_burn_count_l{event_window}d"
    fire_count_col = f"temporal_fire_count_l{event_window}d"
    burn_date_col = f"temporal_burn_date_l{event_window}d"
    burn_timestamp_col = f"temporal_burn_timestamp_l{event_window}d"

    if burn_count_col in frame.columns and burn_timestamp_col not in frame.columns:
        raise ValueError(
            "Local label materialization requires temporal_burn_timestamp features for "
            "burn-derived event dating. Rebuild features with the updated MODIS burn adapter."
        )

    burn_signal = False
    if burn_count_col in frame.columns:
        burn_signal = frame[burn_count_col].fillna(0).astype(float) > 0
    fire_signal = False
    if fire_count_col in frame.columns:
        fire_signal = frame[fire_count_col].fillna(0).astype(float) > 0
    if isinstance(burn_signal, bool):
        observed_event = pd.Series(fire_signal, index=frame.index, dtype=bool)
    elif isinstance(fire_signal, bool):
        observed_event = pd.Series(burn_signal, index=frame.index, dtype=bool)
    else:
        observed_event = burn_signal | fire_signal

    observed_dates: list[date | None] = []
    for row in frame.itertuples(index=False):
        reference_date = pd.Timestamp(row.reference_date).date()
        burn_timestamp = getattr(row, burn_timestamp_col, None)
        burn_day = getattr(row, burn_date_col, None) if burn_date_col in frame.columns else None
        observed_date = infer_observed_burn_timestamp(burn_timestamp)
        if observed_date is None and burn_count_col not in frame.columns:
            observed_date = infer_observed_burn_date(reference_date, burn_day, event_window)
        observed_dates.append(observed_date)

    derived = frame.copy()
    derived["observed_event"] = observed_event.astype(int)
    derived["observed_event_date"] = observed_dates
    missing_dates = derived["observed_event"].eq(1) & derived["observed_event_date"].isna()
    if missing_dates.any():
        raise ValueError(
            "Observed fire/burn signals require an inferable temporal_burn_date label. "
            "Enable MODIS burn-date features or provide a dated event source."
        )
    return derived


def estimate_local_severity(
    group: pd.DataFrame,
    event_date: date,
    severity_window_days: int,
) -> SeverityLabel:
    reference_dates = pd.to_datetime(group["reference_date"]).dt.date
    pre_mask = [
        ref_date < event_date and (event_date - ref_date).days <= severity_window_days
        for ref_date in reference_dates
    ]
    post_mask = [
        ref_date > event_date and (ref_date - event_date).days <= severity_window_days
        for ref_date in reference_dates
    ]
    pre_candidates = group.loc[pre_mask]
    post_candidates = group.loc[post_mask]
    if pre_candidates.empty or post_candidates.empty:
        return SeverityLabel(
            prefire_nbr=None,
            postfire_nbr=None,
            dnbr=None,
            severity_available=0,
            severity_class=None,
        )

    prefire_nbr = select_nbr_value(pre_candidates.iloc[-1])
    postfire_nbr = select_nbr_value(post_candidates.iloc[-1])
    if prefire_nbr is None or postfire_nbr is None:
        return SeverityLabel(
            prefire_nbr=prefire_nbr,
            postfire_nbr=postfire_nbr,
            dnbr=None,
            severity_available=0,
            severity_class=None,
        )

    dnbr = max(prefire_nbr - postfire_nbr, 0.0)
    return SeverityLabel(
        prefire_nbr=prefire_nbr,
        postfire_nbr=postfire_nbr,
        dnbr=dnbr,
        severity_available=1,
        severity_class=classify_severity(dnbr),
    )


def build_geometry_lookup(grid_df: pd.DataFrame) -> dict[str, object]:
    return {
        row["patch_id"]: wkt.loads(row["geometry_wkt"])
        for row in grid_df[["patch_id", "geometry_wkt"]].to_dict(orient="records")
    }


def materialize_labels_from_extractors(
    frame: pd.DataFrame,
    geometry_lookup: dict[str, object],
    horizon_days: int,
    severity_window_days: int,
    occurrence_extractor: Callable[[object, date, int], OccurrenceLabel],
    severity_extractor: Callable[[object, date, int], SeverityLabel],
    materialize_severity: bool,
) -> pd.DataFrame:
    occurrence_col = f"y_occ_{horizon_days}d"
    frame = frame.copy()
    frame[occurrence_col] = 0
    frame["event_date"] = pd.NaT
    frame["optical_nbr_prefire"] = pd.NA
    frame["optical_nbr_postfire"] = pd.NA
    frame["y_sev_available"] = 0
    frame["y_sev_dnbr"] = pd.NA
    frame["y_sev_class"] = pd.NA

    for idx, row in frame.iterrows():
        geometry = geometry_lookup.get(row["patch_id"])
        if geometry is None:
            raise ValueError(f"Missing geometry for patch_id '{row['patch_id']}'")

        reference_date = row["reference_date"].date()
        occurrence = occurrence_extractor(geometry, reference_date, horizon_days)
        frame.at[idx, occurrence_col] = occurrence.label
        if occurrence.event_date is None:
            continue

        frame.at[idx, "event_date"] = pd.Timestamp(occurrence.event_date)
        if materialize_severity:
            severity = severity_extractor(geometry, occurrence.event_date, severity_window_days)
            frame.at[idx, "optical_nbr_prefire"] = severity.prefire_nbr
            frame.at[idx, "optical_nbr_postfire"] = severity.postfire_nbr
            frame.at[idx, "y_sev_available"] = severity.severity_available
            frame.at[idx, "y_sev_dnbr"] = severity.dnbr
            frame.at[idx, "y_sev_class"] = severity.severity_class
    return frame


def materialize_labels_from_observed_events(
    frame: pd.DataFrame,
    geometry_lookup: dict[str, object] | None,
    horizon_days: int,
    severity_window_days: int,
    stride_days: int,
    severity_extractor: Callable[[object, date, int], SeverityLabel] | None,
    materialize_severity: bool,
) -> pd.DataFrame:
    occurrence_col = f"y_occ_{horizon_days}d"
    sorted_frame = frame.sort_values(["patch_id", "reference_date"]).copy()
    frame = derive_observed_events(sorted_frame, stride_days)
    frame[occurrence_col] = 0
    frame["event_date"] = pd.NaT
    frame["optical_nbr_prefire"] = pd.NA
    frame["optical_nbr_postfire"] = pd.NA
    frame["y_sev_available"] = 0
    frame["y_sev_dnbr"] = pd.NA
    frame["y_sev_class"] = pd.NA

    for patch_id, indices in frame.groupby("patch_id", sort=False).groups.items():
        patch_indices = list(indices)
        patch_frame = frame.loc[patch_indices].copy()
        event_dates = sorted(
            {
                event_date
                for is_event, event_date in zip(
                    patch_frame["observed_event"].tolist(),
                    patch_frame["observed_event_date"].tolist(),
                    strict=False,
                )
                if is_event == 1 and isinstance(event_date, date)
            }
        )
        if not event_dates:
            continue

        if materialize_severity and severity_extractor is not None:
            if geometry_lookup is None or patch_id not in geometry_lookup:
                raise ValueError(f"Missing geometry for patch_id '{patch_id}'")
            geometry = geometry_lookup[patch_id]
            severity_by_event = {
                event_date: severity_extractor(geometry, event_date, severity_window_days)
                for event_date in event_dates
            }
        elif materialize_severity:
            severity_by_event = {
                event_date: estimate_local_severity(patch_frame, event_date, severity_window_days)
                for event_date in event_dates
            }
        else:
            severity_by_event = {}

        event_ordinals = np.array([event_date.toordinal() for event_date in event_dates], dtype=int)
        reference_ordinals = np.array(
            [pd.Timestamp(value).date().toordinal() for value in patch_frame["reference_date"]],
            dtype=int,
        )
        candidate_indices = np.searchsorted(event_ordinals, reference_ordinals, side="right")

        for offset, row_index in enumerate(patch_indices):
            next_idx = candidate_indices[offset]
            if next_idx >= len(event_ordinals):
                continue
            next_event_ordinal = int(event_ordinals[next_idx])
            if next_event_ordinal > reference_ordinals[offset] + horizon_days:
                continue

            next_event_date = event_dates[next_idx]
            frame.at[row_index, occurrence_col] = 1
            frame.at[row_index, "event_date"] = pd.Timestamp(next_event_date)
            if materialize_severity:
                severity = severity_by_event[next_event_date]
                frame.at[row_index, "optical_nbr_prefire"] = severity.prefire_nbr
                frame.at[row_index, "optical_nbr_postfire"] = severity.postfire_nbr
                frame.at[row_index, "y_sev_available"] = severity.severity_available
                frame.at[row_index, "y_sev_dnbr"] = severity.dnbr
                frame.at[row_index, "y_sev_class"] = severity.severity_class
    return frame


def compute_label_cutoff(
    frame: pd.DataFrame,
    horizon_days: int,
    label_reference_end: date | None,
) -> pd.Timestamp:
    max_observed_reference = pd.to_datetime(frame["reference_date"]).max()
    fully_observed_cutoff = max_observed_reference - pd.Timedelta(days=horizon_days)
    if label_reference_end is None:
        return fully_observed_cutoff
    return min(pd.Timestamp(label_reference_end), fully_observed_cutoff)


def materialize_labels(
    features_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    horizon_days: int,
    lookback_days: int = 60,
    severity_window_days: int = 30,
    reference_stride_days: int = 7,
    label_reference_end: date | None = None,
    occurrence_extractor: Callable[[object, date, int], OccurrenceLabel] | None = None,
    severity_extractor: Callable[[object, date, int], SeverityLabel] | None = None,
    materialize_severity: bool = True,
) -> pd.DataFrame:
    del lookback_days  # Kept for API compatibility with prior call sites.

    frame = features_df.copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    geometry_lookup = build_geometry_lookup(grid_df)

    if occurrence_extractor is not None:
        if label_reference_end is not None:
            label_cutoff = pd.Timestamp(label_reference_end)
        else:
            label_cutoff = frame["reference_date"].max()
        resolved_severity_extractor = severity_extractor or extract_sentinel2_severity
        frame = materialize_labels_from_extractors(
            frame=frame,
            geometry_lookup=geometry_lookup,
            horizon_days=horizon_days,
            severity_window_days=severity_window_days,
            occurrence_extractor=occurrence_extractor,
            severity_extractor=resolved_severity_extractor,
            materialize_severity=materialize_severity,
        )
    else:
        label_cutoff = compute_label_cutoff(frame, horizon_days, label_reference_end)
        frame = materialize_labels_from_observed_events(
            frame=frame,
            geometry_lookup=geometry_lookup,
            horizon_days=horizon_days,
            severity_window_days=severity_window_days,
            stride_days=reference_stride_days,
            severity_extractor=severity_extractor,
            materialize_severity=materialize_severity,
        )

    frame = frame[frame["reference_date"] <= label_cutoff].copy()
    frame["reference_date"] = frame["reference_date"].dt.date
    return frame


def materialize_labels_from_paths(
    features_path: str | Path,
    grid_path: str | Path,
    horizon_days: int,
    output_path: str | Path,
    lookback_days: int = 60,
    severity_window_days: int = 30,
    reference_stride_days: int = 7,
    label_reference_end: date | None = None,
    materialize_severity: bool = True,
) -> Path:
    features_df = pd.read_parquet(features_path)
    grid_df = pd.read_parquet(grid_path)
    labeled = materialize_labels(
        features_df=features_df,
        grid_df=grid_df,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
        severity_window_days=severity_window_days,
        reference_stride_days=reference_stride_days,
        label_reference_end=label_reference_end,
        materialize_severity=materialize_severity,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output, index=False)
    return output
