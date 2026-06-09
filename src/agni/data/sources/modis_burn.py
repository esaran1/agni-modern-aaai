from __future__ import annotations

from agni.data.sources.base import (
    EESourceAdapter,
    ee_date_subtract,
    materialize_ee,
    month_after_iso,
    month_start_iso,
    ms_timestamp_to_iso_date,
)


class MODISBurnAdapter(EESourceAdapter):
    def collection_id(self) -> str:
        return "MODIS/061/MCD64A1"

    def extract_patch(
        self,
        geometry,
        reference_date: str,
        lookback_days: int,
        temporal_windows: list[int],
    ) -> dict[str, float | None]:
        try:
            import ee
        except ImportError as exc:
            raise RuntimeError("earthengine-api is required for MODIS burn extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        features: dict[str, float | None] = {}
        milliseconds_per_day = 86_400_000

        def burn_timestamp_image(image):
            image = ee.Image(image)
            burn_date = image.select("BurnDate")
            image_year = ee.Date(image.get("system:time_start")).get("year")
            year_start_ms = ee.Date.fromYMD(image_year, 1, 1).millis()
            return (
                burn_date.toDouble()
                .multiply(milliseconds_per_day)
                .add(ee.Number(year_start_ms).subtract(milliseconds_per_day))
                .updateMask(burn_date.gt(0))
                .rename("burn_timestamp")
            )

        def interval_burn_timestamps(image, start_date: str, end_date: str):
            timestamps = burn_timestamp_image(image)
            start_ms = ee.Date(start_date).millis()
            end_ms = ee.Date(end_date).millis()
            # Match the rest of the feature pipeline: [start_date, reference_date)
            valid = timestamps.gte(start_ms).And(timestamps.lt(end_ms))
            return timestamps.updateMask(valid)

        def interval_burn_mask(image, start_date: str, end_date: str):
            timestamps = interval_burn_timestamps(image, start_date, end_date)
            return timestamps.mask().unmask(0).rename("burned")

        for window in temporal_windows:
            interval_start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(
                    month_start_iso(interval_start),
                    month_after_iso(reference_date),
                )
                .filterBounds(ee_geometry)
            )
            burn_presence = collection.map(
                lambda image, start=interval_start, end=reference_date: interval_burn_mask(
                    image,
                    start,
                    end,
                )
            ).max()
            burn_fraction = burn_presence.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=500,
            )
            earliest_burn = collection.map(
                lambda image, start=interval_start, end=reference_date: interval_burn_timestamps(
                    image,
                    start,
                    end,
                )
            ).min().reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=ee_geometry,
                scale=500,
            )
            # Count, timestamp and date all derive from the same masked interval burn
            # pixels, so a positive burn fraction always carries a resolvable date.
            # The timestamp is materialized below and converted to an ISO date in Python
            # so a null (no-burn) timestamp cannot crash a server-side ee.Date.format.
            earliest_burn_ts = earliest_burn.get("burn_timestamp")
            features[f"temporal_burn_count_l{window}d"] = burn_fraction.get("burned")
            features[f"temporal_burn_timestamp_l{window}d"] = earliest_burn_ts
            features[f"_earliest_burn_ts_l{window}d"] = earliest_burn_ts

        lookback_start = ee_date_subtract(reference_date, lookback_days)
        observed_collection = (
            ee.ImageCollection(self.collection_id())
            .filterBounds(ee_geometry)
            .filterDate(
                month_start_iso(lookback_start),
                month_after_iso(reference_date),
            )
        )
        observed_burn = (
            observed_collection
            .map(lambda image: interval_burn_timestamps(image, lookback_start, reference_date))
            .min()
            .reduceRegion(reducer=ee.Reducer.min(), geometry=ee_geometry, scale=500)
        )
        observed_earliest_burn_ts = observed_burn.get("burn_timestamp")
        features["modis_burn_timestamp"] = observed_earliest_burn_ts
        features["_observed_earliest_burn_ts"] = observed_earliest_burn_ts

        resolved = materialize_ee(features)

        for window in temporal_windows:
            earliest = resolved.pop(f"_earliest_burn_ts_l{window}d", None)
            resolved[f"temporal_burn_date_l{window}d"] = ms_timestamp_to_iso_date(earliest)
        observed_earliest = resolved.pop("_observed_earliest_burn_ts", None)
        resolved["modis_burn_date"] = ms_timestamp_to_iso_date(observed_earliest)
        return resolved
