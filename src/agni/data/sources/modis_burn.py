from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract


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

        for window in temporal_windows:
            start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(start, reference_date)
                .filterBounds(ee_geometry)
            )
            burn_date = collection.select("BurnDate").max()
            burn_region = burn_date.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=500,
            )
            features[f"temporal_burn_count_l{window}d"] = collection.size()
            features[f"temporal_burn_date_l{window}d"] = burn_region.get("BurnDate")

        latest_burn = (
            ee.ImageCollection(self.collection_id())
            .filterBounds(ee_geometry)
            .filterDate(ee_date_subtract(reference_date, lookback_days), reference_date)
            .select("BurnDate")
            .max()
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geometry, scale=500)
        )
        features["modis_burn_date"] = latest_burn.get("BurnDate")
        return features
