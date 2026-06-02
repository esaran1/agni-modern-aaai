from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract


class Sentinel2Adapter(EESourceAdapter):
    BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

    def collection_id(self) -> str:
        return "COPERNICUS/S2_SR_HARMONIZED"

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
            raise RuntimeError("earthengine-api is required for Sentinel-2 extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        features: dict[str, float | None] = {}
        for window in temporal_windows:
            start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(start, reference_date)
                .filterBounds(ee_geometry)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            )
            composite = collection.median()
            for band in self.BANDS:
                stats = composite.select(band).reduceRegion(
                    reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), sharedInputs=True)
                    .combine(ee.Reducer.minMax(), sharedInputs=True),
                    geometry=ee_geometry,
                    scale=10,
                )
                band_name = band.lower()
                features[f"optical_{band_name}_mean_l{window}d"] = stats.get(f"{band}_mean")
                features[f"optical_{band_name}_std_l{window}d"] = stats.get(f"{band}_stdDev")
                features[f"optical_{band_name}_min_l{window}d"] = stats.get(f"{band}_min")
                features[f"optical_{band_name}_max_l{window}d"] = stats.get(f"{band}_max")
        return features
