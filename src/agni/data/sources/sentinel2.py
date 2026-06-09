from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract, materialize_ee


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
        reducer = (
            ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.minMax(), sharedInputs=True)
        )

        # Earth Engine values to resolve, and known-empty windows kept as Python None so
        # we never feed a band-less image to ``select`` (which raises) nor a None into
        # ``ee.Dictionary``.
        features: dict[str, object] = {}
        empty: dict[str, float | None] = {}
        for window in temporal_windows:
            start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(start, reference_date)
                .filterBounds(ee_geometry)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            )
            # Cloud/haze filtering frequently leaves zero clear scenes over equatorial
            # peatlands in fire season; an empty composite has no bands, so guard it.
            if int(collection.size().getInfo() or 0) == 0:
                for band in self.BANDS:
                    band_name = band.lower()
                    for stat in ("mean", "std", "min", "max"):
                        empty[f"optical_{band_name}_{stat}_l{window}d"] = None
                continue

            composite = collection.median().select(self.BANDS)
            stats = composite.reduceRegion(
                reducer=reducer,
                geometry=ee_geometry,
                scale=10,
                maxPixels=1_000_000_000,
            )
            for band in self.BANDS:
                band_name = band.lower()
                features[f"optical_{band_name}_mean_l{window}d"] = stats.get(f"{band}_mean")
                features[f"optical_{band_name}_std_l{window}d"] = stats.get(f"{band}_stdDev")
                features[f"optical_{band_name}_min_l{window}d"] = stats.get(f"{band}_min")
                features[f"optical_{band_name}_max_l{window}d"] = stats.get(f"{band}_max")

        resolved = materialize_ee(features)
        resolved.update(empty)
        return resolved
