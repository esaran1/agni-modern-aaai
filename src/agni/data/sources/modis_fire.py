from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract


class MODISFireAdapter(EESourceAdapter):
    def collection_id(self) -> str:
        return "MODIS/061/MOD14A1"

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
            raise RuntimeError("earthengine-api is required for MODIS fire extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        features: dict[str, float | None] = {}
        for window in temporal_windows:
            start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(start, reference_date)
                .filterBounds(ee_geometry)
            )
            fire_mask = collection.select("FireMask").map(lambda img: img.gte(7))
            fire_count = fire_mask.sum().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=1000,
            )
            features[f"temporal_fire_count_l{window}d"] = fire_count.get("FireMask")
        return features
