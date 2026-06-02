from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract


class DynamicWorldAdapter(EESourceAdapter):
    CLASSES = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]

    def collection_id(self) -> str:
        return "GOOGLE/DYNAMICWORLD/V1"

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
            raise RuntimeError("earthengine-api is required for Dynamic World extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        start = ee_date_subtract(reference_date, lookback_days)
        collection = (
            ee.ImageCollection(self.collection_id())
            .filterDate(start, reference_date)
            .filterBounds(ee_geometry)
        )
        mode_image = collection.select("label").mode()

        features: dict[str, float | None] = {}
        for class_idx, class_name in enumerate(self.CLASSES):
            mask = mode_image.eq(class_idx)
            fraction = mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=10,
            )
            features[f"landcover_{class_name}_fraction"] = fraction.get("label")
        return features
