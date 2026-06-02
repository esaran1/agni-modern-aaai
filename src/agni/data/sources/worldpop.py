from __future__ import annotations

from agni.data.sources.base import EESourceAdapter


class WorldPopAdapter(EESourceAdapter):
    def collection_id(self) -> str:
        return "WorldPop/GP/100m/pop_age_sex_cons_unadj"

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
            raise RuntimeError("earthengine-api is required for WorldPop extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        year = int(reference_date[:4])
        image = ee.ImageCollection(self.collection_id()).filter(ee.Filter.eq("year", year)).mean()
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=ee_geometry,
            scale=100,
        )
        return {
            "human_population_mean": stats.get("population_mean"),
            "human_population_max": stats.get("population_max"),
        }
