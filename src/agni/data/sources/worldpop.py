from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, materialize_ee


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
        collection = ee.ImageCollection(self.collection_id()).filter(ee.Filter.eq("year", year))
        # WorldPop annual layers do not cover every year (e.g. years past the dataset's
        # latest release); an empty collection has no bands to reduce.
        if int(collection.size().getInfo() or 0) == 0:
            return {"human_population_mean": None, "human_population_max": None}
        image = collection.mean()
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry=ee_geometry,
            scale=100,
            maxPixels=1_000_000_000,
        )
        return materialize_ee(
            {
                "human_population_mean": stats.get("population_mean"),
                "human_population_max": stats.get("population_max"),
            }
        )
