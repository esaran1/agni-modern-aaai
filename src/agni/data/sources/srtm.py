from __future__ import annotations

from agni.data.sources.base import EESourceAdapter


class SRTMAdapter(EESourceAdapter):
    def collection_id(self) -> str:
        return "CGIAR/SRTM90_V4"

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
            raise RuntimeError("earthengine-api is required for SRTM extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        dem = ee.Image(self.collection_id())
        terrain = ee.Terrain.products(dem)

        elev_stats = dem.select("elevation").reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=ee_geometry,
            scale=90,
        )
        slope_stats = terrain.select("slope").reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=ee_geometry,
            scale=90,
        )

        slope_rad = terrain.select("slope").multiply(3.14159 / 180.0)
        tan_slope = slope_rad.tan().max(0.001)
        twi = tan_slope.log().multiply(-1).rename("twi")
        twi_stats = twi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=ee_geometry,
            scale=90,
        )

        return {
            "terrain_elevation_mean": elev_stats.get("elevation_mean"),
            "terrain_elevation_std": elev_stats.get("elevation_stdDev"),
            "terrain_elevation_min": elev_stats.get("elevation_min"),
            "terrain_elevation_max": elev_stats.get("elevation_max"),
            "terrain_slope_mean": slope_stats.get("slope_mean"),
            "terrain_slope_std": slope_stats.get("slope_stdDev"),
            "terrain_twi_mean": twi_stats.get("twi_mean"),
            "terrain_twi_std": twi_stats.get("twi_stdDev"),
        }
