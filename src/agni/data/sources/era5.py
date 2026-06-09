from __future__ import annotations

from agni.data.sources.base import EESourceAdapter, ee_date_subtract, materialize_ee


class ERA5LandAdapter(EESourceAdapter):
    BANDS = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "total_precipitation_sum",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_pressure",
        "soil_temperature_level_1",
        "volumetric_soil_water_layer_1",
    ]

    def collection_id(self) -> str:
        return "ECMWF/ERA5_LAND/DAILY_AGGR"

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
            raise RuntimeError("earthengine-api is required for ERA5 extraction") from exc

        ee_geometry = ee.Geometry(geometry.__geo_interface__)
        features: dict[str, float | None] = {}

        for window in temporal_windows:
            start = ee_date_subtract(reference_date, window)
            collection = (
                ee.ImageCollection(self.collection_id())
                .filterDate(start, reference_date)
                .filterBounds(ee_geometry)
            )
            for band in self.BANDS:
                stats = collection.select(band).mean().reduceRegion(
                    reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.stdDev(), sharedInputs=True)
                    .combine(ee.Reducer.minMax(), sharedInputs=True),
                    geometry=ee_geometry,
                    scale=11132,
                )
                features[f"weather_{band}_mean_l{window}d"] = stats.get(f"{band}_mean")
                features[f"weather_{band}_std_l{window}d"] = stats.get(f"{band}_stdDev")
                features[f"weather_{band}_min_l{window}d"] = stats.get(f"{band}_min")
                features[f"weather_{band}_max_l{window}d"] = stats.get(f"{band}_max")
        return materialize_ee(features)
