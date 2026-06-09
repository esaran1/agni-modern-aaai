from __future__ import annotations

from pathlib import Path

from rasterio.mask import mask as rio_mask

from agni.data.sources.base import EESourceAdapter, materialize_ee


class PeatExtentAdapter(EESourceAdapter):
    def collection_id(self) -> str:
        return "projects/global-wetlands/assets/cifor_peatlands"

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
            raise RuntimeError("earthengine-api is required for EE peat extraction") from exc

        try:
            ee_geometry = ee.Geometry(geometry.__geo_interface__)
            peat_image = ee.Image(self.collection_id())
            peat_fraction = peat_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=250,
            )
            return materialize_ee({"peat_fraction": peat_fraction.get("peat")})
        except Exception:
            return {"peat_fraction": None}

    @staticmethod
    def from_local_geotiff(
        geotiff_path: str | Path,
        patch_geometries: dict[str, dict],
    ) -> dict[str, float | None]:
        import rasterio

        results: dict[str, float | None] = {}
        with rasterio.open(geotiff_path) as src:
            for patch_id, geometry in patch_geometries.items():
                try:
                    out_image, _ = rio_mask(src, [geometry], crop=True)
                    peat_pixels = int((out_image > 0).sum())
                    total_pixels = int((out_image != src.nodata).sum())
                    results[patch_id] = peat_pixels / max(total_pixels, 1)
                except Exception:
                    results[patch_id] = None
        return results
