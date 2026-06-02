from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

import pandas as pd
from shapely.geometry import Polygon

from agni.config import BBox


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    patch_row: int
    patch_col: int
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float

    @property
    def centroid_lon(self) -> float:
        return (self.min_lon + self.max_lon) / 2.0

    @property
    def centroid_lat(self) -> float:
        return (self.min_lat + self.max_lat) / 2.0

    @property
    def geometry(self) -> Polygon:
        return Polygon(
            [
                (self.min_lon, self.min_lat),
                (self.max_lon, self.min_lat),
                (self.max_lon, self.max_lat),
                (self.min_lon, self.max_lat),
            ]
        )


def km_to_lat_degrees(km: float) -> float:
    return km / 110.574


def km_to_lon_degrees(km: float, latitude: float) -> float:
    return km / max(111.320 * cos(radians(latitude)), 1e-6)


def build_patch_grid(bbox: BBox, grid_km: int) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    lat_step = km_to_lat_degrees(grid_km)

    row = 0
    current_lat = bbox.lat_min
    while current_lat < bbox.lat_max - 1e-9:
        next_lat = min(current_lat + lat_step, bbox.lat_max)
        lon_step = km_to_lon_degrees(grid_km, (current_lat + next_lat) / 2.0)

        col = 0
        current_lon = bbox.lon_min
        while current_lon < bbox.lon_max - 1e-9:
            next_lon = min(current_lon + lon_step, bbox.lon_max)
            patch = PatchRecord(
                patch_id=f"{row}_{col}",
                patch_row=row,
                patch_col=col,
                min_lon=current_lon,
                max_lon=next_lon,
                min_lat=current_lat,
                max_lat=next_lat,
            )
            records.append(
                {
                    "patch_id": patch.patch_id,
                    "patch_row": patch.patch_row,
                    "patch_col": patch.patch_col,
                    "min_lon": patch.min_lon,
                    "max_lon": patch.max_lon,
                    "min_lat": patch.min_lat,
                    "max_lat": patch.max_lat,
                    "centroid_lon": patch.centroid_lon,
                    "centroid_lat": patch.centroid_lat,
                    "geometry_wkt": patch.geometry.wkt,
                }
            )
            current_lon = next_lon
            col += 1

        current_lat = next_lat
        row += 1

    return pd.DataFrame.from_records(records)
