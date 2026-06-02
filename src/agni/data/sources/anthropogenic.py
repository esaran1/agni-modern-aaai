from __future__ import annotations

from typing import Any


class AnthropogenicAdapter:
    """Compute distance-to-infrastructure features."""

    @staticmethod
    def compute_road_features(
        patch_centroid_lat: float,
        patch_centroid_lon: float,
        search_radius_km: int = 50,
    ) -> dict[str, float | None]:
        try:
            import osmnx as ox
            from shapely.geometry import Point
        except ImportError:
            return {
                "anthropogenic_dist_to_road_km": None,
                "anthropogenic_road_density_km": None,
            }

        point = Point(patch_centroid_lon, patch_centroid_lat)
        try:
            graph = ox.graph_from_point(
                (patch_centroid_lat, patch_centroid_lon),
                dist=search_radius_km * 1000,
                network_type="drive",
            )
            edges = ox.graph_to_gdfs(graph, nodes=False)
            distances = edges.geometry.distance(point)
            min_dist_km = float(distances.min()) * 111.0
            nearby = edges[distances < (10 / 111.0)]
            road_length_km = float(nearby.geometry.length.sum()) * 111.0
            return {
                "anthropogenic_dist_to_road_km": min_dist_km,
                "anthropogenic_road_density_km": road_length_km,
            }
        except Exception:
            return {
                "anthropogenic_dist_to_road_km": None,
                "anthropogenic_road_density_km": None,
            }

    @staticmethod
    def compute_river_features_ee(geometry: Any) -> dict[str, float | None]:
        try:
            import ee
        except ImportError:
            return {"anthropogenic_dist_to_river_m": None}

        try:
            ee_geometry = ee.Geometry(geometry.__geo_interface__)
            rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
            nearest = rivers.filterBounds(ee_geometry.buffer(50000))
            centroid = ee_geometry.centroid()
            distances = nearest.map(lambda feature: feature.set("dist", feature.geometry().distance(centroid)))
            min_dist = distances.aggregate_min("dist")
            value = min_dist.getInfo() if hasattr(min_dist, "getInfo") else min_dist
            return {"anthropogenic_dist_to_river_m": value}
        except Exception:
            return {"anthropogenic_dist_to_river_m": None}

    @staticmethod
    def compute_settlement_features(
        patch_centroid_lat: float,
        patch_centroid_lon: float,
    ) -> dict[str, float | None]:
        try:
            import osmnx as ox
            from shapely.geometry import Point
        except ImportError:
            return {"anthropogenic_dist_to_settlement_km": None}

        try:
            tags = {"place": ["city", "town", "village"]}
            settlements = ox.features_from_point(
                (patch_centroid_lat, patch_centroid_lon),
                tags=tags,
                dist=50000,
            )
            if settlements.empty:
                return {"anthropogenic_dist_to_settlement_km": 50.0}
            point = Point(patch_centroid_lon, patch_centroid_lat)
            distances = settlements.geometry.distance(point) * 111.0
            return {"anthropogenic_dist_to_settlement_km": float(distances.min())}
        except Exception:
            return {"anthropogenic_dist_to_settlement_km": None}
