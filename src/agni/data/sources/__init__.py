"""Earth Engine and local-source adapters."""

from agni.data.sources.anthropogenic import AnthropogenicAdapter
from agni.data.sources.dynamic_world import DynamicWorldAdapter
from agni.data.sources.era5 import ERA5LandAdapter
from agni.data.sources.modis_burn import MODISBurnAdapter
from agni.data.sources.modis_fire import MODISFireAdapter
from agni.data.sources.peat import PeatExtentAdapter
from agni.data.sources.sentinel2 import Sentinel2Adapter
from agni.data.sources.srtm import SRTMAdapter
from agni.data.sources.worldpop import WorldPopAdapter

ADAPTER_REGISTRY = {
    "era5": ERA5LandAdapter,
    "sentinel2": Sentinel2Adapter,
    "srtm": SRTMAdapter,
    "modis_fire": MODISFireAdapter,
    "modis_burn": MODISBurnAdapter,
    "dynamic_world": DynamicWorldAdapter,
    "peat": PeatExtentAdapter,
    "worldpop": WorldPopAdapter,
}


def build_adapters(source_configs):
    adapters = []
    for source in source_configs:
        if not source.enabled:
            continue
        adapter_cls = ADAPTER_REGISTRY.get(source.name)
        if adapter_cls is None:
            continue
        adapters.append(adapter_cls(**source.params) if source.params else adapter_cls())
    return adapters


__all__ = [
    "ADAPTER_REGISTRY",
    "AnthropogenicAdapter",
    "DynamicWorldAdapter",
    "ERA5LandAdapter",
    "MODISBurnAdapter",
    "MODISFireAdapter",
    "PeatExtentAdapter",
    "Sentinel2Adapter",
    "SRTMAdapter",
    "WorldPopAdapter",
    "build_adapters",
]
