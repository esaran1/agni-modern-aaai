from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from shapely import wkt

from agni.config import DataConfig
from agni.data.manifest import build_dataset_manifest

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    dataset: pd.DataFrame
    dataset_path: Path
    manifest_path: Path


def iter_reference_dates(config: DataConfig) -> list[pd.Timestamp]:
    dates = []
    current = pd.Timestamp(config.temporal.reference_start)
    end = pd.Timestamp(config.temporal.reference_end)
    stride = timedelta(days=config.temporal.reference_stride_days)
    while current <= end:
        dates.append(current)
        current += stride
    return dates


def build_dataset(
    config: DataConfig,
    patch_df: pd.DataFrame,
    adapters: Iterable[Any],
    output_name: str = "dataset.parquet",
) -> DatasetBuildResult:
    output_dir = Path(config.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / output_name
    manifest_path = output_dir / f"{dataset_path.stem}.manifest.json"

    rows: list[dict[str, Any]] = []
    reference_dates = iter_reference_dates(config)

    for patch in patch_df.to_dict(orient="records"):
        geometry = wkt.loads(patch["geometry_wkt"])
        for reference_date in reference_dates:
            row = {
                "patch_id": patch["patch_id"],
                "patch_row": patch["patch_row"],
                "patch_col": patch["patch_col"],
                "centroid_lon": patch["centroid_lon"],
                "centroid_lat": patch["centroid_lat"],
                "reference_date": reference_date.date(),
            }
            for adapter in adapters:
                try:
                    row.update(
                        adapter.extract_patch(
                            geometry=geometry,
                            reference_date=reference_date.date().isoformat(),
                            lookback_days=config.temporal.lookback_days,
                            temporal_windows=config.temporal.temporal_windows,
                        )
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Adapter %s failed for patch=%s date=%s: %s",
                        adapter.__class__.__name__,
                        patch["patch_id"],
                        reference_date.date(),
                        exc,
                    )
            rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset.to_parquet(dataset_path, index=False)

    manifest = build_dataset_manifest(
        dataset_path=dataset_path,
        config_dict=config.model_dump(mode="json"),
        row_count=len(dataset),
        extra={"output_name": output_name},
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DatasetBuildResult(dataset=dataset, dataset_path=dataset_path, manifest_path=manifest_path)
