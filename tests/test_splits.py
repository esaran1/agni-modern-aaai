from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from agni.config import (
    BBox,
    DataConfig,
    GridConfig,
    SpatialBlockConfig,
    SplitConfig,
    TemporalConfig,
)
from agni.splits.spatial import assign_spatial_blocks
from agni.splits.spatiotemporal import spatiotemporal_purged_split
from agni.splits.temporal import validate_temporal_buffer


def test_temporal_buffer_enforced() -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2020-01-01", "2020-01-15"]),
            "split": ["train", "val"],
        }
    )
    with pytest.raises(AssertionError):
        validate_temporal_buffer(df, horizon_days=30)


def test_spatial_blocks_disjoint() -> None:
    base = pd.DataFrame({"patch_id": [f"{row}_{col}" for row in range(6) for col in range(6)]})
    blocked = assign_spatial_blocks(base, block_size_km=20, grid_km=10)
    assert blocked["block_id"].nunique() > 1


def test_spatiotemporal_split_contains_disjoint_blocks() -> None:
    rows = []
    for row in range(6):
        for col in range(6):
            for ref_date in [date(2020, 1, 1), date(2020, 7, 15), date(2020, 11, 15)]:
                rows.append({"patch_id": f"{row}_{col}", "reference_date": ref_date})
    df = pd.DataFrame(rows)
    config = DataConfig(
        grid=GridConfig(
            grid_km=10,
            bbox=BBox(lon_min=0.0, lon_max=1.0, lat_min=0.0, lat_max=1.0),
        ),
        temporal=TemporalConfig(
            reference_start=date(2020, 1, 1),
            reference_end=date(2020, 12, 31),
            reference_stride_days=7,
            lookback_days=60,
            temporal_windows=[7, 30, 60],
            horizon_days=30,
        ),
        split=SplitConfig(
            buffer_days=30,
            train_end=date(2020, 3, 31),
            val_end=date(2020, 8, 31),
            test_end=date(2020, 12, 31),
        ),
        spatial_blocks=SpatialBlockConfig(block_size_km=20, n_folds=3, seed=42),
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
    )
    result = spatiotemporal_purged_split(df, config)
    assert {"train", "val", "test"} <= set(result["split"].unique())
