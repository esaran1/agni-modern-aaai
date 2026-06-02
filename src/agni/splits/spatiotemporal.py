from __future__ import annotations

import pandas as pd

from agni.config import DataConfig
from agni.splits.spatial import assign_spatial_blocks, spatial_block_split
from agni.splits.temporal import temporal_purged_split, validate_temporal_buffer


def spatiotemporal_purged_split(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    if config.spatial_blocks is None:
        raise ValueError("spatial_blocks config is required for spatiotemporal splitting")

    frame = assign_spatial_blocks(
        df,
        block_size_km=config.spatial_blocks.block_size_km,
        grid_km=config.grid.grid_km,
    )
    frame = spatial_block_split(
        frame,
        n_folds=config.spatial_blocks.n_folds,
        seed=config.spatial_blocks.seed,
    )
    frame = temporal_purged_split(
        frame,
        split_config=config.split,
        horizon_days=config.temporal.horizon_days,
    )

    frame["split"] = frame.apply(
        lambda row: row["split"] if row["split"] == row["spatial_split"] else "embargo",
        axis=1,
    )

    validate_temporal_buffer(
        frame[frame["split"].isin(["train", "val", "test"])],
        horizon_days=config.temporal.horizon_days,
    )
    return frame
