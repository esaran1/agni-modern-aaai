from __future__ import annotations

import numpy as np
import pandas as pd


def assign_spatial_blocks(
    df: pd.DataFrame,
    block_size_km: int = 50,
    grid_km: int = 10,
) -> pd.DataFrame:
    frame = df.copy()
    if "patch_row" not in frame.columns:
        frame["patch_row"] = frame["patch_id"].str.extract(r"(\d+)_\d+$").astype(int)
    if "patch_col" not in frame.columns:
        frame["patch_col"] = frame["patch_id"].str.extract(r"\d+_(\d+)$").astype(int)
    patches_per_block = max(1, block_size_km // grid_km)
    frame["block_id"] = (
        (frame["patch_row"] // patches_per_block).astype(str)
        + "_"
        + (frame["patch_col"] // patches_per_block).astype(str)
    )
    return frame


def spatial_block_split(
    df: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
) -> pd.DataFrame:
    frame = df.copy()
    unique_blocks = frame["block_id"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    if len(unique_blocks) < len(split_names):
        raise ValueError("Not enough unique spatial blocks to assign train/val/test")

    fold_size = max(1, len(unique_blocks) // n_folds)
    test_blocks = set(unique_blocks[:fold_size])
    val_blocks = set(unique_blocks[fold_size : 2 * fold_size])
    train_blocks = set(unique_blocks[2 * fold_size :])

    frame["spatial_split"] = "train"
    frame.loc[frame["block_id"].isin(val_blocks), "spatial_split"] = "val"
    frame.loc[frame["block_id"].isin(test_blocks), "spatial_split"] = "test"
    frame.loc[frame["block_id"].isin(train_blocks), "spatial_split"] = "train"
    return frame


def spatial_block_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_size_km: int = 50,
    grid_km: int = 10,
    seed: int = 42,
) -> list[tuple[pd.Series, pd.Series]]:
    frame = assign_spatial_blocks(df, block_size_km=block_size_km, grid_km=grid_km)
    unique_blocks = frame["block_id"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    block_folds = np.array_split(unique_blocks, min(n_folds, len(unique_blocks)))
    results = []
    for test_blocks in block_folds:
        test_blocks = set(test_blocks.tolist())
        test_mask = frame["block_id"].isin(test_blocks)
        train_mask = ~test_mask
        results.append((train_mask, test_mask))
    return results
