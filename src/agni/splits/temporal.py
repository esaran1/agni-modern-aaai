from __future__ import annotations

import pandas as pd

from agni.config import SplitConfig


def validate_temporal_buffer(df: pd.DataFrame, horizon_days: int, split_col: str = "split") -> None:
    ordered_pairs = [("train", "val"), ("val", "test")]
    frame = df.copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    for left, right in ordered_pairs:
        left_dates = frame.loc[frame[split_col] == left, "reference_date"]
        right_dates = frame.loc[frame[split_col] == right, "reference_date"]
        if left_dates.empty or right_dates.empty:
            continue
        gap = (right_dates.min() - left_dates.max()).days
        if gap < horizon_days:
            raise AssertionError(f"{left}->{right} buffer {gap}d < horizon {horizon_days}d")


def temporal_purged_split(
    df: pd.DataFrame,
    split_config: SplitConfig,
    horizon_days: int,
    date_col: str = "reference_date",
) -> pd.DataFrame:
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["split"] = "embargo"

    train_cutoff = pd.Timestamp(split_config.train_end)
    val_cutoff = pd.Timestamp(split_config.val_end)
    test_cutoff = pd.Timestamp(split_config.test_end)
    buffer = pd.Timedelta(days=split_config.buffer_days)

    train_mask = frame[date_col] <= train_cutoff
    val_mask = (frame[date_col] > train_cutoff + buffer) & (frame[date_col] <= val_cutoff)
    test_mask = (frame[date_col] > val_cutoff + buffer) & (frame[date_col] <= test_cutoff)

    frame.loc[train_mask, "split"] = "train"
    frame.loc[val_mask, "split"] = "val"
    frame.loc[test_mask, "split"] = "test"

    validate_temporal_buffer(frame[frame["split"].isin(["train", "val", "test"])], horizon_days)
    frame[date_col] = frame[date_col].dt.date
    return frame
