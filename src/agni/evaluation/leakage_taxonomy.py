from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from agni.splits.spatial import spatial_block_cv


def apply_temporal_split_with_buffer(
    df: pd.DataFrame,
    buffer_days: int,
    horizon_days: int,
    split_boundaries: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    frame = df.copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    if split_boundaries is None:
        unique_dates = np.sort(frame["reference_date"].dropna().unique())
        if len(unique_dates) < 3:
            raise ValueError("Need at least three unique dates to derive train/val/test boundaries")
        train_end = pd.Timestamp(unique_dates[len(unique_dates) // 3 - 1])
        val_end = pd.Timestamp(unique_dates[(2 * len(unique_dates)) // 3 - 1])
    else:
        train_end, val_end = split_boundaries

    frame["split"] = "embargo"
    frame.loc[frame["reference_date"] <= train_end, "split"] = "train"
    frame.loc[
        (frame["reference_date"] > train_end + pd.Timedelta(days=buffer_days))
        & (frame["reference_date"] <= val_end),
        "split",
    ] = "val"
    frame.loc[
        frame["reference_date"] > val_end + pd.Timedelta(days=buffer_days),
        "split",
    ] = "test"
    return frame


def detect_type2_leakage(df: pd.DataFrame, horizon_days: int) -> dict:
    frame = df.copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    if len(train) == 0 or len(val) == 0:
        return {"leakage_detected": False}

    last_train = train["reference_date"].max()
    first_val = val["reference_date"].min()
    gap_days = int((first_val - last_train).days)
    overlap_days = max(0, horizon_days - gap_days)
    leaking_mask = (train["reference_date"] + pd.Timedelta(days=horizon_days)) >= first_val
    leak_fraction = float(leaking_mask.mean()) if len(train) else 0.0
    return {
        "leakage_detected": overlap_days > 0,
        "gap_days": gap_days,
        "horizon_days": int(horizon_days),
        "overlap_days": int(overlap_days),
        "overlap_ratio": float(overlap_days / horizon_days),
        "leaking_train_fraction": leak_fraction,
    }


def compute_leakage_curve(
    df: pd.DataFrame,
    train_model_fn: Callable,
    evaluate_fn: Callable,
    horizon_days: int,
    buffer_range: list[int] | None = None,
    split_boundaries: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if buffer_range is None:
        buffer_range = list(range(0, 2 * horizon_days + 1, 5))
    results = []
    for buffer in buffer_range:
        df_split = apply_temporal_split_with_buffer(
            df,
            buffer,
            horizon_days,
            split_boundaries=split_boundaries,
        )
        df_train = df_split[df_split["split"] == "train"]
        df_val = df_split[df_split["split"] == "val"]
        df_test = df_split[df_split["split"] == "test"]
        if len(df_train) < 10 or len(df_test) < 10:
            continue
        model = train_model_fn(df_train, df_val)
        auc = evaluate_fn(model, df_test)
        if np.isnan(auc):
            continue
        results.append(
            {
                "buffer_days": int(buffer),
                "roc_auc": float(auc),
                "n_train": int(len(df_train)),
                "n_val": int(len(df_val)),
                "n_test": int(len(df_test)),
                "type2_overlap_days": int(max(0, horizon_days - buffer)),
            }
        )
    return pd.DataFrame(results).sort_values("buffer_days").reset_index(drop=True)


def detect_type3_leakage(
    df: pd.DataFrame,
    train_model_fn: Callable,
    evaluate_fn: Callable,
    block_sizes_km: list[int] | None = None,
    grid_km: int = 10,
) -> pd.DataFrame:
    if block_sizes_km is None:
        block_sizes_km = [10, 25, 50, 100]
    results = []
    for block_km in block_sizes_km:
        folds = spatial_block_cv(df, n_folds=5, block_size_km=block_km, grid_km=grid_km)
        fold_aucs = []
        for train_mask, test_mask in folds:
            df_train = df.loc[train_mask]
            df_test = df.loc[test_mask]
            df_val = df[df["split"] == "val"] if "split" in df.columns else df_test
            if len(df_train) < 10 or len(df_test) < 10:
                continue
            model = train_model_fn(df_train, df_val)
            fold_aucs.append(float(evaluate_fn(model, df_test)))
        if fold_aucs:
            results.append(
                {
                    "block_size_km": int(block_km),
                    "mean_auc": float(np.mean(fold_aucs)),
                    "std_auc": float(np.std(fold_aucs)),
                    "n_folds": int(len(fold_aucs)),
                }
            )
    return pd.DataFrame(results).sort_values("block_size_km").reset_index(drop=True)
