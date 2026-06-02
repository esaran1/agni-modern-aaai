from __future__ import annotations

import pandas as pd


def build_occurrence_labels(
    df: pd.DataFrame,
    event_indicator_col: str = "fire_event",
    horizon_days: int = 30,
    group_col: str = "patch_id",
    date_col: str = "reference_date",
) -> pd.DataFrame:
    frame = df.sort_values([group_col, date_col]).copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    label_col = f"y_occ_{horizon_days}d"
    frame[label_col] = 0

    for patch_id, group in frame.groupby(group_col, sort=False):
        event_dates = group.loc[group[event_indicator_col].fillna(0).astype(int) == 1, date_col]
        if event_dates.empty:
            continue
        for idx, ref_date in zip(group.index, group[date_col], strict=False):
            in_window = ((event_dates > ref_date) & (event_dates <= ref_date + pd.Timedelta(days=horizon_days))).any()
            frame.at[idx, label_col] = int(in_window)

    frame[date_col] = frame[date_col].dt.date
    return frame
