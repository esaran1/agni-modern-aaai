from __future__ import annotations

import pandas as pd


def build_occurrence_labels(
    df: pd.DataFrame,
    event_indicator_col: str = "observed_event",
    event_date_col: str = "observed_event_date",
    horizon_days: int = 30,
    group_col: str = "patch_id",
    date_col: str = "reference_date",
) -> pd.DataFrame:
    required_columns = {event_indicator_col, event_date_col, group_col, date_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Occurrence label building requires the current label-materialization "
            f"columns. Missing: {', '.join(missing_columns)}."
        )

    frame = df.sort_values([group_col, date_col]).copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[event_date_col] = pd.to_datetime(frame[event_date_col])
    invalid_positive_events = (
        frame[event_indicator_col].fillna(0).astype(int).eq(1) & frame[event_date_col].isna()
    )
    if invalid_positive_events.any():
        raise ValueError(
            "Occurrence label building requires dated positive events. "
            f"Found {int(invalid_positive_events.sum())} rows with "
            f"{event_indicator_col}=1 and missing {event_date_col}."
        )
    label_col = f"y_occ_{horizon_days}d"
    frame[label_col] = 0

    for _patch_id, group in frame.groupby(group_col, sort=False):
        event_dates = group.loc[
            group[event_indicator_col].fillna(0).astype(int) == 1,
            event_date_col,
        ].dropna()
        if event_dates.empty:
            continue
        for idx, ref_date in zip(group.index, group[date_col], strict=False):
            in_window = (
                (event_dates > ref_date)
                & (event_dates <= ref_date + pd.Timedelta(days=horizon_days))
            ).any()
            frame.at[idx, label_col] = int(in_window)

    frame[date_col] = frame[date_col].dt.date
    return frame
