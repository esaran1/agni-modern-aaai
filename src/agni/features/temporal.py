from __future__ import annotations

import pandas as pd


def compute_temporal_diffs(
    df: pd.DataFrame,
    columns: list[str],
    stride_days: int = 7,
) -> pd.DataFrame:
    df = df.sort_values(["patch_id", "reference_date"]).copy()
    for col in columns:
        if col in df.columns:
            diff = df.groupby("patch_id")[col].diff()
            df[f"{col}_diff_{stride_days}d"] = diff
            df[f"{col}_rate_{stride_days}d"] = diff / stride_days
    return df


def compute_desiccation_index(df: pd.DataFrame) -> pd.DataFrame:
    for window in [7, 14, 30, 60]:
        temp_col = f"weather_temperature_2m_mean_l{window}d"
        precip_col = f"weather_total_precipitation_sum_mean_l{window}d"
        if temp_col in df.columns and precip_col in df.columns:
            temp_z = (df[temp_col] - df[temp_col].mean()) / (df[temp_col].std() + 1e-8)
            precip_z = (df[precip_col] - df[precip_col].mean()) / (df[precip_col].std() + 1e-8)
            df[f"weather_desiccation_index_l{window}d"] = temp_z - precip_z
    return df


def compute_feature_ratios(df: pd.DataFrame) -> pd.DataFrame:
    ratio_pairs = [(7, 60), (7, 30), (14, 60), (3, 30)]
    base_vars = [
        "weather_total_precipitation_sum_mean",
        "weather_temperature_2m_mean",
        "weather_vpd_mean",
    ]
    for short_window, long_window in ratio_pairs:
        for var in base_vars:
            short_col = f"{var}_l{short_window}d"
            long_col = f"{var}_l{long_window}d"
            if short_col in df.columns and long_col in df.columns:
                df[f"{var}_ratio_{short_window}d_{long_window}d"] = df[short_col] / (
                    df[long_col] + 1e-8
                )
    return df
