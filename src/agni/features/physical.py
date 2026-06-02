from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vpd_features(df: pd.DataFrame) -> pd.DataFrame:
    windows: set[str] = set()
    for col in df.columns:
        if col.startswith("weather_temperature_2m_mean_l") and col.endswith("d"):
            windows.add(col.split("_l")[1].rstrip("d"))

    for window in sorted(windows, key=int):
        temp_col = f"weather_temperature_2m_mean_l{window}d"
        dew_col = f"weather_dewpoint_temperature_2m_mean_l{window}d"
        if temp_col not in df.columns or dew_col not in df.columns:
            continue

        temp_c = df[temp_col].astype(float).copy()
        dew_c = df[dew_col].astype(float).copy()
        if temp_c.dropna().median() > 200:
            temp_c -= 273.15
            dew_c -= 273.15

        e_sat = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        e_act = 0.6108 * np.exp(17.27 * dew_c / (dew_c + 237.3))
        df[f"weather_vpd_mean_l{window}d"] = (e_sat - e_act).clip(lower=0)

        temp_max_col = f"weather_temperature_2m_max_l{window}d"
        if temp_max_col in df.columns:
            temp_max_c = df[temp_max_col].astype(float).copy()
            if temp_max_c.dropna().median() > 200:
                temp_max_c -= 273.15
            e_sat_max = 0.6108 * np.exp(17.27 * temp_max_c / (temp_max_c + 237.3))
            df[f"weather_vpd_max_l{window}d"] = (e_sat_max - e_act).clip(lower=0)
    return df


def compute_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.startswith("weather_u_component_of_wind_10m_mean_l"):
            suffix = col.split("_mean_")[1]
            u_col = f"weather_u_component_of_wind_10m_mean_{suffix}"
            v_col = f"weather_v_component_of_wind_10m_mean_{suffix}"
            if u_col in df.columns and v_col in df.columns:
                df[f"weather_wind_speed_mean_{suffix}"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
    return df


def compute_terrain_range(df: pd.DataFrame) -> pd.DataFrame:
    min_col = "terrain_elevation_min"
    max_col = "terrain_elevation_max"
    if min_col in df.columns and max_col in df.columns:
        df["terrain_elevation_range"] = df[max_col] - df[min_col]
    return df
