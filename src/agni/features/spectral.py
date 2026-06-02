from __future__ import annotations

import pandas as pd


def compute_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    for window in [7, 14, 30, 60]:
        suffix = f"_l{window}d"
        b2 = f"optical_b2_mean{suffix}"
        b3 = f"optical_b3_mean{suffix}"
        b4 = f"optical_b4_mean{suffix}"
        b8 = f"optical_b8_mean{suffix}"
        b11 = f"optical_b11_mean{suffix}"
        b12 = f"optical_b12_mean{suffix}"

        if b8 in df.columns and b4 in df.columns:
            df[f"optical_ndvi_mean{suffix}"] = (df[b8] - df[b4]) / (df[b8] + df[b4] + 1e-8)
        if b8 in df.columns and b12 in df.columns:
            df[f"optical_nbr_mean{suffix}"] = (df[b8] - df[b12]) / (df[b8] + df[b12] + 1e-8)
        if b8 in df.columns and b11 in df.columns:
            df[f"optical_ndmi_mean{suffix}"] = (df[b8] - df[b11]) / (df[b8] + df[b11] + 1e-8)
        if all(col in df.columns for col in [b8, b4, b2]):
            df[f"optical_evi_mean{suffix}"] = 2.5 * (
                (df[b8] - df[b4]) / (df[b8] + 6 * df[b4] - 7.5 * df[b2] + 1 + 1e-8)
            )
        if all(col in df.columns for col in [b8, b3, b11]):
            df[f"optical_savi_mean{suffix}"] = 1.5 * (
                (df[b8] - df[b4]) / (df[b8] + df[b4] + 0.5 + 1e-8)
            )
    return df
