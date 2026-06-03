from __future__ import annotations

import pandas as pd


def build_severity_labels(
    df: pd.DataFrame,
    burn_area_col: str = "modis_burn_date",
    pre_nbr_col: str = "optical_nbr_prefire",
    post_nbr_col: str = "optical_nbr_postfire",
    occurrence_col: str = "y_occ_30d",
) -> pd.DataFrame:
    has_burn = df[burn_area_col].notna() & (df[occurrence_col] == 1)
    df = df.copy()
    df["y_sev_available"] = has_burn.astype(int)
    df.loc[has_burn, "y_sev_dnbr"] = (
        df.loc[has_burn, pre_nbr_col] - df.loc[has_burn, post_nbr_col]
    ).clip(lower=0)
    df.loc[has_burn, "y_sev_class"] = pd.cut(
        df.loc[has_burn, "y_sev_dnbr"],
        bins=[0, 0.1, 0.27, 0.66, float("inf")],
        labels=["low", "moderate", "high", "extreme"],
    )
    return df
