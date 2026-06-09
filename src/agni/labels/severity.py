from __future__ import annotations

import pandas as pd


def build_severity_labels(
    df: pd.DataFrame,
    burn_area_col: str = "event_date",
    pre_nbr_col: str = "label_nbr_prefire",
    post_nbr_col: str = "label_nbr_postfire",
    occurrence_col: str | None = None,
) -> pd.DataFrame:
    if occurrence_col is None:
        occurrence_candidates = sorted(
            column for column in df.columns if column.startswith("y_occ_")
        )
        if len(occurrence_candidates) != 1:
            raise ValueError(
                "Severity label building requires an explicit occurrence column when "
                "the dataframe does not contain exactly one y_occ_* target."
            )
        occurrence_col = occurrence_candidates[0]

    required_columns = {burn_area_col, pre_nbr_col, post_nbr_col, occurrence_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Severity label building requires event-centered label columns. "
            f"Missing: {', '.join(missing_columns)}."
        )

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
