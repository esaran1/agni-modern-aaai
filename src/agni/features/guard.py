ALLOWED_PREFIXES = [
    "weather_",
    "optical_",
    "terrain_",
    "temporal_",
    "landcover_",
    "anthropogenic_",
    "peat_",
    "human_",
]

FORBIDDEN_PATTERNS = [
    "y_occ_",
    "y_sev_",
    "y_sev_available",
    "dnbr",
    "delta_nbr",
    "prefire",
    "postfire",
    "viirs_future_",
    "label_window_",
    "event_date",
    "reference_date",
    "patch_id",
    "split",
    "block_id",
    "patch_row",
    "patch_col",
    "centroid_lat",
    "centroid_lon",
]


def assert_no_leakage(feature_columns: list[str]) -> None:
    for col in feature_columns:
        col_lower = col.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in col_lower:
                raise ValueError(f"LEAKAGE: '{col}' matches forbidden pattern '{pattern}'")
        if not any(col.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            raise ValueError(
                f"UNRECOGNIZED: '{col}' does not match any allowed prefix. "
                "Add it to ALLOWED_PREFIXES or FORBIDDEN_PATTERNS."
            )


def infer_feature_columns(df) -> list[str]:
    candidates = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    ]
    assert_no_leakage(candidates)
    return candidates
