from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from agni.evaluation.leakage_taxonomy import compute_leakage_curve, detect_type2_leakage


def _make_frame() -> pd.DataFrame:
    rows = []
    start = date(2020, 1, 1)
    for idx in range(60):
        rows.append(
            {
                "patch_id": f"{idx // 10}_{idx % 10}",
                "reference_date": start + timedelta(days=idx * 3),
                "y_occ_30d": int(idx % 2 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_type2_detection_catches_known_leak() -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-20"]),
            "split": ["train", "train", "val"],
        }
    )
    result = detect_type2_leakage(df, horizon_days=30)
    assert result["leakage_detected"] is True
    assert result["overlap_days"] > 0


def test_leakage_curve_monotonic_non_increasing() -> None:
    df = _make_frame()

    def train_model_fn(df_train, df_val):
        del df_train, df_val
        return object()

    def evaluate_fn(model, df_test):
        del model, df_test
        return 1.0

    curve = compute_leakage_curve(df, train_model_fn, evaluate_fn, horizon_days=30, buffer_range=[0, 10, 20, 30, 40])
    assert curve["roc_auc"].is_monotonic_decreasing or curve["roc_auc"].nunique() == 1
