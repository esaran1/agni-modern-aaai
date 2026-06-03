from __future__ import annotations

import numpy as np
import pandas as pd

from agni.experiment_utils import carve_conformal_calibration_split
from agni.models.conformal import SplitConformalRiskPredictor


def test_conformal_coverage_guarantee() -> None:
    np.random.seed(42)
    n_cal, n_test = 100, 200
    true = np.random.uniform(0, 1, n_cal + n_test)
    noise = np.random.normal(0, 0.1, n_cal + n_test)
    pred = true + noise

    cp = SplitConformalRiskPredictor(alpha=0.10)
    cp.calibrate(pred[:n_cal], true[:n_cal])
    result = cp.evaluate_coverage(pred[n_cal:], true[n_cal:])
    assert result["empirical_coverage"] >= 0.85


def test_conformal_wider_with_more_noise() -> None:
    rng = np.random.default_rng(42)
    cp_clean = SplitConformalRiskPredictor(alpha=0.10)
    cp_noisy = SplitConformalRiskPredictor(alpha=0.10)
    true = rng.uniform(0, 1, 50)
    cp_clean.calibrate(true + rng.normal(0, 0.01, 50), true)
    cp_noisy.calibrate(true + rng.normal(0, 0.5, 50), true)
    assert cp_noisy.q > cp_clean.q


def test_conformal_uses_higher_order_statistic() -> None:
    cp = SplitConformalRiskPredictor(alpha=0.50)
    cp.calibrate(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 0.0]))
    assert cp.q == 10.0


def test_conformal_calibration_split_is_disjoint_from_tuning_validation() -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-15",
                    "2020-01-22",
                    "2020-01-29",
                    "2020-02-05",
                ]
            ),
            "split": ["train", "val", "val", "val", "val", "test"],
        }
    )
    partitioned = carve_conformal_calibration_split(df, calibration_fraction=0.5)
    tune_dates = partitioned.loc[partitioned["split"] == "val", "reference_date"]
    cal_dates = partitioned.loc[partitioned["split"] == "calibration", "reference_date"]
    assert not tune_dates.empty
    assert not cal_dates.empty
    assert tune_dates.max() < cal_dates.min()


def test_conformal_calibration_split_requires_evaluable_rows() -> None:
    df = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-08",
                    "2020-01-15",
                    "2020-01-22",
                    "2020-01-29",
                    "2020-02-05",
                    "2020-02-12",
                    "2020-02-19",
                ]
            ),
            "split": ["train", "val", "val", "val", "val", "val", "val", "test"],
            "y_sev_available": [0, 1, 0, 1, 1, 0, 1, 1],
            "y_sev_dnbr": [np.nan, 0.2, np.nan, 0.3, 0.4, np.nan, 0.5, 0.6],
        }
    )
    partitioned = carve_conformal_calibration_split(
        df,
        calibration_fraction=0.5,
        min_required_rows=2,
        required_columns=("y_sev_available", "y_sev_dnbr"),
    )
    tune_mask = (
        (partitioned["split"] == "val")
        & partitioned["y_sev_available"].eq(1)
        & partitioned["y_sev_dnbr"].notna()
    )
    calibration_mask = (
        (partitioned["split"] == "calibration")
        & partitioned["y_sev_available"].eq(1)
        & partitioned["y_sev_dnbr"].notna()
    )
    assert int(tune_mask.sum()) >= 2
    assert int(calibration_mask.sum()) >= 2
