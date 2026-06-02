from __future__ import annotations

import numpy as np

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
