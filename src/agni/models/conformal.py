from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _higher_quantile(values: np.ndarray, level: float) -> float:
    level = min(max(level, 0.0), 1.0)
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:  # pragma: no cover
        return float(np.quantile(values, level, interpolation="higher"))


@dataclass
class ConformalRiskSet:
    risk_point: float
    risk_lower: float
    risk_upper: float
    coverage_level: float


class SplitConformalRiskPredictor:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q: float | None = None
        self.cal_scores: np.ndarray | None = None

    def calibrate(self, cal_risk_pred: np.ndarray, cal_risk_true: np.ndarray):
        cal_risk_pred = np.asarray(cal_risk_pred, dtype=float)
        cal_risk_true = np.asarray(cal_risk_true, dtype=float)
        if len(cal_risk_pred) != len(cal_risk_true):
            raise ValueError("Calibration prediction and truth arrays must align")
        if len(cal_risk_pred) < 1:
            raise ValueError("Need at least 1 calibration sample")

        self.cal_scores = np.abs(cal_risk_pred - cal_risk_true)
        n = len(self.cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = _higher_quantile(self.cal_scores, level)
        return self

    def predict(self, test_risk_pred: np.ndarray) -> list[ConformalRiskSet]:
        if self.q is None:
            raise RuntimeError("Must calibrate before predicting")
        results = []
        for risk in np.asarray(test_risk_pred, dtype=float):
            results.append(
                ConformalRiskSet(
                    risk_point=float(risk),
                    risk_lower=float(max(risk - self.q, 0.0)),
                    risk_upper=float(risk + self.q),
                    coverage_level=1 - self.alpha,
                )
            )
        return results

    def evaluate_coverage(self, test_risk_pred: np.ndarray, test_risk_true: np.ndarray) -> dict:
        sets = self.predict(test_risk_pred)
        truth = np.asarray(test_risk_true, dtype=float)
        covered = sum(
            1
            for interval, observed in zip(sets, truth, strict=False)
            if interval.risk_lower <= observed <= interval.risk_upper
        )
        widths = np.array(
            [interval.risk_upper - interval.risk_lower for interval in sets],
            dtype=float,
        )
        result = {
            "empirical_coverage": float(covered / max(len(sets), 1)),
            "target_coverage": float(1 - self.alpha),
            "mean_interval_width": float(np.mean(widths)) if len(widths) else 0.0,
            "median_interval_width": float(np.median(widths)) if len(widths) else 0.0,
            "calibration_q": float(self.q if self.q is not None else 0.0),
            "n_calibration": int(len(self.cal_scores) if self.cal_scores is not None else 0),
            "n_test": int(len(sets)),
        }
        if result["n_calibration"] < 20:
            result["warning"] = (
                "Coverage guarantees are noisy with fewer than 20 calibration samples"
            )
        return result


class AdaptiveConformalRiskPredictor:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q: float | None = None

    def calibrate(self, cal_risk_pred, cal_risk_true, cal_sigma):
        scores = np.abs(np.asarray(cal_risk_pred) - np.asarray(cal_risk_true)) / (
            np.asarray(cal_sigma, dtype=float) + 1e-8
        )
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = _higher_quantile(scores, level)
        return self

    def predict(self, test_risk_pred, test_sigma):
        if self.q is None:
            raise RuntimeError("Must calibrate before predicting")
        results = []
        for risk, sigma in zip(
            np.asarray(test_risk_pred, dtype=float),
            np.asarray(test_sigma, dtype=float),
            strict=False,
        ):
            width = float(self.q * sigma)
            results.append(
                ConformalRiskSet(
                    risk_point=float(risk),
                    risk_lower=float(max(risk - width, 0.0)),
                    risk_upper=float(risk + width),
                    coverage_level=1 - self.alpha,
                )
            )
        return results
