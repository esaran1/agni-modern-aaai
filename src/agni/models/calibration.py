from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("Calibration method must be 'isotonic' or 'platt'")
        self.method = method
        self.model = None

    def fit(self, y_score, y_true) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(y_score, y_true)
        else:
            self.model = LogisticRegression()
            self.model.fit(np.asarray(y_score).reshape(-1, 1), y_true)
        return self

    def predict(self, y_score):
        if self.model is None:
            raise RuntimeError("Calibrator has not been fit")
        if self.method == "isotonic":
            return self.model.predict(y_score)
        return self.model.predict_proba(np.asarray(y_score).reshape(-1, 1))[:, 1]
