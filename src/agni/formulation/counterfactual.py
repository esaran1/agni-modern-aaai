from __future__ import annotations

import numpy as np


def observed_outcome(y_treated, treatment, y_control=None):
    """Compose the observed outcome from potential outcomes."""
    y_treated = np.asarray(y_treated, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if y_control is None:
        y_control = np.zeros_like(y_treated)
    else:
        y_control = np.asarray(y_control, dtype=float)
    return treatment * y_treated + (1.0 - treatment) * y_control


def estimate_ipw_mean_treated_outcome(y_obs, treatment, propensity, clip_min: float = 1e-3) -> float:
    """Horvitz-Thompson/IPW estimate of E[Y(1)]."""
    y_obs = np.asarray(y_obs, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    propensity = np.asarray(propensity, dtype=float)
    weights = treatment / np.clip(propensity, clip_min, 1.0)
    return float(np.mean(weights * y_obs))


def compute_expected_risk_from_potential_outcomes(propensity, conditional_severity):
    """Expected risk R(X) = P(D=1|X) * E[Y(1)|X]."""
    propensity = np.asarray(propensity, dtype=float)
    conditional_severity = np.asarray(conditional_severity, dtype=float)
    return propensity * conditional_severity
