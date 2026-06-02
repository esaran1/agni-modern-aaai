from __future__ import annotations

import numpy as np

from agni.formulation.counterfactual import estimate_ipw_mean_treated_outcome, observed_outcome


def test_ipw_estimator_is_unbiased_on_synthetic_data() -> None:
    rng = np.random.default_rng(42)
    n = 50000
    x = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-x))
    y1 = 2.0 + 0.5 * x
    treatment = rng.binomial(1, propensity, size=n)
    y_obs = observed_outcome(y1, treatment)

    estimate = estimate_ipw_mean_treated_outcome(y_obs, treatment, propensity)
    truth = float(np.mean(y1))
    assert abs(estimate - truth) < 0.05
