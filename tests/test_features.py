from __future__ import annotations

import numpy as np
import pandas as pd

from agni.features.physical import compute_vpd_features
from agni.features.spectral import compute_spectral_indices


def test_vpd_nonnegative() -> None:
    df = pd.DataFrame(
        {
            "weather_temperature_2m_mean_l60d": [300.0, 302.0],
            "weather_temperature_2m_max_l60d": [305.0, 307.0],
            "weather_dewpoint_temperature_2m_mean_l60d": [295.0, 296.0],
        }
    )
    result = compute_vpd_features(df)
    assert (result["weather_vpd_mean_l60d"] >= 0).all()
    assert (result["weather_vpd_max_l60d"] >= 0).all()


def test_spectral_indices_finite() -> None:
    df = pd.DataFrame(
        {
            "optical_b2_mean_l7d": [0.1, 0.2],
            "optical_b3_mean_l7d": [0.2, 0.3],
            "optical_b4_mean_l7d": [0.3, 0.4],
            "optical_b8_mean_l7d": [0.6, 0.8],
            "optical_b11_mean_l7d": [0.25, 0.35],
            "optical_b12_mean_l7d": [0.2, 0.3],
        }
    )
    result = compute_spectral_indices(df)
    cols = ["optical_ndvi_mean_l7d", "optical_nbr_mean_l7d", "optical_ndmi_mean_l7d"]
    assert np.isfinite(result[cols].to_numpy()).all()
