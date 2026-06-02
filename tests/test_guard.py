from __future__ import annotations

import pytest

from agni.features.guard import assert_no_leakage


def test_leakage_catches_postfire() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["postfire_nbr"])


def test_leakage_catches_label() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["y_occ_30d"])


def test_allowed_features_pass() -> None:
    assert_no_leakage(["weather_vpd_mean_l7d", "terrain_twi_mean"])
