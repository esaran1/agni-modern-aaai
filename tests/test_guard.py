from __future__ import annotations

import pandas as pd
import pytest

from agni.features.guard import assert_no_leakage, infer_feature_columns


def test_leakage_catches_postfire() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["postfire_nbr"])


def test_leakage_catches_prefire_optical_columns() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["optical_nbr_prefire"])


def test_leakage_catches_label() -> None:
    with pytest.raises(ValueError):
        assert_no_leakage(["y_occ_30d"])


def test_allowed_features_pass() -> None:
    assert_no_leakage(["weather_vpd_mean_l7d", "terrain_twi_mean"])


def test_severity_labeled_frame_infers_features_without_leakage() -> None:
    # A severity/risk labeled frame carries event-anchored NBR support and y_sev_*
    # targets; feature inference must exclude them (no leakage trip) while keeping
    # the genuine predictors.
    frame = pd.DataFrame(
        {
            "patch_id": ["0_0"],
            "reference_date": ["2019-08-01"],
            "weather_vpd_mean_l7d": [1.0],
            "optical_nbr_mean_l30d": [0.4],
            "temporal_burn_count_l7d": [0.0],
            "label_nbr_prefire": [0.6],
            "label_nbr_postfire": [0.2],
            "y_sev_dnbr": [0.4],
            "y_sev_available": [1],
            "event_date": ["2019-08-10"],
        }
    )
    features = infer_feature_columns(frame)
    assert set(features) == {
        "weather_vpd_mean_l7d",
        "optical_nbr_mean_l30d",
        "temporal_burn_count_l7d",
    }
