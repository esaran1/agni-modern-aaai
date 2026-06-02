from __future__ import annotations

from datetime import date

import pandas as pd

from agni.models.logreg import LogisticRegressionModel


def test_logreg_model_fit_predict() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["0_0", "0_1", "0_2", "0_3"],
            "reference_date": [date(2020, 1, 1)] * 4,
            "weather_vpd_mean_l7d": [1.0, 2.0, 3.0, 4.0],
            "terrain_twi_mean": [0.1, 0.2, 0.3, 0.4],
            "y_occ_30d": [0, 0, 1, 1],
        }
    )
    model = LogisticRegressionModel({"task": "classification", "params": {"max_iter": 200}})
    model.fit(df.iloc[:3], df.iloc[3:], ["weather_vpd_mean_l7d", "terrain_twi_mean"], "y_occ_30d")
    preds = model.predict_proba(df, ["weather_vpd_mean_l7d", "terrain_twi_mean"])
    assert len(preds) == len(df)
