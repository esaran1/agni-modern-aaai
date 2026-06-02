from __future__ import annotations

import pandas as pd


def compute_tree_shap(model, x: pd.DataFrame) -> dict[str, object]:
    try:
        import shap
    except ImportError:
        return {"warning": "shap is not installed"}

    estimator = getattr(model, "model", model)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(x)
    return {
        "values": values,
        "expected_value": explainer.expected_value,
    }
