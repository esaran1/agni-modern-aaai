from __future__ import annotations

import numpy as np

from agni.evaluation.bootstrap import bootstrap_metric
from agni.evaluation.delong import delong_roc_test
from agni.evaluation.metrics import classification_metrics


def test_classification_metrics_and_bootstrap() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    metrics = classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] > 0.9
    boot = bootstrap_metric(y_true, y_prob, lambda a, b: classification_metrics(a, b)["roc_auc"], 50)
    assert boot["n_bootstrap"] > 0


def test_delong_test_runs() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.2, 0.9, 0.85, 0.3, 0.8, 0.35, 0.75])
    b = np.array([0.2, 0.25, 0.8, 0.7, 0.4, 0.65, 0.45, 0.6])
    result = delong_roc_test(y_true, a, b)
    assert "p_value" in result
