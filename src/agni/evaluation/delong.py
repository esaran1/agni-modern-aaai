from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[sorted_idx] = midranks
    return result


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positives = predictions_sorted_transposed[:, :m]
    negatives = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r, :] = _compute_midrank(positives[r, :])
        ty[r, :] = _compute_midrank(negatives[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_variance(y_true, y_score) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions_sorted = y_score[np.newaxis, order]
    aucs, cov = _fast_delong(predictions_sorted, label_1_count)
    return float(aucs[0]), float(cov if np.isscalar(cov) else cov[0, 0])


def delong_roc_test(y_true, y_score_a, y_score_b) -> dict[str, float]:
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds = np.vstack([np.asarray(y_score_a), np.asarray(y_score_b)])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = 0.0 if var <= 0 else abs(diff) / math.sqrt(var)
    p_value = 2 * (1 - norm.cdf(z))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "z": float(z),
        "p_value": float(p_value),
    }
