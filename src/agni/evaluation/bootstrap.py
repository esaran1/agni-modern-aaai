from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_metric(
    y_true,
    y_score,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        if np.unique(sample_true).size < 2:
            continue
        values.append(metric_fn(sample_true, y_score[indices]))
    if not values:
        raise ValueError("Unable to compute bootstrap: no valid resamples")
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "n_bootstrap": len(values),
    }


def patch_block_bootstrap(
    df: pd.DataFrame,
    patch_col: str,
    score_col: str,
    target_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    patches = df[patch_col].dropna().unique()
    values = []
    for _ in range(n_bootstrap):
        sampled_patches = rng.choice(patches, size=len(patches), replace=True)
        sampled = pd.concat(
            [df[df[patch_col] == patch] for patch in sampled_patches],
            ignore_index=True,
        )
        if sampled[target_col].nunique() < 2:
            continue
        values.append(metric_fn(sampled[target_col].to_numpy(), sampled[score_col].to_numpy()))
    if not values:
        raise ValueError("Unable to compute patch bootstrap: no valid resamples")
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "n_bootstrap": len(values),
    }
