from __future__ import annotations

from collections import defaultdict

import pandas as pd

from agni.evaluation.metrics import classification_metrics, regression_metrics


def group_features_by_namespace(feature_columns: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for column in feature_columns:
        namespace = column.split("_", 1)[0]
        grouped[namespace].append(column)
    return dict(grouped)


def leave_one_source_out_ablation(
    model_builder,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    task: str,
) -> pd.DataFrame:
    grouped = group_features_by_namespace(feature_columns)
    rows = []
    for namespace, removed_columns in grouped.items():
        kept_columns = [col for col in feature_columns if col not in removed_columns]
        model = model_builder()
        model.fit(df_train, df_val, kept_columns, target_column)
        preds = model.predict_proba(df_test, kept_columns)
        metrics = (
            classification_metrics(df_test[target_column], preds)
            if task == "classification"
            else regression_metrics(df_test[target_column], preds)
        )
        rows.append({"removed_namespace": namespace, **metrics})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("removed_namespace").reset_index(drop=True)
