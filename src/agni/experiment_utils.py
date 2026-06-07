from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from agni.evaluation.metrics import expected_calibration_error, regression_metrics
from agni.features.guard import infer_feature_columns
from agni.models import build_model
from agni.models.joint_risk_model import (
    JointRiskTrainer,
    JointRiskTransformer,
    build_joint_risk_loader,
)
from agni.models.propensity_severity import (
    SUPPORTED_SEVERITY_ESTIMATORS,
    NaiveSeverityModel,
    PropensityWeightedSeverityModel,
)
from agni.risk.expected_risk import compute_expected_risk, evaluate_risk_ranking


@dataclass
class ModelRunResult:
    model: object
    predictions: pd.DataFrame
    metrics: dict[str, float]
    feature_columns: list[str]


def occurrence_target_column(horizon_days: int) -> str:
    return f"y_occ_{int(horizon_days)}d"


def validate_severity_estimator_name(estimator_name: str) -> str:
    normalized = estimator_name.lower()
    if normalized not in SUPPORTED_SEVERITY_ESTIMATORS:
        supported = ", ".join(sorted(SUPPORTED_SEVERITY_ESTIMATORS))
        raise ValueError(
            f"Severity/risk workflows do not support model family '{estimator_name}'. "
            f"Supported families are: {supported}."
        )
    return normalized


def safe_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    if np.unique(y_true).size < 2:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        from sklearn.metrics import average_precision_score

        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def _test_split_metrics(
    df: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    task: str,
) -> dict[str, float]:
    test_df = df[df["split"] == "test"].copy()
    if task == "classification":
        return safe_classification_metrics(test_df[target_column], test_df[prediction_column])
    return regression_metrics(test_df[target_column], test_df[prediction_column])


def train_model_on_existing_split(
    df: pd.DataFrame,
    model_name: str,
    model_task: str,
    model_params: dict,
    target_column: str,
    feature_columns: list[str] | None = None,
    fit_kwargs: dict | None = None,
) -> ModelRunResult:
    frame = df.copy()
    feature_columns = feature_columns or infer_feature_columns(frame)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split assignment produced an empty partition: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    model = build_model(model_name, {"task": model_task, "params": model_params})
    fit_kwargs = fit_kwargs or {}
    model.fit(train_df, val_df, feature_columns, target_column, **fit_kwargs)
    predictions = frame.copy()
    predictions["prediction"] = model.predict_proba(predictions, feature_columns)
    metrics = _test_split_metrics(predictions, target_column, "prediction", model_task)
    return ModelRunResult(
        model=model,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns,
    )


def attach_occurrence_propensity(
    df: pd.DataFrame,
    horizon_days: int,
    feature_columns: list[str] | None = None,
    model_name: str = "xgboost",
    model_params: dict | None = None,
    n_splits: int = 5,
    group_column: str = "patch_id",
) -> ModelRunResult:
    target_column = occurrence_target_column(horizon_days)
    frame = df.copy()
    feature_columns = feature_columns or infer_feature_columns(frame)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Propensity estimation requires non-empty train, val, and test splits"
        )

    if group_column in train_df.columns:
        groups = train_df[group_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
            split_iter = splitter.split(train_df, train_df[target_column], groups)
        else:
            splitter = KFold(
                n_splits=min(n_splits, len(train_df)),
                shuffle=True,
                random_state=42,
            )
            split_iter = splitter.split(train_df)
    else:
        splitter = KFold(
            n_splits=min(n_splits, len(train_df)),
            shuffle=True,
            random_state=42,
        )
        split_iter = splitter.split(train_df)

    oof_predictions = pd.Series(index=train_df.index, dtype=float)
    params = model_params or {}
    for train_indices, holdout_indices in split_iter:
        fold_train = train_df.iloc[train_indices].copy()
        fold_holdout = train_df.iloc[holdout_indices].copy()
        fold_model = build_model(
            model_name,
            {"task": "classification", "params": params},
        )
        fold_model.fit(fold_train, fold_holdout, feature_columns, target_column)
        holdout_pred = fold_model.predict_proba(fold_holdout, feature_columns)
        oof_predictions.loc[fold_holdout.index] = holdout_pred.to_numpy()

    if oof_predictions.isna().any():
        raise RuntimeError("Cross-fitted train propensities were not fully assigned")

    final_model_result = train_model_on_existing_split(
        df=frame,
        model_name=model_name,
        model_task="classification",
        model_params=params,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    predictions = final_model_result.predictions.copy()
    predictions.loc[train_df.index, "prediction"] = oof_predictions
    predictions["propensity_score"] = predictions["prediction"]
    predictions["occurrence_prediction"] = predictions["prediction"]
    metrics = _test_split_metrics(predictions, target_column, "prediction", "classification")
    return ModelRunResult(
        model=final_model_result.model,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns,
    )


def train_severity_variant(
    df: pd.DataFrame,
    model_type: str,
    model_params: dict,
    feature_columns: list[str],
    estimator_name: str = "xgboost",
    propensity_column: str = "propensity_score",
    target_column: str = "y_sev_dnbr",
) -> ModelRunResult:
    estimator_name = validate_severity_estimator_name(estimator_name)
    severity_df = df[df["y_sev_available"] == 1].copy()
    if severity_df.empty:
        raise ValueError("No severity-available rows found")

    train_df = severity_df[severity_df["split"] == "train"].copy()
    val_df = severity_df[severity_df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Severity training requires non-empty train and val severity splits")

    if model_type == "naive":
        model = NaiveSeverityModel(
            {"params": model_params, "estimator_name": estimator_name},
        )
    elif model_type == "ipw":
        model = PropensityWeightedSeverityModel(
            {"params": model_params, "estimator_name": estimator_name},
        )
    else:
        raise ValueError(f"Unknown severity model type '{model_type}'")

    model.fit(
        train_df,
        val_df,
        feature_columns,
        target_column,
        propensity_column=propensity_column,
    )
    predictions = severity_df.copy()
    predictions["severity_prediction"] = model.predict(predictions, feature_columns)
    test_df = predictions[predictions["split"] == "test"].copy()
    metrics = regression_metrics(test_df[target_column], test_df["severity_prediction"])
    risk = compute_expected_risk(test_df[propensity_column], test_df["severity_prediction"])
    rho, _ = spearmanr(risk, test_df[target_column])
    metrics["risk_spearman"] = float(rho) if not np.isnan(rho) else float("nan")
    return ModelRunResult(
        model=model,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns,
    )


def _build_temporal_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = df.copy()
    if not frame.index.is_unique:
        raise ValueError("Temporal sequence construction requires unique row indices")
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    frame = frame.sort_values(["patch_id", "reference_date"]).copy()
    sequences: dict[int, np.ndarray] = {}
    lengths: dict[int, int] = {}

    for _, group in frame.groupby("patch_id", sort=False):
        group_indices = group.index.to_list()
        for position, row_index in enumerate(group_indices):
            start = max(0, position - sequence_length + 1)
            window_indices = group_indices[start : position + 1]
            window = (
                frame.loc[window_indices, feature_columns]
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            seq_len = window.shape[0]
            padded = np.zeros((sequence_length, len(feature_columns)), dtype=np.float32)
            padded[-seq_len:, :] = window
            sequences[row_index] = padded
            lengths[row_index] = seq_len

    ordered_sequences = np.stack([sequences[idx] for idx in df.index], axis=0)
    ordered_lengths = np.asarray([lengths[idx] for idx in df.index], dtype=np.int64)
    return ordered_sequences, ordered_lengths


def _frame_to_joint_tensors(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    subset_index: pd.Index | None = None,
    occurrence_target_column_name: str = "y_occ_30d",
):
    import torch

    sequence_array, sequence_lengths = _build_temporal_sequences(
        df,
        feature_columns,
        sequence_length=sequence_length,
    )
    if subset_index is None:
        subset_positions = np.arange(len(df))
        subset_df = df
    else:
        subset_positions = df.index.get_indexer(subset_index)
        if np.any(subset_positions < 0):
            raise ValueError("Subset indices must be present in the source dataframe")
        subset_df = df.loc[subset_index]

    x = torch.tensor(sequence_array[subset_positions], dtype=torch.float32)
    seq_lengths = torch.tensor(sequence_lengths[subset_positions], dtype=torch.long)
    y_occ = torch.tensor(
        subset_df[occurrence_target_column_name].to_numpy(dtype=np.float32),
    )
    y_sev = torch.tensor(subset_df["y_sev_dnbr"].fillna(0.0).to_numpy(dtype=np.float32))
    sev_avail = torch.tensor(subset_df["y_sev_available"].to_numpy(dtype=np.float32))
    return x, seq_lengths, y_occ, y_sev, sev_avail


def carve_conformal_calibration_split(
    df: pd.DataFrame,
    calibration_fraction: float = 0.5,
    min_required_rows: int = 1,
    required_columns: tuple[str, str] | None = None,
) -> pd.DataFrame:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    if min_required_rows < 1:
        raise ValueError("min_required_rows must be >= 1")

    frame = df.copy()
    val_mask = frame["split"] == "val"
    if not val_mask.any():
        raise ValueError("Need a non-empty validation split to derive calibration data")

    if required_columns is not None:
        available_column, label_column = required_columns
        required_mask = (
            val_mask
            & frame[available_column].fillna(0).astype(int).eq(1)
            & frame[label_column].notna()
        )
        if int(required_mask.sum()) < 2 * min_required_rows:
            raise ValueError(
                "Need enough evaluable validation rows to form tuning and calibration splits"
            )
        val_dates = (
            pd.to_datetime(frame.loc[required_mask, "reference_date"]).sort_values().unique()
        )
    else:
        required_mask = val_mask
        val_dates = pd.to_datetime(frame.loc[val_mask, "reference_date"]).sort_values().unique()
    if len(val_dates) < 2:
        raise ValueError("Need at least two unique validation dates for calibration partitioning")

    cutoff_idx = int(np.floor(len(val_dates) * (1.0 - calibration_fraction)))
    cutoff_idx = min(max(cutoff_idx, 1), len(val_dates) - 1)
    calibration_start = pd.Timestamp(val_dates[cutoff_idx])
    val_dates_series = pd.to_datetime(frame["reference_date"])
    calibration_mask = val_mask & (val_dates_series >= calibration_start)
    tune_mask = val_mask & ~calibration_mask

    if not tune_mask.any() or not calibration_mask.any():
        raise ValueError("Validation partitioning produced an empty tuning or calibration split")
    if required_columns is not None:
        required_calibration = required_mask & calibration_mask
        required_tuning = required_mask & tune_mask
        if int(required_calibration.sum()) < min_required_rows:
            raise ValueError("Calibration split does not contain enough evaluable severity rows")
        if int(required_tuning.sum()) < min_required_rows:
            raise ValueError(
                "Tuning validation split does not contain enough evaluable severity rows"
            )

    frame.loc[calibration_mask, "split"] = "calibration"
    return frame


def train_joint_risk_variant(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict,
    lambda_rank: float,
    propensity_column: str = "propensity_score",
    occurrence_target_column_name: str = "y_occ_30d",
) -> ModelRunResult:
    import torch

    params = dict(model_params)
    params["lambda_rank"] = lambda_rank
    torch.manual_seed(int(params.get("random_state", params.get("seed", 42))))
    epochs = int(params.get("epochs", 10))
    patience = int(params.get("patience", 3))
    batch_size = int(params.get("batch_size", 64))
    sequence_length = int(params.get("max_seq_len", 8))

    if not df.index.is_unique:
        raise ValueError("Joint-risk training requires unique row indices")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    x_train, train_lengths, y_occ_train, y_sev_train, sev_avail_train = _frame_to_joint_tensors(
        df,
        feature_columns,
        sequence_length=sequence_length,
        subset_index=train_df.index,
        occurrence_target_column_name=occurrence_target_column_name,
    )
    x_val, val_lengths, y_occ_val, y_sev_val, sev_avail_val = _frame_to_joint_tensors(
        df,
        feature_columns,
        sequence_length=sequence_length,
        subset_index=val_df.index,
        occurrence_target_column_name=occurrence_target_column_name,
    )

    train_loader = build_joint_risk_loader(
        x_train,
        train_lengths,
        y_occ_train,
        y_sev_train,
        sev_avail_train,
        batch_size=batch_size,
        include_indices=True,
    )
    val_loader = build_joint_risk_loader(
        x_val,
        val_lengths,
        y_occ_val,
        y_sev_val,
        sev_avail_val,
        batch_size=batch_size,
        shuffle=False,
    )

    model = JointRiskTransformer(input_dim=len(feature_columns), config=params)
    trainer = JointRiskTrainer(model, params)
    propensity_scores = torch.tensor(
        train_df[propensity_column].fillna(0.5).to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    best_state = deepcopy(model.state_dict())
    best_metric = -float("inf")
    epochs_without_improvement = 0
    for _ in range(epochs):
        trainer.train_epoch(train_loader, propensity_scores=propensity_scores)
        val_metric = trainer.evaluate(val_loader)
        if val_metric > best_metric:
            best_metric = val_metric
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    full_x, full_lengths, _, _, _ = _frame_to_joint_tensors(
        df,
        feature_columns,
        sequence_length=sequence_length,
        occurrence_target_column_name=occurrence_target_column_name,
    )
    with torch.no_grad():
        occ_logit, sev_pred = model(full_x, seq_lengths=full_lengths)
        occ_prob = torch.sigmoid(occ_logit).cpu().numpy()
        sev_np = sev_pred.cpu().numpy()

    predictions = df.copy()
    predictions["occurrence_prediction"] = occ_prob
    predictions["severity_prediction"] = sev_np
    predictions["risk_prediction"] = (
        predictions["occurrence_prediction"] * predictions["severity_prediction"]
    )

    test_predictions = predictions[predictions["split"] == "test"].copy()
    occ_metrics = safe_classification_metrics(
        test_predictions[occurrence_target_column_name],
        test_predictions["occurrence_prediction"],
    )
    severity_test = test_predictions[test_predictions["y_sev_available"] == 1].copy()
    sev_metrics = regression_metrics(
        severity_test["y_sev_dnbr"],
        severity_test["severity_prediction"],
    )
    rho, _ = spearmanr(severity_test["risk_prediction"], severity_test["y_sev_dnbr"])
    metrics = {
        "occurrence_roc_auc": occ_metrics["roc_auc"],
        "severity_rmse": sev_metrics["rmse"],
        "severity_mae": sev_metrics["mae"],
        "risk_spearman": float(rho) if not np.isnan(rho) else float("nan"),
    }
    return ModelRunResult(
        model=model,
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns,
    )


def fit_risk_pipeline(
    df: pd.DataFrame,
    horizon_days: int,
    model_params: dict | None = None,
    feature_columns: list[str] | None = None,
    occurrence_model_name: str = "xgboost",
    occurrence_model_params: dict | None = None,
    severity_estimator_name: str | None = None,
    severity_model_params: dict | None = None,
    severity_model_type: str = "ipw",
) -> ModelRunResult:
    frame = df.copy()
    feature_columns = feature_columns or infer_feature_columns(frame)
    occ_target = occurrence_target_column(horizon_days)
    occurrence_params = dict(occurrence_model_params or model_params or {})
    severity_params = dict(severity_model_params or model_params or {})
    severity_estimator_name = validate_severity_estimator_name(
        severity_estimator_name or occurrence_model_name
    )
    occurrence = attach_occurrence_propensity(
        frame,
        horizon_days=horizon_days,
        feature_columns=feature_columns,
        model_name=occurrence_model_name,
        model_params=occurrence_params,
    )
    severity = train_severity_variant(
        occurrence.predictions,
        model_type=severity_model_type,
        model_params=severity_params,
        feature_columns=feature_columns,
        estimator_name=severity_estimator_name,
    )
    predictions = occurrence.predictions.copy()
    predictions["severity_prediction"] = severity.model.predict(predictions, feature_columns)
    predictions["risk_score"] = compute_expected_risk(
        predictions["occurrence_prediction"],
        predictions["severity_prediction"],
    )
    test_df = predictions[predictions["split"] == "test"].copy()
    metrics = evaluate_risk_ranking(
        test_df["risk_score"],
        test_df["y_sev_dnbr"],
        test_df[occ_target],
    )
    return ModelRunResult(
        model={
            "occurrence_model": occurrence.model,
            "severity_model": severity.model,
        },
        predictions=predictions,
        metrics=metrics,
        feature_columns=feature_columns,
    )
