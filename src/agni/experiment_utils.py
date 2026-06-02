from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from agni.evaluation.metrics import expected_calibration_error, regression_metrics
from agni.features.guard import infer_feature_columns
from agni.models import build_model
from agni.models.joint_risk_model import JointRiskTrainer, JointRiskTransformer, build_joint_risk_loader
from agni.models.propensity_severity import NaiveSeverityModel, PropensityWeightedSeverityModel
from agni.risk.expected_risk import compute_expected_risk


@dataclass
class ModelRunResult:
    model: object
    predictions: pd.DataFrame
    metrics: dict[str, float]
    feature_columns: list[str]


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


def _test_split_metrics(df: pd.DataFrame, target_column: str, prediction_column: str, task: str) -> dict[str, float]:
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
    return ModelRunResult(model=model, predictions=predictions, metrics=metrics, feature_columns=feature_columns)


def attach_occurrence_propensity(
    df: pd.DataFrame,
    horizon_days: int,
    feature_columns: list[str] | None = None,
    model_name: str = "xgboost",
    model_params: dict | None = None,
) -> ModelRunResult:
    target_column = f"y_occ_{horizon_days}d"
    result = train_model_on_existing_split(
        df=df,
        model_name=model_name,
        model_task="classification",
        model_params=model_params or {},
        target_column=target_column,
        feature_columns=feature_columns,
    )
    result.predictions["propensity_score"] = result.predictions["prediction"]
    return result


def train_severity_variant(
    df: pd.DataFrame,
    model_type: str,
    model_params: dict,
    feature_columns: list[str],
    propensity_column: str = "propensity_score",
    target_column: str = "y_sev_dnbr",
) -> ModelRunResult:
    severity_df = df[df["y_sev_available"] == 1].copy()
    if severity_df.empty:
        raise ValueError("No severity-available rows found")

    train_df = severity_df[severity_df["split"] == "train"].copy()
    val_df = severity_df[severity_df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Severity training requires non-empty train and val severity splits")

    if model_type == "naive":
        model = NaiveSeverityModel({"params": model_params})
    elif model_type == "ipw":
        model = PropensityWeightedSeverityModel({"params": model_params})
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
    return ModelRunResult(model=model, predictions=predictions, metrics=metrics, feature_columns=feature_columns)


def _frame_to_joint_tensors(df: pd.DataFrame, feature_columns: list[str]):
    import torch

    x = torch.tensor(df[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)).unsqueeze(1)
    y_occ = torch.tensor(df["y_occ_30d"].to_numpy(dtype=np.float32))
    y_sev = torch.tensor(df["y_sev_dnbr"].fillna(0.0).to_numpy(dtype=np.float32))
    sev_avail = torch.tensor(df["y_sev_available"].to_numpy(dtype=np.float32))
    return x, y_occ, y_sev, sev_avail


def train_joint_risk_variant(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict,
    lambda_rank: float,
    propensity_column: str = "propensity_score",
) -> ModelRunResult:
    import torch

    params = dict(model_params)
    params["lambda_rank"] = lambda_rank
    epochs = int(params.get("epochs", 10))
    patience = int(params.get("patience", 3))
    batch_size = int(params.get("batch_size", 64))

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    x_train, y_occ_train, y_sev_train, sev_avail_train = _frame_to_joint_tensors(train_df, feature_columns)
    x_val, y_occ_val, y_sev_val, sev_avail_val = _frame_to_joint_tensors(val_df, feature_columns)

    train_loader = build_joint_risk_loader(
        x_train,
        y_occ_train,
        y_sev_train,
        sev_avail_train,
        batch_size=batch_size,
        include_indices=True,
    )
    val_loader = build_joint_risk_loader(
        x_val,
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
    full_x, _, _, _ = _frame_to_joint_tensors(df, feature_columns)
    with torch.no_grad():
        occ_logit, sev_pred = model(full_x)
        occ_prob = torch.sigmoid(occ_logit).cpu().numpy()
        sev_np = sev_pred.cpu().numpy()

    predictions = df.copy()
    predictions["occurrence_prediction"] = occ_prob
    predictions["severity_prediction"] = sev_np
    predictions["risk_prediction"] = predictions["occurrence_prediction"] * predictions["severity_prediction"]

    test_predictions = predictions[predictions["split"] == "test"].copy()
    occ_metrics = safe_classification_metrics(test_predictions["y_occ_30d"], test_predictions["occurrence_prediction"])
    severity_test = test_predictions[test_predictions["y_sev_available"] == 1].copy()
    sev_metrics = regression_metrics(severity_test["y_sev_dnbr"], severity_test["severity_prediction"])
    rho, _ = spearmanr(severity_test["risk_prediction"], severity_test["y_sev_dnbr"])
    metrics = {
        "occurrence_roc_auc": occ_metrics["roc_auc"],
        "severity_rmse": sev_metrics["rmse"],
        "severity_mae": sev_metrics["mae"],
        "risk_spearman": float(rho) if not np.isnan(rho) else float("nan"),
    }
    return ModelRunResult(model=model, predictions=predictions, metrics=metrics, feature_columns=feature_columns)
