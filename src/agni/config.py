from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agni.models.propensity_severity import SUPPORTED_SEVERITY_ESTIMATORS

DEFAULT_MODEL_FAMILY_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
    },
    "logreg": {
        "max_iter": 500,
        "random_state": 42,
    },
    "transformer": {
        "hidden_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "epochs": 15,
        "lr": 1e-3,
        "batch_size": 128,
    },
}

DEFAULT_SEVERITY_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
    },
    "logreg": {},
}


class BBox(BaseModel):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @model_validator(mode="after")
    def validate_bounds(self) -> BBox:
        if self.lon_min >= self.lon_max:
            raise ValueError("bbox.lon_min must be < bbox.lon_max")
        if self.lat_min >= self.lat_max:
            raise ValueError("bbox.lat_min must be < bbox.lat_max")
        return self


class GridConfig(BaseModel):
    grid_km: int = Field(..., ge=1)
    bbox: BBox


class TemporalConfig(BaseModel):
    reference_start: date
    reference_end: date
    reference_stride_days: int = Field(..., ge=1)
    lookback_days: int = Field(..., ge=1)
    temporal_windows: list[int]
    horizon_days: int = Field(..., ge=1)

    @field_validator("temporal_windows")
    @classmethod
    def validate_windows(cls, windows: list[int]) -> list[int]:
        if not windows:
            raise ValueError("temporal_windows cannot be empty")
        if any(window <= 0 for window in windows):
            raise ValueError("temporal_windows must be positive")
        if windows != sorted(set(windows)):
            raise ValueError("temporal_windows must be unique and sorted ascending")
        return windows

    @model_validator(mode="after")
    def validate_dates(self) -> TemporalConfig:
        if self.reference_start > self.reference_end:
            raise ValueError("reference_start must be <= reference_end")
        if max(self.temporal_windows) > self.lookback_days:
            raise ValueError("max temporal window cannot exceed lookback_days")
        return self


class SplitConfig(BaseModel):
    buffer_days: int = Field(..., ge=0)
    train_end: date
    val_end: date
    test_end: date

    @model_validator(mode="after")
    def validate_order(self) -> SplitConfig:
        if not (self.train_end < self.val_end < self.test_end):
            raise ValueError("train_end < val_end < test_end is required")
        return self


class SpatialBlockConfig(BaseModel):
    block_size_km: int = Field(..., ge=1)
    n_folds: int = Field(..., ge=2)
    seed: int = 42


class SourceConfig(BaseModel):
    name: str
    enabled: bool
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    grid: GridConfig
    temporal: TemporalConfig
    split: SplitConfig
    spatial_blocks: SpatialBlockConfig | None = None
    sources: list[SourceConfig] = Field(default_factory=list)
    raw_dir: Path
    processed_dir: Path

    @model_validator(mode="after")
    def validate_buffer(self) -> DataConfig:
        if self.split.buffer_days < self.temporal.horizon_days:
            raise ValueError("split.buffer_days must be >= temporal.horizon_days")
        return self


class ModelOverrideConfig(BaseModel):
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    task: str
    params: dict[str, Any] = Field(default_factory=dict)
    occurrence: ModelOverrideConfig | None = None
    severity: ModelOverrideConfig | None = None
    comparison_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        allowed = {"classification", "regression"}
        if value not in allowed:
            raise ValueError(f"task must be one of {sorted(allowed)}")
        return value

    @staticmethod
    def _default_occurrence_params(model_name: str) -> dict[str, Any]:
        return dict(DEFAULT_MODEL_FAMILY_PARAMS.get(model_name, {}))

    @staticmethod
    def _default_severity_params(model_name: str) -> dict[str, Any]:
        return dict(DEFAULT_SEVERITY_PARAMS.get(model_name, {}))

    @staticmethod
    def _merged_params(
        base_params: dict[str, Any],
        override_params: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base_params)
        merged.update(override_params)
        return merged

    def resolve_occurrence_model_name(self) -> str:
        if self.occurrence is not None and self.occurrence.name:
            return self.occurrence.name
        return self.name

    def resolve_occurrence_model_params(self) -> dict[str, Any]:
        resolved_name = self.resolve_occurrence_model_name()
        if self.occurrence is not None:
            if resolved_name == self.name:
                base_params = dict(self.params)
            else:
                base_params = self._default_occurrence_params(resolved_name)
            return self._merged_params(base_params, self.occurrence.params)
        return dict(self.params)

    def resolve_severity_model_name(self) -> str:
        if self.severity is not None and self.severity.name:
            return self.severity.name
        return self.name

    def resolve_severity_model_params(self) -> dict[str, Any]:
        resolved_name = self.resolve_severity_model_name()
        if self.severity is not None:
            if resolved_name == self.name and self.name in SUPPORTED_SEVERITY_ESTIMATORS:
                base_params = dict(self.params)
            else:
                base_params = self._default_severity_params(resolved_name)
            return self._merged_params(base_params, self.severity.params)
        if self.name in SUPPORTED_SEVERITY_ESTIMATORS:
            return dict(self.params)
        return self._default_severity_params(self.name)

    def resolve_comparison_model_params(self, model_name: str) -> dict[str, Any]:
        if model_name in self.comparison_params:
            return dict(self.comparison_params[model_name])
        if model_name == self.name:
            return dict(self.params)
        return self._default_occurrence_params(model_name)


class EvaluationConfig(BaseModel):
    bootstrap_samples: int = Field(..., ge=0)
    confidence_level: float = Field(..., gt=0.0, lt=1.0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task: str
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: Path

    @field_validator("task")
    @classmethod
    def validate_experiment_task(cls, value: str) -> str:
        allowed = {"occurrence", "severity", "risk"}
        if value not in allowed:
            raise ValueError(f"experiment task must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_risk_model_compatibility(self) -> ExperimentConfig:
        if self.task == "occurrence" and self.model.task != "classification":
            raise ValueError(
                "Occurrence experiments require model.task == 'classification'."
            )
        if self.task == "severity" and self.model.task != "regression":
            raise ValueError(
                "Severity experiments require model.task == 'regression'."
            )
        if self.task == "risk" and self.model.task != "classification":
            raise ValueError(
                "Risk experiments require model.task == 'classification' for the "
                "occurrence stage."
            )
        if self.task != "risk":
            return self
        severity_name = self.model.resolve_severity_model_name()
        if severity_name not in SUPPORTED_SEVERITY_ESTIMATORS:
            supported = ", ".join(sorted(SUPPORTED_SEVERITY_ESTIMATORS))
            raise ValueError(
                "Risk workflows require a supported severity estimator family. "
                f"Resolved severity model '{severity_name}' is unsupported. "
                f"Set model.severity.name to one of: {supported}."
            )
        return self


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(Path(path)))


def require_experiment_task(
    experiment: ExperimentConfig,
    expected_task: str,
    entrypoint_name: str,
) -> None:
    if experiment.task != expected_task:
        raise ValueError(
            f"{entrypoint_name} requires config.task == '{expected_task}', "
            f"received '{experiment.task}'."
        )
