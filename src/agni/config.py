from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BBox(BaseModel):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "BBox":
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
    def validate_dates(self) -> "TemporalConfig":
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
    def validate_order(self) -> "SplitConfig":
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
    def validate_buffer(self) -> "DataConfig":
        if self.split.buffer_days < self.temporal.horizon_days:
            raise ValueError("split.buffer_days must be >= temporal.horizon_days")
        return self


class ModelConfig(BaseModel):
    name: str
    task: str
    params: dict[str, Any]

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        allowed = {"classification", "regression"}
        if value not in allowed:
            raise ValueError(f"task must be one of {sorted(allowed)}")
        return value


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(Path(path)))
