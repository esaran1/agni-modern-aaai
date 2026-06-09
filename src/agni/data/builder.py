from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from shapely import wkt

from agni.config import DataConfig
from agni.data.manifest import build_dataset_manifest

LOGGER = logging.getLogger(__name__)

# Substrings that indicate a transient (retryable) Earth Engine / network error
# rather than a deterministic bug. Deterministic errors are never retried so real
# adapter problems still fail fast.
_TRANSIENT_KEYWORDS = (
    "timeout",
    "timed out",
    "temporarily",
    "rate limit",
    "too many requests",
    "429",
    "quota",
    "deadline",
    "capacity",
    "try again",
    "connection reset",
    "connection aborted",
    "502",
    "503",
    "504",
    "backend error",
    "service unavailable",
    "internal error",
)


@dataclass
class DatasetBuildResult:
    dataset: pd.DataFrame
    dataset_path: Path
    manifest_path: Path


def iter_reference_dates(config: DataConfig) -> list[pd.Timestamp]:
    dates = []
    current = pd.Timestamp(config.temporal.reference_start)
    event_window_days = config.temporal.event_observation_window_days()
    end = pd.Timestamp(config.temporal.reference_end) + timedelta(
        days=config.temporal.horizon_days + event_window_days
    )
    stride = timedelta(days=config.temporal.reference_stride_days)
    while current <= end:
        dates.append(current)
        current += stride
    return dates


def _is_transient_error(exc: Exception) -> bool:
    # Classify by message content, not exception class. Earth Engine raises a single
    # ``ee.EEException`` type for *both* transient issues (rate limits, 5xx, timeouts)
    # and deterministic bugs (e.g. selecting a band from an empty image). Retrying the
    # latter just wastes minutes of backoff before failing, so only message keywords
    # (or genuine timeout/connection exceptions) count as transient. The outer build
    # loop resumes from shards, so an occasional misclassified transient is recoverable.
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_KEYWORDS)


def _extract_with_retry(
    adapter: Any,
    kwargs: dict[str, Any],
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            return adapter.extract_patch(**kwargs)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _is_transient_error(exc):
                raise
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Transient error from %s (attempt %d/%d); retrying in %.1fs: %s",
                adapter.__class__.__name__,
                attempt,
                max_retries,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)


def _build_patch_rows(
    patch: dict[str, Any],
    reference_dates: list[pd.Timestamp],
    adapters: list[Any],
    config: DataConfig,
    strict: bool,
    max_retries: int,
    backoff_seconds: float,
) -> list[dict[str, Any]]:
    geometry = wkt.loads(patch["geometry_wkt"])
    rows: list[dict[str, Any]] = []
    for reference_date in reference_dates:
        row: dict[str, Any] = {
            "patch_id": patch["patch_id"],
            "patch_row": patch["patch_row"],
            "patch_col": patch["patch_col"],
            "centroid_lon": patch["centroid_lon"],
            "centroid_lat": patch["centroid_lat"],
            "reference_date": reference_date.date(),
        }
        for adapter in adapters:
            kwargs = {
                "geometry": geometry,
                "reference_date": reference_date.date().isoformat(),
                "lookback_days": config.temporal.lookback_days,
                "temporal_windows": config.temporal.temporal_windows,
            }
            try:
                row.update(_extract_with_retry(adapter, kwargs, max_retries, backoff_seconds))
            except Exception as exc:
                message = (
                    f"Adapter {adapter.__class__.__name__} failed for "
                    f"patch={patch['patch_id']} date={reference_date.date()}: {exc}"
                )
                if strict:
                    raise RuntimeError(message) from exc
                LOGGER.warning(message)
        rows.append(row)
    return rows


def _shard_dir_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"_shards_{stem}"


def _shard_file(shard_dir: Path, patch_id: Any) -> Path:
    return shard_dir / f"patch_{patch_id}.parquet"


def _process_patch(
    patch: dict[str, Any],
    reference_dates: list[pd.Timestamp],
    adapters: list[Any],
    config: DataConfig,
    strict: bool,
    max_retries: int,
    backoff_seconds: float,
    shard_dir: Path | None,
) -> tuple[str, pd.DataFrame]:
    rows = _build_patch_rows(
        patch,
        reference_dates,
        adapters,
        config,
        strict,
        max_retries,
        backoff_seconds,
    )
    frame = pd.DataFrame(rows)
    if shard_dir is not None:
        frame.to_parquet(_shard_file(shard_dir, patch["patch_id"]), index=False)
    return str(patch["patch_id"]), frame


def _run_build(
    pending: list[dict[str, Any]],
    worker: Any,
    max_workers: int,
) -> dict[str, pd.DataFrame]:
    built: dict[str, pd.DataFrame] = {}
    if pending and max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(worker, patch): patch for patch in pending}
            for completed, future in enumerate(as_completed(future_map), start=1):
                patch_id, frame = future.result()
                built[patch_id] = frame
                if completed % 25 == 0 or completed == len(pending):
                    LOGGER.info("Built %d/%d pending patches", completed, len(pending))
    else:
        for completed, patch in enumerate(pending, start=1):
            patch_id, frame = worker(patch)
            built[patch_id] = frame
            if completed % 25 == 0 or completed == len(pending):
                LOGGER.info("Built %d/%d pending patches", completed, len(pending))
    return built


def build_patch_shards(
    config: DataConfig,
    patch_df: pd.DataFrame,
    adapters: Iterable[Any],
    output_name: str = "dataset.parquet",
    strict: bool = True,
    max_workers: int = 1,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    resume: bool = True,
) -> int:
    """Build and checkpoint shards for a (possibly partial) set of patches.

    This is the unit of work for cluster job arrays: each task builds a disjoint
    slice of patches and writes one parquet shard per patch. Run ``build_dataset``
    (or the ``--merge`` CLI step) afterwards to concatenate all shards into the
    final dataset. Returns the number of patches built this call.
    """
    adapters = list(adapters)
    output_dir = Path(config.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = _shard_dir_path(output_dir, Path(output_name).stem)
    shard_dir.mkdir(parents=True, exist_ok=True)
    reference_dates = iter_reference_dates(config)
    patches = patch_df.to_dict(orient="records")
    pending = [
        patch
        for patch in patches
        if not (resume and _shard_file(shard_dir, patch["patch_id"]).exists())
    ]
    LOGGER.info(
        "Building %d patch shards (%d pending) into %s",
        len(patches),
        len(pending),
        shard_dir,
    )

    def worker(patch: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        return _process_patch(
            patch,
            reference_dates,
            adapters,
            config,
            strict,
            max_retries,
            backoff_seconds,
            shard_dir,
        )

    _run_build(pending, worker, max_workers)
    return len(pending)


def build_dataset(
    config: DataConfig,
    patch_df: pd.DataFrame,
    adapters: Iterable[Any],
    output_name: str = "dataset.parquet",
    strict: bool = True,
    max_workers: int = 1,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    checkpoint: bool = True,
    resume: bool = True,
) -> DatasetBuildResult:
    """Build the patch x reference-date feature table.

    The build is parallelized across patches (Earth Engine extraction is
    I/O-bound) and checkpointed per patch so a crash or interruption can resume
    without re-fetching completed patches. Transient EE/network errors are
    retried with exponential backoff; deterministic errors still fail fast in
    strict mode so genuine adapter bugs are not masked.
    """
    adapters = list(adapters)
    event_window_days = config.temporal.event_observation_window_days()
    output_dir = Path(config.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / output_name
    manifest_path = output_dir / f"{dataset_path.stem}.manifest.json"
    shard_dir = _shard_dir_path(output_dir, dataset_path.stem) if checkpoint else None
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)

    reference_dates = iter_reference_dates(config)
    adapter_names = [adapter.__class__.__name__ for adapter in adapters]
    patches = patch_df.to_dict(orient="records")
    n_patches = len(patches)

    def has_shard(patch: dict[str, Any]) -> bool:
        return shard_dir is not None and _shard_file(shard_dir, patch["patch_id"]).exists()

    pending = [patch for patch in patches if not (resume and has_shard(patch))]
    LOGGER.info(
        "Building dataset: %d patches (%d pending), %d adapters, max_workers=%d",
        n_patches,
        len(pending),
        len(adapters),
        max_workers,
    )

    def worker(patch: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        return _process_patch(
            patch,
            reference_dates,
            adapters,
            config,
            strict,
            max_retries,
            backoff_seconds,
            shard_dir,
        )

    built = _run_build(pending, worker, max_workers)

    frames: list[pd.DataFrame] = []
    for patch in patches:
        patch_id = str(patch["patch_id"])
        if patch_id in built:
            frames.append(built[patch_id])
        elif has_shard(patch):
            frames.append(pd.read_parquet(_shard_file(shard_dir, patch["patch_id"])))
        else:
            _, frame = worker(patch)
            frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "reference_date" in dataset.columns:
        dataset["reference_date"] = pd.to_datetime(dataset["reference_date"]).dt.date
    dataset.to_parquet(dataset_path, index=False)

    manifest = build_dataset_manifest(
        dataset_path=dataset_path,
        config_dict=config.model_dump(mode="json"),
        row_count=len(dataset),
        extra={
            "output_name": output_name,
            "adapter_names": adapter_names,
            "reference_date_count": len(reference_dates),
            "patch_count": n_patches,
            "future_padding_days": (config.temporal.horizon_days + event_window_days),
            "strict_mode": strict,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "checkpoint_dir": str(shard_dir) if shard_dir is not None else None,
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DatasetBuildResult(
        dataset=dataset,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
    )
