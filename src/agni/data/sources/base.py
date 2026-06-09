from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from typing import Any


def ee_date_subtract(reference_date: str, days: int) -> str:
    value = date.fromisoformat(reference_date) - timedelta(days=days)
    return value.isoformat()


def materialize_ee(features: dict[str, Any]) -> dict[str, Any]:
    """Resolve a mapping of server-side Earth Engine values to client-side scalars.

    Adapters assemble their feature dicts lazily: each value is an unevaluated
    Earth Engine ``ComputedObject`` (e.g. the result of ``reduceRegion(...).get(...)``).
    Those objects cannot be written to parquet and must be pulled to the client with
    ``getInfo``. Wrapping the whole mapping in a single ``ee.Dictionary`` and calling
    ``getInfo`` once materializes every value in **one** network round trip, instead of
    one round trip per band/window, which is both correct and dramatically cheaper at
    scale. Earth Engine ``null`` values come back as Python ``None``.
    """
    if not features:
        return {}
    import ee

    return dict(ee.Dictionary(features).getInfo())


def ms_timestamp_to_iso_date(timestamp_ms: float | int | None) -> str | None:
    """Convert an epoch-milliseconds timestamp (UTC) to an ISO ``YYYY-MM-DD`` date.

    Returns ``None`` for ``None`` so masked/absent burn timestamps stay null.
    """
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).date().isoformat()


def month_start_iso(value: str | date) -> str:
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    return parsed.replace(day=1).isoformat()


def month_after_iso(value: str | date) -> str:
    parsed = date.fromisoformat(value) if isinstance(value, str) else value
    if parsed.month == 12:
        return date(parsed.year + 1, 1, 1).isoformat()
    return date(parsed.year, parsed.month + 1, 1).isoformat()


class EESourceAdapter(ABC):
    """Base class for Earth Engine data source adapters."""

    @abstractmethod
    def collection_id(self) -> str:
        """Return the GEE collection identifier."""

    @abstractmethod
    def extract_patch(
        self,
        geometry: Any,
        reference_date: str,
        lookback_days: int,
        temporal_windows: list[int],
    ) -> dict[str, float | int | None]:
        """Extract features for one `(patch, reference_date)` record."""
