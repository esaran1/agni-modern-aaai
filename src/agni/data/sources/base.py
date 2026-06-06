from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any


def ee_date_subtract(reference_date: str, days: int) -> str:
    value = date.fromisoformat(reference_date) - timedelta(days=days)
    return value.isoformat()


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
