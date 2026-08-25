"""Date helpers shared by the ingestion and retrieval paths.

Journal entries carry an integer ``date_ts`` payload field (epoch seconds at
UTC midnight) so Qdrant can range-filter on them. The writer (ingestion) and
the reader (query filters) must agree on that encoding, so the conversion
lives here rather than in either service.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

# Payload key holding the entry date as epoch seconds. Indexed as an integer
# so ``rest.Range`` filtering is served by the index rather than a full scan.
DATE_PAYLOAD_FIELD = "date_ts"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def coerce_date(value: Any) -> date | None:
    """Best-effort conversion of an LLM- or filename-supplied value to a date.

    Returns ``None`` for anything unparseable so callers can drop the bound
    instead of failing the whole request.
    """
    # datetime is a subclass of date, so it must be checked first.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def to_epoch_seconds(value: date) -> int:
    """Encode a date as epoch seconds at UTC midnight.

    Negative results are expected and supported — the sample corpus is dated
    1771 — and Qdrant range filters handle negative integers fine.
    """
    midnight = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return int((midnight - _EPOCH).total_seconds())
