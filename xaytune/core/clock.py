"""Time helpers for the control-plane core."""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Naive datetimes are never used: control-plane state is compared across
    controller hosts and serialized into durable events.
    """
    return datetime.now(timezone.utc)
