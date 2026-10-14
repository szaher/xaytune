"""``xaytune studio-web``: serve the Studio API on loopback (ADR-023 §S2, §S10)."""

from __future__ import annotations

import ipaddress
from pathlib import Path

__all__ = ["DEFAULT_PORT", "ApiExposureError", "is_loopback", "serve"]

DEFAULT_PORT = 8765


class ApiExposureError(ValueError):
    """The API was asked to listen where it would be reachable without authentication."""


def is_loopback(host: str) -> bool:
    """Whether *host* is a loopback address or ``localhost``."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def serve(state: Path | str, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Serve the read-only API over *state* until interrupted.

    The database is checked before anything listens, so a wrong path fails
    here rather than on the first request.

    Raises:
        ApiExposureError: If *host* is not a loopback address. The API has no
            authentication until UX-05, and it serves the whole record.
        StoreUnavailableError: If *state* is missing or is not a Xaytune
            state database.
    """
    if not is_loopback(host):
        raise ApiExposureError(
            "the Studio API serves the whole experiment record without authentication, "
            "so it listens on loopback only (127.0.0.1, ::1 or localhost)"
        )
    import uvicorn

    from xaytune.api.app import create_app

    app = create_app(state)
    uvicorn.run(
        app,
        host=host,
        port=port,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        log_level="info",
    )
