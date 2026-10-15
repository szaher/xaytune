"""The built Studio web client, served beside the API from the same origin (ADR-023 §S2, §S11).

The client is a static bundle (``web/``, built by Vite into ``static/`` here).
Serving it from the API process keeps one origin, so the API needs no CORS,
and the front door's headers (CSP, ``nosniff``, ``no-store``) apply to every
file. Paths the client routes itself (``/experiments/...``) are answered with
``index.html``; anything under ``/api/`` is never answered with it.

A source checkout that has not built the client still serves the API: ``/``
then explains how to build it, as a problem document.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

__all__ = ["BUNDLE", "install_web_client"]

BUNDLE = Path(__file__).with_name("static")
"""Where ``npm run build`` in ``web/`` writes the client, and where a wheel carries it."""

_INDEX = "index.html"


def install_web_client(app: FastAPI, bundle: Path | None) -> None:
    """Serve *bundle* at ``/``: its files, and ``index.html`` for the client's own routes."""
    root = bundle.resolve() if bundle is not None else None
    built = root is not None and (root / _INDEX).is_file()

    @app.get("/{path:path}", include_in_schema=False)
    def client(path: str) -> Response:
        from xaytune.api.app import problem

        if path == "api" or path.startswith("api/"):
            return problem(404, "not-found", "Not found", "no such resource")
        if not built or root is None:
            return problem(
                404,
                "web-client-not-built",
                "Web client not built",
                "this installation serves the API only: build the client with "
                "`npm ci && npm run build` in web/, or install a release wheel",
            )
        target = _file(root, path)
        if target is None:
            if Path(path).suffix:  # a missing asset is a 404, not the app shell
                return problem(404, "not-found", "Not found", "no such resource")
            target = root / _INDEX
        media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type)


def _file(root: Path, path: str) -> Path | None:
    """*path* inside *root*, if it names a regular file there; never outside it."""
    if not path:
        return None
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate
