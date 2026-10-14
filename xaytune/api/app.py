"""The Studio API application: read-only routes behind a strict front door (ADR-023).

One ASGI guard in front of every route:

- **§S1** anything but ``GET`` or ``HEAD`` is ``405``, whether or not a route
  exists for it, so a write path cannot appear by adding a route;
- **§S10** a request whose ``Host`` is not a loopback name is refused, which
  is what stops a DNS-rebinding page reading the API from a browser;
- **§S10** every response carries a CSP, ``nosniff``, ``no-referrer`` and
  ``no-store``;
- **§S7** a ``200`` gets an ``ETag``, and a matching ``If-None-Match`` gets
  ``304`` with no body.

Errors are ``application/problem+json`` with fixed text (§S8): never a value
read from the store, never the id or parameter value sent, never an
exception's own message.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from xaytune.api.store import StoreUnavailableError, StudioStore
from xaytune.api.v1.models import Problem
from xaytune.api.v1.routes import InvalidParameterError, NotFoundError, router

__all__ = ["LOOPBACK_HOSTS", "application", "create_app"]

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_READ_METHODS = frozenset({"GET", "HEAD"})
_PROBLEM = "application/problem+json"
_SECURITY_HEADERS = (
    (b"content-security-policy", b"default-src 'self'; frame-ancestors 'none'"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store"),
    (b"x-frame-options", b"DENY"),
)


def create_app(store: StudioStore | Path | str) -> FastAPI:
    """The Studio API over *store*, a state database path or an opened :class:`StudioStore`.

    Raises:
        StoreUnavailableError: If the path is missing or is not a Xaytune
            state database.
    """
    studio_store = store if isinstance(store, StudioStore) else StudioStore(store)
    app = application()
    app.state.store = studio_store
    return app


def application() -> FastAPI:
    """The application without a store: what the OpenAPI document is generated from."""
    app = FastAPI(
        title="Xaytune Studio API",
        # The contract's version, not the package's: a release that changes
        # nothing on the wire must not change the committed document.
        version="v1",
        summary="A read-only view of the Xaytune control-plane store (ADR-023).",
        openapi_url="/api/v1/openapi.json",
        # The interactive docs load scripts from a CDN, which the CSP forbids.
        docs_url=None,
        redoc_url=None,
    )
    app.include_router(router)
    _install_error_handlers(app)
    app.add_middleware(_FrontDoor)
    return app


# ---- errors ------------------------------------------------------------------------------------


def _problem(
    status: int, kind: str, title: str, detail: str, correlation_id: str | None = None
) -> JSONResponse:
    body = Problem(
        type=f"urn:xaytune:problem:{kind}",
        title=title,
        status=status,
        detail=detail,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        body.model_dump(mode="json", exclude_none=True), status_code=status, media_type=_PROBLEM
    )


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, error: NotFoundError) -> JSONResponse:
        return _problem(404, "not-found", "Not found", f"no such {error.kind}")

    @app.exception_handler(InvalidParameterError)
    async def invalid(request: Request, error: InvalidParameterError) -> JSONResponse:
        return _problem(
            422, "invalid-parameter", "Invalid parameter", f"{error.parameter}: {error.reason}"
        )

    @app.exception_handler(RequestValidationError)
    async def unvalidated(request: Request, error: RequestValidationError) -> JSONResponse:
        # Names only: FastAPI's own message would echo the value sent.
        names = sorted(
            {".".join(str(part) for part in item.get("loc", ())[1:]) for item in error.errors()}
        )
        return _problem(
            422,
            "invalid-parameter",
            "Invalid parameter",
            "invalid value for: " + ", ".join(name or "request" for name in names),
        )

    @app.exception_handler(StoreUnavailableError)
    async def unavailable(request: Request, error: StoreUnavailableError) -> JSONResponse:
        # StoreUnavailableError messages are fixed text by construction.
        return _problem(503, "store-unavailable", "State database unavailable", str(error))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        if error.status_code == 404:
            return _problem(404, "not-found", "Not found", "no such resource")
        if error.status_code == 405:
            return _problem(405, "method-not-allowed", "Method not allowed", "read-only API")
        return _problem(error.status_code, "http-error", "Request refused", "request refused")


# ---- the front door ----------------------------------------------------------------------------


def _host_name(scope: Scope) -> str | None:
    for name, value in scope.get("headers", ()):
        if name == b"host":
            host = str(value.decode("latin-1")).strip().lower()
            if host.startswith("["):  # [::1]:8765
                end = host.find("]")
                return host[1:end] if end > 0 else None
            return host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return None


class _FrontDoor:
    """Method filter, Host allow-list, security headers and ETags, for every request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        guarded = _secured(send)
        if _host_name(scope) not in LOOPBACK_HOSTS:
            response = _problem(
                421, "host-refused", "Host refused", "the Studio API answers loopback hosts only"
            )
            await response(scope, receive, guarded)
            return
        if scope["method"] not in _READ_METHODS:
            response = _problem(405, "method-not-allowed", "Method not allowed", "read-only API")
            response.headers["allow"] = "GET, HEAD"
            await response(scope, receive, guarded)
            return
        started = False

        async def tracked(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await guarded(message)

        inner: Send = tracked
        method = scope["method"]
        if method == "HEAD":
            # Answered as GET with the body dropped, so HEAD needs no routes
            # of its own and its headers (ETag included) are GET's.
            inner = _headless(tracked)
            method = "GET"
        try:
            await self.app({**scope, "method": method}, receive, _etagged(scope, inner))
        except Exception:
            # Here rather than in an exception handler: Starlette answers those
            # from outside every middleware, so the headers above would be lost.
            correlation_id = uuid.uuid4().hex
            logger.exception("studio API error %s", correlation_id)
            if started:
                raise
            response = _problem(
                500, "internal-error", "Internal error", "see the server log", correlation_id
            )
            await response(scope, receive, guarded)


def _secured(send: Send) -> Send:
    async def secured(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = [
                (name, value)
                for name, value in message.get("headers", [])
                if name.lower() not in {header for header, _ in _SECURITY_HEADERS}
            ]
            message = {**message, "headers": [*headers, *_SECURITY_HEADERS]}
        await send(message)

    return secured


def _headless(send: Send) -> Send:
    async def headless(message: Message) -> None:
        if message["type"] == "http.response.body":
            message = {**message, "body": b""}
        await send(message)

    return headless


def _etagged(scope: Scope, send: Send) -> Send:
    """Buffer a ``200``, tag it, and answer ``304`` when the client already has it."""
    wanted = _header(scope, b"if-none-match")
    start: dict[str, Any] = {}
    body: list[bytes] = []

    async def tagged(message: Message) -> None:
        if message["type"] == "http.response.start":
            if message["status"] != 200:
                await send(message)
                return
            start.update(message)
            return
        if not start:
            await send(message)
            return
        body.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        content = b"".join(body)
        etag = '"' + hashlib.sha256(content).hexdigest()[:32] + '"'
        headers = [
            (name, value)
            for name, value in start.get("headers", [])
            if name.lower() not in (b"etag", b"content-length")
        ]
        headers.append((b"etag", etag.encode()))
        if wanted is not None and etag in {tag.strip() for tag in wanted.split(",")}:
            await send({**start, "status": 304, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return
        headers.append((b"content-length", str(len(content)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": content})

    return tagged


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key == name:
            return str(value.decode("latin-1"))
    return None
