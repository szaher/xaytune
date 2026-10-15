"""Methods, hosts, headers, ETags and errors (ADR-023 §S1, §S7, §S8, §S10)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import Recorded, client

META = "/api/v1/meta"
SECURITY_HEADERS = {
    "content-security-policy": "default-src 'self'; frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "x-frame-options": "DENY",
}


def _assert_secured(response: object) -> None:
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value, name  # type: ignore[attr-defined]


# ---- §S1: read-only ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"])
@pytest.mark.parametrize("path", [META, "/api/v1/experiments", "/api/v1/no-such-route"])
def test_every_method_but_a_read_is_refused_on_every_path(
    api: TestClient, method: str, path: str
) -> None:
    response = api.request(method, path)
    assert response.status_code == 405
    assert response.headers["allow"] == "GET, HEAD"
    assert response.headers["content-type"] == "application/problem+json"
    _assert_secured(response)


def test_head_answers_like_get_without_a_body(api: TestClient) -> None:
    get = api.get("/api/v1/experiments")
    head = api.head("/api/v1/experiments")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["etag"] == get.headers["etag"]
    assert head.headers["content-length"] == str(len(get.content))


# ---- §S10: exposure ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:8765", "[::1]", "[::1]:8765"]
)
def test_loopback_hosts_are_answered(recorded: Recorded, host: str) -> None:
    response = client(recorded.state).get(META, headers={"host": host})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "host",
    ["attacker.example", "attacker.example:8765", "127.0.0.1.attacker.example", "10.0.0.5", ""],
)
def test_other_hosts_are_refused(recorded: Recorded, host: str) -> None:
    """S-F4: a rebinding page reaches 127.0.0.1 under its own name, and is refused."""
    response = client(recorded.state).get(META, headers={"host": host})
    assert response.status_code == 421
    assert response.headers["content-type"] == "application/problem+json"
    _assert_secured(response)


def test_no_cross_origin_access_is_granted(api: TestClient) -> None:
    response = api.get(META, headers={"origin": "https://attacker.example"})
    assert "access-control-allow-origin" not in response.headers


def test_the_interactive_docs_are_not_served(api: TestClient) -> None:
    """They load scripts from a CDN, which the CSP forbids; the OpenAPI document is served."""
    assert api.get("/docs").status_code == 404
    assert api.get("/redoc").status_code == 404
    assert api.get("/api/v1/openapi.json").status_code == 200


# ---- §S7: freshness ----------------------------------------------------------------------------


def test_an_unchanged_resource_is_not_sent_again(api: TestClient) -> None:
    first = api.get("/api/v1/experiments")
    assert first.headers["x-xaytune-event-sequence"].isdigit()
    again = api.get("/api/v1/experiments", headers={"if-none-match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["etag"] == first.headers["etag"]
    _assert_secured(again)
    other = api.get("/api/v1/experiments", headers={"if-none-match": '"something-else"'})
    assert other.status_code == 200


def test_the_event_sequence_header_is_the_journal_head(api: TestClient, recorded: Recorded) -> None:
    response = api.get(
        f"/api/v1/experiments/{recorded.experiment_id}/events", params={"limit": 500}
    )
    last = response.json()["items"][-1]["sequence"]
    assert int(response.headers["x-xaytune-event-sequence"]) >= last
    assert api.get(META).json()["latest_event_sequence"] == int(
        response.headers["x-xaytune-event-sequence"]
    )


# ---- §S8: errors never echo --------------------------------------------------------------------

PROBE = "<script>secret-9f2c</script>"


@pytest.mark.parametrize(
    "path",
    [
        f"/api/v1/experiments/{PROBE}",
        f"/api/v1/experiments/{PROBE}/graph",
        f"/api/v1/nodes/{PROBE}",
        f"/api/v1/runs/{PROBE}",
        f"/api/v1/attempts/{PROBE}",
        f"/api/v1/evaluation-runs/{PROBE}",
    ],
)
def test_not_found_never_echoes_the_id(api: TestClient, path: str) -> None:
    response = api.get(path)
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert "secret-9f2c" not in response.text
    assert set(response.json()) == {"type", "title", "status", "detail"}


@pytest.mark.parametrize(
    "params",
    [
        {"limit": PROBE},
        {"limit": 0},
        {"limit": 501},
        {"before": PROBE},
        {"before": "eyJub3QiOiJhIGN1cnNvciJ9"},
    ],
)
def test_an_invalid_parameter_is_named_and_never_echoed(api: TestClient, params: dict) -> None:
    response = api.get("/api/v1/experiments", params=params)
    assert response.status_code == 422
    assert "secret-9f2c" not in response.text
    assert next(iter(params)) in response.json()["detail"]


def test_an_oversized_events_limit_is_refused_not_clamped(
    api: TestClient, recorded: Recorded
) -> None:
    path = f"/api/v1/experiments/{recorded.experiment_id}/events"
    assert api.get(path, params={"limit": 501}).status_code == 422
    assert api.get(path, params={"after": -1}).status_code == 422


def test_compare_refuses_an_unknown_or_missing_candidate(
    api: TestClient, recorded: Recorded
) -> None:
    graph = api.get(f"/api/v1/experiments/{recorded.experiment_id}/graph").json()
    node = graph["nodes"][0]["id"]
    assert api.get("/api/v1/compare", params={"left": node}).status_code == 422
    unknown = api.get("/api/v1/compare", params={"left": node, "right": PROBE})
    assert unknown.status_code == 404
    assert "secret-9f2c" not in unknown.text


def test_an_unexpected_error_returns_only_a_correlation_id(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from xaytune.storage.repository import AggregateStore

    def broken(self: object, **kwargs: object) -> None:
        raise RuntimeError("secret-9f2c at /home/someone/state.db")

    monkeypatch.setattr(AggregateStore, "list_experiments", broken)
    response = client(recorded.state, raise_server_exceptions=False).get("/api/v1/experiments")
    assert response.status_code == 500
    body = response.json()
    assert "secret-9f2c" not in response.text and "/home/" not in response.text
    assert len(body["correlation_id"]) == 32
    assert body["correlation_id"] in caplog.text
    _assert_secured(response)
