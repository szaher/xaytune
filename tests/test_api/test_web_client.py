"""The built web client, served beside the API from one origin (UX-02, ADR-023 §S2, §S11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import Recorded, client

INDEX = "<!doctype html><title>Xaytune Studio</title><div id=root></div>"


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "static"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX)
    (root / "assets" / "index-abc.js").write_text("console.log('studio')")
    (root / "assets" / "index-abc.css").write_text("body{}")
    (tmp_path / "secret.txt").write_text("outside the bundle")
    return root


@pytest.fixture
def web(recorded: Recorded, bundle: Path) -> TestClient:
    return client(recorded.state, web_client=bundle)


def test_the_client_is_served_at_the_root(web: TestClient) -> None:
    response = web.get("/")
    assert response.status_code == 200
    assert response.text == INDEX
    assert response.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize(
    ("path", "media_type"),
    [("/assets/index-abc.js", "javascript"), ("/assets/index-abc.css", "text/css")],
)
def test_assets_are_served_with_their_type(web: TestClient, path: str, media_type: str) -> None:
    response = web.get(path)
    assert response.status_code == 200
    assert media_type in response.headers["content-type"]


@pytest.mark.parametrize(
    "path", ["/experiments", "/experiments/exp_1/candidates/node_2", "/compare", "/runs/run_1"]
)
def test_the_clients_own_routes_get_the_app_shell(web: TestClient, path: str) -> None:
    """A reload on any view the client routes must load the client (the URL is the state)."""
    response = web.get(path, params={"tab": "candidates"})
    assert response.status_code == 200
    assert response.text == INDEX


@pytest.mark.parametrize("path", ["/api", "/api/v1/no-such", "/api/v2/experiments", "/api/v1/"])
def test_an_unknown_api_path_is_never_answered_with_the_client(web: TestClient, path: str) -> None:
    response = web.get(path)
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


def test_a_missing_asset_is_a_404_not_the_app_shell(web: TestClient) -> None:
    response = web.get("/assets/index-missing.js")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize(
    "path", ["/../secret.txt", "/assets/../../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"]
)
def test_no_file_outside_the_bundle_is_served(web: TestClient, path: str) -> None:
    response = web.get(path)
    assert "outside the bundle" not in response.text


def test_a_symlink_out_of_the_bundle_is_not_followed(web: TestClient, bundle: Path) -> None:
    (bundle / "assets" / "escape.txt").symlink_to(bundle.parent / "secret.txt")
    response = web.get("/assets/escape.txt")
    assert "outside the bundle" not in response.text
    assert response.status_code == 404


def test_the_front_door_guards_the_client(web: TestClient) -> None:
    response = web.get("/")
    assert response.headers["content-security-policy"].startswith("default-src 'self'")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert web.post("/").status_code == 405
    assert web.get("/", headers={"host": "attacker.example"}).status_code == 421


def test_the_api_is_unchanged_with_the_client_installed(web: TestClient) -> None:
    meta = web.get("/api/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["schema"] == "xaytune.studio-api/v1"


def test_an_unbuilt_client_says_how_to_build_it(recorded: Recorded, tmp_path: Path) -> None:
    for unbuilt in (None, tmp_path / "empty"):
        response = client(recorded.state, web_client=unbuilt).get("/experiments")
        assert response.status_code == 404
        assert response.json()["type"] == "urn:xaytune:problem:web-client-not-built"
        assert "npm run build" in response.json()["detail"]


def test_the_client_route_is_not_part_of_the_contract(web: TestClient) -> None:
    paths = web.get("/api/v1/openapi.json").json()["paths"]
    assert all(path.startswith("/api/v1/") for path in paths)
