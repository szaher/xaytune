"""The Studio API reads a harness node as it reads any node (ADR-018 D10).

The API serves a node's candidate as a recorded document, so a version 2
snapshot (an envelope holding a harness) needs no new route or schema.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_api.conftest import client
from tests.test_storage.test_candidate_snapshot import ACTOR, _seeded, harness_node


def test_a_harness_node_is_served_with_its_envelope(tmp_path: Path) -> None:
    repo, experiment = _seeded(tmp_path)
    node = repo.create_node(harness_node(experiment), actor=ACTOR)
    repo._connection.close()
    api = client(tmp_path / "state.db")
    detail = api.get(f"/api/v1/nodes/{node.id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["node"]["candidate_fingerprint"] == node.candidate_fingerprint
    assert body["candidate"]["spec_version"] == "2"
    assert body["candidate"]["envelope"]["payload"]["kind"] == "agent-harness"
    assert "candidate" not in body["candidate"]
    graph = api.get(f"/api/v1/experiments/{experiment.id}/graph").json()
    assert [item["id"] for item in graph["nodes"]] == [str(node.id)]
