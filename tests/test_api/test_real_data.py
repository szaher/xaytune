"""A real recorded experiment, read through the Studio API (ADR-023 acceptance 7).

```text
node A  LoRA 16   run A  attempt 1 CUDA OOM → attempt 2 recovered → 0.79 → branch
   ↓ planner → node B  LoRA 32   run B (seed inherited) → 0.83 → stop_succeeded
experiment succeeded, best node B
```

Everything asserted here is read from the file the controller wrote, after
the controller has gone.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import Recorded, client

PREFIX = "/api/v1"


def _get(api: TestClient, path: str, **params: Any) -> Any:
    response = api.get(PREFIX + path, params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema"] == "xaytune.studio-api/v1"
    return body


def _graph(api: TestClient, recorded: Recorded) -> dict[str, Any]:
    return _get(api, f"/experiments/{recorded.experiment_id}/graph")


def _parent_child(api: TestClient, recorded: Recorded) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = _graph(api, recorded)
    (edge,) = graph["edges"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    return nodes[edge["parent_id"]], nodes[edge["child_id"]]


def test_the_experiment_is_listed_and_described(api: TestClient, recorded: Recorded) -> None:
    page = _get(api, "/experiments")
    (summary,) = [item for item in page["items"] if item["id"] == recorded.experiment_id]
    assert summary["status"] == "succeeded" == recorded.status
    assert summary["node_counts"] == {"completed": 2}
    assert page["next_before"] is None

    experiment = _get(api, f"/experiments/{recorded.experiment_id}")
    assert experiment["status"] == "succeeded"
    assert experiment["quiescent"] is True
    assert experiment["next_stage"] is None
    assert experiment["objective"]["primary"]["name"] == "task_success"
    assert experiment["planner"] is not None
    assert experiment["evaluation"] is not None


def test_the_candidate_graph_shows_lineage_decisions_and_the_best_node(
    api: TestClient, recorded: Recorded
) -> None:
    graph = _graph(api, recorded)
    parent, child = _parent_child(api, recorded)
    assert graph["best_node_id"] == child["id"]
    assert child["is_best"] and not parent["is_best"]
    assert child["parent_ids"] == [parent["id"]]
    assert parent["latest_decision"]["outcome"] == "branch"
    assert child["latest_decision"]["outcome"] == "stop_succeeded"
    assert child["branch_origin"] is not None
    assert child["branch_origin"]["mutation"]


def test_a_candidate_detail_has_its_runs_evaluations_and_decision(
    api: TestClient, recorded: Recorded
) -> None:
    parent, child = _parent_child(api, recorded)
    detail = _get(api, f"/nodes/{child['id']}")
    assert detail["lineage_paths"] == [[parent["id"], child["id"]]]
    assert detail["candidate"]["candidate"]["training"]["adapter"]["rank"] == 32
    (run,) = detail["runs"]
    assert run["status"] == "succeeded"
    assert run["seed_origin"] is not None  # inherited from the parent's run
    (evaluation,) = detail["evaluation_runs"]
    assert evaluation["status"] == "succeeded"
    metrics = {metric["name"]: metric["value"] for metric in evaluation["result"]["metrics"]}
    assert metrics == {"task_success": 0.83}
    (decision,) = detail["decisions"]
    assert decision["outcome"] == "stop_succeeded"
    assert decision["evaluation_result_ids"] == [evaluation["result"]["id"]]

    by_id = _get(api, f"/evaluation-runs/{evaluation['id']}")
    assert by_id["evaluation_run"] == evaluation
    assert [attempt["status"] for attempt in by_id["attempts"]] == ["succeeded"]


def test_a_run_shows_the_oom_its_recovery_and_checkpoints(
    api: TestClient, recorded: Recorded
) -> None:
    parent, _ = _parent_child(api, recorded)
    (run,) = _get(api, f"/nodes/{parent['id']}")["runs"]
    detail = _get(api, f"/runs/{run['id']}")
    attempts = detail["run"]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["failed", "succeeded"]
    assert detail["checkpoints"]
    assert attempts[1]["execution_overrides"]  # resized after the OOM

    failed = _get(api, f"/attempts/{attempts[0]['id']}")
    assert [incident["category"] for incident in failed["incidents"]] == ["CUDA_OOM"]
    assert failed["checkpoints"]

    incidents = _get(api, f"/experiments/{recorded.experiment_id}/incidents")
    assert [incident["id"] for incident in incidents["incidents"]] == [
        incident["id"] for incident in failed["incidents"]
    ]
    (episode,) = incidents["recovery_episodes"]
    assert episode["effective_plan"] is not None


def test_actions_carry_their_policy_decision(api: TestClient, recorded: Recorded) -> None:
    actions = _get(api, f"/experiments/{recorded.experiment_id}/actions")["items"]
    resize = [action for action in actions if action["type"] == "resize-microbatch"]
    assert resize and resize[0]["status"] == "succeeded"
    assert resize[0]["policy_decision"]["verdict"] == "allow"


def test_decisions_and_budget_are_read_from_the_record(api: TestClient, recorded: Recorded) -> None:
    decisions = _get(api, f"/experiments/{recorded.experiment_id}/decisions")["items"]
    assert [decision["outcome"] for decision in decisions] == ["branch", "stop_succeeded"]
    budget = _get(api, f"/experiments/{recorded.experiment_id}/budget")
    assert budget["limits"] is not None
    assert budget["entries"]
    assert all(isinstance(entry["amount"], str) for entry in budget["entries"])


def test_compare_shows_lineage_the_spec_difference_and_metrics(
    api: TestClient, recorded: Recorded
) -> None:
    parent, child = _parent_child(api, recorded)
    comparison = _get(api, "/compare", left=child["id"], right=parent["id"])
    lineage = comparison["lineage"]
    assert lineage["distinct_candidates"] is True
    assert lineage["nearest_common_ancestor_ids"] == [parent["id"]]
    differences = {d["path"]: (d["left"], d["right"]) for d in comparison["candidate_differences"]}
    assert differences["candidate.training.adapter.rank"] == (32, 16)
    (pair,) = comparison["metrics"]
    assert pair["name"] == "task_success"
    assert (pair["left"]["metric"]["value"], pair["right"]["metric"]["value"]) == (0.83, 0.79)
    # Both measured under the experiment's one evaluation: comparable, with a difference.
    assert pair["left"]["evaluation_fingerprint"] == pair["right"]["evaluation_fingerprint"]
    assert pair["comparable"] is True and pair["reasons"] == []
    assert pair["difference"] == pytest.approx(0.04)


def test_events_page_by_sequence_without_gaps(api: TestClient, recorded: Recorded) -> None:
    path = f"/experiments/{recorded.experiment_id}/events"
    everything = _get(api, path, limit=500)["items"]
    paged: list[dict[str, Any]] = []
    after = 0
    while True:
        page = _get(api, path, after=after, limit=7)
        paged.extend(page["items"])
        if not page["items"]:
            assert page["next_after"] == after
            break
        after = page["next_after"]
    assert paged == everything
    sequences = [event["sequence"] for event in everything]
    assert sequences == sorted(sequences)


def test_every_answer_survives_an_api_restart(api: TestClient, recorded: Recorded) -> None:
    """No state lives in the API: a second process over the same file answers identically."""
    paths = _every_path(api, recorded)
    before = {path: api.get(path) for path in paths}
    restarted = client(recorded.state)
    for path in paths:
        again = restarted.get(path)
        assert again.status_code == 200
        assert again.content == before[path].content, path
        assert again.headers["etag"] == before[path].headers["etag"]


def _every_path(api: TestClient, recorded: Recorded) -> list[str]:
    """One concrete URL for every route the API serves."""
    parent, child = _parent_child(api, recorded)
    node = _get(api, f"/nodes/{parent['id']}")
    run = node["runs"][0]
    experiment = f"{PREFIX}/experiments/{recorded.experiment_id}"
    return [
        f"{PREFIX}/meta",
        f"{PREFIX}/experiments",
        experiment,
        *(
            f"{experiment}/{part}"
            for part in (
                "graph",
                "events",
                "actions",
                "decisions",
                "budget",
                "incidents",
                "agent-invocations",
            )
        ),
        f"{PREFIX}/nodes/{parent['id']}",
        f"{PREFIX}/nodes/{child['id']}",
        f"{PREFIX}/compare?left={child['id']}&right={parent['id']}",
        f"{PREFIX}/runs/{run['id']}",
        f"{PREFIX}/attempts/{run['attempts'][0]['id']}",
        f"{PREFIX}/attempts/{run['attempts'][0]['id']}/metrics",
        f"{PREFIX}/evaluation-runs/{node['evaluation_runs'][0]['id']}",
    ]


def test_every_route_is_covered_by_the_restart_check(api: TestClient, recorded: Recorded) -> None:
    """Counted from the OpenAPI document: ``app.routes`` differs across FastAPI releases."""
    from xaytune.api.v1.openapi import generate

    routes = set(generate()["paths"])
    assert len(_every_path(api, recorded)) - 1 == len(routes)  # two node URLs, one route
