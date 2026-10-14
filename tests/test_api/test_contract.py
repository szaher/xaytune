"""The v1 wire contract holds still (ADR-023 §S4, §S5)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_api.conftest import Recorded
from tests.test_api.test_real_data import _every_path
from xaytune.api.v1 import mapping
from xaytune.api.v1 import models as m
from xaytune.api.v1.openapi import DOCUMENT, generate
from xaytune.core.checkpoint import RecordedCheckpoint
from xaytune.core.domain.action import Action
from xaytune.core.domain.agent_invocation import AgentInvocation
from xaytune.core.domain.budget import BudgetLedgerEntry
from xaytune.core.domain.decision import Decision
from xaytune.core.domain.evaluation import (
    EvaluationAttempt,
    EvaluationResult,
    EvaluationRun,
    MetricResult,
)
from xaytune.core.domain.event import DomainEvent
from xaytune.core.domain.experiment import Experiment, ExperimentNode
from xaytune.core.domain.incident import Incident
from xaytune.core.domain.policy import PolicyDecision
from xaytune.core.domain.recovery import RecoveryEpisode
from xaytune.core.domain.run import Run, RunAttempt
from xaytune.core.refs import Actor

API_SOURCE = Path(__file__).parents[2] / "xaytune" / "api"


PRESENTATIONAL = frozenset({"title", "description", "summary", "example", "examples"})
"""Keys no client is generated from, and which move with the FastAPI/Pydantic version."""


def _shape(document: dict[str, Any]) -> dict[str, Any]:
    """The contract a client is generated from: everything but presentational text.

    Paths, methods, parameters with their schemas, request bodies, responses
    with their schemas, and every component schema's types, formats,
    nullability (``anyOf`` with ``null``), references, enums, constants,
    defaults and required fields. A field changing from a number to a string
    changes this; a reworded description does not.
    """
    return _strip({"paths": document["paths"], "components": document.get("components", {})})


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip(item)
            for key, item in value.items()
            # A property *named* like a presentational key is still a property.
            if key not in PRESENTATIONAL or not isinstance(item, (str, list))
        }
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def test_the_committed_openapi_document_matches_the_application() -> None:
    committed = json.loads(DOCUMENT.read_text())
    assert _shape(generate()) == _shape(committed), (
        "the wire changed: run `python -m xaytune.api.v1.openapi` and review the diff"
    )


def _mutated(document: dict[str, Any], change: Any) -> dict[str, Any]:
    copy = json.loads(json.dumps(document))
    change(copy)
    return copy


def _schemas(document: dict[str, Any]) -> dict[str, Any]:
    return document["components"]["schemas"]


@pytest.mark.parametrize(
    "change",
    [
        # A response field's type: integer|null -> string|null.
        lambda d: _schemas(d)["MetaOut"]["properties"]["latest_event_sequence"].update(
            {"anyOf": [{"type": "string"}, {"type": "null"}]}
        ),
        # A field becoming nullable.
        lambda d: _schemas(d)["ExperimentSummary"]["properties"].update(
            {"name": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
        ),
        # A format: date-time -> date.
        lambda d: _schemas(d)["ExperimentSummary"]["properties"]["created_at"].update(
            {"format": "date"}
        ),
        # A reference to another schema.
        lambda d: _schemas(d)["CompareOut"]["properties"]["left"].update(
            {"$ref": "#/components/schemas/RunOut"}
        ),
        # An operation's response schema.
        lambda d: d["paths"]["/api/v1/meta"]["get"]["responses"]["200"]["content"][
            "application/json"
        ].update({"schema": {"$ref": "#/components/schemas/RunOut"}}),
        # A parameter's type.
        lambda d: next(
            p
            for p in d["paths"]["/api/v1/experiments"]["get"]["parameters"]
            if p["name"] == "limit"
        )["schema"].update({"type": "string"}),
    ],
    ids=["type", "nullability", "format", "reference", "response-schema", "parameter-type"],
)
def test_a_wire_type_change_fails_the_contract_check(change: Any) -> None:
    document = generate()
    assert _shape(_mutated(document, change)) != _shape(document)


def test_presentational_text_does_not_fail_the_contract_check() -> None:
    document = generate()

    def reword(d: dict[str, Any]) -> None:
        schema = _schemas(d)["MetaOut"]
        schema["title"] = "Something else"
        schema["properties"]["store"]["description"] = "reworded"
        d["paths"]["/api/v1/meta"]["get"]["summary"] = "reworded"

    assert _shape(_mutated(document, reword)) == _shape(document)


def test_the_served_openapi_document_is_the_generated_one(api: TestClient) -> None:
    assert _shape(api.get("/api/v1/openapi.json").json()) == _shape(generate())


def test_every_route_is_a_read() -> None:
    methods = {method for path in generate()["paths"].values() for method in path}
    assert methods == {"get"}


# ---- mapping coverage --------------------------------------------------------------------------

COVERED: dict[type, type] = {
    Experiment: m.ExperimentOut,
    ExperimentNode: m.NodeOut,
    Run: m.RunOut,
    RunAttempt: m.AttemptOut,
    EvaluationRun: m.EvaluationRunOut,
    EvaluationAttempt: m.EvaluationAttemptOut,
    EvaluationResult: m.EvaluationResultOut,
    MetricResult: m.MetricOut,
    Decision: m.DecisionOut,
    Action: m.ActionOut,
    PolicyDecision: m.PolicyDecisionOut,
    DomainEvent: m.EventOut,
    Incident: m.IncidentOut,
    RecoveryEpisode: m.RecoveryEpisodeOut,
    BudgetLedgerEntry: m.LedgerEntryOut,
    AgentInvocation: m.AgentInvocationOut,
}


@pytest.mark.parametrize("domain", list(COVERED), ids=lambda model: model.__name__)
def test_every_domain_field_is_mapped_or_deliberately_omitted(domain: type) -> None:
    """S-F9: a new or renamed domain field fails here until someone decides."""
    wire = set(COVERED[domain].model_fields)
    omitted = mapping.OMITTED[domain]
    fields = set(domain.model_fields)  # type: ignore[attr-defined]
    undecided = fields - wire - set(omitted)
    assert not undecided, f"{domain.__name__}: map or omit {sorted(undecided)}"
    stale = set(omitted) - fields
    assert not stale, f"{domain.__name__}: OMITTED names fields it no longer has: {sorted(stale)}"
    assert not (set(omitted) & wire), "a field is both mapped and omitted"


def test_recorded_checkpoints_are_mapped_whole() -> None:
    assert set(RecordedCheckpoint.model_fields) == set(m.CheckpointOut.model_fields)


# ---- documents ---------------------------------------------------------------------------------


def test_actor_metadata_never_leaves_the_server() -> None:
    secret = Actor(type="human", id="ann", metadata={"email": "ann@example.com"})
    assert mapping.actor(secret).model_dump() == {"type": "human", "id": "ann"}

    nested = mapping.document({"approved_by": secret, "steps": [{"by": secret}]})
    assert "ann@example.com" not in json.dumps(nested)
    assert nested == {
        "approved_by": {"type": "human", "id": "ann"},
        "steps": [{"by": {"type": "human", "id": "ann"}}],
    }


def test_an_actor_serialized_into_a_recorded_payload_is_reduced_too() -> None:
    payload = {"node": {"created_by": {"type": "human", "id": "ann", "metadata": {"k": "v"}}}}
    assert mapping.document(payload) == {"node": {"created_by": {"type": "human", "id": "ann"}}}
    # Not an actor: other keys, or another type.
    other = {"type": "human", "id": "x", "metadata": {}, "extra": 1}
    assert mapping.document(other) == other
    assert mapping.document({"type": "run", "id": "r", "metadata": {}})["metadata"] == {}


def test_no_recorded_actor_metadata_reaches_any_response(
    api: TestClient, recorded: Recorded
) -> None:
    """Every actor in every response is exactly ``{type, id}``, events included."""
    for path in _every_path(api, recorded):
        _assert_bare_actors(
            api.get(path, params={"limit": 500} if path.endswith("/events") else None).json(), path
        )


def _assert_bare_actors(value: Any, path: str) -> None:
    if isinstance(value, dict):
        if set(value) >= {"type", "id"} and value.get("type") in {
            "human",
            "rule",
            "llm_agent",
            "search_provider",
            "system",
        }:
            assert set(value) == {"type", "id"}, f"{path}: actor with {sorted(value)}"
        for item in value.values():
            _assert_bare_actors(item, path)
    elif isinstance(value, list):
        for item in value:
            _assert_bare_actors(item, path)


# ---- §S4: no SQL in the API --------------------------------------------------------------------


def test_the_api_package_holds_no_sql() -> None:
    pattern = re.compile(r"\.execute\(|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE FROM\b")
    offenders = [
        str(path.relative_to(API_SOURCE))
        for path in API_SOURCE.rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert offenders == []
