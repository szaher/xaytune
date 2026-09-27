"""Typed actions where intent becomes durable (PR-022).

```text
write   only an action its registered schema accepts        malformed → nothing written
read    any row, whatever its type                          history outlives a plugin
spec    spec_of(row) needs the type registered              fails closed otherwise
```
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from tests.test_core.test_typed_actions import (
    _LEGACY_CANCELLATION,
    ACME,
    BUILT_IN,
    ReduceSequenceLength,
)
from xaytune.core import Actor
from xaytune.core.domain.action import Action, ActionTarget, UnknownActionTypeError
from xaytune.core.domain.actions import (
    ActionDescriptor,
    ActionPayloadError,
    ActionSpec,
    CancelAttempt,
    action_from_spec,
    register_action,
    spec_of,
)
from xaytune.core.domain.actions import contract as action_types
from xaytune.core.ids import ActionId
from xaytune.storage import ControlPlaneRepository, write_transaction

from .conftest import make_experiment

ACTOR = Actor(type="llm_agent", id="planner")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def experiment_id(repo: ControlPlaneRepository) -> Any:
    return repo.create_experiment(make_experiment(), actor=ACTOR).id


def _record(repo: ControlPlaneRepository, action: Action) -> None:
    with write_transaction(repo._connection):
        repo.actions._insert(action)


def _row(repo: ControlPlaneRepository, action: Action) -> str:
    row = repo._connection.execute(
        "SELECT payload_json FROM actions WHERE id = ?", (str(action.id),)
    ).fetchone()
    return str(row["payload_json"])


def _action(spec: ActionSpec, experiment_id: Any) -> Action:
    return action_from_spec(spec, experiment_id=experiment_id, proposed_by=ACTOR, reason="why")


@pytest.mark.parametrize("name", sorted(BUILT_IN))
def test_every_built_in_is_stored_and_read_back_as_the_same_spec(
    repo: ControlPlaneRepository, experiment_id: Any, name: str
) -> None:
    action = _action(BUILT_IN[name], experiment_id)
    _record(repo, action)

    loaded = repo.actions.get(str(action.id))

    assert loaded == action
    assert spec_of(loaded) == BUILT_IN[name]
    assert _row(repo, action) == json.dumps(action.model_dump(mode="json"), sort_keys=True)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"micro_batch_size": 4}, ActionPayloadError),
        ({"schema_version": "1", "parameters": {"micro_batch_size": 0}}, ActionPayloadError),
        ({"schema_version": "1", "parameters": {"do": "something"}}, ActionPayloadError),
        ({"schema_version": "7", "parameters": {"micro_batch_size": 4}}, UnknownActionTypeError),
    ],
    ids=["bare", "bad-value", "unknown-parameter", "unknown-version"],
)
def test_a_malformed_action_never_becomes_durable_intent(
    repo: ControlPlaneRepository, experiment_id: Any, payload: dict, error: type[Exception]
) -> None:
    valid = _action(BUILT_IN["resize-microbatch"], experiment_id)
    malformed = Action.model_validate({**valid.model_dump(mode="python"), "payload": payload})

    with pytest.raises(error):
        _record(repo, malformed)
    assert repo.actions.for_experiment(str(experiment_id)) == ()


def test_an_action_on_a_target_its_type_does_not_act_on_is_not_written(
    repo: ControlPlaneRepository, experiment_id: Any
) -> None:
    valid = _action(BUILT_IN["change-learning-rate"], experiment_id)
    wrong = Action.model_validate(
        {
            **valid.model_dump(mode="python"),
            "target": ActionTarget(kind="experiment", id=str(experiment_id)),
        }
    )
    with pytest.raises(ActionPayloadError, match="acts on run"):
        _record(repo, wrong)
    assert repo.actions.for_experiment(str(experiment_id)) == ()


def test_a_cancellation_row_from_before_typed_actions_is_read_unchanged(
    repo: ControlPlaneRepository, experiment_id: Any
) -> None:
    """Written as 1.0.0a1 wrote it; loaded, typed and left byte-for-byte alone."""
    legacy = json.loads(_LEGACY_CANCELLATION)
    legacy["experiment_id"] = str(experiment_id)
    legacy["parent_action_id"] = None  # its parent row is not part of this fixture
    stored = json.dumps(legacy, sort_keys=True)
    with write_transaction(repo._connection):
        repo._connection.execute(
            "INSERT INTO actions (id, experiment_id, type, status, outcome, target_kind, "
            "target_id, proposed_by_json, reason, payload_json, policy_decision_id, revision, "
            "created_at, updated_at, parent_action_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL)",
            (
                legacy["id"],
                legacy["experiment_id"],
                legacy["type"],
                legacy["status"],
                legacy["outcome"],
                legacy["target"]["kind"],
                legacy["target"]["id"],
                json.dumps(legacy["proposed_by"], sort_keys=True),
                legacy["reason"],
                stored,
                legacy["revision"],
                legacy["created_at"],
                legacy["updated_at"],
            ),
        )

    loaded = repo.actions.get(legacy["id"])

    assert loaded is not None
    assert spec_of(loaded) == CancelAttempt(target=loaded.target)
    assert json.dumps(loaded.model_dump(mode="json"), sort_keys=True) == stored
    assert _row(repo, loaded) == stored


def test_a_plugin_action_is_recorded_and_outlives_its_plugin(
    repo: ControlPlaneRepository, experiment_id: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = dict(action_types._DESCRIPTORS)
    monkeypatch.setattr(action_types, "_DESCRIPTORS", dict(before))
    register_action(ActionDescriptor.for_spec(ReduceSequenceLength, provider=ACME))
    spec = ReduceSequenceLength(target=ActionTarget(kind="run", id="run_1"), max_length=512)
    action = _action(spec, experiment_id)
    _record(repo, action)
    assert spec_of(repo.actions.get(str(action.id))) == spec  # type: ignore[arg-type]

    monkeypatch.setattr(action_types, "_DESCRIPTORS", dict(before))  # uninstalled

    (loaded,) = repo.actions.for_experiment(str(experiment_id))
    assert loaded == action, "history still reads, by every path"
    assert repo.actions.get(str(action.id)) == action
    assert repo.actions.for_target("run", "run_1") == (action,)
    assert repo.actions.unresolved() == (action,)
    with pytest.raises(UnknownActionTypeError, match="acme/reduce-sequence-length"):
        spec_of(loaded)
    with pytest.raises(UnknownActionTypeError):
        _record(
            repo,
            Action.model_validate({**action.model_dump(mode="python"), "id": ActionId.generate()}),
        )
