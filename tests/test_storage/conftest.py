"""Fixtures for the persistence tests.

The database is a real file rather than ``:memory:``. An in-memory database
cannot demonstrate that a commit survives the process that made it, which is the
property most of these tests exist to check.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from xaytune.core import (
    Actor,
    CandidateSpecSnapshot,
    ControllerHostRef,
    Experiment,
    ExperimentId,
    ExperimentNode,
    ExperimentNodeId,
    Objective,
    ObjectiveMetric,
    Run,
    RunAttempt,
    RunAttemptId,
    RunId,
)
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.refs import DatasetRef, ModelRef
from xaytune.storage import AggregateStore, connect, migrate, write_transaction


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "control-plane.sqlite3"


@pytest.fixture
def connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    migrate(conn)
    yield conn
    conn.close()


@pytest.fixture
def store(connection: sqlite3.Connection) -> AggregateStore:
    return AggregateStore(connection)


def _snapshot() -> CandidateSpecSnapshot:
    """A minimal candidate, for tests that care about lineage rather than spec."""
    return CandidateSpecSnapshot(
        candidate=CandidateSpec(
            model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
            data=DataSpec(dataset=DatasetRef(uri="./data/support-v4.jsonl")),
            training=TrainingSpec(kind=TrainingKind.SFT),
        )
    )


def make_experiment(name: str = "support-qwen") -> Experiment:
    return Experiment(
        id=ExperimentId.generate(),
        name=name,
        objective=Objective(
            primary=ObjectiveMetric(name="task_success", direction="maximize"),
            target=0.82,
        ),
        controller_host=ControllerHostRef(kind="embedded", id="local"),
    )


def make_node(
    experiment: Experiment,
    fingerprint: str = "sha256:cand-a",
    parents: tuple[ExperimentNodeId, ...] = (),
) -> ExperimentNode:
    return ExperimentNode(
        id=ExperimentNodeId.generate(),
        experiment_id=experiment.id,
        parent_ids=parents,
        candidate=_snapshot(),
        candidate_fingerprint=fingerprint,
        created_by=Actor(type="system", id="controller"),
    )


def make_run(node: ExperimentNode) -> Run:
    return Run(
        id=RunId.generate(),
        node_id=node.id,
        experiment_id=node.experiment_id,
        candidate_fingerprint=node.candidate_fingerprint,
    )


def make_attempt(run: Run, attempt_number: int = 1) -> RunAttempt:
    return RunAttempt(
        id=RunAttemptId.generate(),
        run_id=run.id,
        attempt_number=attempt_number,
    )


@pytest.fixture
def seeded(store: AggregateStore, connection: sqlite3.Connection) -> dict[str, object]:
    """An experiment with one node, one run and one attempt, committed."""
    experiment = make_experiment()
    node = make_node(experiment)
    run = make_run(node)
    attempt = make_attempt(run)

    with write_transaction(connection):
        store._insert_experiment(experiment)
        store._insert_node(node)
        store._insert_run(run)
        store._insert_attempt(attempt)

    return {"experiment": experiment, "node": node, "run": run, "attempt": attempt}
