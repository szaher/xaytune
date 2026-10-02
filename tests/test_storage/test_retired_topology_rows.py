"""Rows written before the topology fields were retired still load (open question 15)."""

from __future__ import annotations

import json

import pytest

from xaytune.core.errors import InvalidDomainValueError


def _rewrite(connection, table, row_id, extra):
    payload = json.loads(
        connection.execute(f"SELECT payload_json FROM {table} WHERE id = ?", (row_id,)).fetchone()[
            0
        ]
    )
    connection.execute(
        f"UPDATE {table} SET payload_json = ? WHERE id = ?",
        (json.dumps({**payload, **extra}), row_id),
    )


LEGACY = [
    ("experiments", "experiment", "load_experiment", {"active_node_ids": []}),
    ("experiment_nodes", "node", "load_node", {"run_ids": [], "evaluation_run_ids": []}),
    ("runs", "run", "load_run", {"attempt_ids": [], "final_attempt_id": None}),
]


@pytest.mark.parametrize(("table", "key", "loader", "extra"), LEGACY)
def test_a_pre_retirement_row_loads_as_the_same_aggregate(
    store, connection, seeded, table, key, loader, extra
):
    aggregate = seeded[key]
    _rewrite(connection, table, str(aggregate.id), extra)
    assert getattr(store, loader)(str(aggregate.id)) == aggregate


def test_topology_still_comes_from_the_relationships(store, seeded):
    experiment, node, run, attempt = (
        seeded["experiment"],
        seeded["node"],
        seeded["run"],
        seeded["attempt"],
    )
    assert [n.id for n in store.nodes_for_experiment(str(experiment.id))] == [node.id]
    assert [r.id for r in store.runs_for_node(str(node.id))] == [run.id]
    assert [a.id for a in store.attempts_for_run(str(run.id))] == [attempt.id]
    assert store.evaluation_runs_for_node(str(node.id)) == ()


def test_a_row_carrying_a_real_topology_value_fails_closed(store, connection, seeded):
    run = seeded["run"]
    _rewrite(connection, "runs", str(run.id), {"attempt_ids": [str(seeded["attempt"].id)]})
    with pytest.raises((InvalidDomainValueError, ValueError), match="retired"):
        store.load_run(str(run.id))
