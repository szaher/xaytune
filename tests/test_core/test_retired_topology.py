"""Topology has one source of truth: the repository's relationships (open question 15).

The aggregates used to carry child lists -- ``Experiment.active_node_ids``,
``ExperimentNode.run_ids`` / ``evaluation_run_ids``, ``Run.attempt_ids`` /
``final_attempt_id`` -- that no writer ever filled. They are gone; records
stored with their empty defaults still load, and a stored value fails closed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_core.test_domain import make_experiment, make_node, make_run
from xaytune.core import Experiment, ExperimentNode, ExperimentNodeId, Run, RunAttemptId, RunId
from xaytune.core.domain.evaluation import EvaluationRun

RETIRED = {
    Experiment: {"active_node_ids": []},
    ExperimentNode: {"run_ids": [], "evaluation_run_ids": []},
    Run: {"attempt_ids": [], "final_attempt_id": None},
}
BUILDERS = {Experiment: make_experiment, ExperimentNode: make_node, Run: make_run}
POPULATED = [
    (Experiment, "active_node_ids", [str(ExperimentNodeId.generate())]),
    (ExperimentNode, "run_ids", [str(RunId.generate())]),
    (ExperimentNode, "evaluation_run_ids", ["evalrun_x"]),
    (Run, "attempt_ids", [str(RunAttemptId.generate())]),
    (Run, "final_attempt_id", str(RunAttemptId.generate())),
]


@pytest.mark.parametrize("model", list(RETIRED))
def test_the_aggregate_has_no_child_lists(model):
    assert not set(RETIRED[model]) & set(model.model_fields)
    dumped = BUILDERS[model]().model_dump(mode="json")
    assert not set(RETIRED[model]) & set(dumped)


@pytest.mark.parametrize("model", list(RETIRED))
def test_a_record_stored_with_the_never_written_defaults_still_loads(model):
    current = BUILDERS[model]()
    stored = {**current.model_dump(mode="json"), **RETIRED[model]}
    assert model.model_validate(stored) == current
    assert model.model_validate_json(current.model_dump_json()) == current


@pytest.mark.parametrize(("model", "name", "value"), POPULATED)
def test_a_stored_topology_value_fails_closed(model, name, value):
    stored = {**BUILDERS[model]().model_dump(mode="json"), name: value}
    with pytest.raises(ValidationError, match="retired"):
        model.model_validate(stored)


def test_evaluation_runs_never_carried_attempt_lists():
    assert "attempt_ids" not in EvaluationRun.model_fields


def test_kept_fields_are_decisions_and_lineage_not_topology():
    assert "best_node_id" in Experiment.model_fields
    assert {"parent_ids", "decision_ids"} <= set(ExperimentNode.model_fields)
