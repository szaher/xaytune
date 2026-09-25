"""Decisions on the record: written with what they cause, once per cycle, never edited.

```text
record_decision   decision + node + (for STOP outcomes) experiment + events    one commit
                    STOP_SUCCEEDED  node COMPLETED  experiment SUCCEEDED, best_node_id
                    STOP_FAILED     node REJECTED   experiment FAILED
                    REJECT          node REJECTED   experiment stays ACTIVE
defer_decision    DecisionDeferred on the node, once per cycle; the node stays DECIDING
```
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tests.test_storage.test_evaluation_lifecycle import (
    _ACTOR,
    _begin,
    _run,
    _succeed,
)
from tests.test_storage.test_evaluation_lifecycle import node as node  # noqa: F401 (fixture)
from tests.test_storage.test_evaluation_lifecycle import repo as repo  # noqa: F401 (fixture)
from xaytune.core.domain.decision import Decision, DecisionContext, DecisionOutcome
from xaytune.core.errors import ConcurrentModificationError, InvalidTransitionError
from xaytune.core.ids import EvaluationId
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.decision import ThresholdDecisionEngine
from xaytune.storage import write_transaction
from xaytune.storage.control_plane import (
    ControlPlaneRepository,
    DecisionConflictError,
    EvaluationReconciliation,
    ProvenanceError,
)
from xaytune.storage.errors import StorageError


def _deciding(repo: ControlPlaneRepository, node: Any) -> tuple[Any, Any]:
    """Evaluate the node once and reconcile it into DECIDING; return it and its result."""
    run = _run(node)
    _begin(repo, node, run)
    result = _succeed(repo, run)
    assert (
        repo.reconcile_evaluating_node(node.id, actor=_ACTOR) is EvaluationReconciliation.DECIDING
    )
    return repo.aggregates.load_node(str(node.id)), result


def _context(repo: ControlPlaneRepository, node: Any, *results: Any, **objective: Any) -> Any:
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    return DecisionContext(
        experiment_id=node.experiment_id,
        node_id=node.id,
        evaluation_cycle=node.evaluation_cycle,
        objective=experiment.objective.model_copy(update=objective),
        results=results,
    )


def _decide(repo: ControlPlaneRepository, node: Any, *results: Any, **objective: Any) -> Any:
    """The fixture experiment maximizes task_success to 0.82; the fixture metric is accuracy."""
    from xaytune.core.domain.objective import ObjectiveMetric

    objective.setdefault("primary", ObjectiveMetric(name="accuracy", direction="maximize"))
    return ThresholdDecisionEngine().decide(_context(repo, node, *results, **objective))


# ---- applied in one commit ---------------------------------------------------------


def test_a_decision_is_recorded_and_applied_together(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    proposal = _decide(repo, deciding, result, target=0.5)  # accuracy 0.8 >= 0.5

    recorded = repo.record_decision(
        proposal, expected_node_revision=deciding.revision, actor=_ACTOR
    )

    assert recorded.proposal() == proposal
    assert recorded.actor == _ACTOR, "the record's actor, added when recorded"
    decided = repo.aggregates.load_node(str(node.id))
    assert decided.status is ExperimentNodeStatus.COMPLETED
    assert decided.decision_ids == (recorded.id,)
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert experiment.status is ExperimentStatus.SUCCEEDED
    assert experiment.best_node_id == node.id, "the candidate that succeeded, named"
    assert repo.aggregates.decision_for_cycle(str(node.id), 1) == recorded
    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert kinds[-3:] == [
        "DecisionRecorded",
        "ExperimentNodeStatusChanged",
        "ExperimentStatusChanged",
    ]


@pytest.mark.parametrize(
    ("target", "node_status", "experiment_status"),
    [
        (0.9, ExperimentNodeStatus.REJECTED, ExperimentStatus.FAILED),
        (0.5, ExperimentNodeStatus.COMPLETED, ExperimentStatus.SUCCEEDED),
    ],
    ids=["target-missed", "target-met"],
)
def test_the_outcome_decides_where_node_and_experiment_end(
    repo: ControlPlaneRepository,
    node: Any,
    target: float,
    node_status: ExperimentNodeStatus,
    experiment_status: ExperimentStatus,
) -> None:
    deciding, result = _deciding(repo, node)
    repo.record_decision(
        _decide(repo, deciding, result, target=target),
        expected_node_revision=deciding.revision,
        actor=_ACTOR,
    )
    assert repo.aggregates.load_node(str(node.id)).status is node_status
    assert repo.aggregates.load_experiment(str(node.experiment_id)).status is experiment_status


def test_a_rejected_candidate_leaves_the_experiment_open(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """REJECT judges the candidate, not the experiment: another may yet be proposed."""
    from xaytune.core.domain.objective import MetricConstraint

    deciding, result = _deciding(repo, node)
    proposal = _decide(
        repo,
        deciding,
        result,
        target=0.5,
        constraints=(MetricConstraint(name="accuracy", operator=">=", value=0.95),),
    )
    assert proposal.outcome is DecisionOutcome.REJECT

    repo.record_decision(proposal, expected_node_revision=deciding.revision, actor=_ACTOR)

    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.REJECTED
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert experiment.status is ExperimentStatus.ACTIVE
    assert experiment.best_node_id is None
    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert kinds[-2:] == ["DecisionRecorded", "ExperimentNodeStatusChanged"]


def test_a_failed_stop_names_no_best_candidate(repo: ControlPlaneRepository, node: Any) -> None:
    deciding, result = _deciding(repo, node)
    repo.record_decision(
        _decide(repo, deciding, result, target=0.9),
        expected_node_revision=deciding.revision,
        actor=_ACTOR,
    )
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert (experiment.status, experiment.best_node_id) == (ExperimentStatus.FAILED, None)


def test_a_failure_partway_leaves_nothing_behind(
    repo: ControlPlaneRepository, node: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision, node and experiment roll back together: none without the others."""
    deciding, result = _deciding(repo, node)
    decision = _decide(repo, deciding, result, target=0.5)

    def fail(*_: Any) -> None:
        raise RuntimeError("the disk filled up")

    monkeypatch.setattr(repo.aggregates, "_update_experiment", fail)
    with pytest.raises(RuntimeError, match="disk"):
        repo.record_decision(decision, expected_node_revision=deciding.revision, actor=_ACTOR)

    assert repo.aggregates.decision_for_cycle(str(node.id), 1) is None
    after = repo.aggregates.load_node(str(node.id))
    assert (after.status, after.decision_ids) == (ExperimentNodeStatus.DECIDING, ())
    assert (
        repo.aggregates.load_experiment(str(node.experiment_id)).status is ExperimentStatus.ACTIVE
    )
    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert "DecisionRecorded" not in kinds


def test_a_node_read_before_it_moved_is_refused(repo: ControlPlaneRepository, node: Any) -> None:
    deciding, result = _deciding(repo, node)
    with pytest.raises(ConcurrentModificationError):
        repo.record_decision(
            _decide(repo, deciding, result, target=0.5),
            expected_node_revision=deciding.revision - 1,
            actor=_ACTOR,
        )


def test_only_a_deciding_node_is_decided(repo: ControlPlaneRepository, node: Any) -> None:
    run = _run(node)
    evaluating = _begin(repo, node, run)
    result = _succeed(repo, run)  # the node is still EVALUATING: not reconciled
    decision = _decide(repo, evaluating, result, target=0.5)
    with pytest.raises(InvalidTransitionError):
        repo.record_decision(
            decision,
            expected_node_revision=repo.aggregates.load_node(str(node.id)).revision,
            actor=_ACTOR,
        )


# ---- once per cycle -------------------------------------------------------------------


def test_deciding_the_same_cycle_again_returns_the_decision_on_record(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """What a controller that restarted after deciding does: no duplicate, no error."""
    deciding, result = _deciding(repo, node)
    first = repo.record_decision(
        _decide(repo, deciding, result, target=0.5),
        expected_node_revision=deciding.revision,
        actor=_ACTOR,
    )
    again = _decide(repo, deciding, result, target=0.5)

    assert repo.record_decision(again, expected_node_revision=0, actor=_ACTOR) == first
    assert repo.aggregates.decisions_for_node(str(node.id)) == (first,)
    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert kinds.count("DecisionRecorded") == 1


def test_a_different_decision_for_a_decided_cycle_is_refused(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    repo.record_decision(
        _decide(repo, deciding, result, target=0.5),
        expected_node_revision=deciding.revision,
        actor=_ACTOR,
    )
    with pytest.raises(DecisionConflictError, match="was decided stop_succeeded"):
        repo.record_decision(
            _decide(repo, deciding, result, target=0.9),
            expected_node_revision=deciding.revision,
            actor=_ACTOR,
        )


# ---- attributable to the cycle it decides ------------------------------------------


def test_a_decision_must_name_exactly_the_cycles_results(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    foreign = _decide(repo, deciding, result, target=0.5).model_copy(
        update={"evaluation_result_ids": (result.id, EvaluationId.generate())}
    )
    with pytest.raises(ProvenanceError, match="names results"):
        repo.record_decision(foreign, expected_node_revision=deciding.revision, actor=_ACTOR)
    assert repo.aggregates.decision_for_cycle(str(node.id), 1) is None


def test_a_decision_about_another_cycle_is_refused(repo: ControlPlaneRepository, node: Any) -> None:
    deciding, result = _deciding(repo, node)
    wrong = _decide(repo, deciding, result, target=0.5).model_copy(update={"evaluation_cycle": 2})
    with pytest.raises(ProvenanceError, match="decides cycle 2"):
        repo.record_decision(wrong, expected_node_revision=deciding.revision, actor=_ACTOR)


def test_results_of_an_earlier_cycle_do_not_decide_this_one(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """ADR-015 §5: the first round's result cannot stand in for the second's."""
    deciding, first_result = _deciding(repo, node)
    active = repo.transition_node(
        node.id,
        expected_revision=deciding.revision,
        new_status=ExperimentNodeStatus.ACTIVE,
        actor=_ACTOR,
    )
    second, second_result = _deciding(repo, active)
    assert second.evaluation_cycle == 2

    stale = _decide(repo, second, first_result, target=0.5)
    with pytest.raises(ProvenanceError, match="names results"):
        repo.record_decision(stale, expected_node_revision=second.revision, actor=_ACTOR)

    current = _decide(repo, second, second_result, target=0.5)
    repo.record_decision(current, expected_node_revision=second.revision, actor=_ACTOR)
    recorded = repo.aggregates.decision_for_cycle(str(node.id), 2)
    assert recorded is not None and recorded.proposal() == current
    assert repo.aggregates.decision_for_cycle(str(node.id), 1) is None


# ---- deferred --------------------------------------------------------------------


def test_an_undecidable_cycle_is_recorded_once_and_the_node_waits(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, _ = _deciding(repo, node)
    assert repo.defer_decision(
        node.id, engine="threshold 1.0.0", reasons=("no target",), actor=_ACTOR
    )
    assert not repo.defer_decision(
        node.id, engine="threshold 1.0.0", reasons=("no target",), actor=_ACTOR
    )

    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.DECIDING
    (deferred,) = [
        e
        for e in repo.events.events_for_experiment(str(node.experiment_id))
        if e.event_type == "DecisionDeferred"
    ]
    assert deferred.payload["evaluation_cycle"] == 1
    assert list(deferred.payload["reasons"]) == ["no target"]


def test_only_a_deciding_node_can_be_deferred(repo: ControlPlaneRepository, node: Any) -> None:
    with pytest.raises(StorageError, match="not deciding"):
        repo.defer_decision(node.id, engine="threshold 1.0.0", reasons=("x",), actor=_ACTOR)


# ---- what the database itself refuses --------------------------------------------


def test_a_decision_is_never_edited(repo: ControlPlaneRepository, node: Any) -> None:
    deciding, result = _deciding(repo, node)
    decision = _decide(repo, deciding, result, target=0.5)
    repo.record_decision(decision, expected_node_revision=deciding.revision, actor=_ACTOR)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with write_transaction(repo._connection):
            repo._connection.execute("UPDATE decisions SET outcome = 'reject'")


def test_the_database_refuses_a_second_decision_for_a_cycle(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    repo.record_decision(
        _decide(repo, deciding, result, target=0.5),
        expected_node_revision=deciding.revision,
        actor=_ACTOR,
    )
    with pytest.raises(sqlite3.IntegrityError):
        with write_transaction(repo._connection):
            repo.aggregates._insert_decision(
                Decision.record(_decide(repo, deciding, result, target=0.9), actor=_ACTOR)
            )


def test_the_database_refuses_a_decision_filed_under_another_experiment(
    repo: ControlPlaneRepository, node: Any
) -> None:
    from tests.test_storage.conftest import make_experiment

    other = repo.create_experiment(make_experiment("other"), actor=_ACTOR)
    deciding, result = _deciding(repo, node)
    misfiled = _decide(repo, deciding, result, target=0.5).model_copy(
        update={"experiment_id": other.id}
    )
    with pytest.raises(sqlite3.IntegrityError, match="does not belong"):
        with write_transaction(repo._connection):
            repo.aggregates._insert_decision(Decision.record(misfiled, actor=_ACTOR))
