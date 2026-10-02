"""A BRANCH decision on the record: the candidate finishes valid, the experiment goes on.

```text
record_decision(BRANCH)   decision + node COMPLETED + events     one commit
                          experiment stays ACTIVE, best_node_id unset
```
"""

from __future__ import annotations

from typing import Any

from tests.test_storage.test_decisions import _context, _deciding
from tests.test_storage.test_evaluation_lifecycle import _ACTOR
from tests.test_storage.test_evaluation_lifecycle import node as node  # noqa: F401 (fixture)
from tests.test_storage.test_evaluation_lifecycle import repo as repo  # noqa: F401 (fixture)
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import ObjectiveMetric
from xaytune.core.state.status import ExperimentNodeStatus, ExperimentStatus
from xaytune.decision import AdaptiveThresholdDecisionEngine
from xaytune.storage.control_plane import ControlPlaneRepository


def _adaptive(repo: ControlPlaneRepository, node: Any, *results: Any, **objective: Any) -> Any:
    objective.setdefault("primary", ObjectiveMetric(name="accuracy", direction="maximize"))
    return AdaptiveThresholdDecisionEngine().decide(_context(repo, node, *results, **objective))


def test_a_branch_completes_the_candidate_and_leaves_the_experiment_open(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    proposal = _adaptive(repo, deciding, result, target=0.9)  # accuracy 0.8 < 0.9
    assert proposal.outcome is DecisionOutcome.BRANCH

    recorded = repo.record_decision(
        proposal, expected_node_revision=deciding.revision, actor=_ACTOR
    )

    decided = repo.aggregates.load_node(str(node.id))
    assert decided.status is ExperimentNodeStatus.COMPLETED
    assert decided.decision_ids == (recorded.id,)
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert (experiment.status, experiment.best_node_id) == (ExperimentStatus.ACTIVE, None)
    kinds = [e.event_type for e in repo.events.events_for_experiment(str(node.experiment_id))]
    assert kinds[-2:] == ["DecisionRecorded", "ExperimentNodeStatusChanged"]
    assert repo.aggregates.decision_for_cycle(str(node.id), 1) == recorded


def test_deciding_a_branched_cycle_again_returns_the_record(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    proposal = _adaptive(repo, deciding, result, target=0.9)
    first = repo.record_decision(proposal, expected_node_revision=deciding.revision, actor=_ACTOR)
    again = repo.record_decision(proposal, expected_node_revision=deciding.revision, actor=_ACTOR)
    assert again == first


def test_the_adaptive_engine_still_succeeds_the_experiment_on_target(
    repo: ControlPlaneRepository, node: Any
) -> None:
    deciding, result = _deciding(repo, node)
    proposal = _adaptive(repo, deciding, result, target=0.5)
    assert proposal.outcome is DecisionOutcome.STOP_SUCCEEDED
    repo.record_decision(proposal, expected_node_revision=deciding.revision, actor=_ACTOR)
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert (experiment.status, experiment.best_node_id) == (ExperimentStatus.SUCCEEDED, node.id)


# ---- every outcome is handled by name ---------------------------------------------


def test_every_implemented_outcome_is_handled_explicitly() -> None:
    """Adding an outcome must update both maps, never fall through to a default."""
    from xaytune.core.domain.decision import _NODE_STATUS

    assert set(DecisionOutcome) == {
        DecisionOutcome.STOP_SUCCEEDED,
        DecisionOutcome.STOP_FAILED,
        DecisionOutcome.REJECT,
        DecisionOutcome.BRANCH,
    }
    assert set(_NODE_STATUS) == set(DecisionOutcome)


def test_an_outcome_with_no_experiment_transition_fails_closed(
    repo: ControlPlaneRepository, node: Any
) -> None:
    """A future outcome reaching the experiment step raises; it never fails the experiment."""
    from enum import Enum
    from types import SimpleNamespace

    import pytest

    from xaytune.core.ids import DecisionId
    from xaytune.storage import write_transaction
    from xaytune.storage.errors import StorageError

    class Future(str, Enum):
        EVALUATE_MORE = "evaluate_more"

    decision = SimpleNamespace(
        id=DecisionId.generate(),
        outcome=Future.EVALUATE_MORE,
        experiment_id=node.experiment_id,
        node_id=node.id,
    )
    with pytest.raises(StorageError, match="evaluate_more has no experiment transition"):
        with write_transaction(repo._connection):
            repo._conclude_experiment(decision, _ACTOR, ())  # type: ignore[arg-type]
    experiment = repo.aggregates.load_experiment(str(node.experiment_id))
    assert (experiment.status, experiment.best_node_id) == (ExperimentStatus.ACTIVE, None)


def test_an_outcome_with_no_node_status_records_nothing(
    repo: ControlPlaneRepository, node: Any, monkeypatch: Any
) -> None:
    import pytest

    from xaytune.core.domain import decision as decision_module

    deciding, result = _deciding(repo, node)
    proposal = _adaptive(repo, deciding, result, target=0.9)
    monkeypatch.delitem(decision_module._NODE_STATUS, DecisionOutcome.BRANCH)
    with pytest.raises(ValueError, match="branch names no node status"):
        repo.record_decision(proposal, expected_node_revision=deciding.revision, actor=_ACTOR)
    assert repo.aggregates.decision_for_cycle(str(node.id), 1) is None
    assert repo.aggregates.load_node(str(node.id)).status is ExperimentNodeStatus.DECIDING
