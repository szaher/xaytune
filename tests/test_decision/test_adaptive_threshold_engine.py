"""AdaptiveThresholdDecisionEngine: a missed target finishes the candidate, not the experiment.

The comparison is the non-adaptive engine's. Only the meaning of a missed
target differs, and the non-adaptive engine keeps its own.
"""

from __future__ import annotations

import pytest

from tests.test_decision.test_threshold_engine import _context, _objective, _result
from xaytune.core.domain.decision import DecisionOutcome
from xaytune.core.domain.objective import MetricConstraint
from xaytune.core.state.status import ExperimentNodeStatus
from xaytune.decision import (
    AdaptiveThresholdDecisionEngine,
    DecisionEngine,
    ThresholdDecisionEngine,
    UndecidableError,
)

ADAPTIVE = AdaptiveThresholdDecisionEngine()
SINGLE = ThresholdDecisionEngine()


def test_it_is_a_distinct_decision_engine() -> None:
    assert isinstance(ADAPTIVE, DecisionEngine)
    assert (ADAPTIVE.name, ADAPTIVE.version) == ("adaptive-threshold", "1.0.0")
    assert (SINGLE.name, SINGLE.version) == ("threshold", "1.0.0")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.79, DecisionOutcome.BRANCH),
        (0.83, DecisionOutcome.STOP_SUCCEEDED),
        (0.82, DecisionOutcome.STOP_SUCCEEDED),
    ],
)
def test_target_met_stops_and_target_missed_branches(value, expected) -> None:
    proposal = ADAPTIVE.decide(_context(_objective(target=0.82), _result(("accuracy", value))))
    assert proposal.outcome is expected
    assert (proposal.engine_name, proposal.engine_version) == ("adaptive-threshold", "1.0.0")


def test_a_branch_finishes_the_candidate_as_valid() -> None:
    assert DecisionOutcome.BRANCH.node_status is ExperimentNodeStatus.COMPLETED
    assert DecisionOutcome.REJECT.node_status is ExperimentNodeStatus.REJECTED


def test_a_violated_constraint_still_rejects_even_below_target() -> None:
    objective = _objective(
        "accuracy", "maximize", 0.82, MetricConstraint(name="latency_ms", operator="<=", value=100)
    )
    proposal = ADAPTIVE.decide(
        _context(objective, _result(("accuracy", 0.79), ("latency_ms", 150.0)))
    )
    assert proposal.outcome is DecisionOutcome.REJECT


def test_a_branch_records_the_evidence_and_why() -> None:
    objective = _objective(
        "loss", "minimize", 0.5, MetricConstraint(name="latency_ms", operator="<=", value=100)
    )
    proposal = ADAPTIVE.decide(_context(objective, _result(("loss", 0.7), ("latency_ms", 50.0))))
    assert proposal.outcome is DecisionOutcome.BRANCH
    assert proposal.reason.startswith("target not met: loss 0.7 is not <= 0.5")
    assert "another may be explored" in proposal.reason
    assert "constraints held" in proposal.reason
    assert [e.role for e in proposal.evidence] == ["objective", "constraint"]
    assert not proposal.evidence[0].satisfied and proposal.evidence[1].satisfied


def test_the_non_adaptive_engine_is_unchanged() -> None:
    context = _context(_objective(target=0.82), _result(("accuracy", 0.79)))
    single = SINGLE.decide(context)
    adaptive = ADAPTIVE.decide(context)
    assert single.outcome is DecisionOutcome.STOP_FAILED
    assert single.reason == "target not met: accuracy 0.79 is not >= 0.82"
    assert single.input_fingerprint == adaptive.input_fingerprint, "same inputs, same identity"
    assert single.evidence == adaptive.evidence


@pytest.mark.parametrize(
    "case",
    ["no-target", "missing-metric"],
)
def test_what_cannot_be_decided_stays_undecidable(case) -> None:
    if case == "no-target":
        context = _context(_objective(target=None), _result(("accuracy", 0.9)))
    else:
        context = _context(_objective(target=0.8), _result(("other", 0.9)))
    with pytest.raises(UndecidableError):
        ADAPTIVE.decide(context)


def test_deciding_the_same_context_twice_is_identical() -> None:
    context = _context(_objective(target=0.82), _result(("accuracy", 0.79)))
    assert ADAPTIVE.decide(context) == ADAPTIVE.decide(context)
