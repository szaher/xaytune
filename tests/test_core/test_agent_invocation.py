"""Agent invocations: intent before the call, an outcome after it, nothing invented."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions import RejectCandidate
from xaytune.core.domain.agent_invocation import (
    AgentInvocation,
    AgentInvocationConflictError,
    AgentInvocationFailure,
    AgentInvocationIntent,
    AgentInvocationMismatchError,
    AgentInvocationStatus,
    InMemoryAgentInvocationJournal,
    next_invocation,
    require_derived_from,
)
from xaytune.core.domain.planning import (
    ActionProposal,
    ProposalProvenance,
    action_proposal_identity_v1,
)
from xaytune.core.errors import ConcurrentModificationError, InvalidTransitionError
from xaytune.core.ids import AgentInvocationId, ExperimentId, ExperimentNodeId
from xaytune.core.immutable import FrozenDict

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)
EXPERIMENT = ExperimentId.generate()
NODE = ExperimentNodeId.generate()
S = AgentInvocationStatus


def intent(**changes: Any) -> AgentInvocationIntent:
    return AgentInvocationIntent.model_validate(
        {
            "experiment_id": EXPERIMENT,
            "planner_kind": "llm",
            "planner_version": "1.0.0",
            "planner_spec_fingerprint": "sha256:spec",
            "context_identity_version": 1,
            "context_fingerprint": "sha256:context",
            "prompt_version": "xaytune.llm-planner/v1",
            "prompt_fingerprint": "sha256:prompt",
            "request_identity_version": 1,
            "request_fingerprint": "sha256:request",
            "request": {"system": "s"},
            "agent_model": {"plugin": {"name": "adapter"}, "model": {"name": "m"}},
            **changes,
        }
    )


def fresh(**changes: Any) -> AgentInvocation:
    return AgentInvocation(
        id=AgentInvocationId.generate(), attempt=1, intent=intent(**changes), created_at=NOW
    )


PROVENANCE = ProposalProvenance(
    planner_provider="xaytune",
    planner_name="llm",
    planner_version="1.0.0",
    planner_api_version="xaytune.plugins/v1alpha1",
    planner_spec_kind="llm",
    planner_spec_version="1.0.0",
    planner_spec_identity_version=1,
    planner_spec_fingerprint="sha256:spec",
    context_identity_version=1,
    context_fingerprint="sha256:context",
)


def proposal(invocation: AgentInvocation, **changes: Any) -> ActionProposal:
    return ActionProposal(
        action=RejectCandidate(target=ActionTarget(kind="node", id=str(NODE))),
        reason=changes.pop("reason", "below target"),
        provenance=changes.pop("provenance", PROVENANCE),
        agent_invocation_id=changes.pop("agent_invocation_id", invocation.id),
    )


FAILURE = AgentInvocationFailure(kind="invocation-failed", error_type="ConnectionError")
ANSWER = {"content": {"proposal": None}, "model": "m"}


# ---- the lifecycle -----------------------------------------------------------------------


def test_an_invocation_begins_intended_with_nothing_settled() -> None:
    invocation = fresh()
    assert invocation.status is S.INTENDED
    assert (invocation.response, invocation.failure, invocation.settled_at) == (None, None, None)


def test_an_answer_is_recorded_before_anything_is_derived_from_it() -> None:
    answered = fresh().answered(ANSWER, at=NOW)
    assert answered.status is S.ANSWERED
    assert answered.settled_at is None and answered.revision == 1
    done = answered.completed(proposal(answered), at=NOW)
    assert done.status is S.COMPLETED and done.settled_at == NOW and done.revision == 2
    assert done.proposal_fingerprint == fingerprint_of(proposal(answered))
    assert done.response == answered.response


def fingerprint_of(p: ActionProposal) -> str:
    return p.proposal_fingerprint()


def test_completing_with_no_proposal_records_none() -> None:
    done = fresh().answered(ANSWER, at=NOW).completed(None, at=NOW)
    assert (done.proposal, done.proposal_fingerprint) == (None, None)


@pytest.mark.parametrize(
    ("start", "move"),
    [
        (lambda: fresh(), lambda i: i.completed(None, at=NOW)),  # no answer to derive from
        (lambda: fresh().outcome_unknown(at=NOW), lambda i: i.answered(ANSWER, at=NOW)),
        (lambda: fresh().failed(FAILURE, at=NOW), lambda i: i.answered(ANSWER, at=NOW)),
        (
            lambda: fresh().answered(ANSWER, at=NOW).completed(None, at=NOW),
            lambda i: i.refused(FAILURE, at=NOW),
        ),
        (lambda: fresh().answered(ANSWER, at=NOW), lambda i: i.outcome_unknown(at=NOW)),
    ],
    ids=[
        "complete-unanswered",
        "unknown-is-final",
        "failed-is-final",
        "completed-is-final",
        "answered-is-known",
    ],
)
def test_only_forward_transitions(start: Any, move: Any) -> None:
    with pytest.raises(InvalidTransitionError):
        move(start())


def test_a_proposal_naming_another_invocation_cannot_complete_this_one() -> None:
    answered = fresh().answered(ANSWER, at=NOW)
    with pytest.raises(AgentInvocationMismatchError):
        answered.completed(
            proposal(answered, agent_invocation_id=AgentInvocationId.generate()), at=NOW
        )


def test_a_record_inconsistent_with_its_status_cannot_exist() -> None:
    with pytest.raises(ValidationError, match="failure is recorded exactly"):
        AgentInvocation(
            id=AgentInvocationId.generate(),
            attempt=1,
            intent=intent(),
            status=S.FAILED,
            created_at=NOW,
            settled_at=NOW,
        )
    with pytest.raises(ValidationError, match="records its answer"):
        AgentInvocation(
            id=AgentInvocationId.generate(),
            attempt=1,
            intent=intent(),
            status=S.ANSWERED,
            created_at=NOW,
        )


def test_a_failure_carries_a_classification_never_free_text() -> None:
    assert set(AgentInvocationFailure.model_fields) == {"kind", "error_type", "reasons"}
    with pytest.raises(ValidationError):
        AgentInvocationFailure(kind="stack-trace", error_type="X")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgentInvocationFailure.model_validate(
            {"kind": "invocation-failed", "error_type": "X", "message": "Bearer sk-live"}
        )


# ---- the round: replay, retry, unknown ------------------------------------------------------


def test_a_round_with_nothing_recorded_begins_attempt_one() -> None:
    closed, invocation, replayed = next_invocation(None, intent(), at=NOW)
    assert (closed, invocation.attempt, invocation.status, replayed) == (None, 1, S.INTENDED, False)


@pytest.mark.parametrize(
    "latest",
    [
        lambda: fresh().answered(ANSWER, at=NOW),
        lambda: fresh().answered(ANSWER, at=NOW).completed(None, at=NOW),
        lambda: fresh().refused(FAILURE.model_copy(update={"kind": "output-refused"}), at=NOW),
    ],
    ids=["answered", "completed", "refused"],
)
def test_a_round_that_got_an_answer_is_replayed_never_asked_again(latest: Any) -> None:
    recorded = latest()
    closed, invocation, replayed = next_invocation(recorded, intent(), at=NOW)
    assert (closed, invocation, replayed) == (None, recorded, True)


def test_an_attempt_still_intended_ends_unknown_and_the_round_tries_again() -> None:
    orphan = fresh()
    closed, invocation, replayed = next_invocation(orphan, intent(), at=NOW)
    assert closed is not None and closed.id == orphan.id
    assert closed.status is S.OUTCOME_UNKNOWN and closed.settled_at == NOW
    assert (invocation.attempt, invocation.status, replayed) == (2, S.INTENDED, False)


@pytest.mark.parametrize(
    "latest", [lambda: fresh().failed(FAILURE, at=NOW), lambda: fresh().outcome_unknown(at=NOW)]
)
def test_a_failed_or_unknown_round_gets_a_new_numbered_attempt(latest: Any) -> None:
    closed, invocation, replayed = next_invocation(latest(), intent(), at=NOW)
    assert (closed, invocation.attempt, replayed) == (None, 2, False)


def test_a_round_that_recorded_another_request_is_a_conflict() -> None:
    with pytest.raises(AgentInvocationConflictError, match="sha256:other"):
        next_invocation(fresh(), intent(request_fingerprint="sha256:other"), at=NOW)


def test_the_in_memory_journal_follows_the_same_rules_and_guards_revisions() -> None:
    journal = InMemoryAgentInvocationJournal(clock=lambda: NOW)
    first = journal.begin(intent())
    assert journal.begin(intent()).attempt == 2  # first was still INTENDED
    assert journal.get(first.id).status is S.OUTCOME_UNKNOWN  # type: ignore[union-attr]
    second = journal.invocations[-1]
    answered = journal.answered(second, ANSWER)
    with pytest.raises(ConcurrentModificationError):
        journal.answered(second, ANSWER)  # moved from a stale revision
    assert journal.begin(intent()) == answered  # replayed


# ---- binding a proposal to the call that produced it ----------------------------------------


def _completed() -> tuple[AgentInvocation, ActionProposal]:
    answered = fresh().answered(ANSWER, at=NOW)
    made = proposal(answered)
    return answered.completed(made, at=NOW), made


def test_a_proposal_is_bound_to_the_invocation_that_recorded_it() -> None:
    invocation, made = _completed()
    require_derived_from(made, invocation)


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        (lambda i: (proposal(i, agent_invocation_id=None), i), "names no agent"),
        (lambda i: (proposal(i), None), "not on record"),
        (lambda i: (proposal(i, reason="something else"), i), "not the proposal"),
        (
            lambda i: (
                proposal(
                    i,
                    provenance=PROVENANCE.model_copy(
                        update={"context_fingerprint": "sha256:elsewhere"}
                    ),
                ),
                i,
            ),
            "planning context",
        ),
        (
            lambda i: (proposal(i, agent_invocation_id=AgentInvocationId.generate()), i),
            "names invocation",
        ),
    ],
    ids=["unbound", "unrecorded", "altered", "other-context", "other-invocation"],
)
def test_a_proposal_cannot_claim_a_call_that_did_not_produce_it(tamper: Any, match: str) -> None:
    invocation, _ = _completed()
    claimed, record = tamper(invocation)
    with pytest.raises(AgentInvocationMismatchError, match=match):
        require_derived_from(claimed, record)


def test_an_unfinished_invocation_vouches_for_no_proposal() -> None:
    answered = fresh().answered(ANSWER, at=NOW)
    with pytest.raises(AgentInvocationMismatchError, match="answered, not completed"):
        require_derived_from(proposal(answered), answered)


def test_the_invocation_id_is_not_part_of_the_proposals_identity() -> None:
    invocation, made = _completed()
    other = proposal(invocation, agent_invocation_id=AgentInvocationId.generate())
    assert action_proposal_identity_v1(made) == action_proposal_identity_v1(other)
    assert "agent_invocation_id" not in str(action_proposal_identity_v1(made))
    assert set(ActionProposal.model_fields) == {
        "kind",
        "action",
        "reason",
        "evidence_refs",
        "provenance",
        "agent_invocation_id",
    }


def test_the_intent_carries_the_whole_request_and_descriptor_as_canonical_json() -> None:
    recorded = intent(request={"messages": [{"role": "user", "content": "x"}]})
    assert isinstance(recorded.request, FrozenDict)
    restored = AgentInvocationIntent.model_validate_json(recorded.model_dump_json())
    assert restored == recorded


def test_a_recorded_answer_never_fails_so_the_round_never_asks_again() -> None:
    """Review 1 (S1): ANSWERED -> FAILED would make the round retryable."""
    answered = fresh().answered(ANSWER, at=NOW)
    with pytest.raises(InvalidTransitionError):
        answered.failed(FAILURE, at=NOW)
