"""Governed actions, recorded (PR-023).

```text
PROPOSED → VALIDATING ─ not applicable ──────▶ REJECTED           no policy decision
                      └ VALIDATED → policy
                          ALLOW ─────────────▶ VALIDATED          + decision (authorized)
                          DENY ──────────────▶ REJECTED           + decision
                          REQUIRE_APPROVAL ──▶ APPROVAL_PENDING   + decision
                                               ├ human approves ─▶ APPROVED
                                               └ human rejects ──▶ REJECTED
```

The Action, its decision and every step's event commit together. The decision
records the snapshot it judged; a snapshot that changed before it could be
recorded is refused. Nothing here carries an action out.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from tests.test_storage.test_budget_ledger import _node
from xaytune.core.capabilities import CapabilityDocument, ElasticityCapabilities
from xaytune.core.domain.action import ActionStatus, ActionTarget
from xaytune.core.domain.actions import (
    ActionSpec,
    CancelExperiment,
    CancelRun,
    ChangeLearningRate,
    ChangeWorkerCount,
    MutationClass,
    RejectCandidate,
    ResizeMicrobatch,
)
from xaytune.core.domain.policy import (
    PolicyContext,
    PolicyProposal,
    PolicyProposer,
    PolicyVerdict,
)
from xaytune.core.ids import ActionId
from xaytune.core.refs import Actor
from xaytune.core.state.status import RunStatus
from xaytune.policy import DenyAllPolicy, PolicyRule, RulePolicyEngine
from xaytune.storage import ControlPlaneRepository, write_transaction
from xaytune.storage.control_plane import (
    ApprovalConflictError,
    ApprovalError,
    CancellationNotGovernedError,
    ProvenanceError,
    StalePolicyContextError,
)
from xaytune.storage.journal import IdempotencyConflictError

from .conftest import make_run

AGENT = Actor(type="llm_agent", id="planner")
ANA = Actor(type="human", id="ana")
CONTROLLER = Actor(type="system", id="controller")

POLICY = RulePolicyEngine(
    [
        PolicyRule(
            id="science-reviewed",
            verdict=PolicyVerdict.REQUIRE_APPROVAL,
            reason="scientific interventions are reviewed",
            mutation_classes=(MutationClass.SCIENTIFIC_INTERVENTION,),
        ),
        PolicyRule(
            id="operations-allowed",
            verdict=PolicyVerdict.ALLOW,
            reason="operational changes are fine",
            mutation_classes=(MutationClass.OPERATIONAL,),
        ),
    ],
    default=PolicyVerdict.DENY,
)


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def run(repo: ControlPlaneRepository) -> Any:
    """An active run, in an active experiment."""
    node = _node(repo)
    created = repo.create_run(make_run(node), actor=CONTROLLER)
    return repo.transition_run(
        created.id,
        expected_revision=created.revision,
        new_status=RunStatus.ACTIVE,
        actor=CONTROLLER,
    )


def _target(run: Any) -> ActionTarget:
    return ActionTarget(kind="run", id=str(run.id))


def _propose(
    repo: ControlPlaneRepository,
    run: Any,
    spec: ActionSpec,
    *,
    policy: Any = POLICY,
    capabilities: CapabilityDocument | None = None,
    **options: Any,
) -> Any:
    return repo.propose_action(
        spec,
        experiment_id=run.experiment_id,
        proposed_by=options.pop("proposed_by", AGENT),
        reason=options.pop("reason", "the loss plateaued"),
        policy=policy,
        capabilities=capabilities,
        **options,
    )


def _event_types(repo: ControlPlaneRepository, action: Any) -> list[str]:
    return [e.event_type for e in repo.events.events_for_aggregate(str(action.id))]


def _rows(repo: ControlPlaneRepository, table: str) -> int:
    return int(repo._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


# ---- the three verdicts ------------------------------------------------------------------


def test_an_allowed_action_stays_validated_with_its_decision(
    repo: ControlPlaneRepository, run: Any
) -> None:
    governed = _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2))

    action, decision = governed.action, governed.decision
    assert action.status is ActionStatus.VALIDATED, "authorization is the decision, not a state"
    assert decision is not None and decision.verdict is PolicyVerdict.ALLOW
    assert action.policy_decision_id == decision.id
    assert decision.rule_ids == ("operations-allowed",)
    assert repo.policy.for_action(str(action.id)) == decision
    assert _event_types(repo, action) == [
        "ActionProposed",
        "ActionValidating",
        "ActionValidated",
        "ActionAuthorized",
    ]


def test_with_no_policy_configured_the_denial_is_recorded(
    repo: ControlPlaneRepository, run: Any
) -> None:
    governed = _propose(
        repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2), policy=DenyAllPolicy()
    )
    assert governed.action.status is ActionStatus.REJECTED
    assert governed.decision is not None
    assert (governed.decision.verdict, governed.decision.engine_name) == (
        PolicyVerdict.DENY,
        "deny-all",
    )
    assert _event_types(repo, governed.action)[-1] == "ActionRejected"


def test_an_action_needing_approval_waits_for_a_human(
    repo: ControlPlaneRepository, run: Any
) -> None:
    governed = _propose(repo, run, ChangeLearningRate(target=_target(run), learning_rate=1e-5))
    assert governed.action.status is ActionStatus.APPROVAL_PENDING
    assert governed.decision is not None
    assert governed.decision.verdict is PolicyVerdict.REQUIRE_APPROVAL


def test_the_decision_records_the_snapshot_it_judged(
    repo: ControlPlaneRepository, run: Any
) -> None:
    elastic = CapabilityDocument(elasticity=ElasticityCapabilities(supported=True, max_workers=4))
    governed = _propose(
        repo, run, ChangeWorkerCount(target=_target(run), workers=2), capabilities=elastic
    )
    decision = governed.decision
    assert decision is not None
    assert decision.context.capabilities == elastic, "kept, because nothing else keeps it"
    assert decision.context.target_status == "active"
    assert decision.input_fingerprint == decision.context.input_fingerprint()

    row = repo._connection.execute(
        "SELECT verdict, input_fingerprint, payload_json FROM policy_decisions"
    ).fetchone()
    assert (row["verdict"], row["input_fingerprint"]) == ("allow", decision.input_fingerprint)
    assert json.loads(row["payload_json"])["context"]["capabilities"]["elasticity"]["supported"]


class _TrustsAdmins:
    """A policy that would branch on the proposer's metadata, if it could see it."""

    name = "trusts-admins"
    version = "1"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        metadata = getattr(context.proposed_by, "metadata", {})
        verdict = PolicyVerdict.ALLOW if metadata.get("role") == "admin" else PolicyVerdict.DENY
        return POLICY.evaluate(spec, context).model_copy(
            update={"verdict": verdict, "engine_name": self.name, "engine_version": self.version}
        )


def test_policy_sees_who_proposed_not_their_metadata(
    repo: ControlPlaneRepository, run: Any
) -> None:
    admin = Actor(type="llm_agent", id="planner", metadata={"role": "admin"})
    governed = _propose(
        repo,
        run,
        ResizeMicrobatch(target=_target(run), micro_batch_size=2),
        policy=_TrustsAdmins(),
        proposed_by=admin,
    )
    decision = governed.decision
    assert decision is not None
    assert decision.verdict is PolicyVerdict.DENY, "metadata is not policy input"
    assert decision.context.proposed_by == PolicyProposer(type="llm_agent", id="planner")
    row = repo._connection.execute("SELECT payload_json FROM policy_decisions").fetchone()
    assert json.loads(row["payload_json"])["context"]["proposed_by"] == {
        "type": "llm_agent",
        "id": "planner",
    }
    assert governed.action.proposed_by == admin, "the Action keeps the full actor, for audit"


# ---- validation comes first --------------------------------------------------------------


class _MustNotBeAsked:
    name = "must-not-be-asked"
    version = "1"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        raise AssertionError("policy was consulted about an action that does not apply")


def test_an_action_that_does_not_apply_is_rejected_before_any_policy_is_asked(
    repo: ControlPlaneRepository, run: Any
) -> None:
    governed = _propose(
        repo,
        run,
        ChangeWorkerCount(target=_target(run), workers=2),
        policy=_MustNotBeAsked(),
        capabilities=None,
    )

    assert governed.action.status is ActionStatus.REJECTED
    assert governed.decision is None and governed.action.policy_decision_id is None
    assert governed.problems == (
        "the runtime does not declare whether its worker count can change",
    )
    assert _rows(repo, "policy_decisions") == 0
    assert repo.governed_action(governed.action.id) == governed, "the problems are kept"


def test_a_target_outside_the_experiment_is_rejected(
    repo: ControlPlaneRepository, run: Any
) -> None:
    other = _node(repo)
    foreign = repo.create_run(make_run(other), actor=CONTROLLER)
    governed = _propose(
        repo,
        run,
        ResizeMicrobatch(target=ActionTarget(kind="run", id=str(foreign.id)), micro_batch_size=2),
        policy=_MustNotBeAsked(),
    )
    assert governed.problems == (
        f"run {foreign.id} does not exist in experiment {run.experiment_id}",
    )


@pytest.mark.parametrize(
    "spec",
    [
        CancelRun(target=ActionTarget(kind="run", id="run_x")),
        CancelExperiment(target=ActionTarget(kind="experiment", id="exp_x")),
    ],
    ids=lambda s: s.type,
)
def test_cancellation_is_never_proposed_to_policy(
    repo: ControlPlaneRepository, run: Any, spec: ActionSpec
) -> None:
    with pytest.raises(CancellationNotGovernedError, match="cancellation API"):
        _propose(repo, run, spec, policy=_MustNotBeAsked())
    assert _rows(repo, "actions") == 0


# ---- one commit, one snapshot ------------------------------------------------------------


def test_an_action_and_its_decision_are_written_together_or_not_at_all(
    repo: ControlPlaneRepository, run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(decision: Any) -> None:
        raise RuntimeError("the disk filled up")

    monkeypatch.setattr(repo.policy, "_insert", fail)
    with pytest.raises(RuntimeError, match="disk"):
        _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2))

    assert (_rows(repo, "actions"), _rows(repo, "policy_decisions")) == (0, 0)
    assert repo.events.events_for_experiment(str(run.experiment_id))[-1].event_type != (
        "ActionProposed"
    )


class _WhileItThinks:
    """A policy slow enough that the run ends while it deliberates."""

    name, version = POLICY.name, POLICY.version

    def __init__(self, repo: ControlPlaneRepository, run: Any) -> None:
        self.repo, self.run = repo, run

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        current = self.repo.aggregates.load_run(str(self.run.id))
        self.repo.transition_run(
            current.id,
            expected_revision=current.revision,
            new_status=RunStatus.SUCCEEDED,
            actor=CONTROLLER,
        )
        return POLICY.evaluate(spec, context)


def test_a_decision_about_a_state_that_has_since_changed_is_not_recorded(
    repo: ControlPlaneRepository, run: Any
) -> None:
    with pytest.raises(StalePolicyContextError, match="nothing was written"):
        _propose(
            repo,
            run,
            ResizeMicrobatch(target=_target(run), micro_batch_size=2),
            policy=_WhileItThinks(repo, run),
        )
    assert (_rows(repo, "actions"), _rows(repo, "policy_decisions")) == (0, 0)


class _Misremembers:
    name, version = POLICY.name, POLICY.version

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        honest = POLICY.evaluate(spec, context)
        return honest.model_copy(update={"input_fingerprint": "sha256:" + "0" * 64})


def test_an_engine_that_claims_other_inputs_is_refused(
    repo: ControlPlaneRepository, run: Any
) -> None:
    with pytest.raises(ProvenanceError, match="claims input"):
        _propose(
            repo,
            run,
            ResizeMicrobatch(target=_target(run), micro_batch_size=2),
            policy=_Misremembers(),
        )
    assert _rows(repo, "actions") == 0


class _LaterCapabilities(CapabilityDocument):
    topology: str | None = None


class _LaterElasticity(ElasticityCapabilities):
    rebalance_seconds: int | None = None


class _Probes:
    """Records what it was shown; allows only if a field v1 ignores leaked through."""

    name, version = POLICY.name, POLICY.version

    def __init__(self) -> None:
        self.seen: PolicyContext | None = None

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        self.seen = context
        caps = context.capabilities
        leaked = getattr(caps, "topology", None) or getattr(
            getattr(caps, "elasticity", None), "rebalance_seconds", None
        )
        verdict = PolicyVerdict.ALLOW if leaked else PolicyVerdict.DENY
        return POLICY.evaluate(spec, context).model_copy(update={"verdict": verdict})


def test_a_runtime_declaring_more_than_v1_shows_policy_only_v1(
    repo: ControlPlaneRepository, run: Any
) -> None:
    newer = _LaterCapabilities(
        elasticity=_LaterElasticity(supported=True, max_workers=4, rebalance_seconds=30),
        topology="ring",
        extensions={"rack": "b"},
    )
    exact = CapabilityDocument(
        elasticity=ElasticityCapabilities(supported=True, max_workers=4),
        extensions={"rack": "b"},
    )
    probe = _Probes()
    governed = _propose(
        repo,
        run,
        ChangeWorkerCount(target=_target(run), workers=2),
        policy=probe,
        capabilities=newer,
    )

    assert probe.seen is not None and probe.seen.capabilities == exact
    assert type(probe.seen.capabilities.elasticity) is ElasticityCapabilities  # type: ignore[union-attr]
    decision = governed.decision
    assert decision is not None and decision.verdict is PolicyVerdict.DENY
    assert decision.context.capabilities == exact, "the record is what policy saw"
    stored = json.loads(
        repo._connection.execute("SELECT payload_json FROM policy_decisions").fetchone()[0]
    )["context"]["capabilities"]
    assert "topology" not in stored and "rebalance_seconds" not in stored["elasticity"]
    assert stored["extensions"] == {"rack": "b"}, "extensions are how a runtime tells policy more"


class _Impersonates:
    """Honest inputs, forged signature: claims to be the trusted rule engine."""

    name = "impersonates"
    version = "1"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        return POLICY.evaluate(spec, context)


class _BumpsItsVersion:
    name = POLICY.name
    version = "2"

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        return POLICY.evaluate(spec, context)


@pytest.mark.parametrize("policy", [_Impersonates(), _BumpsItsVersion()], ids=["name", "version"])
def test_an_engine_that_signs_as_another_is_refused_and_nothing_is_written(
    repo: ControlPlaneRepository, run: Any, policy: Any
) -> None:
    events = len(repo.events.events_for_experiment(str(run.experiment_id)))
    with pytest.raises(ProvenanceError, match="returned a proposal signed rules"):
        _propose(
            repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2), policy=policy
        )
    assert (_rows(repo, "actions"), _rows(repo, "policy_decisions")) == (0, 0)
    assert len(repo.events.events_for_experiment(str(run.experiment_id))) == events


class _Counting:
    name, version = POLICY.name, POLICY.version

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, spec: ActionSpec, context: PolicyContext) -> PolicyProposal:
        self.calls += 1
        return POLICY.evaluate(spec, context)


def test_proposing_the_same_action_again_returns_it_without_judging_again(
    repo: ControlPlaneRepository, run: Any
) -> None:
    engine = _Counting()
    action_id = ActionId.generate()
    spec = ResizeMicrobatch(target=_target(run), micro_batch_size=2)

    first = _propose(repo, run, spec, policy=engine, action_id=action_id)
    again = _propose(repo, run, spec, policy=engine, action_id=action_id)

    assert again == first and engine.calls == 1
    assert _rows(repo, "policy_decisions") == 1
    with pytest.raises(IdempotencyConflictError):
        _propose(repo, run, spec, policy=engine, action_id=action_id, reason="another reason")


# ---- approval ----------------------------------------------------------------------------


@pytest.fixture
def pending(repo: ControlPlaneRepository, run: Any) -> Any:
    return _propose(repo, run, ChangeLearningRate(target=_target(run), learning_rate=1e-5)).action


def test_a_human_approves_the_recorded_proposal(repo: ControlPlaneRepository, pending: Any) -> None:
    decision = repo.policy.for_action(str(pending.id))

    approved = repo.approve_action(pending.id, approver=ANA, reason="checked the curves")

    assert approved.status is ActionStatus.APPROVED
    assert repo.policy.for_action(str(pending.id)) == decision, "approval judges nothing again"
    (event,) = [
        e
        for e in repo.events.events_for_aggregate(str(pending.id))
        if e.event_type == "ActionApproved"
    ]
    assert (event.actor, event.payload["reason"]) == (ANA, "checked the curves")


def test_approving_again_the_same_way_writes_nothing_and_any_other_answer_is_refused(
    repo: ControlPlaneRepository, pending: Any
) -> None:
    repo.approve_action(pending.id, approver=ANA, reason="checked the curves")
    events = len(repo.events.events_for_aggregate(str(pending.id)))

    again = repo.approve_action(pending.id, approver=ANA, reason="checked the curves")

    assert again.status is ActionStatus.APPROVED
    assert len(repo.events.events_for_aggregate(str(pending.id))) == events
    for change in (
        {"approver": Actor(type="human", id="bo"), "reason": "checked the curves"},
        {"approver": ANA, "reason": "a different reason"},
    ):
        with pytest.raises(ApprovalConflictError, match="approved by human ana"):
            repo.approve_action(pending.id, **change)
    with pytest.raises(ApprovalConflictError):
        repo.reject_action(pending.id, approver=ANA, reason="checked the curves")


def test_the_same_human_answering_the_same_way_is_the_same_answer_whatever_the_metadata(
    repo: ControlPlaneRepository, pending: Any
) -> None:
    first = Actor(type="human", id="ana", metadata={"session": 1})
    approved = repo.approve_action(pending.id, approver=first, reason="looks good")
    events = len(repo.events.events_for_aggregate(str(pending.id)))

    later = Actor(type="human", id="ana", metadata={"session": 2})
    again = repo.approve_action(pending.id, approver=later, reason="looks good")

    assert again == approved
    assert len(repo.events.events_for_aggregate(str(pending.id))) == events
    (event,) = [
        e
        for e in repo.events.events_for_aggregate(str(pending.id))
        if e.event_type == "ActionApproved"
    ]
    assert event.actor == first, "the original answer keeps its provenance"
    with pytest.raises(ApprovalConflictError):
        repo.reject_action(pending.id, approver=later, reason="looks good")


def test_a_human_may_reject_what_awaits_approval(
    repo: ControlPlaneRepository, pending: Any
) -> None:
    rejected = repo.reject_action(pending.id, approver=ANA, reason="too aggressive")
    assert rejected.status is ActionStatus.REJECTED
    assert repo.reject_action(pending.id, approver=ANA, reason="too aggressive") == rejected
    with pytest.raises(ApprovalConflictError):
        repo.approve_action(pending.id, approver=ANA, reason="changed my mind")


@pytest.mark.parametrize("who", ["llm_agent", "rule", "system", "search_provider"])
def test_only_a_human_approves_or_rejects(
    repo: ControlPlaneRepository, pending: Any, who: str
) -> None:
    impostor = Actor(type=who, id="x")  # type: ignore[arg-type]
    with pytest.raises(ApprovalError, match="only a human"):
        repo.approve_action(pending.id, approver=impostor, reason="fine")
    with pytest.raises(ApprovalError, match="only a human"):
        repo.reject_action(pending.id, approver=impostor, reason="no")
    assert repo.actions.get(str(pending.id)).status is ActionStatus.APPROVAL_PENDING  # type: ignore[union-attr]


def test_nothing_but_an_action_awaiting_approval_can_be_approved(
    repo: ControlPlaneRepository, run: Any
) -> None:
    allowed = _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2)).action
    with pytest.raises(ApprovalError, match="not awaiting approval"):
        repo.approve_action(allowed.id, approver=ANA, reason="sure")
    with pytest.raises(ApprovalError, match="says why"):
        repo.approve_action(allowed.id, approver=ANA, reason="  ")


# ---- the record --------------------------------------------------------------------------


def test_the_database_keeps_policy_decisions_append_only(
    repo: ControlPlaneRepository, run: Any
) -> None:
    _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2))
    for statement in (
        "UPDATE policy_decisions SET verdict = 'deny'",
        "DELETE FROM policy_decisions",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            with write_transaction(repo._connection):
                repo._connection.execute(statement)


def test_a_policy_decision_belongs_to_its_actions_experiment(
    repo: ControlPlaneRepository, run: Any
) -> None:
    governed = _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2))
    other = _node(repo)
    assert governed.decision is not None
    forged = governed.decision.model_copy(
        update={"id": "policy_01J9ZQ0000000000000000FRGD", "experiment_id": other.experiment_id}
    )
    with pytest.raises(sqlite3.IntegrityError, match="does not belong"):
        with write_transaction(repo._connection):
            repo.policy._insert(forged)


def test_a_candidate_judgement_is_governed_like_any_other_action(
    repo: ControlPlaneRepository, run: Any
) -> None:
    node = ActionTarget(kind="node", id=str(run.node_id))
    governed = _propose(repo, run, RejectCandidate(target=node))
    assert governed.action.status is ActionStatus.REJECTED, "no rule matches: the default denies"
    assert governed.decision is not None and governed.decision.rule_ids == ("default",)


def test_a_governed_action_at_rest_is_not_work_in_flight(
    repo: ControlPlaneRepository, run: Any
) -> None:
    """Waiting for a person, or for an executor, is not a controller working."""
    allowed = _propose(repo, run, ResizeMicrobatch(target=_target(run), micro_batch_size=2))
    pending = _propose(repo, run, ChangeLearningRate(target=_target(run), learning_rate=1e-5))
    approved = _propose(repo, run, ChangeLearningRate(target=_target(run), learning_rate=2e-5))
    repo.approve_action(approved.action.id, approver=ANA, reason="fine")

    operations, actions = repo.unsettled_work(str(run.experiment_id))

    assert (operations, actions) == ((), ())
    statuses = {
        repo.actions.get(str(g.action.id)).status  # type: ignore[union-attr]
        for g in (allowed, pending, approved)
    }
    assert statuses == {
        ActionStatus.VALIDATED,
        ActionStatus.APPROVAL_PENDING,
        ActionStatus.APPROVED,
    }
    awaiting_approval, awaiting_execution = repo.resting_actions(str(run.experiment_id))
    assert [a.id for a in awaiting_approval] == [pending.action.id]
    assert [a.id for a in awaiting_execution] == [allowed.action.id, approved.action.id]
