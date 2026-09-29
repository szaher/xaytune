"""Pure PR-020 OOM contract tests against real recorded episode conclusions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.test_storage.test_recovery_episodes import decide, incident, signal
from xaytune.core.domain.action import ActionTarget
from xaytune.core.domain.actions.builtin import ResizeMicrobatch
from xaytune.core.domain.oom_recovery import (
    OOMEscalation,
    OOMEscalationCode,
    OOMRecoveryInputsV1,
    OOMResizeProposal,
    PriorOOMResize,
)
from xaytune.core.domain.recovery import RecoveryRequest
from xaytune.core.ids import ActionId, RunAttemptId, RunId
from xaytune.resilience.oom import OOMRecoveryPlanner
from xaytune.storage import ControlPlaneRepository


@pytest.fixture
def oom_inputs(connection, seeded):
    repo = ControlPlaneRepository(connection)
    observed = incident(repo, seeded["attempt"], signal=signal("cuda-oom"))
    plan = decide(repo, observed, RecoveryRequest())
    inputs = OOMRecoveryInputsV1(
        plan=plan,
        run_id=RunId.validate(plan.inputs.context.run_id),
        candidate_fingerprint=plan.inputs.candidate_fingerprint,
        execution_state_fingerprint=plan.execution_state_fingerprint,
        current_micro_batch_size=4,
        current_gradient_accumulation=8,
        world_size=8,
    )
    return repo, inputs


def test_oom_planner_proposes_one_atomic_action_with_preserved_effective_batch(oom_inputs):
    repo, inputs = oom_inputs
    before = {
        table: repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("actions", "run_attempts", "runtime_operations")
    }
    proposal = OOMRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, OOMResizeProposal)
    assert proposal.action_spec == ResizeMicrobatch(
        target=ActionTarget(kind="run", id=str(inputs.run_id)),
        micro_batch_size=2,
        gradient_accumulation=16,
    )
    assert proposal.effective_batch_size == 256
    assert proposal.preserves == ("effective_batch_size",)
    assert proposal.episode_id == inputs.plan.episode_id
    assert proposal.plan_id == inputs.plan.id
    assert proposal.plan_sequence == inputs.plan.sequence
    assert proposal.input_fingerprint == inputs.input_fingerprint
    assert OOMResizeProposal.model_validate_json(proposal.model_dump_json()) == proposal
    assert OOMRecoveryPlanner().plan(inputs) == proposal
    assert before == {
        table: repo._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in before
    }


@pytest.mark.parametrize(
    "micro,accum,minimum,maximum,expected_micro,expected_accum",
    [(8, 4, 1, None, 4, 8), (4, 8, 1, None, 2, 16), (2, 16, 1, None, 1, 32), (5, 6, 3, 10, 3, 10)],
)
def test_halving_and_floor_are_deterministic(
    oom_inputs, micro, accum, minimum, maximum, expected_micro, expected_accum
):
    _, inputs = oom_inputs
    configured = inputs.model_copy(
        update={
            "current_micro_batch_size": micro,
            "current_gradient_accumulation": accum,
            "min_micro_batch_size": minimum,
            "max_gradient_accumulation": maximum,
        }
    )
    proposal = OOMRecoveryPlanner().plan(configured)
    assert isinstance(proposal, OOMResizeProposal)
    assert proposal.action_spec.micro_batch_size == expected_micro
    assert proposal.action_spec.gradient_accumulation == expected_accum
    assert proposal.action_spec.micro_batch_size * expected_accum * configured.world_size == (
        configured.effective_batch_size
    )


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"current_micro_batch_size": 1}, OOMEscalationCode.MINIMUM_MICRO_BATCH_REACHED),
        ({"preserve_effective_batch": False}, OOMEscalationCode.PRESERVATION_NOT_AUTHORIZED),
        (
            {"current_micro_batch_size": 5, "current_gradient_accumulation": 1},
            OOMEscalationCode.NONINTEGRAL_ACCUMULATION,
        ),
        ({"max_gradient_accumulation": 15}, OOMEscalationCode.ACCUMULATION_LIMIT),
    ],
)
def test_constraints_escalate_without_an_unsafe_proposal(oom_inputs, updates, code):
    _, inputs = oom_inputs
    result = OOMRecoveryPlanner().plan(inputs.model_copy(update=updates))
    assert isinstance(result, OOMEscalation)
    assert result.code is code
    assert result.plan_id == inputs.plan.id


def test_prior_promised_resize_must_be_visible_in_current_execution(oom_inputs):
    _, inputs = oom_inputs
    prior = PriorOOMResize(
        action_id=ActionId.generate(),
        successor_attempt_id=RunAttemptId.validate(inputs.plan.inputs.context.target.id),
        prior_execution_state_fingerprint=inputs.execution_state_fingerprint,
        promised_micro_batch_size=2,
        promised_gradient_accumulation=16,
    )
    same_fingerprint = inputs.model_copy(update={"prior_resize": prior})
    result = OOMRecoveryPlanner().plan(same_fingerprint)
    assert isinstance(result, OOMEscalation)
    assert result.code is OOMEscalationCode.PRIOR_OVERRIDE_NOT_APPLIED
    changed = inputs.model_copy(
        update={
            "prior_resize": prior.model_copy(
                update={"prior_execution_state_fingerprint": "sha256:" + "1" * 64}
            )
        }
    )
    assert OOMRecoveryPlanner().plan(changed).code is OOMEscalationCode.PRIOR_OVERRIDE_NOT_APPLIED
    applied = changed.model_copy(
        update={
            "prior_resize": changed.prior_resize.model_copy(
                update={"promised_micro_batch_size": 4, "promised_gradient_accumulation": 8}
            )
        }
    )
    assert isinstance(OOMRecoveryPlanner().plan(applied), OOMResizeProposal)


@pytest.mark.parametrize("reason", ["process-failure", "numerical-nan", "config-error"])
def test_plan_authority_rejects_non_oom_or_conflicting_evidence(oom_inputs, reason):
    repo, inputs = oom_inputs
    second = incident(
        repo,
        repo.aggregates.load_attempt(inputs.plan.inputs.context.target.id),
        sequence=10,
        signal=signal(reason),
    )
    effective = decide(repo, second)
    if reason == "process-failure":
        updated = inputs.model_copy(update={"plan": effective})
        assert isinstance(OOMRecoveryPlanner().plan(updated), OOMResizeProposal)
    else:
        with pytest.raises(ValidationError):
            inputs.model_copy(update={"plan": effective})


def test_input_values_and_proposal_cannot_hide_nonpreservation(oom_inputs):
    _, inputs = oom_inputs
    with pytest.raises(ValidationError):
        inputs.model_copy(update={"current_micro_batch_size": True})
    with pytest.raises(ValidationError, match="paused training episode decision"):
        inputs.model_copy(
            update={"plan": inputs.plan.model_copy(update={"reason": "experiment is paused"})}
        )
    proposal = OOMRecoveryPlanner().plan(inputs)
    assert isinstance(proposal, OOMResizeProposal)
    with pytest.raises(ValidationError, match="preserve effective batch"):
        proposal.model_copy(
            update={
                "action_spec": ResizeMicrobatch(
                    target=ActionTarget(kind="run", id=str(inputs.run_id)),
                    micro_batch_size=2,
                    gradient_accumulation=8,
                )
            }
        )
