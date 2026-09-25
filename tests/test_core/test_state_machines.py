"""Transition tables: every legal edge allowed, everything else rejected."""

from __future__ import annotations

import itertools

import pytest

from xaytune.core.errors import InvalidTransitionError
from xaytune.core.state import (
    ATTEMPT_MACHINE,
    EVALUATION_ATTEMPT_MACHINE,
    EVALUATION_RUN_MACHINE,
    EXPERIMENT_MACHINE,
    NODE_MACHINE,
    RUN_MACHINE,
    EvaluationAttemptStatus,
    EvaluationRunStatus,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
    StateMachine,
)

# The agreed transition tables, transcribed independently of the implementation
# so that a change to either side shows up as a failure.
#
# Two rules run through all of them: every non-terminal state can reach FAILED,
# because anything unfinished can break; and every non-terminal state can reach
# CANCELLED, because an operator can stop work at any point.
EXPERIMENT_EDGES = {
    (ExperimentStatus.CREATED, ExperimentStatus.ACTIVE),
    (ExperimentStatus.CREATED, ExperimentStatus.CANCELLED),
    (ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED),
    (ExperimentStatus.ACTIVE, ExperimentStatus.SUCCEEDED),
    (ExperimentStatus.ACTIVE, ExperimentStatus.FAILED),
    (ExperimentStatus.ACTIVE, ExperimentStatus.CANCELLED),
    (ExperimentStatus.ACTIVE, ExperimentStatus.BUDGET_EXHAUSTED),
    (ExperimentStatus.PAUSED, ExperimentStatus.ACTIVE),
    (ExperimentStatus.PAUSED, ExperimentStatus.CANCELLED),
    (ExperimentStatus.PAUSED, ExperimentStatus.FAILED),
}

_NODE_PROGRESS = [
    (ExperimentNodeStatus.CREATED, ExperimentNodeStatus.PLANNED),
    (ExperimentNodeStatus.PLANNED, ExperimentNodeStatus.READY),
    (ExperimentNodeStatus.READY, ExperimentNodeStatus.ACTIVE),
    (ExperimentNodeStatus.ACTIVE, ExperimentNodeStatus.EVALUATING),
    (ExperimentNodeStatus.EVALUATING, ExperimentNodeStatus.DECIDING),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.ACTIVE),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.COMPLETED),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.REJECTED),
]
_NODE_NON_TERMINAL = [
    ExperimentNodeStatus.CREATED,
    ExperimentNodeStatus.PLANNED,
    ExperimentNodeStatus.READY,
    ExperimentNodeStatus.ACTIVE,
    ExperimentNodeStatus.EVALUATING,
    ExperimentNodeStatus.DECIDING,
]
NODE_EDGES = set(_NODE_PROGRESS) | {
    (state, outcome)
    for state in _NODE_NON_TERMINAL
    for outcome in (ExperimentNodeStatus.CANCELLED, ExperimentNodeStatus.FAILED)
}

RUN_EDGES = {
    (RunStatus.CREATED, RunStatus.ACTIVE),
    (RunStatus.CREATED, RunStatus.CANCELLED),
    (RunStatus.CREATED, RunStatus.FAILED),
    (RunStatus.ACTIVE, RunStatus.SUCCEEDED),
    (RunStatus.ACTIVE, RunStatus.FAILED),
    (RunStatus.ACTIVE, RunStatus.CANCELLED),
}

_ATTEMPT_PROGRESS = [
    (RunAttemptStatus.CREATED, RunAttemptStatus.QUEUED),
    (RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING),
    (RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.CHECKPOINTING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.RECOVERING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.SUCCEEDED),
    (RunAttemptStatus.CHECKPOINTING, RunAttemptStatus.RUNNING),
    (RunAttemptStatus.CHECKPOINTING, RunAttemptStatus.RECOVERING),
    (RunAttemptStatus.RECOVERING, RunAttemptStatus.RUNNING),
]
_ATTEMPT_NON_TERMINAL = [
    RunAttemptStatus.CREATED,
    RunAttemptStatus.QUEUED,
    RunAttemptStatus.STARTING,
    RunAttemptStatus.RUNNING,
    RunAttemptStatus.CHECKPOINTING,
    RunAttemptStatus.RECOVERING,
]
ATTEMPT_EDGES = (
    set(_ATTEMPT_PROGRESS)
    | {
        (state, outcome)
        for state in _ATTEMPT_NON_TERMINAL
        for outcome in (RunAttemptStatus.CANCELLED, RunAttemptStatus.FAILED)
    }
    | {
        # Preemption applies once the runtime knows about the workload, which is
        # from QUEUED onwards -- a CREATED attempt has not been submitted.
        (state, RunAttemptStatus.PREEMPTED)
        for state in _ATTEMPT_NON_TERMINAL
        if state is not RunAttemptStatus.CREATED
    }
)

# ADR-015 §1, transcribed from its tables.
EVALUATION_RUN_EDGES = {
    (EvaluationRunStatus.CREATED, EvaluationRunStatus.ACTIVE),
    (EvaluationRunStatus.CREATED, EvaluationRunStatus.CANCELLED),
    (EvaluationRunStatus.CREATED, EvaluationRunStatus.FAILED),
    (EvaluationRunStatus.ACTIVE, EvaluationRunStatus.SUCCEEDED),
    (EvaluationRunStatus.ACTIVE, EvaluationRunStatus.FAILED),
    (EvaluationRunStatus.ACTIVE, EvaluationRunStatus.CANCELLED),
}

_EVALUATION_ATTEMPT_NON_TERMINAL = [
    EvaluationAttemptStatus.CREATED,
    EvaluationAttemptStatus.QUEUED,
    EvaluationAttemptStatus.STARTING,
    EvaluationAttemptStatus.RUNNING,
]
EVALUATION_ATTEMPT_EDGES = (
    {
        (EvaluationAttemptStatus.CREATED, EvaluationAttemptStatus.QUEUED),
        (EvaluationAttemptStatus.QUEUED, EvaluationAttemptStatus.STARTING),
        (EvaluationAttemptStatus.STARTING, EvaluationAttemptStatus.RUNNING),
        (EvaluationAttemptStatus.RUNNING, EvaluationAttemptStatus.SUCCEEDED),
    }
    | {
        (state, outcome)
        for state in _EVALUATION_ATTEMPT_NON_TERMINAL
        for outcome in (EvaluationAttemptStatus.CANCELLED, EvaluationAttemptStatus.FAILED)
    }
    | {
        (state, EvaluationAttemptStatus.PREEMPTED)
        for state in _EVALUATION_ATTEMPT_NON_TERMINAL
        if state is not EvaluationAttemptStatus.CREATED
    }
)

MACHINES = [
    pytest.param(EXPERIMENT_MACHINE, ExperimentStatus, EXPERIMENT_EDGES, id="experiment"),
    pytest.param(NODE_MACHINE, ExperimentNodeStatus, NODE_EDGES, id="node"),
    pytest.param(RUN_MACHINE, RunStatus, RUN_EDGES, id="run"),
    pytest.param(ATTEMPT_MACHINE, RunAttemptStatus, ATTEMPT_EDGES, id="attempt"),
    pytest.param(
        EVALUATION_RUN_MACHINE, EvaluationRunStatus, EVALUATION_RUN_EDGES, id="evaluation-run"
    ),
    pytest.param(
        EVALUATION_ATTEMPT_MACHINE,
        EvaluationAttemptStatus,
        EVALUATION_ATTEMPT_EDGES,
        id="evaluation-attempt",
    ),
]


class TestTransitionTables:
    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_exactly_the_specified_edges_are_allowed(self, machine, status_enum, edges):
        for current, requested in itertools.product(status_enum, status_enum):
            expected = (current, requested) in edges
            assert machine.can(current, requested) is expected, (
                f"{machine.aggregate}: {current.value} -> {requested.value} "
                f"should be {'allowed' if expected else 'rejected'}"
            )

    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_illegal_transitions_raise(self, machine, status_enum, edges):
        for current, requested in itertools.product(status_enum, status_enum):
            if (current, requested) in edges:
                machine.validate(current, requested)
            else:
                with pytest.raises(InvalidTransitionError):
                    machine.validate(current, requested)

    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_no_self_transitions(self, machine, status_enum, edges):
        for state in status_enum:
            assert not machine.can(state, state)

    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_terminal_states_accept_nothing(self, machine, status_enum, edges):
        assert machine.terminal_states, f"{machine.aggregate} has no terminal state"
        for state in machine.terminal_states:
            assert machine.is_terminal(state)
            for requested in status_enum:
                with pytest.raises(InvalidTransitionError):
                    machine.validate(state, requested)

    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_every_state_is_reachable_from_the_initial_state(self, machine, status_enum, edges):
        seen = {machine.initial}
        frontier = [machine.initial]
        while frontier:
            for nxt in machine.allowed_from(frontier.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        assert seen == set(status_enum), (
            f"{machine.aggregate} unreachable states: "
            f"{sorted(s.value for s in set(status_enum) - seen)}"
        )

    @pytest.mark.parametrize(("machine", "status_enum", "edges"), MACHINES)
    def test_initial_state_is_not_terminal(self, machine, status_enum, edges):
        assert not machine.is_terminal(machine.initial)


class TestFailureAndCancellationCoverage:
    """The gaps that the spec-literal tables left open.

    Each of these was rejected before: a node could not fail while ACTIVE, an
    attempt could not fail while STARTING, CHECKPOINTING or RECOVERING, and
    nothing could be cancelled before it started.
    """

    @pytest.mark.parametrize(
        "state",
        [
            ExperimentNodeStatus.CREATED,
            ExperimentNodeStatus.PLANNED,
            ExperimentNodeStatus.READY,
            ExperimentNodeStatus.ACTIVE,
            ExperimentNodeStatus.EVALUATING,
            ExperimentNodeStatus.DECIDING,
        ],
    )
    def test_a_node_can_fail_or_be_cancelled_from_any_live_state(self, state):
        NODE_MACHINE.validate(state, ExperimentNodeStatus.FAILED)
        NODE_MACHINE.validate(state, ExperimentNodeStatus.CANCELLED)

    @pytest.mark.parametrize(
        "state",
        [
            RunAttemptStatus.CREATED,
            RunAttemptStatus.QUEUED,
            RunAttemptStatus.STARTING,
            RunAttemptStatus.RUNNING,
            RunAttemptStatus.CHECKPOINTING,
            RunAttemptStatus.RECOVERING,
        ],
    )
    def test_an_attempt_can_fail_or_be_cancelled_from_any_live_state(self, state):
        ATTEMPT_MACHINE.validate(state, RunAttemptStatus.FAILED)
        ATTEMPT_MACHINE.validate(state, RunAttemptStatus.CANCELLED)

    @pytest.mark.parametrize(
        "state",
        [
            RunAttemptStatus.QUEUED,
            RunAttemptStatus.STARTING,
            RunAttemptStatus.RUNNING,
            RunAttemptStatus.CHECKPOINTING,
            RunAttemptStatus.RECOVERING,
        ],
    )
    def test_an_attempt_can_be_preempted_once_the_runtime_knows_about_it(self, state):
        ATTEMPT_MACHINE.validate(state, RunAttemptStatus.PREEMPTED)

    def test_a_created_attempt_cannot_be_preempted(self):
        """Nothing has been submitted yet, so there is nothing to reclaim."""
        with pytest.raises(InvalidTransitionError):
            ATTEMPT_MACHINE.validate(RunAttemptStatus.CREATED, RunAttemptStatus.PREEMPTED)

    def test_a_checkpoint_write_can_fail(self):
        ATTEMPT_MACHINE.validate(RunAttemptStatus.CHECKPOINTING, RunAttemptStatus.FAILED)

    def test_recovery_can_itself_fail(self):
        ATTEMPT_MACHINE.validate(RunAttemptStatus.RECOVERING, RunAttemptStatus.FAILED)

    def test_a_run_can_be_cancelled_before_it_starts(self):
        RUN_MACHINE.validate(RunStatus.CREATED, RunStatus.CANCELLED)

    def test_rejected_and_cancelled_are_distinct_outcomes(self):
        """Rejected is a judgement on merit; cancelled is work stopped early."""
        assert ExperimentNodeStatus.REJECTED is not ExperimentNodeStatus.CANCELLED
        assert NODE_MACHINE.is_terminal(ExperimentNodeStatus.REJECTED)
        assert NODE_MACHINE.is_terminal(ExperimentNodeStatus.CANCELLED)
        # Only a decision can reject; anything live can be cancelled.
        rejecting = [
            s for s in ExperimentNodeStatus if NODE_MACHINE.can(s, ExperimentNodeStatus.REJECTED)
        ]
        assert rejecting == [ExperimentNodeStatus.DECIDING]


class TestErrorMessage:
    def test_names_the_aggregate_and_both_states(self):
        with pytest.raises(InvalidTransitionError) as excinfo:
            EXPERIMENT_MACHINE.validate(ExperimentStatus.SUCCEEDED, ExperimentStatus.ACTIVE)
        message = str(excinfo.value)
        assert "Experiment" in message
        assert "succeeded" in message
        assert "active" in message

    def test_carries_structured_fields(self):
        with pytest.raises(InvalidTransitionError) as excinfo:
            RUN_MACHINE.validate(RunStatus.FAILED, RunStatus.ACTIVE)
        assert excinfo.value.aggregate == "Run"
        assert excinfo.value.current is RunStatus.FAILED
        assert excinfo.value.requested is RunStatus.ACTIVE


class TestTableConstruction:
    def test_incomplete_table_is_rejected(self):
        """A forgotten state must fail loudly at import time, not at runtime."""
        with pytest.raises(ValueError, match="missing states"):
            StateMachine(
                "Partial",
                RunStatus.CREATED,
                {RunStatus.CREATED: {RunStatus.ACTIVE}},
            )

    def test_foreign_target_is_rejected(self):
        with pytest.raises(ValueError, match="foreign targets"):
            StateMachine(
                "Mixed",
                RunStatus.CREATED,
                {
                    RunStatus.CREATED: {ExperimentStatus.ACTIVE},
                    RunStatus.ACTIVE: set(),
                    RunStatus.SUCCEEDED: set(),
                    RunStatus.FAILED: set(),
                    RunStatus.CANCELLED: set(),
                },
            )


class TestEvaluationMachines:
    """ADR-015 AC-1 and AC-2: evaluation's own states, not training's."""

    def test_evaluation_attempts_have_no_checkpointing_or_recovering(self):
        names = {status.name for status in EvaluationAttemptStatus}
        assert "CHECKPOINTING" not in names
        assert "RECOVERING" not in names

    def test_preemption_applies_from_queued_onwards_only(self):
        preemptable = {
            state
            for state in EvaluationAttemptStatus
            if EVALUATION_ATTEMPT_MACHINE.can(state, EvaluationAttemptStatus.PREEMPTED)
        }
        assert preemptable == {
            EvaluationAttemptStatus.QUEUED,
            EvaluationAttemptStatus.STARTING,
            EvaluationAttemptStatus.RUNNING,
        }
