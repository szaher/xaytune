"""Transition tables: every legal edge allowed, everything else rejected."""

from __future__ import annotations

import itertools

import pytest

from xaytune.core.errors import InvalidTransitionError
from xaytune.core.state import (
    ATTEMPT_MACHINE,
    EXPERIMENT_MACHINE,
    NODE_MACHINE,
    RUN_MACHINE,
    ExperimentNodeStatus,
    ExperimentStatus,
    RunAttemptStatus,
    RunStatus,
    StateMachine,
)

# The edges drawn in the architecture specification, transcribed independently
# of the implementation so that a change to either side shows up as a failure.
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

NODE_EDGES = {
    (ExperimentNodeStatus.CREATED, ExperimentNodeStatus.PLANNED),
    (ExperimentNodeStatus.PLANNED, ExperimentNodeStatus.READY),
    (ExperimentNodeStatus.READY, ExperimentNodeStatus.ACTIVE),
    (ExperimentNodeStatus.ACTIVE, ExperimentNodeStatus.EVALUATING),
    (ExperimentNodeStatus.EVALUATING, ExperimentNodeStatus.DECIDING),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.ACTIVE),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.COMPLETED),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.REJECTED),
    (ExperimentNodeStatus.DECIDING, ExperimentNodeStatus.FAILED),
}

RUN_EDGES = {
    (RunStatus.CREATED, RunStatus.ACTIVE),
    (RunStatus.ACTIVE, RunStatus.SUCCEEDED),
    (RunStatus.ACTIVE, RunStatus.FAILED),
    (RunStatus.ACTIVE, RunStatus.CANCELLED),
}

ATTEMPT_EDGES = {
    (RunAttemptStatus.CREATED, RunAttemptStatus.QUEUED),
    (RunAttemptStatus.QUEUED, RunAttemptStatus.STARTING),
    (RunAttemptStatus.STARTING, RunAttemptStatus.RUNNING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.CHECKPOINTING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.RECOVERING),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.SUCCEEDED),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.FAILED),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.PREEMPTED),
    (RunAttemptStatus.RUNNING, RunAttemptStatus.CANCELLED),
    (RunAttemptStatus.CHECKPOINTING, RunAttemptStatus.RUNNING),
    (RunAttemptStatus.RECOVERING, RunAttemptStatus.RUNNING),
}

MACHINES = [
    pytest.param(EXPERIMENT_MACHINE, ExperimentStatus, EXPERIMENT_EDGES, id="experiment"),
    pytest.param(NODE_MACHINE, ExperimentNodeStatus, NODE_EDGES, id="node"),
    pytest.param(RUN_MACHINE, RunStatus, RUN_EDGES, id="run"),
    pytest.param(ATTEMPT_MACHINE, RunAttemptStatus, ATTEMPT_EDGES, id="attempt"),
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
