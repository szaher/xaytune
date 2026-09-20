"""Domain aggregates: serialization, immutability, and guarded transitions."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from xaytune.core import (
    Actor,
    ArtifactId,
    ArtifactRef,
    BudgetSpec,
    CheckpointId,
    CheckpointRef,
    ControllerHostRef,
    DatasetRef,
    ExecutionOverride,
    Experiment,
    ExperimentId,
    ExperimentNode,
    ExperimentNodeId,
    ExperimentNodeStatus,
    ExperimentStatus,
    IncidentId,
    MetricConstraint,
    ModelRef,
    Objective,
    ObjectiveMetric,
    ResourceUsage,
    Run,
    RunAttempt,
    RunAttemptId,
    RunAttemptStatus,
    RunId,
    RunStatus,
    TrainingSpecSnapshot,
)
from xaytune.core.errors import InvalidTransitionError


def make_experiment(**overrides) -> Experiment:
    defaults = dict(
        id=ExperimentId.generate(),
        name="support-sft",
        objective=Objective(
            primary=ObjectiveMetric(name="task_success", direction="maximize"),
            target=0.85,
            constraints=[MetricConstraint(name="latency_ms", operator="<=", value=120)],
        ),
        budget=BudgetSpec(max_runs=8, max_gpu_hours=20.0, max_cost=Decimal("125.50")),
        controller_host=ControllerHostRef(kind="embedded"),
    )
    defaults.update(overrides)
    return Experiment(**defaults)


def make_node(experiment_id: ExperimentId | None = None, **overrides) -> ExperimentNode:
    defaults = dict(
        id=ExperimentNodeId.generate(),
        experiment_id=experiment_id or ExperimentId.generate(),
        training_spec=TrainingSpecSnapshot(
            kind="sft",
            model=ModelRef(uri="Qwen/Qwen3-8B", revision="main"),
            dataset=DatasetRef(uri="support-v4", revision="2026-01-01"),
            payload={"learning_rate": 2e-5},
        ),
        training_fingerprint="sha256:abc",
        created_by=Actor(type="rule", id="plateau-v1"),
    )
    defaults.update(overrides)
    return ExperimentNode(**defaults)


def make_run(**overrides) -> Run:
    defaults = dict(
        id=RunId.generate(),
        node_id=ExperimentNodeId.generate(),
        experiment_id=ExperimentId.generate(),
        training_fingerprint="sha256:abc",
        seed=42,
    )
    defaults.update(overrides)
    return Run(**defaults)


def make_attempt(**overrides) -> RunAttempt:
    defaults = dict(
        id=RunAttemptId.generate(),
        run_id=RunId.generate(),
        attempt_number=1,
    )
    defaults.update(overrides)
    return RunAttempt(**defaults)


ALL_FACTORIES = [
    pytest.param(make_experiment, Experiment, id="experiment"),
    pytest.param(make_node, ExperimentNode, id="node"),
    pytest.param(make_run, Run, id="run"),
    pytest.param(make_attempt, RunAttempt, id="attempt"),
]


class TestSerialization:
    @pytest.mark.parametrize(("factory", "model_type"), ALL_FACTORIES)
    def test_json_round_trip(self, factory, model_type):
        original = factory()
        restored = model_type.model_validate_json(original.model_dump_json())
        assert restored == original

    @pytest.mark.parametrize(("factory", "model_type"), ALL_FACTORIES)
    def test_unknown_fields_are_rejected(self, factory, model_type):
        """Silently dropping a field would lose provenance."""
        payload = factory().model_dump(mode="json")
        payload["not_a_real_field"] = 1
        with pytest.raises(ValidationError):
            model_type.model_validate(payload)

    def test_ids_survive_the_round_trip_as_typed_ids(self):
        node = make_node()
        restored = ExperimentNode.model_validate_json(node.model_dump_json())
        assert isinstance(restored.id, ExperimentNodeId)
        assert isinstance(restored.experiment_id, ExperimentId)

    def test_decimal_budget_survives_the_round_trip(self):
        experiment = make_experiment()
        restored = Experiment.model_validate_json(experiment.model_dump_json())
        assert restored.budget is not None
        assert restored.budget.max_cost == Decimal("125.50")

    def test_nested_refs_round_trip(self):
        attempt = make_attempt(
            artifact_refs=[ArtifactRef(id=ArtifactId.generate(), kind="adapter", uri="s3://a")],
            checkpoint_ref=CheckpointRef(
                id=CheckpointId.generate(), uri="s3://ckpt", global_step=100
            ),
            resource_usage=ResourceUsage(gpu_hours=1.5, cost=Decimal("3.20")),
        )
        restored = RunAttempt.model_validate_json(attempt.model_dump_json())
        assert restored == attempt


class TestImmutability:
    @pytest.mark.parametrize(("factory", "model_type"), ALL_FACTORIES)
    def test_status_cannot_be_assigned_directly(self, factory, model_type):
        """Rule 7: transitions go through the transition API, never assignment."""
        aggregate = factory()
        with pytest.raises(ValidationError):
            aggregate.status = aggregate.status

    def test_training_spec_snapshot_is_frozen(self):
        """Rule 5: a scientific change creates a child node, never an edit."""
        node = make_node()
        with pytest.raises(ValidationError):
            node.training_spec.payload = {}
        with pytest.raises(ValidationError):
            node.training_fingerprint = "sha256:other"


class TestExperimentTransitions:
    def test_valid_transition_bumps_revision_and_timestamp(self):
        experiment = make_experiment()
        active = experiment.with_status(ExperimentStatus.ACTIVE)

        assert active.status is ExperimentStatus.ACTIVE
        assert active.revision == experiment.revision + 1
        assert active.updated_at >= experiment.updated_at

    def test_original_is_untouched(self):
        experiment = make_experiment()
        experiment.with_status(ExperimentStatus.ACTIVE)
        assert experiment.status is ExperimentStatus.CREATED
        assert experiment.revision == 0

    def test_invalid_transition_raises(self):
        experiment = make_experiment()
        with pytest.raises(InvalidTransitionError):
            experiment.with_status(ExperimentStatus.SUCCEEDED)

    def test_pause_and_resume(self):
        experiment = make_experiment().with_status(ExperimentStatus.ACTIVE)
        paused = experiment.with_status(ExperimentStatus.PAUSED)
        resumed = paused.with_status(ExperimentStatus.ACTIVE)
        assert resumed.status is ExperimentStatus.ACTIVE
        assert resumed.revision == 3

    def test_terminal_state_is_reported(self):
        experiment = make_experiment().with_status(ExperimentStatus.ACTIVE)
        assert not experiment.is_terminal
        assert experiment.with_status(ExperimentStatus.SUCCEEDED).is_terminal


class TestNodeLineage:
    def test_node_without_parents_is_a_root(self):
        assert make_node().is_root

    def test_node_with_parents_is_not_a_root(self):
        parent = make_node()
        child = make_node(parent_ids=[parent.id])
        assert not child.is_root
        assert child.parent_ids == [parent.id]

    def test_node_supports_multiple_parents(self):
        first, second = make_node(), make_node()
        merged = make_node(parent_ids=[first.id, second.id])
        assert len(merged.parent_ids) == 2

    def test_full_happy_path(self):
        node = make_node()
        for status in (
            ExperimentNodeStatus.PLANNED,
            ExperimentNodeStatus.READY,
            ExperimentNodeStatus.ACTIVE,
            ExperimentNodeStatus.EVALUATING,
            ExperimentNodeStatus.DECIDING,
            ExperimentNodeStatus.COMPLETED,
        ):
            node = node.with_status(status)
        assert node.is_terminal
        assert node.revision == 6

    def test_deciding_can_send_the_node_back_to_active(self):
        """A decision may ask for another run or replicate."""
        node = make_node()
        for status in (
            ExperimentNodeStatus.PLANNED,
            ExperimentNodeStatus.READY,
            ExperimentNodeStatus.ACTIVE,
            ExperimentNodeStatus.EVALUATING,
            ExperimentNodeStatus.DECIDING,
            ExperimentNodeStatus.ACTIVE,
        ):
            node = node.with_status(status)
        assert node.status is ExperimentNodeStatus.ACTIVE


class TestRunAndAttempt:
    def test_run_happy_path(self):
        run = make_run().with_status(RunStatus.ACTIVE)
        assert run.with_status(RunStatus.SUCCEEDED).is_terminal

    def test_attempt_stamps_started_at_on_running(self):
        attempt = make_attempt()
        assert attempt.started_at is None
        running = (
            attempt.with_status(RunAttemptStatus.QUEUED)
            .with_status(RunAttemptStatus.STARTING)
            .with_status(RunAttemptStatus.RUNNING)
        )
        assert running.started_at is not None
        assert running.ended_at is None

    def test_attempt_stamps_ended_at_on_terminal(self):
        running = (
            make_attempt()
            .with_status(RunAttemptStatus.QUEUED)
            .with_status(RunAttemptStatus.STARTING)
            .with_status(RunAttemptStatus.RUNNING)
        )
        failed = running.with_status(RunAttemptStatus.FAILED)
        assert failed.ended_at is not None
        assert failed.is_terminal

    def test_checkpoint_round_trip_preserves_start_time(self):
        """CHECKPOINTING -> RUNNING must not reset the attempt's start time."""
        running = (
            make_attempt()
            .with_status(RunAttemptStatus.QUEUED)
            .with_status(RunAttemptStatus.STARTING)
            .with_status(RunAttemptStatus.RUNNING)
        )
        resumed = running.with_status(RunAttemptStatus.CHECKPOINTING).with_status(
            RunAttemptStatus.RUNNING
        )
        assert resumed.started_at == running.started_at

    def test_attempt_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            make_attempt(attempt_number=0)

    def test_recovery_stays_on_the_same_run(self):
        """ADR-003: operational recovery never branches the scientific graph."""
        run = make_run().with_status(RunStatus.ACTIVE)
        first = make_attempt(run_id=run.id, attempt_number=1)
        second = make_attempt(run_id=run.id, attempt_number=2)
        assert first.run_id == second.run_id == run.id


class TestExecutionOverride:
    def test_records_what_it_preserves(self):
        override = ExecutionOverride(
            id="ovr-1",
            kind="micro_batch_resize",
            reason="CUDA OOM at step 400",
            values={"micro_batch_size": 2, "gradient_accumulation": 4},
            preserves=["effective_batch_size"],
            incident_id=IncidentId.generate(),
        )
        attempt = make_attempt(execution_overrides=[override])
        restored = RunAttempt.model_validate_json(attempt.model_dump_json())
        assert restored.execution_overrides[0].preserves == ["effective_batch_size"]

    def test_rejects_a_scientific_change_as_an_override_kind(self):
        """LR and optimizer changes are new nodes, not execution overrides."""
        with pytest.raises(ValidationError):
            ExecutionOverride(
                id="ovr-2",
                kind="learning_rate_change",
                reason="loss plateau",
            )

    def test_is_frozen(self):
        override = ExecutionOverride(id="ovr-3", kind="checkpoint_restore", reason="retry")
        with pytest.raises(ValidationError):
            override.reason = "something else"
