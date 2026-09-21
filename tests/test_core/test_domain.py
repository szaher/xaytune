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
    CandidateSpecSnapshot,
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
)
from xaytune.core.domain.candidate import (
    CandidateSpec,
    DataSpec,
    ModelSpec,
    TrainingKind,
    TrainingSpec,
)
from xaytune.core.errors import InvalidDomainValueError, InvalidTransitionError
from xaytune.core.immutable import FrozenDict, deep_freeze, thaw


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
        candidate=CandidateSpecSnapshot(
            candidate=CandidateSpec(
                model=ModelSpec(model=ModelRef(uri="Qwen/Qwen3-8B")),
                data=DataSpec(dataset=DatasetRef(uri="./data.jsonl")),
                training=TrainingSpec(
                    kind=TrainingKind.SFT,
                    metadata={"learning_rate": 2e-5},
                ),
            )
        ),
        candidate_fingerprint="sha256:abc",
        created_by=Actor(type="rule", id="plateau-v1"),
    )
    defaults.update(overrides)
    return ExperimentNode(**defaults)


def make_run(**overrides) -> Run:
    defaults = dict(
        id=RunId.generate(),
        node_id=ExperimentNodeId.generate(),
        experiment_id=ExperimentId.generate(),
        candidate_fingerprint="sha256:abc",
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


class TestDeepImmutability:
    """Frozen must mean frozen all the way down.

    Pydantic's ``frozen=True`` only blocks attribute assignment. Container
    fields stayed mutable and kept a reference to whatever the caller passed,
    so a scientific record could be changed after construction from either
    side -- which would make any fingerprint taken at construction a lie.
    """

    def test_caller_cannot_mutate_a_snapshot_through_a_retained_reference(self):
        source = {"optimizer": {"lr": 2e-5, "betas": [0.9, 0.95]}}
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata=source)

        source["optimizer"]["lr"] = 7

        assert snapshot.metadata["optimizer"]["lr"] == 2e-5

    def test_nested_mapping_cannot_be_mutated(self):
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata={"optimizer": {"lr": 2e-5}})

        with pytest.raises(TypeError):
            snapshot.metadata["optimizer"]["lr"] = 7
        with pytest.raises(TypeError):
            snapshot.metadata["new_key"] = 1
        with pytest.raises(TypeError):
            snapshot.metadata.update({"new_key": 1})

        assert snapshot.metadata["optimizer"]["lr"] == 2e-5

    def test_nested_sequence_cannot_be_mutated(self):
        snapshot = TrainingSpec(
            kind=TrainingKind.SFT, metadata={"optimizer": {"betas": [0.9, 0.95]}}
        )

        # Lists become tuples on the way in.
        assert snapshot.metadata["optimizer"]["betas"] == (0.9, 0.95)
        with pytest.raises(AttributeError):
            snapshot.metadata["optimizer"]["betas"].append(1.0)

    def test_aggregate_id_lists_cannot_be_appended_to(self):
        """An id list that accepts post-construction appends also skips validation."""
        experiment = make_experiment()
        with pytest.raises(AttributeError):
            experiment.active_node_ids.append("node_bogus")

        node = make_node()
        with pytest.raises(AttributeError):
            node.run_ids.append("run_bogus")

    def test_metadata_cannot_be_mutated(self):
        experiment = make_experiment(metadata={"owner": "team-a"})
        with pytest.raises(TypeError):
            experiment.metadata["injected"] = True

        actor = Actor(type="human", id="alex", metadata={"team": "a"})
        with pytest.raises(TypeError):
            actor.metadata["team"] = "b"

    def test_execution_override_values_cannot_be_mutated(self):
        override = ExecutionOverride(
            id="ovr-1",
            kind="micro_batch_resize",
            reason="oom",
            values={"micro_batch_size": 2},
        )
        with pytest.raises(TypeError):
            override.values["micro_batch_size"] = 8

    def test_frozen_containers_still_round_trip_as_plain_json(self):
        snapshot = TrainingSpec(
            kind=TrainingKind.SFT,
            metadata={"optimizer": {"lr": 2e-5, "betas": [0.9, 0.95]}},
        )
        payload = snapshot.model_dump_json()

        assert '"betas":[0.9,0.95]' in payload
        assert TrainingSpec.model_validate_json(payload) == snapshot

    def test_thaw_returns_a_mutable_copy_without_affecting_the_record(self):
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata={"optimizer": {"lr": 2e-5}})

        working = thaw(snapshot.metadata)
        working["optimizer"]["lr"] = 7

        assert snapshot.metadata["optimizer"]["lr"] == 2e-5


class TestFrozenDictInvariants:
    """FrozenDict must be deeply immutable by construction, not by usage.

    The first version froze only at validation time, so a caller could build a
    FrozenDict containing mutable dictionaries and hand it in; validation saw an
    instance of the right class and passed it through untouched.
    """

    def test_a_preconstructed_frozen_dict_is_still_deeply_frozen(self):
        source = FrozenDict({"optimizer": {"lr": 2e-5}})
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata=source)

        with pytest.raises(TypeError):
            source["optimizer"]["lr"] = 7

        assert snapshot.metadata["optimizer"]["lr"] == 2e-5

    def test_the_backing_store_cannot_be_reached_through(self):
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata={"a": 1})

        with pytest.raises(TypeError):
            snapshot.metadata._data["injected"] = True

        assert "injected" not in snapshot.metadata

    def test_attributes_cannot_be_replaced_or_deleted(self):
        frozen = FrozenDict({"a": 1})
        with pytest.raises(TypeError):
            frozen._data = {}
        with pytest.raises(TypeError):
            del frozen._data


class TestValidatedModelCopy:
    """Value objects may be evolved, but only through validation.

    Pydantic's `model_copy(update=...)` assigns update values untouched, which
    defeats every guarantee the field types provide.
    """

    def test_updated_mapping_is_refrozen(self):
        original = TrainingSpec(kind=TrainingKind.SFT, metadata={"a": {"b": 1}})

        copied = original.model_copy(update={"metadata": {"a": {"b": 2}}})

        assert isinstance(copied.metadata, FrozenDict)
        with pytest.raises(TypeError):
            copied.metadata["a"]["b"] = 3

    def test_updated_metadata_is_refrozen(self):
        ref = DatasetRef(uri="s3://x")

        copied = ref.model_copy(update={"metadata": {"nested": []}})

        assert isinstance(copied.metadata, FrozenDict)
        with pytest.raises(AttributeError):
            copied.metadata["nested"].append(1)

    def test_the_value_contract_applies_to_updates(self):
        ref = DatasetRef(uri="s3://x")

        with pytest.raises(ValidationError):
            ref.model_copy(update={"metadata": {"features": {"a", "b"}}})

    def test_an_invalid_value_is_rejected_on_copy(self):
        override = ExecutionOverride(id="ovr", kind="checkpoint_restore", reason="r")

        with pytest.raises(ValidationError):
            override.model_copy(update={"kind": "learning_rate_change"})

    def test_a_plain_copy_is_unchanged(self):
        ref = DatasetRef(uri="s3://x", revision="v4")
        assert ref.model_copy() == ref
        assert ref.model_copy(deep=True) == ref

    def test_typed_ids_survive_the_round_trip(self):
        ref = ArtifactRef(id=ArtifactId.generate(), kind="adapter", uri="s3://x")

        copied = ref.model_copy(update={"uri": "s3://y"})

        assert isinstance(copied.id, ArtifactId)
        assert copied.id == ref.id


class TestAggregateUpdatesAreRefused:
    """Schema validation is not enough for an aggregate.

    Re-validating an update checks field types. It cannot check that the
    transition is legal, that the revision moved, or that the timestamps are
    consistent -- so an aggregate refuses updates entirely and changes only
    through its transition methods. Rule 7 forbids direct status mutation, and
    an unrestricted `model_copy` is that mutation with extra steps.
    """

    def test_experiment_status_cannot_bypass_the_state_machine(self):
        experiment = make_experiment()
        assert experiment.status is ExperimentStatus.CREATED

        with pytest.raises(TypeError, match="with_status"):
            experiment.model_copy(update={"status": ExperimentStatus.SUCCEEDED})

    def test_attempt_cannot_skip_its_lifecycle(self):
        """Otherwise: SUCCEEDED with no started_at, no ended_at, revision 0."""
        attempt = make_attempt()

        with pytest.raises(TypeError):
            attempt.model_copy(update={"status": RunAttemptStatus.SUCCEEDED})

    def test_node_scientific_identity_cannot_be_rewritten(self):
        node = make_node()

        with pytest.raises(TypeError):
            node.model_copy(update={"candidate_fingerprint": "sha256:different"})

    def test_aggregate_id_cannot_be_rewritten(self):
        experiment = make_experiment()

        with pytest.raises(TypeError):
            experiment.model_copy(update={"id": ExperimentId.generate()})

    def test_revision_cannot_be_set_directly(self):
        run = make_run()

        with pytest.raises(TypeError):
            run.model_copy(update={"revision": 99})

    @pytest.mark.parametrize(("factory", "model_type"), ALL_FACTORIES)
    def test_no_aggregate_accepts_an_update(self, factory, model_type):
        with pytest.raises(TypeError, match="cannot be updated"):
            factory().model_copy(update={"revision": 1})

    @pytest.mark.parametrize(("factory", "model_type"), ALL_FACTORIES)
    def test_a_plain_copy_is_still_allowed(self, factory, model_type):
        aggregate = factory()
        assert aggregate.model_copy() == aggregate
        assert aggregate.model_copy(deep=True) == aggregate

    def test_the_error_names_the_offending_fields(self):
        experiment = make_experiment()

        with pytest.raises(TypeError, match="revision"):
            experiment.model_copy(update={"revision": 3})

    def test_transitions_still_work_and_stay_validated(self):
        """The supported path keeps every guarantee the refused one skipped."""
        experiment = make_experiment()
        active = experiment.with_status(ExperimentStatus.ACTIVE)

        assert active.status is ExperimentStatus.ACTIVE
        assert active.revision == experiment.revision + 1
        assert isinstance(active.metadata, FrozenDict)
        assert isinstance(active.id, ExperimentId)
        assert active.budget is not None
        assert active.budget.max_cost == Decimal("125.50")

    def test_transitions_still_refuse_illegal_moves(self):
        experiment = make_experiment()

        with pytest.raises(InvalidTransitionError):
            experiment.with_status(ExperimentStatus.SUCCEEDED)


class TestDomainValueContract:
    """Domain payloads must be canonically persistable.

    A record that cannot round-trip cannot be fingerprinted, so the contract is
    enforced at construction rather than discovered at serialization.
    """

    def test_non_string_keys_are_rejected(self):
        """{1: 'x'} serializes to {'1': 'x'} and stops comparing equal."""
        with pytest.raises(ValidationError):
            TrainingSpec(kind=TrainingKind.SFT, metadata={1: "x"})

    def test_sets_are_rejected(self):
        """Sets have no stable order, so fingerprints would vary by process."""
        with pytest.raises(InvalidDomainValueError, match="stable"):
            deep_freeze({"features": {"a", "b", "c"}})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_floats_are_rejected(self, value):
        with pytest.raises(InvalidDomainValueError, match="finite"):
            deep_freeze({"metric": value})

    def test_arbitrary_objects_are_rejected(self):
        class Custom:
            pass

        with pytest.raises(InvalidDomainValueError):
            deep_freeze({"thing": Custom()})

    def test_bytes_are_rejected(self):
        with pytest.raises(InvalidDomainValueError):
            deep_freeze({"blob": b"\x00"})

    def test_the_contract_is_a_value_error_so_pydantic_reports_it(self):
        assert issubclass(InvalidDomainValueError, ValueError)

    @pytest.mark.parametrize(
        "value",
        [None, True, False, 0, -1, 2.5, "text", {"a": {"b": [1, 2]}}, [1, "a", None]],
    )
    def test_json_shaped_values_are_accepted(self, value):
        deep_freeze({"v": value})

    def test_nested_structures_round_trip_unchanged(self):
        payload = {"a": {"b": [1, {"c": "d"}]}, "e": None, "f": True}
        snapshot = TrainingSpec(kind=TrainingKind.SFT, metadata=payload)

        restored = TrainingSpec.model_validate_json(snapshot.model_dump_json())
        assert restored == snapshot
        assert restored.metadata["a"]["b"][1]["c"] == "d"


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
            node.candidate.metadata = {}
        with pytest.raises(ValidationError):
            node.candidate_fingerprint = "sha256:other"


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
        assert child.parent_ids == (parent.id,)

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
        assert restored.execution_overrides[0].preserves == ("effective_batch_size",)

    def test_rejects_a_scientific_change_as_an_override_kind(self):
        """A scientific change is never an ExecutionOverride.

        Per ADR-011 it is either a TrainingIntervention on a continuing
        trajectory or a new ExperimentNode, decided by comparability — but
        either way the override vocabulary must reject it.
        """
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
