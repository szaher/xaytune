"""Typed identifier behaviour: prefixes, validation, ordering, serialization."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from xaytune.core.errors import InvalidIdError
from xaytune.core.ids import (
    ActionId,
    ArtifactId,
    CheckpointId,
    DecisionId,
    EvaluationId,
    EventId,
    ExperimentId,
    ExperimentNodeId,
    IncidentId,
    RunAttemptId,
    RunId,
)

ALL_ID_TYPES = [
    (ExperimentId, "exp_"),
    (ExperimentNodeId, "node_"),
    (RunId, "run_"),
    (RunAttemptId, "attempt_"),
    (ActionId, "act_"),
    (IncidentId, "inc_"),
    (EvaluationId, "eval_"),
    (ArtifactId, "artifact_"),
    (CheckpointId, "ckpt_"),
    (DecisionId, "decision_"),
    (EventId, "event_"),
]


class TestPrefixes:
    @pytest.mark.parametrize(("id_type", "prefix"), ALL_ID_TYPES)
    def test_generated_id_carries_its_prefix(self, id_type, prefix):
        assert id_type.generate().startswith(prefix)

    @pytest.mark.parametrize(("id_type", "prefix"), ALL_ID_TYPES)
    def test_generated_id_validates_as_its_own_type(self, id_type, prefix):
        value = id_type.generate()
        assert id_type.validate(value) == value

    def test_prefixes_are_unique(self):
        prefixes = [prefix for _, prefix in ALL_ID_TYPES]
        assert len(set(prefixes)) == len(prefixes)

    def test_id_is_a_plain_string(self):
        """Ids must be usable directly as dict keys and SQL parameters."""
        value = ExperimentId.generate()
        assert isinstance(value, str)
        assert {value: 1}[str(value)] == 1


class TestValidation:
    def test_rejects_wrong_type_prefix(self):
        run_id = RunId.generate()
        with pytest.raises(InvalidIdError, match="must start with 'exp_'"):
            ExperimentId.validate(run_id)

    def test_rejects_missing_prefix(self):
        with pytest.raises(InvalidIdError):
            ExperimentId.validate("01ARZ3NDEKTSV4RRFFQ69G5FAV")

    def test_rejects_short_body(self):
        with pytest.raises(InvalidIdError, match="body must be"):
            ExperimentId.validate("exp_TOOSHORT")

    def test_rejects_non_crockford_characters(self):
        # I, L, O and U are excluded from the alphabet.
        with pytest.raises(InvalidIdError, match="Crockford"):
            ExperimentId.validate("exp_" + "I" * 26)

    def test_rejects_non_string(self):
        with pytest.raises(InvalidIdError, match="must be a string"):
            ExperimentId.validate(42)

    def test_invalid_id_error_is_a_value_error(self):
        """So Pydantic reports it as a validation error, not a crash."""
        assert issubclass(InvalidIdError, ValueError)


class TestOrdering:
    def test_ids_sort_in_creation_order(self):
        ids = [ExperimentId.generate() for _ in range(500)]
        assert ids == sorted(ids)

    def test_ids_are_unique(self):
        ids = [ExperimentId.generate() for _ in range(500)]
        assert len(set(ids)) == len(ids)

    def test_ids_of_different_types_still_sort_within_a_type(self):
        first = RunId.generate()
        RunAttemptId.generate()
        second = RunId.generate()
        assert first < second

    def test_created_at_ms_is_recoverable(self):
        import time

        before = int(time.time() * 1000)
        value = ExperimentId.generate()
        after = int(time.time() * 1000)
        assert before <= value.created_at_ms <= after + 1


class _Holder(BaseModel):
    experiment_id: ExperimentId
    run_id: RunId | None = None


class TestPydanticIntegration:
    def test_serializes_as_a_plain_string(self):
        holder = _Holder(experiment_id=ExperimentId.generate())
        payload = holder.model_dump_json()
        assert f'"{holder.experiment_id}"' in payload

    def test_round_trips(self):
        holder = _Holder(experiment_id=ExperimentId.generate(), run_id=RunId.generate())
        restored = _Holder.model_validate_json(holder.model_dump_json())
        assert restored == holder
        assert isinstance(restored.experiment_id, ExperimentId)

    def test_wrong_id_type_is_a_validation_error(self):
        with pytest.raises(ValidationError):
            _Holder(experiment_id=RunId.generate())

    def test_parses_from_a_raw_string(self):
        raw = str(ExperimentId.generate())
        holder = _Holder(experiment_id=raw)
        assert isinstance(holder.experiment_id, ExperimentId)
