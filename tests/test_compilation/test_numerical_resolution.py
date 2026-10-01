"""Numerical-recovery requests in the resolved plan: opaque, durable and deterministic."""

from __future__ import annotations

import pytest

from tests.test_compilation.test_attempt_resolution import _candidate
from xaytune.compilation import CompilationContext
from xaytune.compilation.attempt_resolution import (
    resolve_training_attempt,
    training_execution_fingerprint,
)
from xaytune.compilation.native import NativeCompiler
from xaytune.core.domain.intervention import (
    InterventionDirective,
    InterventionDirectiveKind,
    LearningRateMutation,
)
from xaytune.core.domain.run import ExecutionOverride, RunAttempt
from xaytune.core.execution_controls import (
    MANAGED_NUMERICAL_RECOVERY,
    TRAINING_INTERVENTIONS,
    TrainingInterventionDirectives,
)
from xaytune.core.ids import (
    ActionId,
    CheckpointId,
    InterventionApplicationId,
    InterventionId,
    RunAttemptId,
    RunId,
)
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import CheckpointRef


@pytest.fixture
def compiled(tmp_path):
    return NativeCompiler().compile(
        _candidate(tmp_path),
        CompilationContext(run_id="run-contract", seed=7, output_uri=str(tmp_path / "out")),
    )


def successor(action_id=None):
    checkpoint = CheckpointRef(
        id=CheckpointId.generate(),
        uri="file:///checkpoint",
        digest="sha256:" + "a" * 64,
        compatibility_key="test-compatible",
    )
    return RunAttempt(
        id=RunAttemptId.generate(),
        run_id=RunId.generate(),
        attempt_number=2,
        checkpoint_ref=checkpoint,
        execution_overrides=(
            ExecutionOverride(
                id="numerical-restore",
                kind="checkpoint_restore",
                reason="restore validated FULL+EXACT checkpoint",
                values=FrozenDict({"checkpoint_id": str(checkpoint.id)}),
                action_id=action_id,
            ),
        ),
    )


def directive_for(attempt, ordinal=0, previous=2e-4, applied=1e-4):
    return InterventionDirective(
        application_id=InterventionApplicationId.generate(),
        intervention_id=InterventionId.generate(),
        attempt_id=attempt.id,
        ordinal=ordinal,
        kind=InterventionDirectiveKind.INITIAL,
        mutation=LearningRateMutation(learning_rate=applied),
        expected_previous_value=previous,
    )


def test_defaults_leave_the_plan_unchanged(compiled):
    attempt = successor()
    plain = resolve_training_attempt(compiled, attempt, "local")
    assert set(plain.runtime_options) == {"checkpoint_restore"}


def test_numerical_successor_carries_directives_and_control(compiled):
    action_id = str(ActionId.generate())
    attempt = successor(action_id)
    with pytest.raises(ValueError, match="another governing Action"):
        resolve_training_attempt(compiled, attempt, "local")
    directives = (directive_for(attempt),)
    resolved = resolve_training_attempt(
        compiled,
        attempt,
        "local",
        directives=directives,
        numerical_recovery=True,
        restore_action_id=action_id,
    )
    carried = TrainingInterventionDirectives.model_validate(
        thaw(resolved.runtime_options[TRAINING_INTERVENTIONS])
    )
    assert carried.directives == directives
    assert resolved.runtime_options[MANAGED_NUMERICAL_RECOVERY]["fail_on_nonfinite_loss"] is True
    # The LR is never an operational config change.
    assert resolved.spec == compiled
    again = resolve_training_attempt(
        compiled,
        attempt,
        "local",
        directives=directives,
        numerical_recovery=True,
        restore_action_id=action_id,
    )
    assert again.request_digest("submit") == resolved.request_digest("submit")


def test_execution_identity_includes_the_control_but_not_the_directives(compiled):
    action_id = str(ActionId.generate())
    attempt = successor(action_id)

    def identity(**options):
        return training_execution_fingerprint(
            resolve_training_attempt(
                compiled, attempt, "local", restore_action_id=action_id, **options
            )
        )

    assert identity() != identity(numerical_recovery=True)
    assert identity(numerical_recovery=True) == identity(
        numerical_recovery=True, directives=(directive_for(attempt),)
    )


def test_directives_need_a_restore_and_their_own_attempt(compiled):
    fresh = RunAttempt(id=RunAttemptId.generate(), run_id=RunId.generate(), attempt_number=1)
    with pytest.raises(ValueError, match="checkpoint restore"):
        resolve_training_attempt(compiled, fresh, "local", directives=(directive_for(fresh),))
    attempt = successor()
    with pytest.raises(ValueError, match="checkpoint restore"):
        resolve_training_attempt(compiled, attempt, "local", directives=(directive_for(fresh),))
