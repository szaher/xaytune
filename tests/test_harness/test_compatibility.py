"""H03 moves nothing that is already recorded (ADR-019 §"The boundary H03 stops at").

The literals here were produced by ``main`` before H03: the training request
digest is the one ``test_evaluation_transport.py`` already pins, and the other
two were computed from modules H03 does not modify. A change to any of them
re-keys recorded submissions or re-identifies recorded executions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.test_core.test_candidate import _plain_candidate
from tests.test_core.test_candidate_envelope import V1, V2
from tests.test_core.test_harness import BASE, PINNED_V1, _harness
from tests.test_harness.fixtures import compile_spec
from xaytune.compilation.attempt_resolution import training_execution_fingerprint
from xaytune.core.domain.candidate_envelope import ExperimentCandidate
from xaytune.core.domain.operation import OperationTargetKind, RuntimeOperationTarget
from xaytune.core.execution import (
    CompilerIdentity,
    EvaluationExecutionSpec,
    EvaluatorIdentity,
    ExecutionSpec,
    PythonModuleEntrypoint,
    ResolvedExecutionPlan,
    TrainingExecutionSpec,
)
from xaytune.core.ids import ArtifactId
from xaytune.core.refs import ArtifactRef
from xaytune.runtimes import _PAYLOAD_WORKLOAD_FOR_TARGET

PINNED_TRAINING_REQUEST = "sha256:bb3feb32896bf5f54c793ffb2aeeeaec3540398de892eb737cd6ee8f5c3c3813"
PINNED_TRAINING_EXECUTION = (
    "sha256:3c9c5b11e87bdc0aa054bde8becf1574f8f7a0e373e9a331f86dff985f91fd72"
)
PINNED_EVALUATION_REQUEST = (
    "sha256:75512db86b9c263919ec470a243682352815288b47410570c3250f1698be3397"
)


def _training_plan() -> ResolvedExecutionPlan:
    return ResolvedExecutionPlan(
        spec=TrainingExecutionSpec(
            compiler=CompilerIdentity(name="native", version="1.0.0"),
            candidate_fingerprint="sha256:" + "0" * 64,
            entrypoint=PythonModuleEntrypoint(module="xaytune.workers.native"),
            config={"learning_rate": 0.001},
        ),
        runtime="local",
        target=RuntimeOperationTarget(kind="training-attempt", id="attempt_x"),
    )


def _evaluation_plan() -> ResolvedExecutionPlan:
    return ResolvedExecutionPlan(
        spec=EvaluationExecutionSpec(
            evaluator=EvaluatorIdentity(name="native", version="0.1.0"),
            evaluation_fingerprint="sha256:" + "e" * 64,
            subject=ArtifactRef(
                id=ArtifactId("artifact_" + "0" * 26),
                kind="model",
                uri="/models/m",
                digest="sha256:m",
            ),
            entrypoint=PythonModuleEntrypoint(module="xaytune.workers.native_eval"),
        ),
        runtime="local",
        target=RuntimeOperationTarget(kind="evaluation-attempt", id="evalattempt_x"),
    )


def test_recorded_training_and_evaluation_requests_keep_their_digests() -> None:
    assert _training_plan().request_digest("submit") == PINNED_TRAINING_REQUEST
    assert training_execution_fingerprint(_training_plan()) == PINNED_TRAINING_EXECUTION
    assert _evaluation_plan().request_digest("submit") == PINNED_EVALUATION_REQUEST


def test_scientific_identities_are_unchanged() -> None:
    assert ExperimentCandidate.training(_plain_candidate()).fingerprints_for_lookup() == (V2, V1)
    assert _harness().harness_fingerprint() == BASE == PINNED_V1


def test_no_target_kind_union_member_or_payload_family_was_added() -> None:
    assert get_args(OperationTargetKind) == ("training-attempt", "evaluation-attempt")
    assert set(_PAYLOAD_WORKLOAD_FOR_TARGET) == {"training-attempt", "evaluation-attempt"}
    members = get_args(get_args(ExecutionSpec)[0])
    assert members == (TrainingExecutionSpec, EvaluationExecutionSpec)


def test_a_harness_spec_cannot_ride_a_plan_or_the_wire_union(tmp_path: Path) -> None:
    spec = compile_spec(tmp_path)
    with pytest.raises(ValidationError):
        ResolvedExecutionPlan(
            spec=spec,  # type: ignore[arg-type]
            runtime="local",
            target=RuntimeOperationTarget(kind="training-attempt", id="attempt_x"),
        )
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionSpec).validate_python(spec.model_dump(mode="json"))


def test_the_execution_contract_imports_nothing_it_should_not() -> None:
    """``xaytune.core.agent_execution`` is core: no harness package, agent SDK or trainer."""
    code = (
        "import sys, xaytune.core.agent_execution\n"
        "bad = [m for m in ('xaytune.harness', 'xaytune.agent', 'xaytune.runtimes', "
        "'xaytune.storage', 'torch', 'transformers') if m in sys.modules]\n"
        "print(','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""
