"""ADR-022 V2: a pinned task wired into execution, and an unbound spec left exactly as it was."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.test_harness.fixtures import context, harness
from tests.test_harness_evaluation import fixtures as fx
from xaytune.core.agent_execution import (
    HarnessTaskInput,
    agent_execution_fingerprint,
    agent_run_request_digest,
)
from xaytune.core.domain.benchmark import FirstTasks, TaskSpec, agent_visible, record_assignment
from xaytune.core.harness_evaluation import (
    HarnessTaskBinding,
    bind_task,
    task_binding_refusals,
)
from xaytune.core.immutable import FrozenDict
from xaytune.harness import (
    HarnessCompilationContext,
    HarnessRunRefusedError,
    ReferenceHarnessCompiler,
    UnsupportedHarnessError,
)
from xaytune.harness.evidence import HarnessEvidenceStore
from xaytune.trajectory import LocalTrajectoryStore
from xaytune.trajectory.sealing import read_manifest

TASK = "support/refund-002"


@pytest.fixture
def suite():
    return fx.suite()


@pytest.fixture
def assignment(suite):
    return fx.assignment(the_suite=suite)


def _compile(tmp_path: Path, **overrides):
    spec = harness()
    return ReferenceHarnessCompiler().compile(spec, context(spec, tmp_path, **overrides))


def _bound(suite, assignment, task_id=TASK):
    task_input, binding = bind_task(suite, assignment, task_id)
    return {
        "task": task_input,
        "task_binding": binding,
        "suite": suite,
        "trajectory": fx.recording(),
    }


class TestBoundSpec:
    def test_carries_the_tasks_agent_visible_inputs_and_its_pin(self, tmp_path, suite, assignment):
        spec = _compile(tmp_path, **_bound(suite, assignment))
        task = suite.task(TASK)
        assert spec.task.task_id == TASK
        assert spec.task.variables == agent_visible(task).variables
        binding = HarnessTaskBinding.model_validate(dict(spec.config["task"]))
        assert binding.suite == suite.ref
        assert (binding.task_id, binding.revision) == (task.id, task.revision)
        assert binding.task_fingerprint == task.task_fingerprint()
        assert binding.position == fx.TASK_IDS.index(TASK)
        assert binding.cohort == assignment.cohort

    def test_binding_is_execution_context_never_harness_identity(self, tmp_path, suite, assignment):
        bound = _compile(tmp_path, **_bound(suite, assignment))
        recorded = _compile(tmp_path, task=bound.task, trajectory=fx.recording())
        assert bound.harness_fingerprint == recorded.harness_fingerprint
        assert agent_execution_fingerprint(bound) != agent_execution_fingerprint(recorded)
        assert agent_run_request_digest(bound) != agent_run_request_digest(recorded)

    def test_the_trajectory_attribution_records_the_task_pin_and_cohort(
        self, tmp_path, suite, assignment
    ):
        spec = fx.bound_spec(HarnessEvidenceStore(tmp_path), suite, assignment, TASK)
        fx.run(spec)
        store = LocalTrajectoryStore(Path(spec.outputs[1].uri))
        (paths,) = store.attempts()
        manifest = read_manifest(paths.manifest)
        assert manifest is not None and manifest.completeness == "complete"
        attribution = manifest.attribution
        assert attribution.task.task_id == TASK
        assert attribution.task.revision == suite.task(TASK).revision
        assert attribution.task.task_fingerprint == suite.task(TASK).task_fingerprint()
        assert attribution.assignment_cohort == assignment.cohort

    def test_no_reference_byte_or_locator_reaches_the_spec(self, tmp_path, suite, assignment):
        spec = _compile(tmp_path, **_bound(suite, assignment))
        task = suite.task(TASK)
        assert task.reference is not None
        text = spec.model_dump_json()
        assert fx.REFERENCES[TASK].strip() not in text
        assert task.reference.uri not in text
        assert task.reference.digest not in text


class TestUnboundSpec:
    def test_is_exactly_h03s(self, tmp_path, suite):
        unbound = _compile(tmp_path)
        assert "task" not in unbound.config
        assert set(unbound.config) == {"compiler_api", "token_counter"}
        explicit = _compile(tmp_path, task_binding=None, suite=None)
        assert agent_run_request_digest(unbound) == agent_run_request_digest(explicit)
        assert unbound == explicit

    def test_a_recording_spec_without_a_binding_is_h05s(self, tmp_path, suite):
        spec = _compile(tmp_path, trajectory=fx.recording())
        assert set(spec.config) == {"compiler_api", "token_counter", "trajectory"}


class TestRefusals:
    def test_different_variables(self, tmp_path, suite, assignment):
        fields = _bound(suite, assignment)
        fields["task"] = HarnessTaskInput(
            task_id=TASK, variables=FrozenDict({"customer": "Bo", "ticket": "Something else"})
        )
        with pytest.raises(UnsupportedHarnessError) as caught:
            _compile(tmp_path, **fields)
        assert any("agent-visible variables" in r for r in caught.value.reasons)

    def test_a_different_task_id(self, suite, assignment):
        task_input, binding = bind_task(suite, assignment, TASK)
        other = task_input.model_copy(update={"task_id": "support/refund-001"})
        assert "the task input's id is not the bound task's" in task_binding_refusals(
            binding, other, suite
        )

    def test_a_wrong_fingerprint(self, tmp_path, suite, assignment):
        fields = _bound(suite, assignment)
        fields["task_binding"] = fields["task_binding"].model_copy(
            update={"task_fingerprint": "sha256:" + "0" * 64}
        )
        with pytest.raises(UnsupportedHarnessError) as caught:
            _compile(tmp_path, **fields)
        assert any("task fingerprint" in r for r in caught.value.reasons)

    def test_a_wrong_position(self, tmp_path, suite, assignment):
        fields = _bound(suite, assignment)
        fields["task_binding"] = fields["task_binding"].model_copy(update={"position": 0})
        with pytest.raises(UnsupportedHarnessError) as caught:
            _compile(tmp_path, **fields)
        assert any("position" in r for r in caught.value.reasons)

    def test_a_position_beyond_the_suite(self, suite, assignment):
        task_input, binding = bind_task(suite, assignment, TASK)
        beyond = binding.model_copy(update={"position": 9})
        assert any("beyond" in r for r in task_binding_refusals(beyond, task_input, suite))

    def test_another_suite(self, suite, assignment):
        task_input, binding = bind_task(suite, assignment, TASK)
        other = binding.model_copy(
            update={"suite": suite.ref.model_copy(update={"revision": "v4"})}
        )
        assert any("another suite" in r for r in task_binding_refusals(other, task_input, suite))

    def test_a_task_with_attachments(self, tmp_path):
        record = fx.task_record(TASK)
        record["inputs"]["attachments"] = [
            {"name": "policy", "content": {"kind": "inline", "text": "Refund policy."}}
        ]
        members = tuple(
            TaskSpec.model_validate(record) if i == TASK else fx.task(i) for i in fx.TASK_IDS
        )
        attached = fx.suite(members)
        assignment = fx.assignment(the_suite=attached)
        with pytest.raises(UnsupportedHarnessError) as caught:
            _compile(tmp_path, **_bound(attached, assignment))
        assert any("attachments" in r for r in caught.value.reasons)

    def test_a_bound_spec_must_record(self, tmp_path, suite, assignment):
        fields = _bound(suite, assignment)
        fields["trajectory"] = None
        with pytest.raises(UnsupportedHarnessError) as caught:
            _compile(tmp_path, **fields)
        assert any("must record" in r for r in caught.value.reasons)

    def test_a_binding_comes_with_its_suite(self, tmp_path, suite, assignment):
        spec = harness()
        task_input, binding = bind_task(suite, assignment, TASK)
        with pytest.raises(ValidationError, match="come together"):
            context(spec, tmp_path, task=task_input, task_binding=binding)

    def test_a_task_outside_the_assignment_cannot_be_bound(self, suite):
        partial = record_assignment(suite, FirstTasks(count=1), 3)
        with pytest.raises(KeyError):
            bind_task(suite, partial, TASK)


class TestRunnerChecksTheBinding:
    def test_a_binding_for_another_task_is_refused(self, tmp_path, suite, assignment):
        spec = fx.bound_spec(HarnessEvidenceStore(tmp_path), suite, assignment, TASK)
        block = dict(spec.config["task"])
        block["task_id"] = "support/refund-001"
        tampered = spec.model_copy(
            update={"config": FrozenDict({**dict(spec.config), "task": FrozenDict(block)})}
        )
        with pytest.raises(HarnessRunRefusedError, match="binds another task"):
            fx.run(tampered)

    def test_a_malformed_binding_is_refused(self, tmp_path, suite, assignment):
        spec = fx.bound_spec(HarnessEvidenceStore(tmp_path), suite, assignment, TASK)
        tampered = spec.model_copy(
            update={"config": FrozenDict({**dict(spec.config), "task": FrozenDict({"x": 1})})}
        )
        with pytest.raises(HarnessRunRefusedError, match="not a valid"):
            fx.run(tampered)

    def test_an_unrecorded_bound_spec_is_refused(self, tmp_path, suite, assignment):
        spec = fx.bound_spec(HarnessEvidenceStore(tmp_path), suite, assignment, TASK)
        config = {k: v for k, v in dict(spec.config).items() if k != "trajectory"}
        tampered = spec.model_copy(
            update={"config": FrozenDict(config), "outputs": spec.outputs[:1]}
        )
        with pytest.raises(HarnessRunRefusedError, match="must record"):
            fx.run(tampered)


def test_context_round_trips(tmp_path, suite, assignment):
    spec = harness()
    built = context(spec, tmp_path, **_bound(suite, assignment))
    assert HarnessCompilationContext.model_validate_json(built.model_dump_json()) == built
    assert json.loads(built.model_dump_json())["task_binding"]["task_id"] == TASK
