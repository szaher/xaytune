"""A small pinned suite without attachments, and bound, recorded scripted runs over it.

The scripted harness answers every task with the same text, so the held-out
references decide which tasks succeed: ``support/refund-001`` and
``support/refund-002`` hold the scripted answer, ``support/shipping-001``
does not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tests.test_harness.fixtures import SCRIPT, RecordingTools, context, harness
from xaytune.benchmarks import MappingBenchmarkSource, resolve_benchmark_suite

__all__ = ["AllTasks", "MappingBenchmarkSource"]
from xaytune.agent import AgentModelIdentity, ScriptedAgentModel
from xaytune.core.agent_execution import AgentExecutionSpec, HarnessRunResult
from xaytune.core.domain.benchmark import (
    AllTasks,
    BenchmarkSuiteManifest,
    BenchmarkSuiteRef,
    ResolvedBenchmarkSuite,
    TaskAssignment,
    TaskSpec,
    record_assignment,
)
from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec
from xaytune.core.domain.harness import AgentHarnessSpec
from xaytune.core.harness_evaluation import bind_task
from xaytune.core.immutable import FrozenDict
from xaytune.core.trajectory import TrajectoryRecordingSpec
from xaytune.evaluation.harness import publish_evidence
from xaytune.harness import ReferenceHarnessCompiler, ScriptedHarnessModel, ScriptedHarnessRunner
from xaytune.harness.evidence import HarnessEvidenceStore, assemble_evidence
from xaytune.trajectory import WRITER

ANSWER = "Refunds are issued within five days."
"""What the scripted harness answers on every task."""

TASK_IDS = ("support/refund-001", "support/refund-002", "support/shipping-001")
REFERENCES = {
    "support/refund-001": ANSWER,
    "support/refund-002": "  refunds are issued within five days.  ",
    "support/shipping-001": "Orders ship within two days.",
}
AUTHORIZED = frozenset({"kb.search", "tickets.close"})


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def reference_uri(task_id: str) -> str:
    return f"https://bench.example.com/held-out/{task_id.replace('/', '-')}.txt"


def task_record(task_id: str, **changes: Any) -> dict[str, Any]:
    text = REFERENCES.get(task_id, f"Held-out answer for {task_id}.").encode()
    record: dict[str, Any] = {
        "schema_version": "xaytune.task/v1alpha1",
        "id": task_id,
        "revision": "2026.10.1",
        "inputs": {
            "variables": {"customer": "Ann Lee", "ticket": f"Question in {task_id}"},
            "attachments": [],
        },
        "output": {"kind": "text-answer", "media_type": "text/plain"},
        "reference": {
            "kind": "ref",
            "uri": reference_uri(task_id),
            "digest": digest(text),
            "media_type": "text/plain",
        },
    }
    record.update(changes)
    return record


def task(task_id: str, **changes: Any) -> TaskSpec:
    return TaskSpec.model_validate(task_record(task_id, **changes))


def tasks(ids: Iterable[str] = TASK_IDS) -> tuple[TaskSpec, ...]:
    return tuple(task(i) for i in ids)


def manifest(members: tuple[TaskSpec, ...] | None = None, **changes: Any) -> BenchmarkSuiteManifest:
    members = members if members is not None else tasks()
    record: dict[str, Any] = {
        "schema_version": "xaytune.benchmark-suite/v1alpha1",
        "id": "support/core",
        "revision": "v3",
        "provider": {"name": "acme-bench", "version": "1.4.0"},
        "members": [
            {"task_id": t.id, "revision": t.revision, "task_fingerprint": t.task_fingerprint()}
            for t in members
        ],
        "reproducibility": "static",
        "environment_snapshot": None,
        "reset": "per-task",
    }
    record.update(changes)
    return BenchmarkSuiteManifest.model_validate(record)


def source(members: tuple[TaskSpec, ...] | None = None) -> MappingBenchmarkSource:
    members = members if members is not None else tasks()
    held = {
        reference_uri(t.id): REFERENCES.get(t.id, f"Held-out answer for {t.id}.").encode()
        for t in members
    }
    return MappingBenchmarkSource(
        manifests=(manifest(members),),
        tasks=members,
        blobs=held,
        aliases={("support/core", "stable"): "v3"},
    )


def suite(members: tuple[TaskSpec, ...] | None = None) -> ResolvedBenchmarkSuite:
    ref = BenchmarkSuiteRef(
        suite_id="support/core",
        revision="v3",
        suite_fingerprint=manifest(members).suite_fingerprint(),
    )
    return resolve_benchmark_suite(ref, source(members))


def recording(**changes: Any) -> TrajectoryRecordingSpec:
    return TrajectoryRecordingSpec(writer=WRITER, **changes)


def bound_spec(
    store: HarnessEvidenceStore,
    the_suite: ResolvedBenchmarkSuite,
    assignment: TaskAssignment,
    task_id: str,
    *,
    harness_spec: AgentHarnessSpec | None = None,
    seed: int | None = None,
    replicate: int = 0,
    recording_spec: TrajectoryRecordingSpec | None = None,
) -> AgentExecutionSpec:
    """The compiled, recording spec for *task_id*, at its place in the evidence store."""
    spec = harness_spec if harness_spec is not None else harness()
    task_input, binding = bind_task(the_suite, assignment, task_id)
    return ReferenceHarnessCompiler().compile(
        spec,
        context(
            spec,
            store.task_output(binding.position),
            task=task_input,
            seed=assignment.seed if seed is None else seed,
            replicate=replicate,
            trajectory=recording_spec or recording(),
            task_binding=binding,
            suite=the_suite,
        ),
    )


def run(
    spec: AgentExecutionSpec,
    script: Iterable[Mapping[str, Any]] = SCRIPT,
    tools: RecordingTools | None = None,
    **options: Any,
) -> HarnessRunResult:
    options.setdefault("authorized_tools", AUTHORIZED)
    tools = tools or RecordingTools()
    return ScriptedHarnessRunner().run(
        spec, model=ScriptedHarnessModel(script), toolbox=tools.toolbox(), **options
    )


def assignment(seed: int = 11, the_suite: ResolvedBenchmarkSuite | None = None) -> TaskAssignment:
    return record_assignment(the_suite or suite(), AllTasks(), seed)


def replicate_specs(
    store: HarnessEvidenceStore,
    the_suite: ResolvedBenchmarkSuite,
    the_assignment: TaskAssignment,
    **options: Any,
) -> list[AgentExecutionSpec]:
    """One bound, recording spec per assigned task, in assignment order."""
    return [
        bound_spec(store, the_suite, the_assignment, task_id, **options)
        for task_id in the_assignment.task_ids
    ]


def execute(specs: Iterable[AgentExecutionSpec], **options: Any) -> None:
    for spec in specs:
        run(spec, **options)


# ---- evaluation --------------------------------------------------------------------------------

JUDGE = AgentModelIdentity(provider="acme", name="judge", revision="2026-10-01")
PROMPT = "You grade support answers. Reply with a score only."
RUBRIC = "5: correct and complete. 3: partly correct. 1: wrong."


def resolved_content(text: str) -> dict[str, str]:
    return {"digest": digest(text.encode()), "text": text}


def outcome_config(**changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": "xaytune.harness-evaluation/v1alpha1",
        "evaluator": "harness-outcome",
        "suite": suite().ref.model_dump(mode="json"),
        "sampling": {"kind": "all"},
        "catalog": "xaytune.harness-metrics/v1",
        "success": {"kind": "exact-match", "normalization": "casefold-strip"},
        "partial_evidence": "refuse",
        "timing_environment": None,
    }
    config.update(changes)
    return config


def judge_config(**changes: Any) -> dict[str, Any]:
    config = outcome_config(evaluator="harness-judge")
    config.update(
        {
            "judge_model": {
                "provider": JUDGE.provider,
                "model": JUDGE.name,
                "revision": JUDGE.revision,
                "artifact_digest": None,
                "generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 16,
                    "stop_sequences": [],
                    "provider_parameters": {},
                },
            },
            "prompt": resolved_content(PROMPT),
            "rubric": resolved_content(RUBRIC),
            "scale": {"minimum": 1, "maximum": 5},
            "reference_access": "none",
        }
    )
    config.update(changes)
    return config


def evaluation_spec(config: Mapping[str, Any], **metadata: Any) -> EvaluationSpec:
    return EvaluationSpec(
        evaluator=EvaluatorSpec(name=str(config["evaluator"]), config=FrozenDict(config)),
        metadata=FrozenDict(metadata),
    )


def judge_model(*answers: Any) -> ScriptedAgentModel:
    return ScriptedAgentModel(answers, model=JUDGE)


class Replicate:
    """One executed, assembled and published execution replicate."""

    def __init__(
        self,
        root: Path,
        *,
        seed: int = 11,
        harness_spec: AgentHarnessSpec | None = None,
        script: Iterable[Mapping[str, Any]] = SCRIPT,
        sampling: Any = None,
        timing_environment: str | None = None,
        per_task: Mapping[str, Any] | None = None,
        members: tuple[TaskSpec, ...] | None = None,
    ) -> None:
        self.suite = suite(members)
        self.source = source(members)
        self.sampling = sampling if sampling is not None else AllTasks()
        self.assignment = record_assignment(self.suite, self.sampling, seed)
        self.store = HarnessEvidenceStore(root)
        self.specs = replicate_specs(
            self.store, self.suite, self.assignment, harness_spec=harness_spec
        )
        handlers = per_task or {}
        for spec in self.specs:
            handler = handlers.get(spec.task.task_id)
            if handler is None:
                run(spec, script)
            else:
                handler(spec)
        self.evidence = assemble_evidence(
            self.specs,
            self.suite,
            self.sampling,
            self.assignment,
            self.source,
            self.store,
            timing_environment=timing_environment,
        )
        self.subject = publish_evidence(self.evidence, self.store)
