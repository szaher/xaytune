"""Pinned tasks and benchmark suites: what a harness is measured on (H04, ADR-020).

```text
BenchmarkSuiteRequest(id, revision or alias)         what was asked for
BenchmarkSuiteRef(id, pinned revision, fingerprint)  what it resolved to
ResolvedBenchmarkSuite
  manifest  BenchmarkSuiteManifest   ordered members (task id, revision, task_fingerprint)
  tasks     TaskSpec, in manifest order
            inputs     agent-visible: variables + attachments
            reference  held out: a ContentRef, never inline, never agent-visible
```

A benchmark is **evaluation and comparison identity, never harness identity**
(ADR-018): the same harness on another suite is the same hypothesis, measured
differently. So nothing here touches ``AgentHarnessSpec`` or its fingerprint.

Every identity is an explicit, versioned projection hashed by the canonical
encoder, as candidates' and harnesses' are. Content enters by digest, and
metadata and locators never do. Pure core: no fetching, no scoring, no
persistence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from xaytune.core.domain.harness import (
    _MUTABLE_ALIASES,
    ContentRef,
    HarnessContent,
    content_digest,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.immutable import FrozenDict, FrozenDomainModel

__all__ = [
    "BENCHMARK_SUITE_SCHEMA",
    "TASK_SCHEMA",
    "AllTasks",
    "BenchmarkProvider",
    "BenchmarkSuiteManifest",
    "BenchmarkSuiteRef",
    "BenchmarkSuiteRequest",
    "FirstTasks",
    "ResolvedBenchmarkSuite",
    "SeededSample",
    "SuiteMember",
    "TaskAssignment",
    "TaskAttachment",
    "TaskInputs",
    "TaskOutputContract",
    "TaskSampling",
    "TaskSpec",
    "UnverifiedAssignmentError",
    "agent_visible",
    "assign_tasks",
    "assignment_mismatches",
    "benchmark_cohort",
    "record_assignment",
    "suite_fingerprint",
    "suite_identity_v1",
    "task_fingerprint",
    "task_identity_v1",
    "verify_assignment",
]

TASK_SCHEMA = "xaytune.task/v1alpha1"
BENCHMARK_SUITE_SCHEMA = "xaytune.benchmark-suite/v1alpha1"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class _Pinned(FrozenDomainModel):
    """Frozen, closed, and silent about refused input, like the harness contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


def _task_id(value: str) -> str:
    if len(value) > 200 or not _TASK_ID.match(value):
        raise ValueError("a task or suite id is lowercase, '/'-namespaced, at most 200 chars")
    return value


def _pinned_revision(value: str) -> str:
    if not _REVISION.match(value):
        raise ValueError("a revision is letters, digits and ._:+-, at most 128 chars")
    if value.lower() in _MUTABLE_ALIASES or value.lower().endswith("-latest"):
        raise ValueError(
            f"revision {value!r} is a moving alias, not a pin; resolve it before recording it"
        )
    return value


def _provider_name(value: str) -> str:
    if not _NAME.match(value):
        raise ValueError("a provider name is lowercase letters, digits and ._-")
    return value


def _digest(value: str) -> str:
    if not _DIGEST.match(value):
        raise ValueError("a digest is 'sha256:' and 64 hex digits")
    return value


# ---- tasks ----------------------------------------------------------------------------------


class TaskAttachment(_Pinned):
    """A named piece of content the agent may see, inline or pinned by digest."""

    name: str
    content: HarnessContent

    @field_validator("name")
    @classmethod
    def _named(cls, value: str) -> str:
        if not _VARIABLE.match(value):
            raise ValueError("attachment names are identifiers")
        return value


class TaskInputs(_Pinned):
    """Everything about a task the agent may see. Nothing held out is ever here."""

    variables: FrozenDict
    attachments: tuple[TaskAttachment, ...]

    @field_validator("variables")
    @classmethod
    def _strings(cls, value: FrozenDict) -> FrozenDict:
        if not all(_VARIABLE.match(key) for key in value):
            raise ValueError("task variable names are identifiers")
        if not all(isinstance(text, str) for text in value.values()):
            raise ValueError("task variable values are strings")
        return value

    @field_validator("attachments")
    @classmethod
    def _by_name(cls, value: tuple[TaskAttachment, ...]) -> tuple[TaskAttachment, ...]:
        names = [attachment.name for attachment in value]
        if names != sorted(set(names)):
            raise ValueError("attachments are listed by name, once each, sorted")
        return value


class TaskOutputContract(_Pinned):
    """What the task expects back. v1: one text answer."""

    kind: Literal["text-answer"]
    media_type: Literal["text/plain", "text/markdown"]


class TaskSpec(_Pinned):
    """One task, pinned: a stable id, a pinned revision, and content by digest (ADR-020 B1).

    ``reference`` is the held-out answer or rubric input. It is a
    :class:`ContentRef` only -- the bytes live elsewhere -- so a task record
    can travel without carrying its answer, and :func:`agent_visible` never
    returns it.
    """

    schema_version: Literal["xaytune.task/v1alpha1"]
    id: str
    revision: str
    inputs: TaskInputs
    output: TaskOutputContract
    reference: ContentRef | None
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return _task_id(value)

    @field_validator("revision")
    @classmethod
    def _revision_pinned(cls, value: str) -> str:
        return _pinned_revision(value)

    def task_fingerprint(self) -> str:
        return task_fingerprint(self)


def agent_visible(task: TaskSpec) -> TaskInputs:
    """What an agent may be shown of *task*: its inputs, and never its reference (B2)."""
    return task.inputs


def task_identity_v1(task: TaskSpec) -> Mapping[str, Any]:
    """The frozen v1 projection of a task. Content by digest; no metadata, no locators."""
    reference = task.reference
    return {
        "type": "task",
        "version": 1,
        "id": task.id,
        "revision": task.revision,
        "variables": task.inputs.variables,
        "attachments": [
            {"name": a.name, "content": content_digest(a.content)} for a in task.inputs.attachments
        ],
        "output": {"kind": task.output.kind, "media_type": task.output.media_type},
        "reference": reference.digest if reference is not None else None,
    }


def task_fingerprint(task: TaskSpec) -> str:
    """*Is this the same task?* Id, revision and every input byte, by digest."""
    return fingerprint(task_identity_v1(task))


# ---- suites -----------------------------------------------------------------------------------


class BenchmarkProvider(_Pinned):
    """Who publishes a suite, and at what version of their tooling."""

    name: str
    version: str

    @field_validator("name")
    @classmethod
    def _name_shape(cls, value: str) -> str:
        return _provider_name(value)

    @field_validator("version")
    @classmethod
    def _version_pinned(cls, value: str) -> str:
        return _pinned_revision(value)


class SuiteMember(_Pinned):
    """One task in a suite, pinned by revision and fingerprint."""

    task_id: str
    revision: str
    task_fingerprint: str

    @field_validator("task_id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return _task_id(value)

    @field_validator("revision")
    @classmethod
    def _revision_pinned(cls, value: str) -> str:
        return _pinned_revision(value)

    @field_validator("task_fingerprint")
    @classmethod
    def _fingerprint_shape(cls, value: str) -> str:
        return _digest(value)


class BenchmarkSuiteManifest(_Pinned):
    """An ordered, pinned suite (ADR-020 B3). Order is identity.

    ``reproducibility`` says what the pinned inputs cover (B6): ``static`` --
    the tasks' inputs are all the agent can observe -- or ``snapshot`` -- the
    environment's state is pinned too, by ``environment_snapshot``. Dynamic
    services without a snapshot are not representable in v1.

    ``reset`` records the reset semantics. v1 has one: ``per-task``, every
    task starts from the pinned state, so no task observes another's effects.
    It is recorded rather than implied so that a later mode (shared state
    across a suite) is a visible identity change, never a silent one.
    """

    schema_version: Literal["xaytune.benchmark-suite/v1alpha1"]
    id: str
    revision: str
    provider: BenchmarkProvider
    members: tuple[SuiteMember, ...] = Field(min_length=1)
    reproducibility: Literal["static", "snapshot"]
    environment_snapshot: ContentRef | None
    reset: Literal["per-task"]
    metadata: FrozenDict = Field(default_factory=FrozenDict)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return _task_id(value)

    @field_validator("revision")
    @classmethod
    def _revision_pinned(cls, value: str) -> str:
        return _pinned_revision(value)

    @model_validator(mode="after")
    def _consistent(self) -> BenchmarkSuiteManifest:
        ids = [member.task_id for member in self.members]
        if len(ids) != len(set(ids)):
            raise ValueError("a suite lists each task once")
        if (self.reproducibility == "snapshot") != (self.environment_snapshot is not None):
            raise ValueError(
                "a snapshot suite pins environment_snapshot, and only a snapshot suite has one"
            )
        return self

    def suite_fingerprint(self) -> str:
        return suite_fingerprint(self)


def suite_identity_v1(manifest: BenchmarkSuiteManifest) -> Mapping[str, Any]:
    """The frozen v1 projection of a suite: ordered members, environment, reset. No metadata."""
    snapshot = manifest.environment_snapshot
    return {
        "type": "benchmark-suite",
        "version": 1,
        "id": manifest.id,
        "revision": manifest.revision,
        "provider": {"name": manifest.provider.name, "version": manifest.provider.version},
        "members": [
            {"task_id": m.task_id, "revision": m.revision, "task_fingerprint": m.task_fingerprint}
            for m in manifest.members
        ],
        "reproducibility": manifest.reproducibility,
        "environment_snapshot": snapshot.digest if snapshot is not None else None,
        "reset": manifest.reset,
    }


def suite_fingerprint(manifest: BenchmarkSuiteManifest) -> str:
    """*Is this the same suite?* Same tasks, same order, same reproducibility."""
    return fingerprint(suite_identity_v1(manifest))


class BenchmarkSuiteRequest(_Pinned):
    """What was asked for. ``revision`` may be an alias; resolution pins it (B4)."""

    suite_id: str
    revision: str

    @field_validator("suite_id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return _task_id(value)

    @field_validator("revision")
    @classmethod
    def _shape(cls, value: str) -> str:
        if not _REVISION.match(value):
            raise ValueError("a revision is letters, digits and ._:+-, at most 128 chars")
        return value


class BenchmarkSuiteRef(_Pinned):
    """What a request resolved to: a pinned revision and the suite's fingerprint."""

    suite_id: str
    revision: str
    suite_fingerprint: str

    @field_validator("suite_id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return _task_id(value)

    @field_validator("revision")
    @classmethod
    def _revision_pinned(cls, value: str) -> str:
        return _pinned_revision(value)

    @field_validator("suite_fingerprint")
    @classmethod
    def _fingerprint_shape(cls, value: str) -> str:
        return _digest(value)


class ResolvedBenchmarkSuite(_Pinned):
    """A request, what it resolved to, and every task -- each verified whenever built or parsed.

    Building or parsing one verifies its **structure**: the ref, the manifest's
    fingerprint, and every task's position and fingerprint. It carries no
    bytes, so it cannot prove that the content those fingerprints name is
    still held or unchanged. Only resolution against a source does that
    (``xaytune.benchmarks``), and a loaded record must be checked against a
    source again (``verify_suite_content``) before anything executes or
    evaluates on it (ADR-020 B4).
    """

    request: BenchmarkSuiteRequest
    ref: BenchmarkSuiteRef
    manifest: BenchmarkSuiteManifest
    tasks: tuple[TaskSpec, ...]

    @model_validator(mode="after")
    def _verified(self) -> ResolvedBenchmarkSuite:
        problems: list[str] = []
        manifest, ref = self.manifest, self.ref
        if self.request.suite_id != ref.suite_id:
            problems.append("the ref names a different suite than the request")
        if (ref.suite_id, ref.revision) != (manifest.id, manifest.revision):
            problems.append("the ref and the manifest disagree on suite id or revision")
        if ref.suite_fingerprint != manifest.suite_fingerprint():
            problems.append("the ref's suite_fingerprint is not the manifest's")
        if len(self.tasks) != len(manifest.members):
            problems.append("the tasks are not exactly the manifest's members")
        else:
            for position, (member, task) in enumerate(zip(manifest.members, self.tasks)):
                if (task.id, task.revision) != (member.task_id, member.revision):
                    problems.append(f"task {position} is not the member at that position")
                elif task.task_fingerprint() != member.task_fingerprint:
                    problems.append(f"task {position} does not have its pinned fingerprint")
        if problems:
            raise ValueError("unverified benchmark suite: " + "; ".join(problems))
        return self

    def task(self, task_id: str) -> TaskSpec:
        """The member task *task_id*.

        Raises:
            KeyError: If it is not a member.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError("not a member of this suite")


# ---- sampling and comparison ------------------------------------------------------------------


class AllTasks(_Pinned):
    """Every member, in manifest order."""

    kind: Literal["all"] = "all"


class FirstTasks(_Pinned):
    """The first *count* members, in manifest order."""

    kind: Literal["first"] = "first"
    count: int = Field(gt=0, strict=True)


class SeededSample(_Pinned):
    """*count* members chosen by the run's seed, returned in manifest order."""

    kind: Literal["seeded-sample"] = "seeded-sample"
    count: int = Field(gt=0, strict=True)


TaskSampling = Annotated[AllTasks | FirstTasks | SeededSample, Field(discriminator="kind")]


def assign_tasks(
    suite: ResolvedBenchmarkSuite, sampling: AllTasks | FirstTasks | SeededSample, seed: int
) -> tuple[str, ...]:
    """The task ids one run performs, deterministically (ADR-020 B5).

    A seeded sample ranks members by ``sha256(seed, suite fingerprint, task
    id)`` -- never Python's ``random``, whose algorithm is not a stable
    contract -- keeps the first *count*, and returns them in manifest order.
    The same suite, policy and seed give the same assignment on any machine,
    forever.

    Raises:
        ValueError: If the policy asks for more tasks than the suite has.
    """
    ids = tuple(member.task_id for member in suite.manifest.members)
    if isinstance(sampling, AllTasks):
        return ids
    if sampling.count > len(ids):
        raise ValueError(
            f"the sampling policy asks for {sampling.count} tasks; the suite has {len(ids)}"
        )
    if isinstance(sampling, FirstTasks):
        return ids[: sampling.count]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("a seeded sample needs an integer seed")
    suite_fp = suite.ref.suite_fingerprint

    def rank(task_id: str) -> str:
        return hashlib.sha256(f"{seed}\x00{suite_fp}\x00{task_id}".encode()).hexdigest()

    chosen = set(sorted(ids, key=rank)[: sampling.count])
    return tuple(task_id for task_id in ids if task_id in chosen)


def benchmark_cohort(ref: BenchmarkSuiteRef, sampling: AllTasks | FirstTasks | SeededSample) -> str:
    """The benchmark part of a comparison key: **necessary, not sufficient** (ADR-020 B7).

    The suite's fingerprint -- which covers its members, their order, its
    reproducibility, environment snapshot and reset semantics -- and the
    sampling policy. Not the seed: replicates under one policy share it.

    Equal cohorts do **not** make two results comparable. A comparison also
    needs matched task assignments (:func:`assignment_mismatches`), the same
    scoring rules and measurement units (H06's evaluation identity), and
    compatible execution conditions (declared, not inferred from a
    fingerprint). Nothing in H04 decides comparability; it supplies the parts
    H04 owns.
    """
    return fingerprint(
        {
            "type": "benchmark-cohort",
            "version": 1,
            "suite_fingerprint": ref.suite_fingerprint,
            "sampling": sampling.model_dump(mode="json"),
        }
    )


class TaskAssignment(_Pinned):
    """Which tasks one run performed: its cohort, its seed, and the ids, in manifest order.

    Run provenance, never harness identity. Recorded so that a comparison can
    prove its runs were matched rather than assume it.

    A record is only a claim. Parsing one checks its shape; only
    :func:`verify_assignment` proves it is what the suite, policy and seed
    assign.
    """

    cohort: str
    seed: int = Field(strict=True)
    task_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("cohort")
    @classmethod
    def _cohort_shape(cls, value: str) -> str:
        return _digest(value)

    @field_validator("task_ids")
    @classmethod
    def _each_task_once(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("an assignment lists each task once")
        return value


def record_assignment(
    suite: ResolvedBenchmarkSuite, sampling: AllTasks | FirstTasks | SeededSample, seed: int
) -> TaskAssignment:
    """Assign tasks for one run and record the assignment with its cohort and seed."""
    return TaskAssignment(
        cohort=benchmark_cohort(suite.ref, sampling),
        seed=seed,
        task_ids=assign_tasks(suite, sampling, seed),
    )


class UnverifiedAssignmentError(ValueError):
    """A recorded assignment is not what its suite, policy and seed assign. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the task assignment is not verified: " + "; ".join(reasons))


def verify_assignment(
    suite: ResolvedBenchmarkSuite,
    sampling: AllTasks | FirstTasks | SeededSample,
    assignment: TaskAssignment,
) -> TaskAssignment:
    """*assignment*, once it is proven to be exactly what *suite*, *sampling* and its seed assign.

    Recomputes the cohort and the selected tasks and refuses any difference:
    a cohort of another suite or policy, ids that are not members, repeated
    or reordered ids, or tasks another seed would assign. A record parsed from
    storage or another process is a claim until it passes here (ADR-020 B7).

    For :class:`AllTasks` and :class:`FirstTasks` the seed selects nothing,
    so it is verified only as the replicate label it records.

    Raises:
        UnverifiedAssignmentError: With every reason.
    """
    reasons: list[str] = []
    if assignment.cohort != benchmark_cohort(suite.ref, sampling):
        reasons.append("the cohort is not this suite's and sampling policy's")
    ids = assignment.task_ids
    members = [member.task_id for member in suite.manifest.members]
    if len(set(ids)) != len(ids):
        reasons.append("the assignment lists a task more than once")
    if not set(ids) <= set(members):
        reasons.append("the assignment names tasks that are not members of the suite")
    else:
        positions = [members.index(task_id) for task_id in ids]
        if positions != sorted(positions):
            reasons.append("the tasks are not in manifest order")
    try:
        expected: tuple[str, ...] | None = assign_tasks(suite, sampling, assignment.seed)
    except ValueError as error:
        reasons.append(f"the sampling policy does not apply to this suite: {error}")
        expected = None
    if expected is not None and ids != expected:
        reasons.append("the tasks are not the ones this suite, policy and seed assign")
    if reasons:
        raise UnverifiedAssignmentError(tuple(reasons))
    return assignment


def assignment_mismatches(left: TaskAssignment, right: TaskAssignment) -> tuple[str, ...]:
    """Every reason two runs' assignments are not matched; ``()`` when they are (ADR-020 B7).

    This compares two claims: it does not check them against a suite. Two
    fabricated records can agree. Callers must first pass **both** through
    :func:`verify_assignment`.

    Matched means the same cohort, the same replicate seed and the same tasks.
    Candidates run under the same replicate seeds therefore receive identical
    assignments -- :func:`assign_tasks` depends on the suite, the policy and
    the seed, never on the candidate -- and are compared task for task. An
    unmatched comparison is not refused here; it is reported, and H06 must
    either refuse it or account for the sampling difference explicitly.
    """
    reasons: list[str] = []
    if left.cohort != right.cohort:
        reasons.append("different benchmark cohorts: different suite or sampling policy")
    if left.seed != right.seed:
        reasons.append("different replicate seeds: the runs are not paired")
    if left.task_ids != right.task_ids:
        reasons.append("different task assignments")
    return tuple(reasons)
