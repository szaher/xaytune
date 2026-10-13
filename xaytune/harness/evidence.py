"""Assembling and verifying a harness run's evidence against the store (ADR-022 V3).

```text
<evidence root>/
  tasks/<position>/              one assigned task's output_uri
    agent-run.json               the published result
    trajectory/attempts/<id>/    every attempt instance of the task's request digest (ADR-021)
```

**One operation verifies evidence, wherever it is consumed:**
:func:`verify_run_evidence`. A digest proves only that a record is
self-consistent: an edited record whose digest was recomputed is
self-consistent too. So nothing trusts the record's own contents. The
verifier rebuilds every fact the record states from independent sources --
the suite, the source, and the store the **caller** names -- and refuses
with every reason. Assembly calls it as its last step; every evaluator calls
it at the start of execution.

**Locations are never read from the record.** A task's outputs are where
the layout puts them under the root the caller supplies, so a record cannot
point a verifier at another store.

**Its limit.** Membership is checked against the attempts the store still
lists. An attempt whose whole directory was deleted cannot be listed, so
local evidence cannot prove its absence; that needs the durable path's
attempt ordinals (ADR-022 V13).

Reading only: nothing here writes to a store or a published result.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from xaytune.benchmarks import BenchmarkSource, UnverifiedBenchmarkError, verify_suite_content
from xaytune.core.agent_execution import (
    AgentExecutionSpec,
    HarnessRunResult,
    agent_run_request_digest,
)
from xaytune.core.domain.benchmark import (
    AllTasks,
    FirstTasks,
    ResolvedBenchmarkSuite,
    SeededSample,
    TaskAssignment,
    UnverifiedAssignmentError,
    verify_assignment,
)
from xaytune.core.fingerprint import fingerprint
from xaytune.core.harness_evaluation import (
    EvidenceAttempt,
    EvidenceEntry,
    EvidenceOutcome,
    ExecutionFacets,
    HarnessRunEvidence,
    HarnessRunEvidenceBody,
    HarnessTaskBinding,
    configuration_digest,
    run_evidence_digest,
)
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.trajectory import (
    TrajectoryManifest,
    TrajectoryOpenRecord,
    TrajectoryRecordingSpec,
    validate_manifest,
)
from xaytune.harness.compiler import RESULT_FILE, TRAJECTORY_DIRECTORY
from xaytune.harness.runner import RUNNER
from xaytune.trajectory._files import read_bytes
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore
from xaytune.trajectory.verify import load_events, verify_publication, verify_trajectory

__all__ = [
    "EvidenceRefusedError",
    "HarnessEvidenceStore",
    "ListedAttempt",
    "assemble_evidence",
    "verify_run_evidence",
]


class EvidenceRefusedError(ValueError):
    """Evidence could not be assembled, or did not verify. Carries every reason.

    Reasons name tasks, positions and attempt ids, never content.
    """

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the harness run evidence is refused: " + "; ".join(reasons))


@dataclass(frozen=True)
class ListedAttempt:
    """An attempt instance the store lists, read from disk, never from a record."""

    paths: AttemptPaths
    store: LocalTrajectoryStore
    open_record: TrajectoryOpenRecord
    manifest: TrajectoryManifest | None
    """``None`` while the attempt is open."""
    manifest_problem: str | None = None
    """Why a manifest that exists does not validate."""

    @property
    def trajectory_id(self) -> str:
        return self.paths.trajectory_id


class HarnessEvidenceStore:
    """The trajectories of one replicate's tasks, under a root the caller names.

    Opening reads only: a missing store is an empty one, never created.
    """

    def __init__(self, root: Path) -> None:
        root = Path(root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("the evidence root must be an absolute path without '..'")
        self.root = root

    @property
    def tasks_directory(self) -> Path:
        return self.root / "tasks"

    def task_output(self, position: int) -> Path:
        """Where the task at manifest *position* is compiled to publish (its ``output_uri``)."""
        return self.tasks_directory / f"{position:04d}"

    def result_location(self, position: int) -> Path:
        return self.task_output(position) / RESULT_FILE

    def _stores(self) -> tuple[LocalTrajectoryStore, ...]:
        if not self.tasks_directory.is_dir():
            return ()
        found = []
        for name in sorted(os.listdir(self.tasks_directory)):
            task = self.tasks_directory / name
            directory = task / TRAJECTORY_DIRECTORY
            if task.is_symlink() or directory.is_symlink():
                raise ValueError("a task directory in the evidence store is a symlink")
            if (directory / "attempts").is_dir():
                found.append(LocalTrajectoryStore(directory))
        return tuple(found)

    def attempts_for(
        self, request_digest: str
    ) -> tuple[tuple[ListedAttempt, ...], tuple[str, ...]]:
        """Every attempt instance anywhere under the root that names *request_digest*.

        Also every reason the listing itself is not trustworthy: an attempt
        whose open record cannot be read (so it cannot be attributed), or an
        attempt of this digest held in quarantine, whose evidence cannot be
        measured and whose spend would otherwise vanish.
        """
        listed: list[ListedAttempt] = []
        problems: list[str] = []
        for store in self._stores():
            for paths in store.attempts():
                record = store.read_open_record(paths)
                if record is None:
                    problems.append(
                        f"attempt {paths.trajectory_id} has no readable open record, so it "
                        f"cannot be attributed"
                    )
                    continue
                if record.attribution.request_digest != request_digest:
                    continue
                manifest, problem = _manifest(paths)
                listed.append(ListedAttempt(paths, store, record, manifest, problem))
            for name in sorted(os.listdir(store.quarantine_directory)):
                held = AttemptPaths(store.quarantine_directory / name)
                record = store.read_open_record(held)
                if record is not None and record.attribution.request_digest == request_digest:
                    problems.append(
                        f"attempt {name} of this task is quarantined: its evidence cannot be "
                        f"measured, and its spend cannot be accounted for"
                    )
        listed.sort(key=lambda attempt: attempt.trajectory_id)
        return tuple(listed), tuple(problems)


def _manifest(paths: AttemptPaths) -> tuple[TrajectoryManifest | None, str | None]:
    if not os.path.lexists(paths.manifest):
        return None, None
    try:
        return validate_manifest(read_bytes(paths.manifest)), None
    except (OSError, ValidationError):
        return None, f"attempt {paths.trajectory_id}'s sealed manifest does not validate"


def _result(paths: AttemptPaths) -> HarnessRunResult | None:
    try:
        return HarnessRunResult.model_validate_json(read_bytes(paths.result))
    except (OSError, ValidationError):
        return None


# ---- verification ---------------------------------------------------------------------------


Sampling = AllTasks | FirstTasks | SeededSample


def verify_run_evidence(
    evidence: HarnessRunEvidence,
    suite: ResolvedBenchmarkSuite,
    source: BenchmarkSource,
    store: HarnessEvidenceStore,
) -> HarnessRunEvidence:
    """*evidence*, once every fact it states is rebuilt from the suite and the store.

    Refuses, listing every reason (ADR-022 V3):

    - the suite's bytes (``verify_suite_content``) and the assignment
      (``verify_assignment``);
    - one entry per assigned task, in assignment order, none missing or
      repeated, and at least one attempt in each (G3g);
    - attempt membership: each entry's attempts must **equal** the attempts
      the store lists for its request digest -- none left out, none added,
      none still open, none quarantined;
    - attribution: task pin, cohort, seed, replicate and harness, on every
      attempt;
    - integrity: ``validate_manifest`` and ``verify_trajectory`` on every
      attempt, complete or partial (G3d, G4);
    - the facets and configuration the trajectories record;
    - the entry state: ``attested`` needs its outcome attempt sealed
      ``complete``, verified and published here (``verify_publication``);
      ``no-attested-result`` is refused when any attempt is complete (G3a,
      G3b);
    - the digest.

    Raises:
        EvidenceRefusedError: With every reason.
    """
    reasons: list[str] = []
    try:
        verify_suite_content(suite, source)
    except UnverifiedBenchmarkError as error:
        reasons.extend(f"suite: {reason}" for reason in error.reasons)
    if evidence.suite != suite.ref:
        reasons.append("the evidence names another suite than the one verified against")
    try:
        verify_assignment(suite, evidence.sampling, evidence.assignment)
    except UnverifiedAssignmentError as error:
        reasons.extend(f"assignment: {reason}" for reason in error.reasons)
    if evidence.assignment.seed != evidence.seed:
        reasons.append("the assignment's seed is not the replicate's seed")

    entries = evidence.entries
    recorded_ids = [entry.task_id for entry in entries]
    assigned = list(evidence.assignment.task_ids)
    if recorded_ids != assigned:
        missing = [t for t in assigned if t not in recorded_ids]
        repeated = sorted({t for t in recorded_ids if recorded_ids.count(t) > 1})
        extra = [t for t in recorded_ids if t not in assigned]
        if missing:
            reasons.append(f"entries are missing for assigned tasks {missing}")
        if repeated:
            reasons.append(f"tasks {repeated} have more than one entry")
        if extra:
            reasons.append(f"entries name tasks that are not assigned: {extra}")
        if not (missing or repeated or extra):
            reasons.append("the entries are not in assignment order")

    expected_configuration = configuration_digest(evidence.configuration)
    for entry in entries:
        reasons.extend(_entry_reasons(evidence, entry, suite, store, expected_configuration))

    if evidence.evidence_digest != run_evidence_digest(evidence):
        reasons.append("the evidence digest is not the digest of its contents")
    if reasons:
        raise EvidenceRefusedError(tuple(dict.fromkeys(reasons)))
    return evidence


def _entry_reasons(
    evidence: HarnessRunEvidence,
    entry: EvidenceEntry,
    suite: ResolvedBenchmarkSuite,
    store: HarnessEvidenceStore,
    configuration: str,
) -> list[str]:
    where = f"task {entry.task_id}"
    reasons: list[str] = []
    members = suite.manifest.members
    member = members[entry.position] if entry.position < len(members) else None
    if member is None or member.task_id != entry.task_id:
        reasons.append(f"{where}: its position is not its place in the suite")
    elif (member.revision, member.task_fingerprint) != (entry.revision, entry.task_fingerprint):
        reasons.append(f"{where}: its revision or task fingerprint is not the suite's pin")

    if not entry.attempts:
        reasons.append(f"{where}: the entry lists no attempts, and that is not an execution")
        return reasons

    listed, listing_problems = store.attempts_for(entry.request_digest)
    reasons.extend(f"{where}: {problem}" for problem in listing_problems)
    recorded = [attempt.trajectory_id for attempt in entry.attempts]
    if len(set(recorded)) != len(recorded):
        reasons.append(f"{where}: an attempt is listed more than once")
    in_store = {attempt.trajectory_id: attempt for attempt in listed}
    for trajectory_id, attempt in in_store.items():
        if attempt.manifest is None and attempt.manifest_problem is None:
            reasons.append(
                f"{where}: attempt {trajectory_id} is still open; reconcile it before "
                f"assembling evidence"
            )
        elif trajectory_id not in recorded:
            reasons.append(f"{where}: attempt {trajectory_id} is in the store but left out")
    for trajectory_id in recorded:
        if trajectory_id not in in_store:
            reasons.append(
                f"{where}: attempt {trajectory_id} is not in the store for this request digest"
            )
    if not listed:
        return reasons

    complete_and_verified: list[str] = []
    for claimed in entry.attempts:
        actual = in_store.get(claimed.trajectory_id)
        if actual is None:
            continue
        problems, verified = _attempt_reasons(
            evidence, entry, claimed, actual, configuration, where
        )
        reasons.extend(problems)
        if verified and actual.manifest is not None and actual.manifest.completeness == "complete":
            complete_and_verified.append(actual.trajectory_id)

    reasons.extend(_state_reasons(evidence, entry, in_store, complete_and_verified, store, where))
    return reasons


def _attempt_reasons(
    evidence: HarnessRunEvidence,
    entry: EvidenceEntry,
    claimed: EvidenceAttempt,
    actual: ListedAttempt,
    configuration: str,
    where: str,
) -> tuple[list[str], bool]:
    """Every way *claimed* is not the sealed *actual*; and whether *actual* verifies."""
    attempt = f"{where}: attempt {actual.trajectory_id}"
    if actual.manifest_problem is not None:
        return [f"{where}: {actual.manifest_problem}"], False
    manifest = actual.manifest
    if manifest is None:
        return [f"{attempt} is still open; reconcile it before assembling evidence"], False
    reasons: list[str] = []
    if (claimed.trajectory_fingerprint, claimed.completeness, claimed.abandonment) != (
        manifest.trajectory_fingerprint,
        manifest.completeness,
        manifest.abandonment,
    ):
        reasons.append(f"{attempt}: the entry does not record it as it is sealed")
    attribution = manifest.attribution
    for name, stated, expected in (
        ("request digest", attribution.request_digest, entry.request_digest),
        ("task", attribution.task.task_id, entry.task_id),
        ("task revision", attribution.task.revision, entry.revision),
        ("task fingerprint", attribution.task.task_fingerprint, entry.task_fingerprint),
        ("cohort", attribution.assignment_cohort, evidence.assignment.cohort),
        ("seed", attribution.seed, evidence.seed),
        ("replicate", attribution.replicate, evidence.replicate),
        ("harness", attribution.harness_fingerprint, evidence.harness_fingerprint),
    ):
        if stated != expected:
            reasons.append(f"{attempt}: its {name} is not the evidence's")
    facets = evidence.facets
    if (manifest.runner.name, manifest.runner.version) != (
        facets.runner_name,
        facets.runner_version,
    ):
        reasons.append(f"{attempt}: it ran under another runner than the facets declare")
    if manifest.recording.digest() != facets.recording:
        reasons.append(f"{attempt}: it was recorded under another recording spec")
    integrity = verify_trajectory(actual.store, actual.paths, manifest)
    reasons.extend(f"{attempt} does not verify: {problem}" for problem in integrity)
    if not integrity:
        events, _ = load_events(actual.store, actual.paths, manifest)
        # An empty prefix (all evidence lost) records no configuration to contradict.
        if events and getattr(events[0], "configuration_digest", None) != configuration:
            reasons.append(f"{attempt}: it ran under another harness configuration")
    return reasons, not integrity


def _state_reasons(
    evidence: HarnessRunEvidence,
    entry: EvidenceEntry,
    in_store: dict[str, ListedAttempt],
    complete_and_verified: list[str],
    store: HarnessEvidenceStore,
    where: str,
) -> list[str]:
    reasons: list[str] = []
    if len(complete_and_verified) > 1:
        reasons.append(f"{where}: more than one attempt attests a result")
    if entry.state == "no-attested-result":
        if entry.outcome is not None:
            reasons.append(f"{where}: an entry with no attested result names an outcome")
        if complete_and_verified:
            reasons.append(f"{where}: an attempt attests a result, which cannot be declared absent")
        return reasons
    outcome = entry.outcome
    if outcome is None:
        return [*reasons, f"{where}: an attested entry names no outcome attempt"]
    actual = in_store.get(outcome.trajectory_id)
    manifest = actual.manifest if actual is not None else None
    if actual is None or manifest is None:
        return [*reasons, f"{where}: the outcome attempt is not a sealed attempt of this task"]
    if manifest.completeness != "complete":
        return [*reasons, f"{where}: the outcome attempt is sealed partial and attests nothing"]
    if outcome.trajectory_id not in complete_and_verified:
        return [*reasons, f"{where}: the outcome attempt does not verify"]
    if (outcome.trajectory_fingerprint, outcome.result_digest, outcome.termination) != (
        manifest.trajectory_fingerprint,
        manifest.result_digest,
        manifest.termination,
    ):
        reasons.append(f"{where}: the outcome is not recorded as its attempt sealed it")
    published = verify_publication(actual.paths, manifest, store.result_location(entry.position))
    reasons.extend(f"{where}: the outcome is not published here: {p}" for p in published)
    result = _result(actual.paths)
    if result is None:
        reasons.append(f"{where}: the outcome attempt's result does not parse")
    else:
        facets = evidence.facets
        if (result.compiler.name, result.compiler.version, result.token_counter) != (
            facets.compiler_name,
            facets.compiler_version,
            facets.token_counter,
        ):
            reasons.append(f"{where}: the result names another compiler or token counter")
    return reasons


# ---- assembly -------------------------------------------------------------------------------


def _facet_digest(value: Any) -> str:
    return fingerprint({"type": "harness-execution-facet", "value": value})


def _facets(spec: AgentExecutionSpec, timing_environment: str | None) -> ExecutionFacets:
    recording = spec.config.get("trajectory")
    return ExecutionFacets(
        runner_name=RUNNER.name,
        runner_version=RUNNER.version,
        compiler_name=spec.compiler.name,
        compiler_version=spec.compiler.version,
        token_counter=str(spec.config.get("token_counter")),
        entrypoint=_facet_digest(spec.entrypoint.model_dump(mode="json")),
        container=_facet_digest(
            spec.container.model_dump(mode="json") if spec.container is not None else None
        ),
        dependencies=_facet_digest(spec.dependencies.model_dump(mode="json")),
        resources=_facet_digest(spec.resources.model_dump(mode="json")),
        recording=TrajectoryRecordingSpec.model_validate(thaw(recording)).digest()
        if recording is not None
        else _facet_digest(None),
        timing_environment=timing_environment,
    )


def _configuration(spec: AgentExecutionSpec) -> FrozenDict:
    harness = spec.harness
    return FrozenDict(
        {
            "context": harness.context.model_dump(mode="json"),
            "stopping": harness.stopping.model_dump(mode="json"),
            "generation": harness.model.generation.model_dump(mode="json"),
        }
    )


def assemble_evidence(
    specs: Sequence[AgentExecutionSpec],
    suite: ResolvedBenchmarkSuite,
    sampling: Sampling,
    assignment: TaskAssignment,
    source: BenchmarkSource,
    store: HarnessEvidenceStore,
    *,
    timing_environment: str | None = None,
) -> HarnessRunEvidence:
    """The evidence of one execution replicate, read from the store and then verified.

    *specs* are the task-bound specs of the replicate, one per assigned task,
    in assignment order, each compiled to ``store.task_output(position)``.
    Each entry records every attempt instance the store lists for its
    request digest; its state is ``attested`` when one attempt is sealed
    ``complete``, and ``no-attested-result`` otherwise. Whether that is true
    is :func:`verify_run_evidence`'s to decide, which assembly calls last, so
    nothing is returned that would not verify.

    Raises:
        EvidenceRefusedError: With every reason.
    """
    reasons: list[str] = []
    try:
        verify_suite_content(suite, source)
        verify_assignment(suite, sampling, assignment)
    except (UnverifiedBenchmarkError, UnverifiedAssignmentError) as error:
        raise EvidenceRefusedError(tuple(error.reasons)) from None
    if [spec.task.task_id for spec in specs] != list(assignment.task_ids):
        raise EvidenceRefusedError(
            ("the specs are not one per assigned task, in assignment order",)
        )
    first = specs[0]
    facets = _facets(first, timing_environment)
    configuration = _configuration(first)
    entries: list[EvidenceEntry] = []
    for spec in specs:
        block = spec.config.get("task")
        if block is None:
            reasons.append(f"task {spec.task.task_id}: the spec is not task-bound")
            continue
        binding = HarnessTaskBinding.model_validate(thaw(block))
        if binding.suite != suite.ref or binding.cohort != assignment.cohort:
            reasons.append(f"task {spec.task.task_id}: bound to another suite or cohort")
        if (spec.harness_fingerprint, spec.seed, spec.replicate) != (
            first.harness_fingerprint,
            first.seed,
            first.replicate,
        ):
            reasons.append(f"task {spec.task.task_id}: another harness, seed or replicate")
        if spec.seed != assignment.seed:
            reasons.append(f"task {spec.task.task_id}: its seed is not the assignment's")
        if _facets(spec, timing_environment) != facets or _configuration(spec) != configuration:
            reasons.append(f"task {spec.task.task_id}: it runs under other execution conditions")
        if Path(spec.outputs[0].uri) != store.result_location(binding.position):
            reasons.append(
                f"task {spec.task.task_id}: it is not compiled to its place in the evidence store"
            )
        entries.append(_entry(spec, binding, store))
    if reasons:
        raise EvidenceRefusedError(tuple(reasons))
    body = HarnessRunEvidenceBody(
        suite=suite.ref,
        sampling=sampling,
        assignment=assignment,
        harness_fingerprint=first.harness_fingerprint,
        seed=first.seed,
        replicate=first.replicate,
        facets=facets,
        configuration=configuration,
        entries=tuple(entries),
    )
    evidence = HarnessRunEvidence(
        **{name: getattr(body, name) for name in HarnessRunEvidenceBody.model_fields},
        evidence_digest=run_evidence_digest(body),
    )
    return verify_run_evidence(evidence, suite, source, store)


def _entry(
    spec: AgentExecutionSpec, binding: HarnessTaskBinding, store: HarnessEvidenceStore
) -> EvidenceEntry:
    request_digest = agent_run_request_digest(spec)
    listed, _ = store.attempts_for(request_digest)
    attempts = tuple(
        EvidenceAttempt(
            trajectory_id=attempt.trajectory_id,
            trajectory_fingerprint=attempt.manifest.trajectory_fingerprint,
            completeness=attempt.manifest.completeness,
            abandonment=attempt.manifest.abandonment,
        )
        for attempt in listed
        if attempt.manifest is not None
    )
    complete = [a.manifest for a in listed if a.manifest and a.manifest.completeness == "complete"]
    outcome = None
    if complete:
        manifest = complete[0]
        assert manifest.result_digest is not None and manifest.termination is not None
        outcome = EvidenceOutcome(
            trajectory_id=str(manifest.trajectory_id),
            trajectory_fingerprint=manifest.trajectory_fingerprint,
            result_digest=manifest.result_digest,
            termination=manifest.termination,
        )
    return EvidenceEntry(
        task_id=binding.task_id,
        revision=binding.revision,
        task_fingerprint=binding.task_fingerprint,
        position=binding.position,
        request_digest=request_digest,
        attempts=attempts,
        state="attested" if outcome is not None else "no-attested-result",
        outcome=outcome,
    )
