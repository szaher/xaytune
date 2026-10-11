"""Resolving a benchmark suite to pinned, verified tasks (H04, ADR-020 B4).

```text
BenchmarkSuiteRequest | BenchmarkSuiteRef
      ↓ resolve_benchmark_suite(source)
ResolvedBenchmarkSuite        manifest + every task, every fingerprint and byte verified
      ├─ verify_suite_content(suite, source) its bytes, again, after it was loaded
      ├─ resolve_task_inputs(task, source)   agent-visible attachment text
      └─ resolve_reference(task, source)     held-out text, for evaluation only
```

Resolution is the recorded step between asking for a suite and comparing on
it. It pins an alias to a revision and checks everything the suite reaches:

- the manifest has the id and revision asked for (or pinned);
- the manifest has the fingerprint a pinned ref names, so an alias that has
  moved upstream cannot change a pinned suite;
- each member task has its pinned revision and fingerprint;
- every attachment, held-out reference and environment snapshot has its
  digest. Text content must also be ``text/*`` and UTF-8.

There are two levels of verification. Parsing a ``ResolvedBenchmarkSuite``
checks its **structure** only, because it carries fingerprints and digests,
not bytes. Checking **bytes** needs a source: resolution does it once, and
:func:`verify_suite_content` repeats it for a record that was stored, passed
between processes or simply kept. Whatever executes or evaluates on a suite
(H06) must repeat it first.

Every failure is reported, not just the first. Reasons name a role such as
``support/refund-001 attachments.policy``, never a locator or bytes.

Nothing here fetches. A :class:`BenchmarkSource` is supplied by the host;
H04 ships only :class:`MappingBenchmarkSource`, which serves what the host
already holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from xaytune.core.domain.benchmark import (
    BenchmarkSuiteManifest,
    BenchmarkSuiteRef,
    BenchmarkSuiteRequest,
    ResolvedBenchmarkSuite,
    TaskSpec,
    agent_visible,
)
from xaytune.core.domain.harness import ContentRef
from xaytune.core.immutable import FrozenDict

__all__ = [
    "BenchmarkSource",
    "MappingBenchmarkSource",
    "UnverifiedBenchmarkError",
    "resolve_benchmark_suite",
    "resolve_reference",
    "resolve_task_inputs",
    "verify_suite_content",
]


class UnverifiedBenchmarkError(ValueError):
    """A suite, task or content did not resolve to what was pinned. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the benchmark is not verified: " + "; ".join(reasons))


@runtime_checkable
class BenchmarkSource(Protocol):
    """Where manifests, tasks and content come from. ``None`` means "not held here"."""

    def manifest(self, suite_id: str, revision: str) -> BenchmarkSuiteManifest | None:
        """The manifest for *revision*, which may be an alias; the manifest's own is pinned."""
        ...

    def task(self, task_id: str, revision: str) -> TaskSpec | None: ...

    def content(self, ref: ContentRef) -> bytes | None: ...


class MappingBenchmarkSource:
    """What the host already holds: manifests, tasks, aliases and bytes. No network, no files."""

    def __init__(
        self,
        *,
        manifests: Iterable[BenchmarkSuiteManifest],
        tasks: Iterable[TaskSpec],
        blobs: Mapping[str, bytes],
        aliases: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self._manifests = {(m.id, m.revision): m for m in manifests}
        self._tasks = {(t.id, t.revision): t for t in tasks}
        self._blobs = dict(blobs)
        self._aliases = dict(aliases or {})

    def manifest(self, suite_id: str, revision: str) -> BenchmarkSuiteManifest | None:
        pinned = self._aliases.get((suite_id, revision), revision)
        return self._manifests.get((suite_id, pinned))

    def task(self, task_id: str, revision: str) -> TaskSpec | None:
        return self._tasks.get((task_id, revision))

    def content(self, ref: ContentRef) -> bytes | None:
        return self._blobs.get(ref.uri)


def resolve_benchmark_suite(
    requested: BenchmarkSuiteRequest | BenchmarkSuiteRef, source: BenchmarkSource
) -> ResolvedBenchmarkSuite:
    """Resolve *requested* to a pinned suite whose every task and byte is verified.

    A :class:`BenchmarkSuiteRef` is re-resolved to exactly its fingerprint, or
    refused: a pin cannot move.

    Raises:
        UnverifiedBenchmarkError: With every reason.
    """
    if isinstance(requested, BenchmarkSuiteRef):
        request = BenchmarkSuiteRequest(suite_id=requested.suite_id, revision=requested.revision)
        pinned: BenchmarkSuiteRef | None = requested
    else:
        request, pinned = requested, None

    manifest = source.manifest(request.suite_id, request.revision)
    if manifest is None:
        raise UnverifiedBenchmarkError(("the source does not hold the requested suite",))
    reasons: list[str] = []
    if manifest.id != request.suite_id:
        reasons.append("the source returned a different suite than the one requested")
    fingerprint = manifest.suite_fingerprint()
    if pinned is not None:
        if manifest.revision != pinned.revision:
            reasons.append("the source returned a different revision than the pinned one")
        if fingerprint != pinned.suite_fingerprint:
            reasons.append(
                "the pinned suite has changed: its manifest no longer has the pinned fingerprint"
            )

    tasks: list[TaskSpec] = []
    for member in manifest.members:
        task = source.task(member.task_id, member.revision)
        if task is None:
            reasons.append(f"{member.task_id}: the source does not hold the pinned revision")
            continue
        if (task.id, task.revision) != (member.task_id, member.revision):
            reasons.append(f"{member.task_id}: the source returned a different task")
            continue
        if task.task_fingerprint() != member.task_fingerprint:
            reasons.append(f"{member.task_id}: the task does not have its pinned fingerprint")
            continue
        reasons.extend(_task_content_refusals(task, source))
        tasks.append(task)

    if manifest.environment_snapshot is not None:
        problem = _verify(manifest.environment_snapshot, source, text=False)
        if problem:
            reasons.append(f"environment_snapshot: {problem}")

    if reasons:
        raise UnverifiedBenchmarkError(tuple(reasons))
    try:
        return ResolvedBenchmarkSuite(
            request=request,
            ref=BenchmarkSuiteRef(
                suite_id=manifest.id, revision=manifest.revision, suite_fingerprint=fingerprint
            ),
            manifest=manifest,
            tasks=tuple(tasks),
        )
    except ValidationError as error:  # pragma: no cover - every check above precedes it
        raise UnverifiedBenchmarkError(("the resolved suite is inconsistent",)) from error


def verify_suite_content(suite: ResolvedBenchmarkSuite, source: BenchmarkSource) -> None:
    """Check every byte *suite* names against *source*: attachments, references and snapshot.

    A parsed :class:`ResolvedBenchmarkSuite` proves its structure, not that
    its content is still held or unchanged. Call this before executing or
    evaluating on a suite that was loaded rather than just resolved
    (ADR-020 B4).

    Raises:
        UnverifiedBenchmarkError: With every reason.
    """
    reasons: list[str] = []
    for task in suite.tasks:
        reasons.extend(_task_content_refusals(task, source))
    if suite.manifest.environment_snapshot is not None:
        problem = _verify(suite.manifest.environment_snapshot, source, text=False)
        if problem:
            reasons.append(f"environment_snapshot: {problem}")
    if reasons:
        raise UnverifiedBenchmarkError(tuple(reasons))


def resolve_task_inputs(task: TaskSpec, source: BenchmarkSource) -> FrozenDict:
    """The verified text of every attachment the agent may see, by name.

    Reads only :func:`~xaytune.core.domain.benchmark.agent_visible` inputs,
    so a held-out reference can never come back from here.

    Raises:
        UnverifiedBenchmarkError: With every reason.
    """
    texts: dict[str, str] = {}
    reasons: list[str] = []
    for attachment in agent_visible(task).attachments:
        content = attachment.content
        if not isinstance(content, ContentRef):
            texts[attachment.name] = content.text
            continue
        data, problem = _fetch(content, source, text=True)
        if problem:
            reasons.append(f"{task.id} attachments.{attachment.name}: {problem}")
        else:
            texts[attachment.name] = data.decode("utf-8")
    if reasons:
        raise UnverifiedBenchmarkError(tuple(reasons))
    return FrozenDict(texts)


def resolve_reference(task: TaskSpec, source: BenchmarkSource) -> str | None:
    """The verified held-out reference text of *task*, for evaluation only (ADR-020 B2).

    Never pass the result to an agent. ``None`` when the task has no reference.

    Raises:
        UnverifiedBenchmarkError: If the reference does not verify.
    """
    if task.reference is None:
        return None
    data, problem = _fetch(task.reference, source, text=True)
    if problem:
        raise UnverifiedBenchmarkError((f"{task.id} reference: {problem}",))
    return data.decode("utf-8")


def _task_content_refusals(task: TaskSpec, source: BenchmarkSource) -> list[str]:
    reasons: list[str] = []
    for attachment in task.inputs.attachments:
        if isinstance(attachment.content, ContentRef):
            problem = _verify(attachment.content, source, text=True)
            if problem:
                reasons.append(f"{task.id} attachments.{attachment.name}: {problem}")
    if task.reference is not None:
        problem = _verify(task.reference, source, text=True)
        if problem:
            reasons.append(f"{task.id} reference: {problem}")
    return reasons


def _verify(ref: ContentRef, source: BenchmarkSource, *, text: bool) -> str | None:
    return _fetch(ref, source, text=text)[1]


def _fetch(ref: ContentRef, source: BenchmarkSource, *, text: bool) -> tuple[bytes, str | None]:
    """*ref*'s bytes and ``None``, or empty bytes and why they are refused."""
    if text and not ref.media_type.lower().startswith("text/"):
        return b"", "media type is not text/*"
    data = source.content(ref)
    if data is None:
        return b"", "the source does not hold it"
    if "sha256:" + hashlib.sha256(data).hexdigest() != ref.digest:
        return b"", "the bytes do not have the pinned digest"
    if text:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return b"", "the content is not valid UTF-8"
    return data, None
