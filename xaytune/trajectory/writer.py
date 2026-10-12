"""The crash-safe trajectory writer (ADR-021 T3, T4, T8).

One writer records one attempt instance, from its open record to its seal:

```text
open()            mint the attempt instance id; open.json, no-overwrite; hold the lock
content()         scan, redact, write a blob durably: before any event names it
record()          scan, append one event to the tail, fsynced: before the call it announces
stage(result)     roll the final chunk; result.json; manifest.staged.json; verify_trajectory
commit(location)  ══ the commit point ══ hard-link result.json to the declared result
seal_complete()   verify_publication; link the staged manifest to manifest.json
abandon(reason)   before the commit point: seal partial, attesting no result
contain(reason)   a secret discovered late: quarantine the whole attempt
```

**Secrets.** The registered set is frozen at ``open``; registering one later
is refused. Every blob and every event line is scanned for that set before
it is written, so no value of the set, and no digest of one, is ever
written. Nothing is claimed about values outside it.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

from xaytune.core.clock import utc_now
from xaytune.core.ids import AttemptInstanceId
from xaytune.core.trajectory import (
    Abandonment,
    CapturedContent,
    ContentSlot,
    ErrorObserved,
    RedactedContent,
    TrajectoryAttribution,
    TrajectoryManifest,
    TrajectoryOpenRecord,
    TrajectoryRecordingSpec,
    WriterIdentity,
    chunk_bytes,
    event_id_for,
    seal,
)
from xaytune.trajectory._files import (
    UnsafePathError,
    append_durably,
    create_durably,
    device_of,
    digest_of,
    link_new,
    refuse_symlinks,
    same_file,
    truncate_durably,
    write_new,
)
from xaytune.trajectory.sealing import manifest_body, manifest_bytes, read_manifest, roll
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore, TrajectoryProgress
from xaytune.trajectory.verify import (
    TrajectoryVerificationError,
    verify_publication,
    verify_trajectory,
)

__all__ = [
    "WRITER",
    "Redactor",
    "SecretDetectedError",
    "SecretRegistrationClosedError",
    "TrajectoryRecordingError",
    "TrajectoryWriter",
    "contain_attempt",
]

WRITER = WriterIdentity(name="xaytune.local-trajectory-writer", version="1")


class TrajectoryRecordingError(RuntimeError):
    """The writer can no longer record. Before the commit point, no result is published."""


class SecretDetectedError(TrajectoryRecordingError):
    """A registered secret value was about to be written. It was not written.

    Names the field role only, never the value.
    """

    def __init__(self, role: str) -> None:
        self.role = role
        super().__init__(f"a registered secret value appeared in {role}; nothing was written")


class SecretRegistrationClosedError(RuntimeError):
    """The registered-secret set is frozen when the attempt opens (T8)."""


class Redactor(Protocol):
    """A host's redaction rules, named by the digest the recording spec carries."""

    @property
    def policy_digest(self) -> str: ...

    def rule_for(self, role: str, text: str) -> str | None:
        """The name of the rule that removes *text* from *role*, or ``None`` to keep it."""
        ...


Clock = Callable[[], str]
ErrorCode = Literal["secret-detected", "recording-failed"]


def _now() -> str:
    return utc_now().isoformat()


class TrajectoryWriter:
    """Records one attempt instance. Use :meth:`open`; close with a seal, an abandon or contain."""

    def __init__(
        self,
        store: LocalTrajectoryStore,
        paths: AttemptPaths,
        record: TrajectoryOpenRecord,
        lock: int,
        secrets: tuple[str, ...],
        redactor: Redactor | None,
        clock: Clock,
    ) -> None:
        self.store = store
        self.paths = paths
        self.opened = record
        self._lock: int | None = lock
        self._secrets = secrets
        self._redactor = redactor
        self._clock = clock
        self._events: list[Any] = []
        self._tail: list[Any] = []
        self._tail_bytes = 0
        self._progress = TrajectoryProgress(trajectory_id=str(record.trajectory_id))
        self._staged: TrajectoryManifest | None = None
        self._state: Literal["open", "staged", "committed", "sealed", "contained", "released"] = (
            "open"
        )

    # ---- opening -----------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        store: LocalTrajectoryStore,
        *,
        attribution: TrajectoryAttribution,
        recording: TrajectoryRecordingSpec,
        runner: WriterIdentity,
        secrets: Iterable[str] = (),
        secret_sources: Iterable[str] = (),
        redactor: Redactor | None = None,
        clock: Clock | None = None,
    ) -> TrajectoryWriter:
        """Mint an attempt instance and persist its open record before anything else.

        Raises:
            ValueError: If the recording spec names a redaction policy and no
                redactor with that digest is supplied, or a writer other than
                this one.
        """
        if recording.writer != WRITER:
            raise ValueError("the recording spec names another trajectory writer")
        if recording.redaction_policy_digest is not None and (
            redactor is None or redactor.policy_digest != recording.redaction_policy_digest
        ):
            raise ValueError("the recording spec's redaction policy has no matching redactor")
        frozen = tuple(sorted({value for value in secrets if value}))
        trajectory_id = AttemptInstanceId.generate()
        paths = store.create_attempt(trajectory_id)
        lock = os.open(paths.lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        clock = clock or _now
        record = TrajectoryOpenRecord(
            trajectory_id=trajectory_id,
            attribution=attribution,
            writer=WRITER,
            runner=runner,
            recording=recording,
            secrets_registered=len(frozen),
            secret_sources=tuple(secret_sources),
            opened_at=clock(),
        )
        writer = cls(store, paths, record, lock, frozen, redactor, clock)
        try:
            writer._scan_value(record.model_dump(mode="json"), "the open record")
            write_new(paths.open_record, record.model_dump_json().encode("utf-8"))
            # The tail exists, durably named by its directory, before any event
            # depends on it: the first start event is a write-ahead point too.
            create_durably(paths.tail)
        except BaseException:
            writer.close()
            raise
        return writer

    @property
    def trajectory_id(self) -> AttemptInstanceId:
        return self.opened.trajectory_id

    @property
    def events(self) -> tuple[Any, ...]:
        return tuple(self._events)

    @property
    def next_sequence(self) -> int:
        return len(self._events)

    @property
    def committed(self) -> bool:
        """Whether the result is published: past the commit point, nothing is rolled back."""
        return self._state in ("committed", "sealed")

    def register_secret(self, value: str) -> None:
        """Always refused: the set was frozen when the attempt opened (T8)."""
        raise SecretRegistrationClosedError(
            "secrets are registered before the attempt opens; contain() the attempt instead"
        )

    # ---- recording ---------------------------------------------------------------------------

    def _scan(self, text: str, role: str) -> None:
        for value in self._secrets:
            if value in text:
                raise SecretDetectedError(role)

    def _scan_value(self, value: Any, role: str) -> None:
        """Scan every string in *value*, keys included, as the raw text it is.

        Before serialization, never after: JSON escapes quotes, backslashes,
        control characters and (by default) non-ASCII text, so a secret
        containing any of them no longer occurs literally in the serialized
        form.
        """
        if isinstance(value, str):
            self._scan(value, role)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                self._scan_value(key, role)
                self._scan_value(item, role)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._scan_value(item, role)

    def _require_open(self) -> None:
        if self._state != "open":
            raise TrajectoryRecordingError(
                f"the attempt is {self._state}; nothing more is recorded"
            )

    def content(
        self,
        text: str,
        *,
        role: str,
        media_type: str = "text/plain; charset=utf-8",
        sensitive: bool = False,
    ) -> ContentSlot:
        """Capture *text* as a blob, durably, before any event names it (write-ahead rule 2)."""
        self._require_open()
        self._scan(text, role)
        if media_type.startswith("application/json"):
            try:
                self._scan_value(json.loads(text), role)
            except ValueError as error:
                raise TrajectoryRecordingError(f"{role} is not the JSON it claims") from error
        if self._redactor is not None:
            rule = self._redactor.rule_for(role, text)
            if rule is not None:
                return RedactedContent(
                    rule=rule, secret=False, removed_digest=digest_of(text.encode("utf-8"))
                )
        data = text.encode("utf-8")
        digest = digest_of(data)
        tier: Literal["ordinary", "restricted"] = "restricted" if sensitive else "ordinary"
        try:
            self.store.put_blob(self.paths, tier, digest, data)
        except OSError as error:
            raise TrajectoryRecordingError("the store refused a blob") from error
        return CapturedContent(
            disposition="captured-restricted" if sensitive else "captured",
            digest=digest,
            size=len(data),
            media_type=media_type,
        )

    def record(self, event_type: type[Any], *, parent: int | None, **fields: Any) -> Any:
        """Append one event, durably, and return it (write-ahead rule 3)."""
        self._require_open()
        sequence = len(self._events)
        event = event_type(
            trajectory_id=self.trajectory_id,
            sequence=sequence,
            event_id=event_id_for(self.trajectory_id, sequence),
            parent=parent,
            emitted_at=self._clock(),
            **fields,
        )
        line = (
            json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self._scan_value(event.model_dump(mode="json"), f"the {event.kind} event")
        self._scan(line.decode("utf-8"), f"the {event.kind} event")
        recording = self.opened.recording
        header = len(chunk_bytes(str(self.trajectory_id), 999_999, "sha256:" + "0" * 64, []))
        if header + len(line) > recording.max_bytes:
            raise TrajectoryRecordingError("an event is larger than the chunk's max_bytes")
        if self._tail and (
            len(self._tail) >= recording.max_events
            or header + self._tail_bytes + len(line) > recording.max_bytes
        ):
            self._roll()
        try:
            append_durably(self.paths.tail, line)
        except OSError as error:
            raise TrajectoryRecordingError("the store refused an event") from error
        self._events.append(event)
        self._tail.append(event)
        self._tail_bytes += len(line)
        return event

    def _roll(self) -> None:
        try:
            self._progress = roll(self.store, self.paths, self._progress, self._tail)
            truncate_durably(self.paths.tail)
        except OSError as error:
            raise TrajectoryRecordingError("the store refused a chunk") from error
        self._tail = []
        self._tail_bytes = 0

    # ---- before the commit point ------------------------------------------------------------

    def stage(self, result_bytes: bytes) -> TrajectoryManifest:
        """Stage everything the commit point needs, then verify it (steps 2-4 of T4).

        Raises:
            TrajectoryRecordingError: If anything fails to write or verify. No
                result has been published.
        """
        self._require_open()
        terminated = self._events[-1] if self._events else None
        if terminated is None or terminated.kind != "run-terminated":
            raise TrajectoryRecordingError("a run is staged only after run-terminated")
        if terminated.result_digest != digest_of(result_bytes):
            raise TrajectoryRecordingError("the result is not the one run-terminated named")
        text = result_bytes.decode("utf-8")
        self._scan(text, "the result")
        try:
            self._scan_value(json.loads(text), "the result")
        except ValueError as error:
            raise TrajectoryRecordingError("the result is not JSON") from error
        self._roll()
        try:
            write_new(self.paths.result, result_bytes)
            staged = seal(
                manifest_body(
                    self.opened,
                    self._progress.chunks,
                    self._events,
                    completeness="complete",
                    result_digest=terminated.result_digest,
                )
            )
            write_new(self.paths.staged, manifest_bytes(staged))
        except OSError as error:
            raise TrajectoryRecordingError("the store refused the staged evidence") from error
        problems = verify_trajectory(self.store, self.paths, staged)
        if problems:
            raise TrajectoryRecordingError(
                "the staged evidence does not verify: " + "; ".join(problems)
            )
        self._staged = staged
        self._state = "staged"
        return staged

    def abandon(
        self,
        reason: Abandonment,
        *,
        error_code: Literal["secret-detected", "recording-failed"] | None = None,
        role: str | None = None,
    ) -> TrajectoryManifest:
        """Seal partial, attesting no result. Records a sanitized error first, if it can."""
        if self._state not in ("open", "staged"):
            raise TrajectoryRecordingError(f"the attempt is {self._state}; it cannot be abandoned")
        try:
            return self._seal_partial(reason, error_code, role)
        finally:
            # Sealed or not, the lock is released: if the partial seal could
            # not be written (a full disk), reconciliation recovers the attempt.
            if self._state != "sealed":
                self._state = "released"
            self.close()

    def _seal_partial(
        self,
        reason: Abandonment,
        error_code: Literal["secret-detected", "recording-failed"] | None,
        role: str | None,
    ) -> TrajectoryManifest:
        self._state = "open"
        if error_code is not None and (
            not self._events or self._events[-1].kind != "run-terminated"
        ):
            try:
                self.record(
                    ErrorObserved,
                    parent=self._events[-1].sequence if self._events else None,
                    code=error_code,
                    role=role,
                )
            except TrajectoryRecordingError:
                pass
        try:
            self._roll()
        except TrajectoryRecordingError:
            pass
        manifest = seal(
            manifest_body(
                self.opened,
                self._progress.chunks,
                [e for e in self._events if e.sequence <= self._last_rolled()],
                completeness="partial",
                abandonment=reason,
            )
        )
        write_new(self.paths.manifest, manifest_bytes(manifest))
        self._state = "sealed"
        return manifest

    def _last_rolled(self) -> int:
        return self._progress.chunks[-1].last_sequence if self._progress.chunks else -1

    # ---- the commit point and after ---------------------------------------------------------

    def check_publishable(self, location: Path) -> None:
        """Before opening: the result must be linkable from this attempt (same filesystem)."""
        location.parent.mkdir(parents=True, exist_ok=True)
        if device_of(location.parent) != device_of(self.paths.directory):
            raise UnsafePathError("the result and the trajectory are on different filesystems")

    def commit(self, location: Path) -> bool:
        """Publish ``result.json`` by a no-overwrite hard link: the commit point (T4 step 5).

        Returns ``False`` without publishing if the location already holds a
        file.

        Raises:
            UnsafePathError: If the location's directory or the attempt is on
                another filesystem, or a path component is a symlink.
        """
        if self._state != "staged":
            raise TrajectoryRecordingError("only staged evidence can be committed")
        refuse_symlinks(location, below=location.parent)
        refuse_symlinks(self.paths.result, below=self.store.root)
        if device_of(location.parent) != device_of(self.paths.directory):
            raise UnsafePathError("the result and the trajectory are on different filesystems")
        try:
            linked = link_new(self.paths.result, location)
        except BaseException:
            # The link may exist although a later step (syncing its directory)
            # failed. Past the commit point nothing is rolled back, so the
            # state follows the filesystem, not the exception.
            if same_file(location, self.paths.result):
                self._state = "committed"
            raise
        if not linked:
            return False
        self._state = "committed"
        return True

    def seal_complete(self, location: Path) -> TrajectoryManifest:
        """Verify the publication, then link the staged manifest into place (T4 step 6).

        Raises:
            TrajectoryVerificationError: If the publication does not verify; the
                result stands, and reconciliation completes or reports it.
        """
        if self._state != "committed" or self._staged is None:
            raise TrajectoryRecordingError("only a committed attempt is sealed complete")
        problems = verify_publication(self.paths, self._staged, location)
        if problems:
            raise TrajectoryVerificationError(problems)
        if not link_new(self.paths.staged, self.paths.manifest):
            existing = read_manifest(self.paths.manifest)
            if existing != self._staged:
                raise TrajectoryRecordingError("a different manifest is already sealed")
        self._state = "sealed"
        self.close()
        return self._staged

    def contain(self, reason: str, *, location: Path | None = None) -> AttemptPaths:
        """Quarantine this attempt for a late-discovered secret (T8): :func:`contain_attempt`."""
        committed = self.committed
        self.close()
        self._state = "contained"
        return contain_attempt(
            self.store, self.paths, reason, location=location if committed else None
        )

    def close(self) -> None:
        """Release the liveness lock. Idempotent."""
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None


def contain_attempt(
    store: LocalTrajectoryStore,
    paths: AttemptPaths,
    reason: str,
    *,
    location: Path | None = None,
) -> AttemptPaths:
    """Quarantine a whole attempt, blobs included, because a secret was discovered late (T8).

    Works on an open, abandoned or sealed attempt alike. It seals nothing, and
    claims nothing about what earlier writes contained: no scan is made to
    prove the secret absent. The incident record carries the reason and the
    attempt id, never a value or a digest of one. If *location* is this
    attempt's published result, its inode is made owner-only, which restricts
    the published path as well, and the result can no longer be attested:
    replay refuses it.
    """
    published = location is not None and same_file(location, paths.result)
    incident = {
        "schema_version": "xaytune.trajectory-incident/v1alpha1",
        "trajectory_id": paths.trajectory_id,
        "kind": "late-secret",
        "reason": reason,
        "published_result_restricted": published,
        "at": _now(),
    }
    write_new(paths.incident, json.dumps(incident, sort_keys=True).encode("utf-8"))
    if published:
        assert location is not None
        os.chmod(location, 0o600)
    return store.contain(paths)
