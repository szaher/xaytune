"""Sealing attempts whose writer is gone (ADR-021 T4).

For every attempt that has an open record and no sealed manifest:

- **Writer alive** (its lock is held): skipped. On one host, ``flock`` is the
  liveness proof; across hosts nothing is claimed.
- **Past the commit point** (the published result is this attempt's file):
  re-verify the staged evidence and the publication, then link the staged
  manifest, sealing ``complete``. If the evidence no longer verifies, seal
  ``partial`` with ``post-commit-evidence-lost``, naming the result as
  unattested, and write an incident record.
- **Before the commit point:** rebuild the events from the rolled chunks and
  the tail, discard a torn final line (keeping only its size), keep the
  verified prefix, and seal ``partial`` with the reason that fits.

Sealing is a no-overwrite link, so two reconcilers produce one manifest.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from xaytune.core.trajectory import Abandonment, TrajectoryManifest, seal, structural_refusals
from xaytune.trajectory._files import (
    StoreConflictError,
    digest_of,
    link_new,
    read_bytes,
    same_file,
    truncate_durably,
    write_new,
)
from xaytune.trajectory.sealing import (
    manifest_body,
    manifest_bytes,
    parse_tail,
    read_manifest,
    roll,
)
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore
from xaytune.trajectory.verify import load_events, verify_publication, verify_trajectory

__all__ = ["Reconciled", "reconcile_attempt", "reconcile_trajectories"]


@dataclass(frozen=True)
class Reconciled:
    """What reconciliation did to one attempt."""

    trajectory_id: str
    outcome: Literal["skipped-live", "already-sealed", "sealed", "unreadable"]
    manifest: TrajectoryManifest | None = None


@contextmanager
def _unlocked(paths: AttemptPaths) -> Iterator[bool]:
    """Hold the attempt's lock if its writer is gone; yield whether it was free."""
    try:
        fd = os.open(paths.lock, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        yield True
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(fd)


def reconcile_trajectories(
    store: LocalTrajectoryStore, result_location: Path
) -> tuple[Reconciled, ...]:
    """Seal every abandoned attempt under *store*. Idempotent."""
    return tuple(reconcile_attempt(store, paths, result_location) for paths in store.attempts())


def reconcile_attempt(
    store: LocalTrajectoryStore, paths: AttemptPaths, result_location: Path
) -> Reconciled:
    existing = read_manifest(paths.manifest)
    if existing is not None:
        return Reconciled(paths.trajectory_id, "already-sealed", existing)
    with _unlocked(paths) as free:
        if not free:
            return Reconciled(paths.trajectory_id, "skipped-live")
        record = store.read_open_record(paths)
        if record is None:
            return Reconciled(paths.trajectory_id, "unreadable")
        if paths.result.exists() and same_file(result_location, paths.result):
            manifest = _past_commit(store, paths, result_location, record)
        else:
            manifest = _before_commit(store, paths, result_location, record)
        return Reconciled(paths.trajectory_id, "sealed", manifest)


def _write_seal(paths: AttemptPaths, manifest: TrajectoryManifest) -> TrajectoryManifest:
    """Seal once. If another reconciler sealed first, its manifest is the seal."""
    if write_new_or_existing(paths, manifest):
        return manifest
    sealed = read_manifest(paths.manifest)
    return sealed if sealed is not None else manifest


def write_new_or_existing(paths: AttemptPaths, manifest: TrajectoryManifest) -> bool:
    try:
        return write_new(paths.manifest, manifest_bytes(manifest))
    except StoreConflictError:
        return False


def _past_commit(
    store: LocalTrajectoryStore, paths: AttemptPaths, location: Path, record: Any
) -> TrajectoryManifest:
    staged = read_manifest(paths.staged)
    if staged is not None:
        problems = verify_trajectory(store, paths, staged) + verify_publication(
            paths, staged, location
        )
        if not problems:
            if not link_new(paths.staged, paths.manifest):
                sealed = read_manifest(paths.manifest)
                if sealed is not None:
                    return sealed
            return staged
    # The result stands, but its evidence no longer verifies.
    published = digest_of(read_bytes(location))
    chunks = staged.chunks if staged is not None else store.read_progress(paths).chunks
    events: tuple[Any, ...] = ()
    if staged is not None:
        loaded, problems = load_events(store, paths, staged)
        if not problems and not structural_refusals(str(record.trajectory_id), loaded):
            events = loaded
    kept = chunks if events else ()
    body = manifest_body(
        record,
        kept,
        events,
        completeness="partial",
        abandonment="post-commit-evidence-lost",
        unattested_result=published,
    )
    incident = {
        "schema_version": "xaytune.trajectory-incident/v1alpha1",
        "trajectory_id": paths.trajectory_id,
        "kind": "post-commit-evidence-lost",
        "unattested_result": published,
    }
    write_new(paths.incident, json.dumps(incident, sort_keys=True).encode("utf-8"))
    return _write_seal(paths, seal(body))


def _before_commit(
    store: LocalTrajectoryStore, paths: AttemptPaths, location: Path, record: Any
) -> TrajectoryManifest:
    progress = store.read_progress(paths)
    rolled_events: tuple[Any, ...] = ()
    reason: Abandonment | None = None
    if progress.chunks:
        draft = manifest_body(
            record, progress.chunks, (), completeness="partial", abandonment="writer-lost"
        )
        rolled_events, problems = load_events(store, paths, draft)
        if problems:
            rolled_events, reason = (), "unverifiable-tail"
            progress = progress.model_copy(update={"chunks": ()})
    tail, torn, broke = parse_tail(store.read_tail(paths), rolled_events)
    events = list(rolled_events)
    for event in tail:
        if event.sequence != len(events) or structural_refusals(
            str(record.trajectory_id), [*events, event]
        ):
            broke = True
            break
        events.append(event)
    if broke and reason is None:
        reason = "unverifiable-tail"
    progress = roll(store, paths, progress, events[len(rolled_events) :])
    if paths.tail.exists():
        truncate_durably(paths.tail)
    terminated = any(e.kind == "run-terminated" for e in events)
    if reason is None:
        if os.path.lexists(location) and not same_file(location, paths.result) and terminated:
            reason = "result-published-by-another-attempt"
        elif terminated:
            reason = "result-unpublished"
        else:
            reason = "writer-lost"
    body = manifest_body(
        record,
        progress.chunks,
        events,
        completeness="partial",
        abandonment=reason,
        torn_tail_bytes=torn,
    )
    return _write_seal(paths, seal(body))
