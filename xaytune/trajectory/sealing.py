"""Rolling chunks and sealing manifests: shared by the writer and the reconciler (ADR-021 T4)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from xaytune.core.clock import utc_now
from xaytune.core.trajectory import (
    Abandonment,
    ChunkRef,
    TrajectoryManifest,
    TrajectoryManifestBody,
    TrajectoryOpenRecord,
    chunk_bytes,
    content_references,
    parse_event,
    seal,
    validate_manifest,
)
from xaytune.trajectory._files import digest_of, read_bytes, write_new
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore, TrajectoryProgress

__all__ = [
    "manifest_body",
    "manifest_bytes",
    "parse_tail",
    "read_manifest",
    "roll",
    "write_manifest",
]


def roll(
    store: LocalTrajectoryStore,
    paths: AttemptPaths,
    progress: TrajectoryProgress,
    events: Sequence[Any],
) -> TrajectoryProgress:
    """Publish *events* as the next immutable chunk, then advance ``progress.json``.

    The chunk is published before progress names it (write-ahead rule 5). The
    caller truncates the tail only after both.
    """
    if not events:
        return progress
    previous = progress.chunks[-1].digest if progress.chunks else None
    index = len(progress.chunks)
    data = chunk_bytes(str(paths.trajectory_id), index, previous, events)
    ref = ChunkRef(
        index=index,
        first_sequence=events[0].sequence,
        last_sequence=events[-1].sequence,
        digest=digest_of(data),
        previous=previous,
        bytes=len(data),
        events=len(events),
    )
    store.put_chunk(paths, ref, data)
    advanced = TrajectoryProgress(
        trajectory_id=progress.trajectory_id, chunks=(*progress.chunks, ref)
    )
    store.write_progress(paths, advanced)
    return advanced


def parse_tail(data: bytes, rolled: Sequence[Any]) -> tuple[tuple[Any, ...], int, bool]:
    """The tail's events beyond *rolled*, the torn bytes discarded, and whether it broke.

    A crash between rolling a chunk and truncating the tail leaves events in
    both: a tail event already rolled is a redelivery, deduplicated when it is
    identical and a conflict when it is not, which breaks the tail. A final
    line without its newline, or that does not parse, is torn: it is
    discarded and only its size is kept. A line that does not parse anywhere
    else breaks the tail: the events before it are kept, and nothing after.
    """
    events: list[Any] = []
    lines = data.split(b"\n")
    torn = 0
    if lines and lines[-1] != b"":
        torn = len(lines[-1])
    complete = lines[:-1]
    for position, line in enumerate(complete):
        try:
            event = parse_event(line)
        except (ValueError, ValidationError):
            if position == len(complete) - 1 and torn == 0:
                return tuple(events), len(line) + 1, False
            return tuple(events), torn, True
        if event.sequence < len(rolled):
            if rolled[event.sequence] != event:
                return tuple(events), torn, True
            continue
        events.append(event)
    return tuple(events), torn, False


def manifest_body(
    record: TrajectoryOpenRecord,
    chunks: Sequence[ChunkRef],
    events: Sequence[Any],
    *,
    completeness: str,
    abandonment: Abandonment | None = None,
    result_digest: str | None = None,
    unattested_result: str | None = None,
    torn_tail_bytes: int = 0,
) -> TrajectoryManifestBody:
    terminated = next((e for e in events if e.kind == "run-terminated"), None)
    return TrajectoryManifestBody(
        trajectory_id=record.trajectory_id,
        attribution=record.attribution,
        writer=record.writer,
        runner=record.runner,
        recording=record.recording,
        chunks=tuple(chunks),
        content=content_references(events),
        completeness=completeness,  # type: ignore[arg-type]
        abandonment=abandonment,
        termination=terminated.termination if terminated is not None else None,
        result_digest=result_digest,
        unattested_result=unattested_result,
        torn_tail_bytes=torn_tail_bytes,
        sealed_at=utc_now().isoformat(),
    )


def manifest_bytes(manifest: TrajectoryManifestBody) -> bytes:
    return (json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=1) + "\n").encode()


def write_manifest(path: Any, manifest: TrajectoryManifestBody) -> bool:
    """Write a manifest once, no-overwrite. ``False`` if identical bytes were there."""
    return write_new(path, manifest_bytes(manifest))


def read_manifest(path: Any) -> TrajectoryManifest | None:
    """The manifest at *path*, validated alone; ``None`` if absent or invalid."""
    try:
        return validate_manifest(read_bytes(path))
    except (OSError, ValidationError):
        return None


def sealed(body: TrajectoryManifestBody) -> TrajectoryManifest:
    return seal(body)
