"""Verifying a trajectory against the local store (ADR-021 T5, T6).

Four operations, kept apart because they prove different things:

==========================  ===================================================
``validate_manifest``       the manifest alone (in :mod:`xaytune.core.trajectory`)
``verify_trajectory``       the evidence: chunks, chain, events, parents, content
                            references, blob integrity, agreement with the result.
                            Runs before the commit point; claims nothing about
                            publication.
``verify_publication``      after the commit point, on the local host: the
                            published result is this attempt's ``result.json``
                            (``st_dev``, ``st_ino``) with the digest
                            ``run-terminated`` named.
``check_availability``      whether each captured blob is still there now.
==========================  ===================================================

Each returns every problem it finds, as text that names roles and sequences,
never content.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from xaytune.core.agent_execution import HarnessRunResult
from xaytune.core.trajectory import (
    CapturedContent,
    TrajectoryManifestBody,
    content_references,
    parse_chunk,
    result_disagreements,
    structural_refusals,
)
from xaytune.trajectory._files import digest_of, read_bytes, same_file
from xaytune.trajectory.retention import authorized_expiries
from xaytune.trajectory.store import AttemptPaths, LocalTrajectoryStore

__all__ = [
    "Availability",
    "TrajectoryVerificationError",
    "check_availability",
    "load_events",
    "verify_publication",
    "verify_trajectory",
]

Availability = Literal["available", "expired", "missing", "corrupt"]


class TrajectoryVerificationError(RuntimeError):
    """A trajectory did not verify. Carries every reason."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("the trajectory does not verify: " + "; ".join(reasons))


def load_events(
    store: LocalTrajectoryStore, paths: AttemptPaths, manifest: TrajectoryManifestBody
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """The events of every chunk *manifest* names, and every problem reading them."""
    events: list[Any] = []
    problems: list[str] = []
    for ref in manifest.chunks:
        data = store.read_chunk(paths, ref)
        if data is None:
            problems.append(f"chunk {ref.index} is missing")
            continue
        if digest_of(data) != ref.digest:
            problems.append(f"chunk {ref.index} does not have its digest")
            continue
        if len(data) != ref.bytes:
            problems.append(f"chunk {ref.index} is not its recorded size")
        try:
            header, chunk_events = parse_chunk(data)
        except (ValueError, ValidationError):
            problems.append(f"chunk {ref.index} does not parse")
            continue
        if header != {
            "chunk": ref.index,
            "previous": ref.previous,
            "trajectory_id": str(manifest.trajectory_id),
        }:
            problems.append(f"chunk {ref.index}'s header is not its place in the chain")
        sequences = [event.sequence for event in chunk_events]
        if sequences != list(range(ref.first_sequence, ref.last_sequence + 1)):
            problems.append(f"chunk {ref.index} does not hold exactly its range")
        events.extend(chunk_events)
    return tuple(events), tuple(problems)


def check_availability(
    store: LocalTrajectoryStore, paths: AttemptPaths, manifest: TrajectoryManifestBody
) -> Mapping[tuple[str, str], Availability]:
    """The current availability of every captured blob, by (tier, digest) (T6).

    ``expired`` needs an expiry record that *manifest*'s retention authorized:
    its fingerprint, tier, retention and deadline, removed at or after the
    deadline. A blob absent without one is ``missing`` (F14), even if a
    well-formed record names it; present with the wrong bytes, ``corrupt``.
    """
    expiries = authorized_expiries(paths, manifest)
    found: dict[tuple[str, str], Availability] = {}
    for reference in manifest.content:
        slot = reference.slot
        if not isinstance(slot, CapturedContent):
            continue
        key = (slot.tier, slot.digest)
        if key in found:
            continue
        data = store.read_blob(paths, slot.tier, slot.digest)
        if data is not None:
            found[key] = "available" if digest_of(data) == slot.digest else "corrupt"
        elif key in expiries:
            found[key] = "expired"
        else:
            found[key] = "missing"
    return found


def _result(paths: AttemptPaths) -> tuple[bytes | None, HarnessRunResult | None]:
    try:
        data = read_bytes(paths.result)
    except FileNotFoundError:
        return None, None
    try:
        return data, HarnessRunResult.model_validate_json(data)
    except ValidationError:
        return data, None


def verify_trajectory(
    store: LocalTrajectoryStore, paths: AttemptPaths, manifest: TrajectoryManifestBody
) -> tuple[str, ...]:
    """Every way the evidence *manifest* names fails to verify. ``()`` if it verifies.

    Pre-commit evidence verification: it never looks at a published result,
    so its outcome is the same before and after the commit point.
    """
    events, problems = load_events(store, paths, manifest)
    reasons = list(problems)
    if problems:
        return tuple(reasons)
    trajectory_id = str(manifest.trajectory_id)
    reasons.extend(structural_refusals(trajectory_id, events))
    if events:
        started = events[0]
        if getattr(started, "attribution", None) != manifest.attribution:
            reasons.append("run-started's attribution is not the manifest's")
        if getattr(started, "writer", None) != manifest.writer:
            reasons.append("run-started's writer is not the manifest's")
        if getattr(started, "runner", None) != manifest.runner:
            reasons.append("run-started's runner is not the manifest's")
    if content_references(events) != manifest.content:
        reasons.append("the events' content references are not the manifest's disposition table")
    for (tier, _), status in check_availability(store, paths, manifest).items():
        if status in ("missing", "corrupt"):
            reasons.append(f"a captured {tier} blob is {status}")
    terminated = next((e for e in events if getattr(e, "kind", None) == "run-terminated"), None)
    if manifest.termination is not None and (
        terminated is None or terminated.termination != manifest.termination
    ):
        reasons.append("the manifest's termination is not run-terminated's")
    if manifest.completeness == "complete":
        reasons.extend(_complete(paths, manifest, events, terminated))
    return tuple(reasons)


def _complete(
    paths: AttemptPaths,
    manifest: TrajectoryManifestBody,
    events: tuple[Any, ...],
    terminated: Any | None,
) -> list[str]:
    if terminated is None:
        return ["a complete trajectory ends with run-terminated"]
    reasons: list[str] = []
    if terminated.result_digest != manifest.result_digest:
        reasons.append("the manifest's result digest is not the one run-terminated named")
    data, result = _result(paths)
    if data is None:
        return [*reasons, "the attempt's result.json is missing"]
    if digest_of(data) != manifest.result_digest:
        reasons.append("result.json does not have the result digest")
    if result is None:
        return [*reasons, "result.json is not a harness run result"]
    attribution = manifest.attribution
    for name, expected in (
        ("request_digest", attribution.request_digest),
        ("execution_fingerprint", attribution.execution_fingerprint),
        ("harness_fingerprint", attribution.harness_fingerprint),
        ("task_id", attribution.task.task_id),
        ("seed", attribution.seed),
        ("replicate", attribution.replicate),
    ):
        if getattr(result, name) != expected:
            reasons.append(f"the result's {name} is not the attribution's")
    reasons.extend(result_disagreements(events, result))
    return reasons


def verify_publication(
    paths: AttemptPaths, manifest: TrajectoryManifestBody, location: Path
) -> tuple[str, ...]:
    """Every way *location* fails to be this attempt's publication. ``()`` if it is.

    Post-commit, local host only: the published path is not a symlink, it is
    the same file (``st_dev``, ``st_ino``) as this attempt's ``result.json``,
    and it has the digest the manifest attests (or names as unattested). It
    makes no claim across hosts.
    """
    expected = manifest.result_digest or manifest.unattested_result
    reasons: list[str] = []
    try:
        mode = os.lstat(location).st_mode
    except FileNotFoundError:
        return ("no result is published at the declared location",)
    if stat.S_ISLNK(mode):
        return ("the declared result location is a symlink",)
    if not same_file(location, paths.result):
        reasons.append("the published result is not this attempt's file")
    if expected is None:
        reasons.append("the manifest attests no result")
    elif digest_of(read_bytes(location)) != expected:
        reasons.append("the published result does not have the attested digest")
    return tuple(reasons)
