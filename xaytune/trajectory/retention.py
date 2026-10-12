"""Authorized expiry of a sealed trajectory's blobs (ADR-021 T6).

Retention removes **blobs only**: never chunks, manifests or open records,
which are the evidence spine. Each removal is recorded in the attempt's
append-only ``expiry.jsonl`` **before** the blob is deleted, so a blob that is
absent with no record is ``missing`` (an incident), never ``expired``.

The sealed manifest is never touched: its bytes and its fingerprint are the
same before and after any expiry.

Retention is measured from the seal time of the manifest **stored in the
attempt**, never from a manifest a caller supplies. ``sealed_at`` is not part
of the fingerprint (T6), so a caller's copy with an earlier seal time would
still validate; a supplied manifest is therefore accepted only if it is the
stored one, field for field, and the stored seal time must not precede the
attempt's own open record.

Blobs are stored per attempt, so a blob here carries exactly one retention
obligation: its own manifest's, for its tier. Shared, deduplicated blobs would
need the deletion rules T6 fixes, and are not implemented.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, ValidationError

from xaytune.core.clock import utc_now
from xaytune.core.immutable import FrozenDomainModel
from xaytune.core.trajectory import (
    CapturedContent,
    TrajectoryManifest,
    TrajectoryManifestBody,
    TrajectoryOpenRecord,
)
from xaytune.trajectory._files import append_durably, read_bytes, sync_directory
from xaytune.trajectory.sealing import read_manifest
from xaytune.trajectory.store import AttemptPaths, Tier

__all__ = [
    "ExpiryRecord",
    "ExpiryRefusedError",
    "authorized_expiries",
    "expire_blob",
    "read_expiry_log",
]


class ExpiryRefusedError(ValueError):
    """An expiry that retention does not authorize. Nothing was removed."""


class ExpiryRecord(FrozenDomainModel):
    """One authorized removal, recorded before the blob is deleted."""

    schema_version: Literal["xaytune.trajectory-expiry/v1alpha1"] = (
        "xaytune.trajectory-expiry/v1alpha1"
    )
    trajectory_id: str
    trajectory_fingerprint: str
    digest: str
    tier: Tier
    retention_seconds: int = Field(strict=True, gt=0)
    deadline: str
    actor: str = Field(min_length=1, max_length=200)
    at: str


def read_expiry_log(paths: AttemptPaths) -> tuple[ExpiryRecord, ...]:
    """Every well-formed expiry record for this attempt, in log order.

    Well-formed is not authorized: see :func:`authorized_expiries`.
    """
    try:
        data = read_bytes(paths.expiry)
    except FileNotFoundError:
        return ()
    records: list[ExpiryRecord] = []
    for line in data.splitlines():
        try:
            record = ExpiryRecord.model_validate_json(line)
        except ValidationError:
            continue
        if record.trajectory_id == paths.trajectory_id:
            records.append(record)
    return tuple(records)


def _authorizes(record: ExpiryRecord, manifest: TrajectoryManifestBody) -> bool:
    """Whether *record* is an expiry *manifest*'s retention authorized (F14).

    It names this sealed trajectory by fingerprint, a blob the manifest
    captures in that tier, the tier's declared retention, the deadline that
    retention gives from the seal time, and a removal at or after it.
    """
    fingerprint = getattr(manifest, "trajectory_fingerprint", None)
    if fingerprint is None or record.trajectory_fingerprint != fingerprint:
        return False
    if record.trajectory_id != str(manifest.trajectory_id):
        return False
    if not any(
        isinstance(ref.slot, CapturedContent)
        and ref.slot.digest == record.digest
        and ref.slot.tier == record.tier
        for ref in manifest.content
    ):
        return False
    recording = manifest.recording
    policy = recording.ordinary if record.tier == "ordinary" else recording.restricted
    if policy.retention_seconds is None or record.retention_seconds != policy.retention_seconds:
        return False
    sealed = _moment(manifest.sealed_at)
    deadline, at = _moment(record.deadline), _moment(record.at)
    if sealed is None or deadline is None or at is None:
        return False
    return deadline == sealed + timedelta(seconds=policy.retention_seconds) and at >= deadline


def authorized_expiries(
    paths: AttemptPaths, manifest: TrajectoryManifestBody
) -> frozenset[tuple[str, str]]:
    """The (tier, digest) of every blob whose removal *manifest*'s retention authorized.

    A record that is well formed but premature, for another trajectory or
    retention, or for a blob the manifest does not capture authorizes nothing:
    the blob it names, if absent, is ``missing``, never ``expired``.
    """
    return frozenset(
        (record.tier, record.digest)
        for record in read_expiry_log(paths)
        if _authorizes(record, manifest)
    )


def _moment(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _seal_time(paths: AttemptPaths, manifest: TrajectoryManifest) -> datetime:
    """The verified seal time: the stored manifest's, bound to the attempt's open record.

    Raises:
        ExpiryRefusedError: If the attempt has no valid sealed manifest, the
            supplied manifest is not that manifest, or its seal time is not
            a time at or after the attempt was opened.
    """
    stored = read_manifest(paths.manifest)
    if stored is None or str(stored.trajectory_id) != paths.trajectory_id:
        raise ExpiryRefusedError("the attempt has no valid sealed manifest")
    if manifest != stored:
        raise ExpiryRefusedError("the supplied manifest is not the attempt's sealed manifest")
    try:
        opened = TrajectoryOpenRecord.model_validate_json(read_bytes(paths.open_record))
    except (OSError, ValidationError):
        raise ExpiryRefusedError("the attempt has no valid open record") from None
    opened_at, sealed_at = _moment(opened.opened_at), _moment(stored.sealed_at)
    if str(opened.trajectory_id) != paths.trajectory_id or opened_at is None:
        raise ExpiryRefusedError("the attempt has no valid open record")
    if sealed_at is None or sealed_at < opened_at:
        raise ExpiryRefusedError("the sealed manifest's seal time is not a valid seal time")
    return sealed_at


def _deadline(
    paths: AttemptPaths, manifest: TrajectoryManifest, tier: Tier
) -> tuple[int, datetime]:
    policy = manifest.recording.ordinary if tier == "ordinary" else manifest.recording.restricted
    if policy.retention_seconds is None:
        raise ExpiryRefusedError("the tier's retention is indefinite")
    sealed = _seal_time(paths, manifest)
    return policy.retention_seconds, sealed + timedelta(seconds=policy.retention_seconds)


def expire_blob(
    paths: AttemptPaths,
    manifest: TrajectoryManifest,
    digest: str,
    tier: Tier,
    *,
    actor: str,
    now: datetime | None = None,
) -> ExpiryRecord:
    """Remove one captured blob after its retention, recording the authority first.

    Raises:
        ExpiryRefusedError: If the manifest does not capture *digest* in
            *tier*, the tier is kept indefinitely, *manifest* is not the
            attempt's stored sealed manifest, its seal time precedes the
            attempt's opening, or the deadline has not passed. Nothing is
            removed.
    """
    captured = any(
        isinstance(ref.slot, CapturedContent)
        and ref.slot.digest == digest
        and ref.slot.tier == tier
        for ref in manifest.content
    )
    if not captured:
        raise ExpiryRefusedError("the manifest does not capture that blob in that tier")
    seconds, deadline = _deadline(paths, manifest, tier)
    moment = now or utc_now()
    if moment < deadline:
        raise ExpiryRefusedError("the blob's retention has not ended")
    record = ExpiryRecord(
        trajectory_id=paths.trajectory_id,
        trajectory_fingerprint=manifest.trajectory_fingerprint,
        digest=digest,
        tier=tier,
        retention_seconds=seconds,
        deadline=deadline.isoformat(),
        actor=actor,
        at=moment.isoformat(),
    )
    line = json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
    append_durably(paths.expiry, line.encode("utf-8"))
    blob = paths.blob(tier, digest)
    try:
        os.unlink(blob)
    except FileNotFoundError:
        pass
    sync_directory(blob.parent)
    return record
