"""Capture disposition is immutable; availability is observed (ADR-021 T6; criterion 7).

Failure scenarios F13, F14, F15 and F24.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.test_trajectory.fixtures import only_attempt, recorded_spec, recording, run
from xaytune.core.trajectory import CapturedContent, TierPolicy, validate_manifest
from xaytune.trajectory import (
    ExpiryRecord,
    ExpiryRefusedError,
    check_availability,
    expire_blob,
    verify_trajectory,
)
from xaytune.trajectory.sealing import manifest_bytes

DAY = 24 * 60 * 60


def _recorded(tmp_path: Path, name: str = "out", days: int | None = 7) -> Any:
    tier = TierPolicy(readers="experiment", retention_seconds=days * DAY if days else None)
    spec = recorded_spec(tmp_path / name, ordinary=tier)
    run(spec)
    return only_attempt(spec)


def _captured(attempt: Any) -> CapturedContent:
    slot = attempt.manifest.content[0].slot
    assert isinstance(slot, CapturedContent)
    return slot


def _after(attempt: Any, days: float) -> datetime:
    return datetime.fromisoformat(attempt.manifest.sealed_at) + timedelta(days=days)


def test_f13_expiry_removes_a_blob_and_never_touches_the_sealed_manifest(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    before = attempt.paths.manifest.read_bytes()
    fingerprint = attempt.manifest.trajectory_fingerprint
    slot = _captured(attempt)
    record = expire_blob(
        attempt.paths, attempt.manifest, slot.digest, "ordinary", actor="gc", now=_after(attempt, 8)
    )
    assert record.retention_seconds == 7 * DAY
    assert not attempt.paths.blob("ordinary", slot.digest).exists()
    assert attempt.paths.manifest.read_bytes() == before
    assert attempt.manifest.trajectory_fingerprint == fingerprint
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "expired"
    assert verify_trajectory(attempt.store, attempt.paths, attempt.manifest) == ()


def test_f14_a_blob_removed_without_authority_is_missing(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    os.unlink(attempt.paths.blob("ordinary", slot.digest))
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "missing"
    assert any(
        "missing" in p for p in verify_trajectory(attempt.store, attempt.paths, attempt.manifest)
    )


def test_f14_an_early_expiry_is_refused_and_an_early_removal_is_missing(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    with pytest.raises(ExpiryRefusedError, match="has not ended"):
        expire_blob(
            attempt.paths,
            attempt.manifest,
            slot.digest,
            "ordinary",
            actor="gc",
            now=_after(attempt, 6),
        )
    assert attempt.paths.blob("ordinary", slot.digest).exists(), "nothing was removed"
    os.unlink(attempt.paths.blob("ordinary", slot.digest))
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "missing", "never folded into expired"


def test_f15_a_blob_whose_bytes_changed_is_corrupt(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    blob = attempt.paths.blob("ordinary", slot.digest)
    blob.write_bytes(b"something else")
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "corrupt"


def test_indefinite_retention_is_never_expired(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path, days=None)
    with pytest.raises(ExpiryRefusedError, match="indefinite"):
        expire_blob(
            attempt.paths,
            attempt.manifest,
            _captured(attempt).digest,
            "ordinary",
            actor="gc",
            now=_after(attempt, 10_000),
        )


@pytest.mark.parametrize("target", ["chunk", "unreferenced", "other-tier"])
def test_only_a_captured_blob_can_expire(tmp_path: Path, target: str) -> None:
    attempt = _recorded(tmp_path)
    digest = {
        "chunk": attempt.manifest.chunks[0].digest,
        "unreferenced": "sha256:" + "9" * 64,
        "other-tier": _captured(attempt).digest,
    }[target]
    tier = "restricted" if target == "other-tier" else "ordinary"
    with pytest.raises(ExpiryRefusedError, match="does not capture"):
        expire_blob(
            attempt.paths, attempt.manifest, digest, tier, actor="gc", now=_after(attempt, 8)
        )


def test_f24_one_digest_in_two_trajectories_is_two_objects_with_two_retentions(
    tmp_path: Path,
) -> None:
    short = _recorded(tmp_path, "short", days=7)
    long = _recorded(tmp_path, "long", days=90)
    digest = _captured(short).digest
    assert _captured(long).digest == digest, "the same content, by digest"
    assert short.paths.blob("ordinary", digest) != long.paths.blob("ordinary", digest)

    expire_blob(short.paths, short.manifest, digest, "ordinary", actor="gc", now=_after(short, 8))
    assert check_availability(long.store, long.paths, long.manifest)[("ordinary", digest)] == (
        "available"
    )
    with pytest.raises(ExpiryRefusedError, match="has not ended"):
        expire_blob(long.paths, long.manifest, digest, "ordinary", actor="gc", now=_after(long, 8))
    assert long.paths.blob("ordinary", digest).exists()


def test_recording_declares_retention_per_tier() -> None:
    spec = recording(restricted=TierPolicy(readers="owner", retention_seconds=DAY))
    assert spec.ordinary.retention_seconds is None and spec.restricted.retention_seconds == DAY


# ---- the seal time retention is measured from -----------------------------------------------


def _with_sealed_at(attempt: Any, sealed_at: datetime) -> Any:
    """The attempt's manifest with another seal time: still valid, with the same fingerprint."""
    record = attempt.manifest.model_dump(mode="json") | {"sealed_at": sealed_at.isoformat()}
    altered = validate_manifest(record)
    assert altered.trajectory_fingerprint == attempt.manifest.trajectory_fingerprint
    return altered


def test_a_supplied_manifest_with_an_earlier_seal_time_authorizes_nothing(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    sealed = datetime.fromisoformat(attempt.manifest.sealed_at)
    backdated = _with_sealed_at(attempt, sealed - timedelta(days=30))
    with pytest.raises(ExpiryRefusedError, match="not the attempt's sealed manifest"):
        expire_blob(
            attempt.paths, backdated, slot.digest, "ordinary", actor="gc", now=_after(attempt, 3)
        )
    assert attempt.paths.blob("ordinary", slot.digest).exists(), "nothing was removed"
    assert not attempt.paths.expiry.exists(), "and nothing was recorded"


def test_a_supplied_manifest_with_a_later_seal_time_is_refused_too(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    later = _with_sealed_at(attempt, _after(attempt, 1))
    with pytest.raises(ExpiryRefusedError, match="not the attempt's sealed manifest"):
        expire_blob(
            attempt.paths, later, slot.digest, "ordinary", actor="gc", now=_after(attempt, 30)
        )


def test_a_stored_seal_time_before_the_attempt_opened_authorizes_nothing(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    opened = json.loads(attempt.paths.open_record.read_text())["opened_at"]
    backdated = _with_sealed_at(attempt, datetime.fromisoformat(opened) - timedelta(days=30))
    replacement = attempt.paths.directory / "replacement.json"
    replacement.write_bytes(manifest_bytes(backdated))
    os.replace(replacement, attempt.paths.manifest)
    assert attempt.manifest == backdated, "the stored manifest itself was rewritten"
    with pytest.raises(ExpiryRefusedError, match="not a valid seal time"):
        expire_blob(
            attempt.paths, backdated, slot.digest, "ordinary", actor="gc", now=_after(attempt, 3)
        )
    assert attempt.paths.blob("ordinary", slot.digest).exists()


def test_an_attempt_without_a_valid_sealed_manifest_expires_nothing(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    manifest, later = attempt.manifest, _after(attempt, 8)
    attempt.paths.manifest.write_bytes(b"{}")
    with pytest.raises(ExpiryRefusedError, match="no valid sealed manifest"):
        expire_blob(attempt.paths, manifest, slot.digest, "ordinary", actor="gc", now=later)
    os.unlink(attempt.paths.manifest)
    with pytest.raises(ExpiryRefusedError, match="no valid sealed manifest"):
        expire_blob(attempt.paths, manifest, slot.digest, "ordinary", actor="gc", now=later)
    assert attempt.paths.blob("ordinary", slot.digest).exists()


def test_another_attempts_manifest_authorizes_nothing_here(tmp_path: Path) -> None:
    short = _recorded(tmp_path, "short", days=7)
    other = _recorded(tmp_path, "other", days=7)
    digest = _captured(short).digest
    with pytest.raises(ExpiryRefusedError, match="not the attempt's sealed manifest"):
        expire_blob(
            short.paths, other.manifest, digest, "ordinary", actor="gc", now=_after(other, 8)
        )
    assert short.paths.blob("ordinary", digest).exists()


# ---- an expiry record authorizes only what retention allows (F14) --------------------------


def _record(attempt: Any, **changes: Any) -> ExpiryRecord:
    """The record ``expire_blob`` would write for the captured blob, 8 days on, with *changes*."""
    slot = _captured(attempt)
    sealed = datetime.fromisoformat(attempt.manifest.sealed_at)
    fields: dict[str, Any] = {
        "trajectory_id": str(attempt.manifest.trajectory_id),
        "trajectory_fingerprint": attempt.manifest.trajectory_fingerprint,
        "digest": slot.digest,
        "tier": "ordinary",
        "retention_seconds": 7 * DAY,
        "deadline": (sealed + timedelta(days=7)).isoformat(),
        "actor": "gc",
        "at": (sealed + timedelta(days=8)).isoformat(),
    }
    fields.update(changes)
    return ExpiryRecord(**fields)


def _append(attempt: Any, record: ExpiryRecord) -> None:
    line = json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
    with open(attempt.paths.expiry, "a", encoding="utf-8") as log:
        log.write(line)


def _forged(attempt: Any, forgery: str) -> ExpiryRecord:
    sealed = datetime.fromisoformat(attempt.manifest.sealed_at)
    return {
        "premature": lambda: _record(attempt, at=(sealed + timedelta(days=1)).isoformat()),
        "early-deadline": lambda: _record(
            attempt,
            deadline=(sealed + timedelta(hours=1)).isoformat(),
            at=(sealed + timedelta(hours=2)).isoformat(),
        ),
        "short-retention": lambda: _record(
            attempt,
            retention_seconds=60,
            deadline=(sealed + timedelta(seconds=60)).isoformat(),
            at=(sealed + timedelta(minutes=2)).isoformat(),
        ),
        "other-fingerprint": lambda: _record(attempt, trajectory_fingerprint="sha256:" + "e" * 64),
        "other-tier": lambda: _record(attempt, tier="restricted"),
        "naive-times": lambda: _record(
            attempt,
            deadline=(sealed + timedelta(days=7)).replace(tzinfo=None).isoformat(),
            at=(sealed + timedelta(days=8)).replace(tzinfo=None).isoformat(),
        ),
    }[forgery]()


@pytest.mark.parametrize(
    "forgery",
    [
        "premature",
        "early-deadline",
        "short-retention",
        "other-fingerprint",
        "other-tier",
        "naive-times",
    ],
)
def test_f14_a_well_formed_but_unauthorized_expiry_record_leaves_the_blob_missing(
    tmp_path: Path, forgery: str
) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    _append(attempt, _forged(attempt, forgery))
    os.unlink(attempt.paths.blob("ordinary", slot.digest))
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "missing", "an integrity incident"
    assert any(
        "missing" in p for p in verify_trajectory(attempt.store, attempt.paths, attempt.manifest)
    )


def test_f14_a_forged_record_after_an_authorized_one_does_not_hide_it(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    expire_blob(
        attempt.paths, attempt.manifest, slot.digest, "ordinary", actor="gc", now=_after(attempt, 8)
    )
    _append(attempt, _forged(attempt, "premature"))
    availability = check_availability(attempt.store, attempt.paths, attempt.manifest)
    assert availability[("ordinary", slot.digest)] == "expired"


def test_f14_a_record_expire_blob_writes_is_exactly_an_authorized_one(tmp_path: Path) -> None:
    attempt = _recorded(tmp_path)
    slot = _captured(attempt)
    written = expire_blob(
        attempt.paths, attempt.manifest, slot.digest, "ordinary", actor="gc", now=_after(attempt, 8)
    )
    assert written == _record(attempt)
