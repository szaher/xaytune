"""Recording, storing, verifying and reconciling harness trajectories (H05, ADR-021).

The records themselves are pure and live in :mod:`xaytune.core.trajectory`.
This package is the local, single-host implementation around them:

- :class:`~xaytune.trajectory.writer.TrajectoryWriter`: the crash-safe writer;
- :class:`~xaytune.trajectory.store.LocalTrajectoryStore`: one directory per
  attempt instance, owner-only, no-overwrite;
- :func:`~xaytune.trajectory.verify.verify_trajectory`,
  :func:`~xaytune.trajectory.verify.verify_publication` and
  :func:`~xaytune.trajectory.verify.check_availability`;
- :func:`~xaytune.trajectory.reconcile.reconcile_trajectories`;
- :func:`~xaytune.trajectory.retention.expire_blob`.

Nothing here touches the repository database: there is no table and no
migration. Cross-host liveness, ownership and deduplication are deferred.
"""

from xaytune.trajectory.reconcile import Reconciled, reconcile_attempt, reconcile_trajectories
from xaytune.trajectory.retention import ExpiryRecord, ExpiryRefusedError, expire_blob
from xaytune.trajectory.store import AttemptExistsError, AttemptPaths, LocalTrajectoryStore
from xaytune.trajectory.verify import (
    TrajectoryVerificationError,
    check_availability,
    verify_publication,
    verify_trajectory,
)
from xaytune.trajectory.writer import (
    WRITER,
    Redactor,
    SecretDetectedError,
    SecretRegistrationClosedError,
    TrajectoryRecordingError,
    TrajectoryWriter,
    contain_attempt,
)

__all__ = [
    "WRITER",
    "AttemptExistsError",
    "AttemptPaths",
    "ExpiryRecord",
    "ExpiryRefusedError",
    "LocalTrajectoryStore",
    "Reconciled",
    "Redactor",
    "SecretDetectedError",
    "SecretRegistrationClosedError",
    "TrajectoryRecordingError",
    "TrajectoryVerificationError",
    "TrajectoryWriter",
    "check_availability",
    "contain_attempt",
    "expire_blob",
    "reconcile_attempt",
    "reconcile_trajectories",
    "verify_publication",
    "verify_trajectory",
]
