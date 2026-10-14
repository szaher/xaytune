"""The state database as the API sees it: read-only, version-checked, one snapshot per read.

ADR-023 §S3. The API opens the database with ``mode=ro`` and ``query_only``,
never calls :func:`~xaytune.storage.migrations.migrate`, and checks before
every read that the database is at exactly the migration this build ships.
An older database is one the daemon has not migrated yet; a newer one has a
shape this build does not know. Either way the API says so rather than
guessing at rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from xaytune.core.sqlite import connect_read_only, read_snapshot
from xaytune.storage.control_plane import ControlPlaneRepository
from xaytune.storage.migrations import available_migrations, recorded_versions

__all__ = ["StoreStatus", "StoreUnavailableError", "StudioStore"]


class StoreUnavailableError(Exception):
    """The state database cannot be read by this build. The message is fixed text."""


@dataclass(frozen=True)
class StoreStatus:
    """Whether the database can be read, and at which migration."""

    readable: bool
    reason: str | None
    migration: int | None
    supported_migration: int


_MISSING = "the state database does not exist"
_UNOPENABLE = "the state database cannot be opened"
_NOT_XAYTUNE = "the file is not a Xaytune state database"
_OLDER = (
    "the state database is at an older migration than this version reads: run the "
    "daemon or any xaytune control command of this version once to migrate it"
)
_NEWER = "the state database is at a newer migration than this version can read"


class StudioStore:
    """The state database at *path*, opened read-only for each request.

    Construction refuses a path that is missing or is not a Xaytune state
    database, so the server never starts on the wrong file. A database at
    the wrong migration is accepted at construction -- the daemon may
    migrate it while the API runs -- and refused per read.

    Raises:
        StoreUnavailableError: If *path* does not exist, cannot be opened, or
            has no migration ledger.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) == ":memory:" or not self.path.is_file():
            raise StoreUnavailableError(_MISSING)
        self._supported = available_migrations()[-1].version
        status = self.status()
        if status.reason in (_UNOPENABLE, _NOT_XAYTUNE):
            raise StoreUnavailableError(status.reason)

    def status(self) -> StoreStatus:
        """Read the migration ledger and judge it against this build."""
        try:
            connection = connect_read_only(self.path)
        except sqlite3.Error:
            return self._status(None, _UNOPENABLE)
        try:
            versions = recorded_versions(connection)
        except sqlite3.DatabaseError:
            return self._status(None, _NOT_XAYTUNE)
        finally:
            connection.close()
        if versions is None:
            return self._status(None, _NOT_XAYTUNE)
        current = versions[-1] if versions else 0
        if current < self._supported:
            return self._status(current, _OLDER)
        if current > self._supported:
            return self._status(current, _NEWER)
        return StoreStatus(
            readable=True, reason=None, migration=current, supported_migration=self._supported
        )

    def _status(self, migration: int | None, reason: str) -> StoreStatus:
        return StoreStatus(
            readable=False,
            reason=reason,
            migration=migration,
            supported_migration=self._supported,
        )

    @contextmanager
    def snapshot(self) -> Iterator[ControlPlaneRepository]:
        """One consistent, read-only view of the record (§S3).

        The schema check and every read run inside one ``read_snapshot()``,
        so a daemon migrating or committing in between cannot make a
        response mix two states.

        Raises:
            StoreUnavailableError: If the database cannot be read by this build.
        """
        with self.connection() as connection:
            yield ControlPlaneRepository(connection)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """The read-only connection behind :meth:`snapshot`, inside its snapshot."""
        try:
            connection = connect_read_only(self.path)
        except sqlite3.Error as error:
            raise StoreUnavailableError(_UNOPENABLE) from error
        try:
            with read_snapshot(connection):
                versions = recorded_versions(connection)
                if versions is None:
                    raise StoreUnavailableError(_NOT_XAYTUNE)
                current = versions[-1] if versions else 0
                if current != self._supported:
                    raise StoreUnavailableError(_OLDER if current < self._supported else _NEWER)
                yield connection
        finally:
            connection.close()
