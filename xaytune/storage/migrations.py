"""Versioned schema migrations.

Migrations are numbered SQL files applied in order, each inside one transaction,
with the applied version recorded in ``schema_migrations``. A migration is never
edited after it ships: the next change is the next file, because a database that
already ran 001 will never run it again.

The runner is deliberately small. It does not generate migrations, diff schemas
or support rollback. Down-migrations are the tempting feature and the wrong one
here -- a down-migration that drops a column discards durable provenance, and
ADR-005 §10 makes terminal records immutable. Recovering from a bad migration is
a forward migration plus a restore, not an undo.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from xaytune.storage.errors import MigrationError

__all__ = ["Migration", "available_migrations", "applied_versions", "migrate"]

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_FILENAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    """One numbered migration file."""

    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def available_migrations(directory: Path | None = None) -> tuple[Migration, ...]:
    """Return every migration on disk, ordered by version.

    Raises:
        MigrationError: If a filename does not match ``NNN_name.sql``, or if two
            files claim the same version. Both are refused rather than ordered
            arbitrarily: "whichever sorted first" is not a schema history.
    """
    directory = directory or MIGRATIONS_DIR
    migrations: dict[int, Migration] = {}

    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"migration filename {path.name!r} must look like '001_core_aggregates.sql': "
                f"a three-digit version, an underscore, a lowercase name, then '.sql'"
            )
        version = int(match.group(1))
        if version in migrations:
            raise MigrationError(
                f"two migrations claim version {version}: "
                f"{migrations[version].path.name!r} and {path.name!r}"
            )
        migrations[version] = Migration(version=version, name=match.group(2), path=path)

    return tuple(migrations[version] for version in sorted(migrations))


def applied_versions(connection: sqlite3.Connection) -> tuple[int, ...]:
    """Return the versions already applied, in order."""
    _ensure_ledger(connection)
    rows = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return tuple(int(row["version"]) for row in rows)


def migrate(connection: sqlite3.Connection, directory: Path | None = None) -> tuple[int, ...]:
    """Apply every pending migration in order and return the versions applied.

    Idempotent: running it against an up-to-date database applies nothing and
    returns an empty tuple.

    Each migration runs in its own transaction together with its ledger row, so
    a failure part-way through a file leaves that migration wholly unapplied
    rather than half-applied with no record of it.

    Raises:
        MigrationError: If a migration on disk has a version lower than one
            already applied -- that means a file was inserted behind a database
            that has moved past it, and applying it now would run it out of
            order against a schema it was never written for.
    """
    _ensure_ledger(connection)
    already = set(applied_versions(connection))
    highest = max(already, default=0)

    applied: list[int] = []
    for migration in available_migrations(directory):
        if migration.version in already:
            continue
        if migration.version < highest:
            raise MigrationError(
                f"migration {migration.path.name!r} is version {migration.version}, "
                f"but version {highest} has already been applied. A migration cannot "
                f"be inserted behind a database that has moved past it; give it the "
                f"next free version instead."
            )

        _apply(connection, migration)
        applied.append(migration.version)
        highest = migration.version

    return tuple(applied)


def _apply(connection: sqlite3.Connection, migration: Migration) -> None:
    """Apply one migration and record it, atomically.

    The transaction control lives *inside* the script rather than in
    :func:`~xaytune.storage.database.write_transaction`, because
    ``executescript`` implicitly commits any pending transaction before it runs.
    Wrapping it in an outer transaction would therefore commit the ``BEGIN
    IMMEDIATE`` first and run the schema changes in autocommit mode -- the
    migration and its ledger row would land separately, which is the one thing
    this function exists to prevent. Python performs no other implicit
    transaction control, so an explicit ``BEGIN``/``COMMIT`` in the script is
    honoured exactly as written.

    The ledger values are interpolated rather than bound because
    ``executescript`` takes no parameters. Both are safe by construction: the
    version is an ``int``, and the name has already matched ``[a-z0-9_]+``.
    """
    script = (
        "BEGIN IMMEDIATE;\n"
        f"{migration.sql}\n"
        "INSERT INTO schema_migrations (version, name, applied_at) "
        f"VALUES ({migration.version:d}, '{migration.name}', datetime('now'));\n"
        "COMMIT;"
    )
    try:
        connection.executescript(script)
    except sqlite3.Error as error:
        # The script's own COMMIT never ran, so SQLite has discarded the
        # partial work; clear the aborted transaction if one is still open.
        if connection.in_transaction:
            connection.rollback()
        raise MigrationError(f"migration {migration.path.name!r} failed: {error}") from error


def _ensure_ledger(connection: sqlite3.Connection) -> None:
    """Create ``schema_migrations`` if it does not exist.

    Outside the migration sequence on purpose: the ledger is what lets the
    sequence be tracked, so it cannot be tracked by it.
    """
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version INTEGER PRIMARY KEY,"
        "  name TEXT NOT NULL,"
        "  applied_at TEXT NOT NULL"
        ")"
    )
