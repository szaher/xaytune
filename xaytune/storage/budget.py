"""Row-level persistence for the budget ledger (migration 008).

Private writes, for the reason every writer here is private: an entry records
a consequence of a state change, and must commit with it.
:class:`~xaytune.storage.control_plane.ControlPlaneRepository` composes them
into the transitions that cause them.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from decimal import Decimal

from xaytune.core.domain.budget import (
    BudgetDimension,
    BudgetLedgerEntry,
    BudgetSubjectKind,
    LedgerEntryKind,
)
from xaytune.storage.errors import StorageError
from xaytune.storage.journal import _require_transaction

__all__ = ["BudgetLedgerStore", "LedgerConflictError"]


class LedgerConflictError(StorageError):
    """An entry exists for the same subject, dimension and kind, with another amount.

    Settling a subject again must find exactly what was written the first
    time; a different amount means the record and the settlement disagree
    about what happened, and neither may silently win.
    """


class BudgetLedgerStore:
    """Reads, and idempotent appends, for the budget ledger."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def entries(self, experiment_id: str) -> tuple[BudgetLedgerEntry, ...]:
        """Every entry of an experiment, oldest first."""
        rows = self._connection.execute(
            "SELECT * FROM budget_ledger WHERE experiment_id = ? ORDER BY created_at, id",
            (experiment_id,),
        ).fetchall()
        return tuple(_entry(row) for row in rows)

    def entry(
        self,
        subject_kind: BudgetSubjectKind,
        subject_id: str,
        dimension: BudgetDimension,
        kind: LedgerEntryKind,
    ) -> BudgetLedgerEntry | None:
        """The one entry of *kind* for a subject on a dimension, if written."""
        row = self._connection.execute(
            "SELECT * FROM budget_ledger WHERE subject_kind = ? AND subject_id = ? "
            "AND dimension = ? AND kind = ?",
            (subject_kind.value, subject_id, dimension.value, kind.value),
        ).fetchone()
        return None if row is None else _entry(row)

    def _append(self, entry: BudgetLedgerEntry) -> BudgetLedgerEntry:
        """Write *entry*, or return the identical one already written.

        Raises:
            LedgerConflictError: If one exists for the same subject,
                dimension and kind with a different amount.
        """
        _require_transaction(self._connection, "a budget ledger entry")
        existing = self.entry(entry.subject_kind, entry.subject_id, entry.dimension, entry.kind)
        if existing is not None:
            if existing.amount != entry.amount or existing.experiment_id != entry.experiment_id:
                raise LedgerConflictError(
                    f"{entry.subject_kind.value} {entry.subject_id} already has a "
                    f"{entry.kind.value} of {existing.amount} on {entry.dimension.value}; "
                    f"settling it again as {entry.amount} would rewrite what happened"
                )
            return existing
        self._connection.execute(
            "INSERT INTO budget_ledger (id, experiment_id, dimension, kind, amount, "
            "subject_kind, subject_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.experiment_id,
                entry.dimension.value,
                entry.kind.value,
                str(entry.amount),
                entry.subject_kind.value,
                entry.subject_id,
                entry.created_at.isoformat(),
            ),
        )
        return entry


def _entry(row: sqlite3.Row) -> BudgetLedgerEntry:
    return BudgetLedgerEntry(
        id=row["id"],
        experiment_id=row["experiment_id"],
        dimension=BudgetDimension(row["dimension"]),
        kind=LedgerEntryKind(row["kind"]),
        amount=Decimal(row["amount"]),
        subject_kind=BudgetSubjectKind(row["subject_kind"]),
        subject_id=row["subject_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
