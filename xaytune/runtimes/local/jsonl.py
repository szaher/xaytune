"""Reading an append-only JSONL file without rereading it.

Two files in this backend are written by one process and tailed by another:
the worker's observations, which the launcher wraps into envelopes, and the
launcher's events, which the controller consumes. Both are append-only, and
both were being read by loading the whole file on every poll. That is fine for
the two events a launcher emits on its own and quadratic once a real worker
reports per-step metrics.

**The reader knows bytes and complete lines. It does not know telemetry.** It
has no idea what a generation or a sequence is, which is what lets the same
primitive serve both files and makes its failure modes small enough to test
exhaustively.

```text
byte offset   where this reader stopped
partial tail  bytes after the last newline, not yet a record
```

Both are **process-local optimisation**. Durable replay identity remains
:class:`~xaytune.runtimes.StreamCursor`; a reader built fresh starts at zero
and returns the whole file, which is what a restarted controller relies on.
Persisting a byte offset as protocol state would tie a controller's position to
the exact bytes one writer happened to produce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

__all__ = ["AppendOnlyJsonlReader", "CorruptRecordError"]

T = TypeVar("T")


class CorruptRecordError(Exception):
    """A complete line in an append-only file was not a valid record.

    Raised rather than skipped. A partial write is recoverable -- the rest of
    the line arrives and the record is read -- but a *complete* line that does
    not parse means the writer produced something the reader cannot
    understand, and skipping it would turn a stream with a hole in it into one
    that looks intact. A gap the consumer cannot see is worse than an error it
    can.
    """

    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path} line {line_number} is not a valid record: {reason}")


class AppendOnlyJsonlReader(Generic[T]):
    """Yields records appended since the last call.

    Args:
        path: The file to tail. It need not exist yet; a reader created before
            its writer returns nothing until there is something to return.
        adapter: Validates each complete line into ``T``.
    """

    __slots__ = ("_adapter", "_lines_read", "_offset", "_path", "_tail")

    def __init__(self, path: Path, adapter: TypeAdapter[T]) -> None:
        self._path = path
        self._adapter = adapter
        self._offset = 0
        self._tail = b""
        self._lines_read = 0

    @property
    def offset(self) -> int:
        """Bytes consumed so far. Diagnostic only, never protocol state."""
        return self._offset

    def read_new(self) -> tuple[T, ...]:
        """Return records completed since the previous call.

        Raises:
            CorruptRecordError: If a complete line is not valid JSON, or does
                not validate as ``T``.
        """
        try:
            with self._path.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
        except FileNotFoundError:
            return ()

        if not chunk:
            return ()

        self._offset += len(chunk)

        # Buffered as bytes, never decoded early: a multi-byte character split
        # across two appends would otherwise be decoded as two broken halves.
        buffered = self._tail + chunk
        *complete, self._tail = buffered.split(b"\n")

        records = []
        for raw in complete:
            self._lines_read += 1
            if not raw.strip():
                # A blank line carries no record. Ignored rather than refused,
                # because the alternative makes a writer's trailing newline a
                # fatal error while saying nothing about data integrity.
                continue
            records.append(self._parse(raw, self._lines_read))
        return tuple(records)

    def _parse(self, raw: bytes, line_number: int) -> T:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptRecordError(self._path, line_number, str(exc)) from exc

        try:
            return self._adapter.validate_python(payload)
        except ValidationError as exc:
            raise CorruptRecordError(self._path, line_number, str(exc)) from exc
