"""One daemon per state database, by kernel lock (ADR-004 §5).

```text
<resolved state database path>.lock     fcntl.flock(LOCK_EX | LOCK_NB)
```

The kernel lock is the authority, held on an open file for the daemon's
whole life. The file's content -- pid, instance id, start time, database --
is written after the lock is taken and is diagnostic only: nothing reads it
to decide ownership, a PID alone never establishes it (ADR-013 AC-10), and
there is no stale-lock cleanup, because a process that dies releases the lock
with it.

It keeps two daemons on one machine from driving one database. It is not
distributed ownership: that, and durable controller identity, are PR-028's
leases.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xaytune.core.errors import XaytuneError

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only off POSIX
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "DaemonAlreadyRunningError",
    "StateDatabaseLock",
    "UnsupportedPlatformError",
    "lock_path",
    "require_locking",
]


class DaemonAlreadyRunningError(XaytuneError):
    """Another process holds the state database's lock.

    Attributes:
        holder: What that process wrote in the lock file, for diagnosis only;
            ``None`` if it could not be read.
    """

    def __init__(self, path: Path, holder: Mapping[str, Any] | None) -> None:
        self.path = path
        self.holder = holder
        detail = f" (held by {dict(holder)})" if holder else ""
        super().__init__(f"another daemon holds {path}{detail}")


class UnsupportedPlatformError(XaytuneError):
    """This platform has no ``fcntl`` advisory locking; the daemon will not run unlocked."""


def require_locking() -> None:
    """Refuse, before anything else is set up, a platform the lock cannot be taken on.

    Raises:
        UnsupportedPlatformError: If ``fcntl`` is unavailable.
    """
    if fcntl is None:
        raise UnsupportedPlatformError(
            "the local daemon needs POSIX fcntl advisory locking to own its state "
            "database; this platform has none, and an unsafe fallback is refused"
        )


def lock_path(state_path: Path | str) -> Path:
    """The lock file for *state_path*: resolved, so aliases of one database share it."""
    return Path(f"{Path(state_path).resolve()}.lock")


class StateDatabaseLock:
    """The exclusive lock on one state database, held from :meth:`acquire` to :meth:`release`."""

    def __init__(self, state_path: Path | str) -> None:
        if str(state_path) == ":memory:":
            raise ValueError("an in-memory database has no file to lock, and no daemon to share")
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self.path = lock_path(state_path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, metadata: Mapping[str, Any]) -> None:
        """Take the lock without waiting, then record *metadata* in the file for diagnosis.

        Raises:
            DaemonAlreadyRunningError: If another open file holds it.
            UnsupportedPlatformError: If ``fcntl`` is unavailable.
        """
        require_locking()
        assert fcntl is not None
        if self._fd is not None:
            raise RuntimeError(f"{self.path} is already held by this lock")
        # Not O_TRUNC: the file is opened before the lock is taken, and
        # truncating then would erase a running daemon's diagnostics.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise DaemonAlreadyRunningError(self.path, self.holder(self.path)) from None
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(dict(metadata), sort_keys=True).encode("utf-8"))
        os.fsync(fd)

    def release(self) -> None:
        """Clear the diagnostics, then release the lock. Idempotent."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            os.ftruncate(fd, 0)
            assert fcntl is not None
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def holder(path: Path) -> dict[str, Any] | None:
        """What the holder of the lock at *path* wrote, if anything readable. Diagnostic only."""
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text) if text else None
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None
