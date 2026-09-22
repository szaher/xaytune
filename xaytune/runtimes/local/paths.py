"""The on-disk layout of one local workload.

The launcher and the runtime are separate processes that may not overlap in
time: the launcher outlives a controller restart, and a recreated
:class:`~xaytune.runtimes.local.runtime.LocalRuntime` has no ``Popen`` object
for a workload it did not start. Everything they need to agree on therefore
lives in files with fixed names, and this module is the one place those names
are written down.

```text
<root>/registry.db              operations and workloads (durable identity)
<root>/workloads/<id>/plan.json     what was submitted
                     started.json   written by the launcher once the worker is up
                     finished.json  written by the launcher once it has exited
                     events.jsonl   telemetry, one envelope per line
                     stdout.log
                     stderr.log
```

``started.json`` and ``finished.json`` are written with :func:`write_atomic`
and never appended to, so a reader either sees a whole record or no record.
A half-written ``finished.json`` would be read as a workload that ended in a
way nobody can describe, which is worse than one that has not ended yet.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "EVENTS",
    "FINISHED",
    "PLAN",
    "STARTED",
    "STDERR",
    "STDOUT",
    "WorkloadPaths",
    "read_json",
    "write_atomic",
]

PLAN = "plan.json"
STARTED = "started.json"
FINISHED = "finished.json"
EVENTS = "events.jsonl"
STDOUT = "stdout.log"
STDERR = "stderr.log"


class WorkloadPaths:
    """Where one workload's files live."""

    __slots__ = ("directory",)

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def plan(self) -> Path:
        return self.directory / PLAN

    @property
    def started(self) -> Path:
        return self.directory / STARTED

    @property
    def finished(self) -> Path:
        return self.directory / FINISHED

    @property
    def events(self) -> Path:
        return self.directory / EVENTS

    @property
    def stdout(self) -> Path:
        return self.directory / STDOUT

    @property
    def stderr(self) -> Path:
        return self.directory / STDERR


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as JSON so a reader never sees it half-written.

    Into a temporary file in the same directory, then :func:`os.replace`, which
    is atomic within a filesystem. The directory entry is fsynced as well as
    the file: without that, a crash can leave a durable file the directory does
    not yet point at, which is the same as not having written it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the record at *path*, or ``None`` if it is not there yet.

    A file that exists but does not parse is treated as absent rather than
    raised on: the only way to produce one is a crash mid-write, and the
    caller's answer to "not written yet" is already the conservative one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None
