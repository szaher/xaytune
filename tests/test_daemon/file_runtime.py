"""A runtime whose workloads are files, so they outlive the daemon that submitted them.

No model is trained. What is under test is the daemon's ownership of the
control plane: a workload a daemon submitted must survive that daemon's
shutdown or death, be adopted rather than duplicated by the next one, and be
cancelled by nobody. A file per workload makes each of those observable from
the test process:

```text
<root>/workloads/<operation id>.json    {"target", "state", "cancelled", "model"}
<root>/calls.jsonl                      one line per submit / watch / cancel, with the pid
```

A test ends a workload by writing ``succeeded`` into its file
(:func:`finish`). ``fault`` makes ``submit_or_get`` kill its own process --
``kill-before-submit`` before the workload exists (the runtime never received
it), ``kill-after-submit`` after (the response was lost) -- the two crashes
between admission and confirmation.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from xaytune.core.ids import ArtifactId, OperationId
from xaytune.core.refs import ArtifactRef, RuntimeRef
from xaytune.core.telemetry import ArtifactProducedPayload
from xaytune.runtimes import (
    OperationOutcome,
    RuntimeEventEnvelope,
    RuntimeStatus,
    StreamCursor,
    TrainingEventPayload,
)
from xaytune.runtimes.local.runtime import LocalRuntime

_POLL_SECONDS = 0.05


class FileRuntime:
    descriptor = LocalRuntime.descriptor

    def __init__(self, root: str | Path, *, fault: str | None = None) -> None:
        self.root = Path(root)
        (self.root / "workloads").mkdir(parents=True, exist_ok=True)
        self.fault = fault

    def capabilities(self) -> Any:
        return LocalRuntime.capabilities(self)  # type: ignore[arg-type]

    # ---- effects -----------------------------------------------------------

    async def submit_or_get(self, operation_id: OperationId, plan: Any) -> RuntimeRef:
        _log(self.root, "submit", str(operation_id))
        if self.fault == "kill-before-submit":
            os.kill(os.getpid(), signal.SIGKILL)
        if self.fault == "hang-before-submit":
            await asyncio.Event().wait()
        path = workload_path(self.root, str(operation_id))
        if not path.exists():
            _write(
                path,
                {
                    "target": plan.target.id,
                    "state": "running",
                    "cancelled": False,
                    "model": str(ArtifactId.generate()),
                },
            )
        if self.fault == "kill-after-submit":
            os.kill(os.getpid(), signal.SIGKILL)
        return _reference(str(operation_id))

    async def lookup_operation(self, operation_id: OperationId) -> OperationOutcome | None:
        if not workload_path(self.root, str(operation_id)).exists():
            return None
        return OperationOutcome(
            operation_id=operation_id,
            disposition="accepted",
            runtime_ref=_reference(str(operation_id)),
        )

    async def cancel(self, runtime_ref: RuntimeRef, operation_id: OperationId) -> None:
        _log(self.root, "cancel", runtime_ref.external_id)
        path = workload_path(self.root, runtime_ref.external_id)
        workload = _read(path)
        _write(path, {**workload, "state": "cancelled", "cancelled": True})

    # ---- observation ---------------------------------------------------------

    async def get_status(self, runtime_ref: RuntimeRef) -> RuntimeStatus:
        state = _read(workload_path(self.root, runtime_ref.external_id))["state"]
        if state == "running":
            return RuntimeStatus(state="running")
        return RuntimeStatus(state=state, exit_code=0 if state == "succeeded" else 1)

    async def watch(
        self, runtime_ref: RuntimeRef, cursor: StreamCursor | None = None
    ) -> AsyncIterator[RuntimeEventEnvelope]:
        _log(self.root, "watch", runtime_ref.external_id)
        path = workload_path(self.root, runtime_ref.external_id)
        while (workload := _read(path))["state"] == "running":
            await asyncio.sleep(_POLL_SECONDS)
        if workload["state"] == "succeeded" and (cursor is None or cursor.sequence < 0):
            from xaytune.core.domain.operation import RuntimeOperationTarget

            target = RuntimeOperationTarget(kind="training-attempt", id=workload["target"])
            model = ArtifactRef(
                id=ArtifactId(workload["model"]), kind="model", uri=str(path.with_suffix(""))
            )
            yield RuntimeEventEnvelope(
                event_id=f"{workload['target']}-0",
                sequence=0,
                target=target,
                payload=TrainingEventPayload(data=ArtifactProducedPayload(artifact_ref=model)),
            )

    async def get_logs(self, runtime_ref: RuntimeRef) -> AsyncIterator[Any]:
        return
        yield

    def close(self) -> None:
        pass


# ---- what a test reads and does ------------------------------------------------


def workload_path(root: Path, operation_id: str) -> Path:
    return root / "workloads" / f"{operation_id}.json"


def workloads(root: Path) -> dict[str, dict[str, Any]]:
    """Every workload ever started, by operation id."""
    directory = root / "workloads"
    if not directory.exists():
        return {}
    return {path.stem: _read(path) for path in sorted(directory.glob("*.json"))}


def calls(root: Path, kind: str | None = None) -> list[dict[str, Any]]:
    path = root / "calls.jsonl"
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [entry for entry in entries if kind is None or entry["call"] == kind]


def finish(root: Path, operation_id: str) -> None:
    """The workload ends successfully, as far as the runtime can tell."""
    path = workload_path(root, operation_id)
    _write(path, {**_read(path), "state": "succeeded"})


def _reference(operation_id: str) -> RuntimeRef:
    return RuntimeRef(backend="file", external_id=operation_id)


def _log(root: Path, call: str, subject: str) -> None:
    with (root / "calls.jsonl").open("a") as log:
        log.write(json.dumps({"call": call, "subject": subject, "pid": os.getpid()}) + "\n")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)
