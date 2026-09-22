# Runtime Backends and Controller Hosting

## 1. RuntimeBackend

Runtime owns execution.

```python
class RuntimeBackend(Protocol):
    descriptor: PluginDescriptor

    def capabilities(self) -> CapabilityDocument: ...

    async def submit_or_get(
        self,
        operation_id: OperationId,
        plan: ResolvedExecutionPlan,
    ) -> RuntimeRef: ...
        # Get-or-create, never create (ADR-013). Re-submitting the same
        # operation_id returns the existing RuntimeRef; it never starts a
        # second workload. operation_id is the FIRST parameter because it is
        # the identity of the operation, not a tag on it. The same method
        # submits an evaluation attempt: the operation's target is typed
        # (ADR-013), so the runtime needs no training-specific knowledge.

    async def lookup_operation(
        self,
        operation_id: OperationId,
    ) -> OperationOutcome | None: ...
        # Answers "did this operation ever take effect?" after a controller
        # restart. None means never received. An adapter that cannot report
        # completed operations declares so in its CapabilityDocument, and the
        # controller escalates instead of resubmitting (ADR-013).

    async def get_status(
        self,
        runtime_ref: RuntimeRef,
    ) -> RuntimeStatus: ...

    def watch(
        self,
        runtime_ref: RuntimeRef,
        cursor: StreamCursor | None = None,
    ) -> AsyncIterator[RuntimeEventEnvelope]: ...
        # `def`, not `async def`: an async generator is already declared with a
        # plain `def` returning AsyncIterator, and `async def` here would mean
        # the caller must await before iterating. Both shapes satisfy this, but
        # only one types correctly for an implementation that yields.
        # Yields canonical envelopes per ADR-014
        # (xaytune.telemetry/v1alpha2), in increasing (generation, sequence)
        # order. cursor is the last position the controller DURABLY RECORDED,
        # not the last it received -- an event received and then lost in a
        # crash must be redelivered. It is StreamCursor(generation, sequence):
        # two integers with defined meaning, never an opaque provider token.
        # Delivery is at-least-once; handlers must be idempotent on
        # (target, stream_generation, sequence). A runtime that cannot
        # replay declares supports_event_replay: false and reconnects are
        # treated as gaps.

    async def cancel(
        self,
        runtime_ref: RuntimeRef,
        operation_id: OperationId,
    ) -> None: ...

    def get_logs(
        self,
        runtime_ref: RuntimeRef,
    ) -> AsyncIterator[RuntimeLog]: ...
```

Every mutating operation must be idempotent via `operation_id`. There is no
plain `submit()`: ADR-013 rejects create semantics, because a controller that
crashes between submitting and persisting the `RuntimeRef` cannot otherwise
tell a lost submission from a running workload, and retrying starts a second
one.

**Ordered delivery.** `watch()` MUST yield events in increasing
`(generation, sequence)` order. The runtime adapter buffers out-of-order
transport delivery until the missing sequence arrives, or until the replay/gap
policy declares it unavailable. Without this guarantee a reordered arrival
(`11, 13, 12`) is indistinguishable from a real gap, and the controller would
raise `EventGapDetected` for an event that is merely late.

**A lost telemetry stream is not a lost attempt.** If the telemetry supervisor
dies and its history cannot be replayed, the controller confirms through
`get_status()` whether the same workload is still executing. If it is, the
`RunAttempt` is unchanged: `stream_generation` advances, the sequence restarts
at 0, the unrecoverable range is recorded as `EventGapDetected`, and the
interval is marked observability-degraded. A new `RunAttempt` is created only
when the runtime actually restarts or replaces the workload — minting one for a
telemetry failure would record an execution retry that never happened
(ADR-014 §1a).

## 2. Initial runtimes

### LocalRuntime

Supports:

- subprocess
- single process
- torchrun
- local multi-GPU
- local controller development
- deterministic integration tests

### RayTrainRuntime

Uses Ray Train/Jobs for execution.

It does not own planning or search.

### TrainingHubRuntime

Submits runtime-neutral execution requirements to Training Hub.

It does not manipulate:

- Pods
- JobSet
- Kueue Workloads

directly.

## 3. Direct Kubeflow integration

A direct Kubeflow runtime is optional/community scope.

Preferred OpenShift AI path:

```text
Xaytune
  ↓
Training Hub
  ↓
Kubeflow Trainer / KubeRay
  ↓
Kueue
```

## 4. ControllerHost

The controller must survive client process death.

```python
class ControllerHost(Protocol):
    async def submit(
        self,
        experiment: ExperimentSpec,
    ) -> ExperimentHandle: ...

    async def attach(
        self,
        experiment_id: ExperimentId,
    ) -> ExperimentHandle: ...
```

Initial hosts:

### EmbeddedControllerHost

For:

- local development
- tests
- notebooks
- synchronous convenience

Lifecycle dies with the process.

### LocalDaemonControllerHost

Persistent process on workstation/server.

Requirements:

- SQLite database
- experiment reconciliation after restart
- PID/socket or local API
- controlled shutdown
- singleton locking per state database

### RemoteControllerHost

Future.

Possible forms:

- Training Hub service integration
- platform controller
- dedicated Xaytune service

Contracts must support it before it exists.

## 5. ExperimentHandle

```python
class ExperimentHandle:
    experiment_id: str

    async def status(self) -> ExperimentStatus: ...

    async def wait(self) -> ExperimentResult: ...

    async def cancel(self) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def events(self) -> AsyncIterator[Event]: ...
```

High-level API:

```python
handle = experiment.submit()

print(handle.status())

# process can terminate here

handle = xaytune.attach(experiment_id)
result = handle.wait()
```

`run()` is convenience:

```python
def run(self):
    return self.submit().wait()
```

## 6. Controller reconciliation

On startup:

1. acquire controller identity
2. load active experiments
3. load active attempts
4. query each runtime
5. reconcile runtime state
6. ingest missing runtime events if possible
7. re-evaluate deadlines/budgets
8. resume control loops

Reconciliation must be idempotent.

## 7. RuntimeRef

Submission is get-or-create, never create (ADR-013):

```python
def submit_or_get(operation_id: OperationId, plan: ResolvedExecutionPlan) -> RuntimeRef
def lookup_operation(operation_id: OperationId) -> OperationOutcome | None
```

A runtime adapter must also declare whether it can report *completed* operations. After a
controller restart an operation in `SENT` may mean the workload is running, was never
received, or already finished -- and an adapter that cannot distinguish the last two
forces the controller to escalate rather than resubmit.

```python
class RuntimeRef(BaseModel):
    backend: str
    external_id: str
    namespace: str | None
    metadata: dict[str, Any]
```

The core must not assume Kubernetes identifiers.

## 8. Runtime events

These are what the *backend* observes about the workload, and they are distinct
from the `RuntimeEventEnvelope` telemetry stream of ADR-014. They stay available through
`get_status()` when telemetry is degraded, and that independence is what makes
reconciliation possible at all.

Normalized runtime events include:

- submitted
- queued
- resources admitted
- started
- worker started
- worker failed
- preempted
- driver failed
- completed
- cancelled
- logs available
- artifact available

Backends translate their native events into this model.
