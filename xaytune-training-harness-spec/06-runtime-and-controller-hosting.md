# Runtime Backends and Controller Hosting

## 1. RuntimeBackend

Runtime owns execution.

```python
class RuntimeBackend(Protocol):
    descriptor: PluginDescriptor

    def capabilities(self) -> CapabilityDocument: ...

    async def submit(
        self,
        plan: ResolvedExecutionPlan,
        operation_id: str,
    ) -> RuntimeRef: ...

    async def get_status(
        self,
        runtime_ref: RuntimeRef,
    ) -> RuntimeStatus: ...

    async def watch(
        self,
        runtime_ref: RuntimeRef,
        cursor: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]: ...

    async def cancel(
        self,
        runtime_ref: RuntimeRef,
        operation_id: str,
    ) -> None: ...

    async def get_logs(
        self,
        runtime_ref: RuntimeRef,
    ) -> AsyncIterator[RuntimeLog]: ...
```

Every mutating operation must be idempotent via `operation_id`.

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

```python
class RuntimeRef(BaseModel):
    backend: str
    external_id: str
    namespace: str | None
    metadata: dict[str, Any]
```

The core must not assume Kubernetes identifiers.

## 8. Runtime events

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
