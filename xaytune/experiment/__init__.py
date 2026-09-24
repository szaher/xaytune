"""Submitting and controlling experiments (ADR-004).

```python
host = EmbeddedControllerHost("state.db")
handle = await host.submit(spec)
result = await handle.wait()
```

A handle is a way of asking the durable record about one experiment; see
:mod:`xaytune.experiment.handle` for what ``wait()`` means while evaluation
does not yet exist.
"""

from xaytune.experiment.handle import (
    ExperimentHandle,
    ExperimentResult,
    NodeOutcome,
    RunOutcome,
)
from xaytune.experiment.host import (
    ControllerNotRunningError,
    EmbeddedControllerHost,
    ImplementationMismatchError,
    ReconciliationEscalatedError,
    UnknownImplementationError,
)
from xaytune.experiment.spec import CompilerSpec, ExperimentSpec, RuntimeSpec

__all__ = [
    "CompilerSpec",
    "ControllerNotRunningError",
    "EmbeddedControllerHost",
    "ExperimentHandle",
    "ExperimentResult",
    "ExperimentSpec",
    "ImplementationMismatchError",
    "NodeOutcome",
    "ReconciliationEscalatedError",
    "RunOutcome",
    "RuntimeSpec",
    "UnknownImplementationError",
]
