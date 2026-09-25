"""Submitting and controlling experiments (ADR-004).

```python
host = EmbeddedControllerHost("state.db")
handle = await host.submit(spec)
result = await handle.wait()
```

A handle is a way of asking the durable record about one experiment; see
:mod:`xaytune.experiment.handle` for what ``wait()`` means while nothing yet
decides what an evaluated candidate becomes.
"""

from xaytune.core.domain.evaluation import EvaluationSpec, EvaluatorSpec
from xaytune.experiment.handle import (
    EvaluationOutcome,
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
    "EvaluationOutcome",
    "EvaluationSpec",
    "EvaluatorSpec",
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
