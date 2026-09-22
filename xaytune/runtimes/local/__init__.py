"""The local subprocess runtime.

```text
ResolvedExecutionPlan
      ↓ LocalRuntime.submit_or_get
python -m xaytune.runtimes.local.launcher <workload dir>
      ↓
the worker process
```

The launcher in the middle is not ceremony. It is the parent that reaps the
worker, and therefore the only thing that can record an exit code where a
controller restarted since the submission can still read it.
"""

from __future__ import annotations

from xaytune.runtimes.local.runtime import BACKEND, LocalRuntime, UnsupportedPlanError

__all__ = ["BACKEND", "LocalRuntime", "UnsupportedPlanError"]
