"""The local daemon controller host (ADR-004; PR-027).

A persistent, foreground controller process over one SQLite state database,
which is also its mailbox::

    python -m xaytune.daemon --state state.db --config myproject.xaytune_config:create_config

A client hands it work by committing a request, and reads the durable record:

```python
with DaemonClient("state.db") as client:
    request = client.submit(spec)
```

One daemon per database, by kernel lock; SIGTERM or SIGINT stops it without
cancelling a workload. See :mod:`xaytune.daemon.host`.
"""

from xaytune.core.domain.controller_request import (
    ControllerRequest,
    ControllerRequestKind,
    ControllerRequestState,
)
from xaytune.daemon.client import DaemonClient
from xaytune.daemon.config import DaemonConfig, DaemonConfigurationError, load_config
from xaytune.daemon.host import LocalDaemonControllerHost
from xaytune.daemon.lock import (
    DaemonAlreadyRunningError,
    StateDatabaseLock,
    UnsupportedPlatformError,
    lock_path,
)

__all__ = [
    "ControllerRequest",
    "ControllerRequestKind",
    "ControllerRequestState",
    "DaemonAlreadyRunningError",
    "DaemonClient",
    "DaemonConfig",
    "DaemonConfigurationError",
    "LocalDaemonControllerHost",
    "StateDatabaseLock",
    "UnsupportedPlatformError",
    "load_config",
    "lock_path",
]
