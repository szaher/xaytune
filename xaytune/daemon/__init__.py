"""The local daemon: a controller process and its mailbox client (ADR-004; PR-027).

:class:`LocalDaemonControllerServer` is the persistent, foreground controller
process over one SQLite state database, which is also its mailbox::

    python -m xaytune.daemon --state state.db --config myproject.xaytune_config:create_config

:class:`DaemonClient` hands it work by committing a request, and reads the
durable record. It is the v1 mailbox API, not a ``ControllerHost``: the
caller-side ``LocalDaemonControllerHost``, whose ``submit()`` and ``attach()``
return an ``ExperimentHandle`` over the same mailbox, is PR-029's:

```python
with DaemonClient("state.db") as client:
    request = client.submit(spec)
```

One daemon per database, by kernel lock; SIGTERM or SIGINT stops it without
cancelling a workload. See :mod:`xaytune.daemon.server`.
"""

from xaytune.core.domain.controller_request import (
    ControllerRequest,
    ControllerRequestKind,
    ControllerRequestState,
)
from xaytune.daemon.client import DaemonClient
from xaytune.daemon.config import DaemonConfig, DaemonConfigurationError, load_config
from xaytune.daemon.lock import (
    DaemonAlreadyRunningError,
    StateDatabaseLock,
    UnsupportedPlatformError,
    lock_path,
)
from xaytune.daemon.server import LocalDaemonControllerServer

__all__ = [
    "ControllerRequest",
    "ControllerRequestKind",
    "ControllerRequestState",
    "DaemonAlreadyRunningError",
    "DaemonClient",
    "DaemonConfig",
    "DaemonConfigurationError",
    "LocalDaemonControllerServer",
    "StateDatabaseLock",
    "UnsupportedPlatformError",
    "load_config",
    "lock_path",
]
