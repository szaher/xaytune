"""The local daemon: a controller process, and the caller's side of it (ADR-004; PR-027-PR-029).

:class:`LocalDaemonControllerServer` is the persistent, foreground controller
process over one SQLite state database, which is also its mailbox::

    python -m xaytune.daemon --state state.db --config myproject.xaytune_config:create_config

:class:`LocalDaemonControllerHost` is the ``ControllerHost`` a caller uses
with it: ``submit()`` and ``attach()`` return an ``ExperimentHandle`` that
reads the durable record, and every mutation -- submit, attach, cancel,
propose, approve, reject -- is a mailbox request the daemon carries out. The
caller runs no controller, and may exit at any point:

```python
async with LocalDaemonControllerHost("state.db") as host:
    handle = await host.submit(spec)         # admitted and issued by the daemon
    result = await handle.wait()             # polls the record
```

:class:`DaemonClient` is the mailbox underneath it. ``xaytune submit``,
``attach``, ``status``, ``watch``, ``events``, ``results``, ``actions``,
``cancel``, ``approve`` and ``reject`` are the same host on the command line.

One daemon per database, by kernel lock, and one controller, by a durable
lease whose epoch every controller write proves (PR-028); a restarted daemon
reconciles every experiment it owns. SIGTERM or SIGINT stops it without
cancelling a workload. See :mod:`xaytune.daemon.server`.
"""

from xaytune.core.domain.controller_request import (
    ControllerRequest,
    ControllerRequestKind,
    ControllerRequestState,
    MisdirectedRequestError,
)
from xaytune.daemon.client import DaemonClient
from xaytune.daemon.config import DaemonConfig, DaemonConfigurationError, load_config
from xaytune.daemon.host import ControllerRequestFailedError, LocalDaemonControllerHost
from xaytune.daemon.lock import (
    DaemonAlreadyRunningError,
    StateDatabaseLock,
    UnsupportedPlatformError,
    lock_path,
)
from xaytune.daemon.server import LocalDaemonControllerServer

__all__ = [
    "ControllerRequest",
    "ControllerRequestFailedError",
    "ControllerRequestKind",
    "ControllerRequestState",
    "DaemonAlreadyRunningError",
    "DaemonClient",
    "DaemonConfig",
    "DaemonConfigurationError",
    "LocalDaemonControllerHost",
    "LocalDaemonControllerServer",
    "MisdirectedRequestError",
    "StateDatabaseLock",
    "UnsupportedPlatformError",
    "load_config",
    "lock_path",
]
