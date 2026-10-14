"""Xaytune Studio's read-only API over the control-plane store (UX-01, ADR-023).

```text
xaytune studio-web --state state.db          → http://127.0.0.1:8765/api/v1/meta
```

The API is a view of the durable record and nothing else:

- it opens the state database read-only and never migrates it (§S3);
- it answers ``GET`` and ``HEAD`` only (§S1);
- it reads through the stores the controller and the CLI already use, and
  holds no state of its own (§S2, §S4), so restarting it loses nothing;
- it listens on loopback only, because it has no authentication (§S10).

Requires the ``studio-web`` extra (``fastapi`` and ``uvicorn``).
"""

from __future__ import annotations

from xaytune.api.app import create_app
from xaytune.api.store import StoreUnavailableError, StudioStore

__all__ = ["StoreUnavailableError", "StudioStore", "create_app"]
