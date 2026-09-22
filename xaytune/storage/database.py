"""The control plane's connection setup and write-transaction boundary.

Both live in :mod:`xaytune.core.sqlite` so that a runtime backend can open a
durable registry without importing the control plane. Re-exported here because
this is where the persistence code looks for them, and moving the call sites
would say something about the control plane that is not true.
"""

from __future__ import annotations

from xaytune.core.sqlite import connect, write_transaction

__all__ = ["connect", "write_transaction"]
