"""Plugin contract for consuming committed facts, never controlling training."""

from typing import Protocol, runtime_checkable

from xaytune.core.capabilities import PluginDescriptor
from xaytune.core.domain.event import DomainEvent


@runtime_checkable
class EventSink(Protocol):
    """At-least-once outbox consumer.

    Implementations deduplicate by DomainEvent.id. The delivery host isolates
    failures and retries; a sink failure must never fail training or roll back
    committed state. External dashboards are projections, never authoritative.
    Existing logging backends can be adapted here without changing core.
    """

    descriptor: PluginDescriptor

    async def consume(self, event: DomainEvent) -> None: ...
