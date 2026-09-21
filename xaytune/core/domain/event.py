"""Durable domain events and the outbox that publishes them.

The event log is **durable provenance, not an event-sourcing origin**
(ADR-005 §1). Aggregates are read from their own tables; events exist so that
history is answerable, projections can be rebuilt and checked, and external
sinks can be fed without the control plane depending on them.

That distinction decides the shape below. An event carries the
``aggregate_revision`` it produced, so the pairing in ADR-005 §10 -- an
aggregate's revision equals the revision recorded by its latest event -- is
checkable rather than assumed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from xaytune.core.clock import utc_now
from xaytune.core.ids import EventId
from xaytune.core.immutable import FrozenDict, FrozenDomainModel
from xaytune.core.refs import Actor

__all__ = ["DomainEvent", "OutboxRecord", "OutboxState"]

OutboxState = Literal["pending", "sending", "sent", "failed"]


class DomainEvent(FrozenDomainModel):
    """One recorded fact about an aggregate transition.

    Written in the same transaction as the transition it describes (ADR-005 §3).
    There is no path that writes one without the other, which is what makes the
    log trustworthy as provenance.

    Attributes:
        sequence: Total order within the database, assigned by the repository at
            commit. ``None`` before insert, because the value is the database's
            to choose -- a caller-chosen sequence could collide or leave holes.
            Consumers order on this rather than on ``occurred_at``: clocks skew,
            and under a distributed launcher there are several of them.
        aggregate_revision: The revision this transition *produced*, not the one
            it started from. Pairs each event with exactly one aggregate state.
    """

    id: EventId
    sequence: int | None = None

    aggregate_type: str
    aggregate_id: str
    aggregate_revision: int

    event_type: str
    schema_version: str = "1"

    experiment_id: str

    occurred_at: datetime = Field(default_factory=utc_now)
    actor: Actor

    payload: FrozenDict = Field(default_factory=FrozenDict)


class OutboxRecord(FrozenDomainModel):
    """A pending delivery of an event to an external sink.

    Written with its event, delivered later and separately: core state never
    depends on delivery succeeding, so a sink being down degrades observability
    rather than stalling the control plane (ADR-005 §10.6).

    **The outbox publishes events. It never submits or cancels workloads.**
    Delivery is at-least-once, which is correct for publishing a fact and
    catastrophic for starting a GPU job -- a redelivery would be a second
    workload. External effects go through the operation journal, which is
    get-or-create (ADR-013).
    """

    id: str
    event_id: EventId
    destination: str

    state: OutboxState = "pending"
    attempts: int = Field(default=0, ge=0)
    next_attempt_at: datetime | None = None

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
